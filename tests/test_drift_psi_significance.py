"""`s1 drift` raises a PSI status only when a permutation test puts the PSI outside sampling noise (#72).

PSI of two samples from one distribution grows like (bins - 1)(1/n_ref + 1/n_cur): about 0.33 at 54
records per side, over the CRITICAL threshold. Measured on 0.6.0 with live answers (six questions,
~54 per side, dev and test pooled and split at random 20 times): 73 of 120 comparisons CRITICAL.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.monitor import drift
from s1ops.monitor.drift import (
    PSI_ALPHA_CRITICAL,
    PSI_ALPHA_WARN,
    PSI_CRITICAL,
    PSI_WARN,
    _bin_counts,
    _psi_from_counts,
    audit_drift,
    psi,
    psi_test,
)
from s1ops.web.app import create_app


def _row(item: int, p: float | None, *, qid: str = "q", served: str = "m") -> dict:
    row = {
        "schema": "s1-record/1", "item_id": f"i{item}", "question_id": qid, "primitive": "noul",
        "model_requested": served, "model_served": served if p is not None else None,
    }
    if p is not None:
        row["p"] = p
    return row


def _write(path: Path, scores, *, qid: str = "q", served: str = "m", start: int = 0) -> Path:
    with path.open("a", encoding="utf-8") as f:
        for i, p in enumerate(scores, start):
            f.write(json.dumps(_row(i, p, qid=qid, served=served)) + "\n")
    return path


def _sample(seed, n: int, a: float = 2.0, b: float = 5.0, power: float = 1.0) -> list[float]:
    rng = random.Random(str(seed))
    return [round(min(1.0, rng.betavariate(a, b) ** power), 4) for _ in range(n)]


def _pair(tmp_path: Path, ref, cur, **kw) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    r, c = tmp_path / "ref.jsonl", tmp_path / "cur.jsonl"
    r.unlink(missing_ok=True)
    c.unlink(missing_ok=True)
    return _write(r, ref, **kw), _write(c, cur, **kw)


def _status(tmp_path: Path, ref, cur, **kw) -> dict:
    r, c = _pair(tmp_path, ref, cur)
    return audit_drift(r, c, **kw)["questions"][0]


def test_same_distribution_samples_no_longer_alert(tmp_path: Path) -> None:
    """The regression: same-distribution pairs at 50 per side alert by PSI alone most of the time,
    and almost never under the test."""
    old, new = Counter(), Counter()
    for seed in range(40):
        ref, cur = _sample(("ref", seed), 50), _sample(("cur", seed), 50)
        old[_status(tmp_path / "a", ref, cur, psi_resamples=0)["status"]] += 1
        new[_status(tmp_path / "b", ref, cur)["status"]] += 1
    assert old["CRITICAL"] >= 20, old
    assert new["CRITICAL"] <= 2 and new["OK"] >= 34, new


def test_a_strong_shift_is_critical_and_fails_the_cli(tmp_path: Path) -> None:
    r, c = _pair(tmp_path, _sample("r", 60), _sample("c", 60, a=5.0, b=2.0))
    q = audit_drift(r, c)["questions"][0]
    assert q["status"] == "CRITICAL" and q["psi_p_value"] < PSI_ALPHA_CRITICAL
    assert q["model_drift_detected"] is False and q["failure_rate_cur"] == 0.0
    assert "p=" in q["alerts"][0] and "n_ref=60, n_cur=60" in q["alerts"][0]
    assert cli_main(["drift", str(r), str(c), "--fail-on-critical"]) == 4


def test_noise_over_the_threshold_is_a_note_and_passes_the_cli(tmp_path: Path) -> None:
    for seed in range(200):
        ref, cur = _sample(("ref", seed), 40), _sample(("cur", seed), 40)
        q = _status(tmp_path / "s", ref, cur)
        if q["psi"] >= PSI_CRITICAL and q["status"] == "OK":
            break
    else:
        pytest.fail("no same-distribution pair with PSI >= 0.25 in 200 seeds")
    assert q["psi_p_value"] >= PSI_ALPHA_WARN and q["alerts"] == []
    assert len(q["notes"]) == 1 and "did not reach significance" in q["notes"][0]
    r, c = tmp_path / "s" / "ref.jsonl", tmp_path / "s" / "cur.jsonl"
    assert cli_main(["drift", str(r), str(c), "--fail-on-critical"]) == 0
    assert _status(tmp_path / "old", ref, cur, psi_resamples=0)["status"] == "CRITICAL"


def test_critical_psi_significant_only_at_the_warn_alpha_is_warn(tmp_path: Path) -> None:
    for power, seed in itertools.product((0.85, 0.8, 0.75, 0.7, 0.65), range(80)):
        q = _status(tmp_path / "w", _sample(("ref", seed), 50), _sample(("cur", seed), 50, power=power))
        if q["psi"] >= PSI_CRITICAL and PSI_ALPHA_CRITICAL <= q["psi_p_value"] < PSI_ALPHA_WARN:
            break
    else:
        pytest.fail("no PSI >= 0.25 with 0.01 <= p < 0.05 found")
    assert q["status"] == "WARN"
    assert len(q["alerts"]) == 1 and "significant at 0.05 but not at 0.01" in q["alerts"][0]
    assert q["notes"] == []


@pytest.mark.parametrize(
    ("p", "status", "has_note"),
    [(0.0099, "CRITICAL", False), (0.01, "WARN", False), (0.0499, "WARN", False), (0.05, "OK", True)],
)
def test_alpha_is_strict(tmp_path: Path, monkeypatch, p, status, has_note) -> None:
    monkeypatch.setattr(drift, "psi_test", lambda *a, **k: (0.3, p))
    q = _status(tmp_path, [0.1] * 5, [0.9] * 5)
    assert q["status"] == status and bool(q["notes"]) is has_note


@pytest.mark.parametrize(
    ("psi_full", "status"),
    [(0.2499996, "CRITICAL"), (0.2499994, "WARN"), (0.0999996, "WARN"), (0.0999994, "OK")],
)
def test_thresholds_read_the_reported_rounded_psi(tmp_path: Path, monkeypatch, psi_full, status) -> None:
    """The report shows PSI rounded to 6 decimals, and the thresholds read that number, as before #72."""
    monkeypatch.setattr(drift, "psi_test", lambda *a, **k: (psi_full, 0.001))
    q = _status(tmp_path, [0.1] * 5, [0.9] * 5)
    assert q["psi"] == round(psi_full, 6) and q["status"] == status


def test_p_counts_ties_within_tolerance_and_adds_one(monkeypatch) -> None:
    observed = 0.3  # under 1, so the tolerance 1e-12 * max(1, observed) is 1e-12
    resampled = np.array([
        observed,              # equal: counts
        observed - 0.5e-12,    # inside the tolerance: counts
        observed - 2e-12,      # outside it: does not count
        observed + 1.0,        # larger: counts
        observed - 0.1,        # smaller: does not count
    ])
    real = drift._psi_from_counts
    monkeypatch.setattr(drift, "_psi_from_counts", lambda r, nr, c, nc: resampled if np.ndim(r) == 2 else real(r, nr, c, nc) * 0 + observed)
    psi_obs, p = psi_test([0.1, 0.2], [0.8, 0.9], resamples=5, seed=[0, 3])
    assert psi_obs == observed and p == (1 + 3) / (5 + 1)


def _old_psi(ref_scores, cur_scores, *, bins: int = 10) -> float:
    """`psi` as it was before #72, kept to prove the refactor changed no bit."""
    r = np.asarray(ref_scores, dtype=float)
    c = np.asarray(cur_scores, dtype=float)
    if len(r) == 0 or len(c) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    r_counts, _ = np.histogram(r, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)
    r_prop = np.clip(r_counts / len(r), 1e-4, 1.0)
    c_prop = np.clip(c_counts / len(c), 1e-4, 1.0)
    return float(np.sum((c_prop - r_prop) * np.log(c_prop / r_prop)))


def test_psi_is_bit_identical_to_the_old_formula() -> None:
    rng = random.Random("old-psi")
    edges = [0.0, 0.1, 0.2, 0.5, 0.9, 1.0, 0.0999999, 0.1000001]
    for trial in range(300):
        n_ref, n_cur = rng.randint(1, 80), rng.randint(1, 300)
        pool = [rng.random() for _ in range(40)] + edges
        ref = [rng.choice(pool) for _ in range(n_ref)]
        cur = [round(rng.random(), 4) for _ in range(n_cur)]
        bins = rng.choice((3, 5, 10, 17, 20))
        assert psi(ref, cur, bins=bins) == _old_psi(ref, cur, bins=bins), trial
    assert psi([], [0.5]) == _old_psi([], [0.5]) == 0.0
    assert psi([0.5], []) == _old_psi([0.5], []) == 0.0


def test_an_empty_answered_side_has_no_p_and_keeps_the_failure_alert(tmp_path: Path) -> None:
    q = _status(tmp_path, _sample("r", 30), [None] * 30)
    assert q["psi_p_value"] is None and q["psi"] == 0.0
    assert q["n_cur"] == 30 and q["n_cur_answered"] == 0
    assert q["status"] == "CRITICAL" and any("failure/abstain" in a for a in q["alerts"])
    assert not any("PSI" in a for a in q["alerts"]) and q["notes"] == []


def test_answered_counts_exclude_failures(tmp_path: Path) -> None:
    q = _status(tmp_path, _sample("r", 40) + [None] * 2, _sample("c", 40) + [None] * 2)
    assert (q["n_ref"], q["n_ref_answered"], q["n_cur"], q["n_cur_answered"]) == (42, 40, 42, 40)


@pytest.mark.parametrize(
    ("ref", "cur"),
    [([0.2], [0.9]), ([0.4] * 5, _sample("u", 500)), (_sample("u", 500), [0.4] * 5), ([0.31] * 20, [0.37] * 20)],
)
def test_tiny_unequal_and_one_bin_samples(tmp_path: Path, ref, cur) -> None:
    q = _status(tmp_path, ref, cur)
    assert 0.0 < q["psi_p_value"] <= 1.0
    if ref == [0.31] * 20:  # one bin on both sides: PSI 0, every resample ties it
        assert q["psi"] == 0.0 and q["psi_p_value"] == 1.0 and q["status"] == "OK"


def test_one_by_one_cannot_alert(tmp_path: Path) -> None:
    # two items, two splits: the exact p is 1, and no Monte Carlo estimate can fall under alpha
    q = _status(tmp_path, [0.05], [0.95])
    assert q["psi"] >= PSI_CRITICAL and q["status"] == "OK" and q["psi_p_value"] > 0.5


def test_custom_bins(tmp_path: Path) -> None:
    r, c = _pair(tmp_path, _sample("r", 60), _sample("c", 60, a=5.0, b=2.0))
    q = audit_drift(r, c, bins=5)["questions"][0]
    assert q["status"] == "CRITICAL" and q["psi"] == round(psi(_sample("r", 60), _sample("c", 60, a=5.0, b=2.0), bins=5), 6)


@pytest.mark.parametrize("bad", [True, -1, 1, 999, 1.5, "1000"])
def test_resamples_are_validated(tmp_path: Path, bad) -> None:
    r, c = _pair(tmp_path, [0.1], [0.2])
    with pytest.raises(ValueError, match="psi_resamples"):
        audit_drift(r, c, psi_resamples=bad)


def test_zero_resamples_restores_the_threshold_rule(tmp_path: Path) -> None:
    ref, cur = _sample("r", 40), _sample("c", 40)
    r, c = _pair(tmp_path, ref, cur)
    rep = audit_drift(r, c, psi_resamples=0)
    q = rep["questions"][0]
    assert rep["psi_test"] == {"method": "none", "resamples": 0}
    assert q["psi_p_value"] is None and q["notes"] == []
    expected = "CRITICAL" if q["psi"] >= PSI_CRITICAL else "WARN" if q["psi"] >= PSI_WARN else "OK"
    assert q["status"] == expected


def test_report_carries_the_test_settings(tmp_path: Path) -> None:
    r, c = _pair(tmp_path, [0.1], [0.2])
    assert audit_drift(r, c)["psi_test"] == {
        "method": "permutation-by-bin-counts", "resamples": 1000, "seed": 0,
        "alpha_warn": 0.05, "alpha_critical": 0.01,
    }


def test_p_is_deterministic_order_free_and_per_question(tmp_path: Path) -> None:
    ref, cur = _sample("r", 45), _sample("c", 45, power=0.8)
    first = _status(tmp_path / "a", ref, cur)["psi_p_value"]
    assert _status(tmp_path / "b", ref, cur)["psi_p_value"] == first
    rng = random.Random(7)
    shuffled_ref, shuffled_cur = ref[:], cur[:]
    rng.shuffle(shuffled_ref)
    rng.shuffle(shuffled_cur)
    assert _status(tmp_path / "c", shuffled_ref, shuffled_cur)["psi_p_value"] == first
    r, c = _pair(tmp_path / "d", ref, cur)
    _write(r, _sample("o", 30), qid="other", start=1000)
    _write(c, _sample("p", 30), qid="other", start=1000)
    by_q = {q["question_id"]: q for q in audit_drift(r, c)["questions"]}
    assert by_q["q"]["psi_p_value"] == first


@pytest.mark.parametrize("resamples", [0, 1000])
def test_a_question_id_with_a_lone_surrogate_still_drifts(tmp_path: Path, resamples) -> None:
    """The parser accepts a JSON-escaped lone surrogate; seeding the question's generator must too."""
    line = '{"schema":"s1-record/1","item_id":"i1","question_id":"q\\ud800","primitive":"noul","model_requested":"m","model_served":"m","p":0.3}\n'
    r, c = tmp_path / "r.jsonl", tmp_path / "c.jsonl"
    r.write_text(line, encoding="utf-8")
    c.write_text(line, encoding="utf-8")
    q = audit_drift(r, c, psi_resamples=resamples)["questions"][0]
    assert q["question_id"] == "q\ud800" and q["status"] == "OK"


def test_ordinary_question_ids_keep_their_seed() -> None:
    import hashlib

    assert drift._psi_seed("q-27f3ce0906ba1e1f") == [0, int(hashlib.sha256(b"q-27f3ce0906ba1e1f").hexdigest()[:16], 16)]


def test_a_model_change_is_critical_whatever_the_psi(tmp_path: Path) -> None:
    r, c = tmp_path / "r.jsonl", tmp_path / "c.jsonl"
    _write(r, _sample("r", 40), served="m1")
    _write(c, _sample("c", 40), served="m2")
    q = audit_drift(r, c)["questions"][0]
    assert q["status"] == "CRITICAL" and q["model_drift_detected"] is True


def test_sampler_matches_exhaustive_enumeration() -> None:
    """Independent oracle: every indexed split of a small pooled sample, with repeated values, an
    internal bin edge, 0 and 1."""
    ref = [0.0, 0.0, 0.1, 0.1, 0.35, 1.0]
    cur = [1.0, 0.55, 0.0999, 0.1]
    pooled = ref + cur
    n_ref, n = len(ref), len(pooled)
    observed = psi(ref, cur)
    splits = list(itertools.combinations(range(n), n_ref))
    counts = Counter()
    exceed = 0
    for left in splits:
        r = [pooled[i] for i in left]
        c = [pooled[i] for i in range(n) if i not in left]
        rc, cc = _bin_counts(r, 10), _bin_counts(c, 10)
        assert float(_psi_from_counts(rc, len(r), cc, len(c))) == psi(r, c)
        counts[tuple(rc)] += 1
        exceed += psi(r, c) >= observed - 1e-12 * max(1.0, observed)
    totals = _bin_counts(pooled, 10)
    for rc, k in counts.items():
        weight = math.prod(math.comb(int(t), int(x)) for t, x in zip(totals, rc)) / math.comb(n, n_ref)
        assert k / len(splits) == pytest.approx(weight, abs=1e-12)
    exact = exceed / len(splits)
    _, sampled = psi_test(ref, cur, resamples=40_000, seed=[0, 1])
    assert abs(sampled - exact) < 4 * math.sqrt(exact * (1 - exact) / 40_000) + 1 / 40_001


def test_web_drift_reports_p_notes_and_settings(tmp_path: Path) -> None:
    for seed in range(200):
        ref, cur = _sample(("ref", seed), 40), _sample(("cur", seed), 40)
        if _status(tmp_path / "probe", ref, cur)["notes"]:
            break
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws / "records.jsonl", ref)
    _write(ws / "records_canary.jsonl", cur)
    rep = TestClient(create_app(workspace_dir=ws)).get("/api/drift").json()
    q = rep["questions"][0]
    assert rep["psi_test"]["method"] == "permutation-by-bin-counts"
    assert q["psi_p_value"] is not None and q["notes"] and q["status"] == "OK"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert TestClient(create_app(workspace_dir=empty)).get("/api/drift").json()["psi_test"]["resamples"] == 1000


def test_template_renders_p_and_unbadged_notes() -> None:
    html = (Path(drift.__file__).parent.parent / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert "q.psi_p_value" in html and "n_ref_answered" in html
    assert "(q.notes || []).map(n => `<div class=\"sub\" style=\"margin:4px 0;\">${esc(n)}</div>`)" in html


@pytest.mark.parametrize("n", [10, 2_000, 200_000])
def test_resampling_work_does_not_grow_with_records(monkeypatch, n) -> None:
    """The cost claim, checked by structure rather than wall-clock: whatever the number of records,
    the test makes one hypergeometric draw of shape (resamples, bins) over the pooled bin counts and
    computes PSI on that (resamples, bins) array; records are only binned once per side. Measured at
    200,000 records end to end: 1.86 s without the test, 1.79 s with it."""
    draws, psi_shapes = [], []
    real_rng, real_psi = np.random.default_rng, drift._psi_from_counts

    class Spy:
        def __init__(self, seed):
            self._g = real_rng(seed)

        def multivariate_hypergeometric(self, colors, nsample, size):
            draws.append((np.shape(colors), int(np.sum(colors)), nsample, size))
            return self._g.multivariate_hypergeometric(colors, nsample, size=size)

    def spy_psi(r, nr, c, nc):
        psi_shapes.append(np.shape(r))
        return real_psi(r, nr, c, nc)

    scores = np.random.default_rng(3).random(2 * n).tolist()  # before the patch: drift.np is numpy itself
    monkeypatch.setattr(drift.np.random, "default_rng", Spy)
    monkeypatch.setattr(drift, "_psi_from_counts", spy_psi)
    psi_test(scores[:n], scores[n:], resamples=1000, bins=10, seed=[0, 2])
    assert draws == [((10,), 2 * n, n, 1000)]
    assert psi_shapes == [(10,), (1000, 10)]
