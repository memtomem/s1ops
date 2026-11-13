"""A choice or score answer whose top probability is tied is escalated at every threshold, so tuning,
`s1 policy curve` and the web simulator leave it out of every accepted error and count it only in
`n` (#123)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records, select
from s1ops.contracts.schemas import PolicyEntry, load_policies
from s1ops.demo import _categorical_policies, generate_workspace
from s1ops.eval.metrics import routed_confidence, top_label, top_tied
from s1ops.improve.calibrate import Fit, save_fits
from s1ops.improve.policy import clopper_pearson_upper, select_threshold, threshold_curve
from s1ops.runtime.gate import DecisionGate
from s1ops.runtime.routing import routed_metrics
from s1ops.web.app import create_app

CHOICE, SCORE = "q-000000000000000c", "q-000000000000000s"
COST_S1, COST_SYS2 = 0.0001, 0.01


def _row(n: int, probs: dict, label, primitive: str = "choice", **extra) -> dict:
    top = max(probs.values())
    first = next(k for k, v in probs.items() if v == top)
    row = {"schema": "s1-record/1", "dataset": "d", "item_id": f"i{n}", "primitive": primitive,
           "question_id": CHOICE if primitive == "choice" else SCORE, "model_requested": "m", "model_served": "m",
           "probabilities": probs, "confidence": top, "label": label, **extra}
    if primitive == "choice":
        row["choice"] = first
    else:
        row["score"] = sum(int(k) * v for k, v in probs.items())
    return row


def _rows(*groups: tuple, primitive: str = "choice") -> list[dict]:
    """`(count, probs, label)` groups as rows with distinct item ids."""
    flat = [(probs, label) for count, probs, label in groups for _ in range(count)]
    return [_row(n, probs, label, primitive) for n, (probs, label) in enumerate(flat)]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _tune(records: str, out: Path, *extra: str) -> PolicyEntry:
    assert cli_main(["policy", "tune", records, "-o", str(out), "--error-budget", "0.1", *extra]) == 0
    (policy,) = load_policies(out)
    return policy


def _metrics(policy: PolicyEntry) -> tuple:
    m = policy.metrics_on_split
    return m["accepted"], m["errors"], m["accepted_error"]


def _group(records: str, use_calib: bool = False):
    (group,) = select(read_records(records, use_calib=use_calib), allow_model_mismatch=True).groups
    return group


def _gate_agrees(policy: PolicyEntry, rows: list[dict]) -> None:
    """Each row's escalation equals the rule's; `predicted` is compared only on a unique top, since a
    tied row's answer depends on class order. Totals equal `metrics_on_split`."""
    gate = DecisionGate([policy])
    fn = gate.evaluate_choice if policy.primitive == "choice" else gate.evaluate_score
    accepted = errors = 0
    for row in rows:
        probs = row["probabilities"]
        d = fn(question_id=policy.question_id, model_served="m", probs_raw=probs)
        top = max(probs.values())
        tied = sum(v == top for v in probs.values()) > 1
        takes = not tied and policy.threshold is not None and top >= policy.threshold
        assert d.escalated == (not takes), row
        if not tied:
            answer = next(k for k, v in probs.items() if v == top)
            assert d.predicted == (answer if policy.primitive == "choice" else int(answer))
        if not d.escalated:
            accepted += 1
            errors += d.predicted != row["label"]
    assert (accepted, errors) == _metrics(policy)[:2]


def _curve(records: str, capsys, *extra: str) -> list[dict]:
    capsys.readouterr()  # drop what an earlier `tune` printed
    assert cli_main(["policy", "curve", records, "--cost-s1", str(COST_S1), "--cost-sys2", str(COST_SYS2), *extra]) == 0
    (group,) = json.loads(capsys.readouterr().out)["groups"]
    return group["curve"]


def _assert_parity(records: str, capsys, use_calib: bool = False) -> list[dict]:
    """Every curve point has the counts an `error_budget` policy at its threshold routes, over `n`
    counting the ties."""
    curve = _curve(records, capsys, *(("--use-calib",) if use_calib else ()))
    group = _group(records, use_calib)
    for point in curve:
        policy = PolicyEntry(question_id=group.question_id, model_served="m", primitive=group.primitive, mode="error_budget",
                             split="eval", n=group.n, threshold=point["threshold"], error_budget=0.1, calibrated=use_calib)
        m = routed_metrics(policy, group)
        assert (point["accepted"], point["errors"]) == (m["accepted"], m["errors"]), point
        assert point["coverage"] == point["accepted"] / group.n
    return curve


# ---- tuning ----


def test_the_issue_fixture_has_no_threshold_that_meets_the_budget(tmp_path, capsys) -> None:
    # Without the tie, nine at 0.9 with one wrong is 1 of 9 > 0.1. Counted as a right answer, the
    # tie made it 1 of 10 at 0.5, a policy that routes 1 of 9 wrong.
    rows = _rows((8, {"a": 0.9, "b": 0.1}, "a"), (1, {"a": 0.9, "b": 0.1}, "b"), (1, {"a": 0.5, "b": 0.5}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    policy = _tune(records, tmp_path / "p.json")
    assert policy.threshold is None
    assert _metrics(policy) == (0, 0, None)
    _gate_agrees(policy, rows)


def test_a_threshold_is_chosen_over_the_untied_answers(tmp_path, capsys) -> None:
    # `top_label` takes 0.5 (1 of 10, the tie counted right); routed at 0.5 that is 1 of 9.
    rows = _rows((8, {"a": 0.9, "b": 0.1}, "a"), (1, {"a": 0.8, "b": 0.2}, "b"), (1, {"a": 0.5, "b": 0.5}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    group = _group(records)
    assert select_threshold(*top_label(group.P, group.y), budget=0.1) == 0.5
    policy = _tune(records, tmp_path / "p.json")
    assert policy.threshold == 0.9
    assert _metrics(policy) == (8, 0, 0.0)
    _gate_agrees(policy, rows)


def test_a_tie_labelled_another_class_no_longer_counts_as_an_error(tmp_path, capsys) -> None:
    # Two ties at 0.45 labelled b, where argmax answers a: counted, 2 of 2 and then 2 of 11 break
    # the budget, so no threshold. Left out, the nine right answers at 0.4 take 0.4.
    rows = _rows((2, {"a": 0.45, "b": 0.45, "c": 0.1}, "b"), (9, {"a": 0.4, "b": 0.3, "c": 0.3}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    group = _group(records)
    assert select_threshold(*top_label(group.P, group.y), budget=0.1) is None
    policy = _tune(records, tmp_path / "p.json")
    assert policy.threshold == 0.4
    assert _metrics(policy) == (9, 0, 0.0)
    _gate_agrees(policy, rows)


def test_a_score_tie_is_left_out(tmp_path, capsys) -> None:
    rows = _rows((8, {"0": 0.1, "1": 0.8, "2": 0.1}, 1), (1, {"0": 0.7, "1": 0.2, "2": 0.1}, 2),
                 (1, {"0": 0.4, "1": 0.4, "2": 0.2}, 0), primitive="score")
    records = _write(tmp_path / "r.jsonl", rows)
    policy = _tune(records, tmp_path / "p.json")
    assert (policy.primitive, policy.threshold) == ("score", 0.8)
    assert _metrics(policy) == (8, 0, 0.0)
    _gate_agrees(policy, rows)


def test_ties_at_different_tops_in_one_group(tmp_path, capsys) -> None:
    rows = _rows((8, {"a": 0.9, "b": 0.05, "c": 0.05}, "a"), (1, {"a": 0.5, "b": 0.5, "c": 0.0}, "a"),
                 (1, {"a": 0.4, "b": 0.4, "c": 0.2}, "b"), (1, {"a": 0.3, "b": 0.6, "c": 0.1}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    group = _group(records)
    assert top_tied(group.P).tolist() == [False] * 8 + [True, True, False]
    conf, correct, escalated = routed_confidence("choice", group.P, group.y)
    assert (len(conf), escalated) == (9, 2)
    policy = _tune(records, tmp_path / "p.json")
    assert policy.threshold == 0.9 and _metrics(policy) == (8, 0, 0.0)
    _gate_agrees(policy, rows)
    _assert_parity(records, capsys)


def test_every_answer_tied_leaves_nothing_to_accept(tmp_path, capsys) -> None:
    rows = _rows((3, {"a": 0.5, "b": 0.5}, "a"), (2, {"a": 0.5, "b": 0.5}, "b"))
    records = _write(tmp_path / "r.jsonl", rows)
    policy = _tune(records, tmp_path / "p.json")
    assert policy.threshold is None and _metrics(policy) == (0, 0, None)
    assert _curve(records, capsys) == []
    r = TestClient(create_app(records_path=records)).get("/api/simulate-policy", params={"use_calib": False})
    assert r.status_code == 200 and r.json()["tuned_threshold"]["threshold"] is None and r.json()["threshold_curve"] == []


def test_the_cp_bound_is_not_diluted_by_a_tie(tmp_path, capsys) -> None:
    # 0 of 28 has a 95 % upper bound just over 0.1; counting the tie as a right answer, 0 of 29 is under.
    rows = _rows((28, {"a": 0.9, "b": 0.1}, "a"), (1, {"a": 0.5, "b": 0.5}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    assert clopper_pearson_upper(0, 28, 0.05) > 0.1 > clopper_pearson_upper(0, 29, 0.05)
    group = _group(records)
    assert select_threshold(*top_label(group.P, group.y), budget=0.1, use_cp_upper=True) == 0.5
    policy = _tune(records, tmp_path / "p.json", "--use-cp-upper")
    assert policy.threshold is None and _metrics(policy) == (0, 0, None)


def test_a_tie_does_not_count_toward_min_accepted() -> None:
    P, y = np.array([[0.9, 0.1], [0.5, 0.5]]), np.array([0, 0])
    assert select_threshold(*top_label(P, y), budget=0.1, min_accepted=2) == 0.5
    conf, correct, escalated = routed_confidence("choice", P, y)
    assert escalated == 1 and select_threshold(conf, correct, budget=0.1, min_accepted=2) is None


def test_a_tie_made_by_calibration_is_left_out_only_when_tuning_on_it(tmp_path, capsys) -> None:
    # Raw 0.6 / 0.4 is a unique top; calibrated to 0.5 / 0.5 it is tied.
    rows = [_row(n, {"a": 0.9, "b": 0.1}, "a", calib={"probabilities": {"a": 0.9, "b": 0.1}}) for n in range(8)]
    rows += [_row(8, {"a": 0.8, "b": 0.2}, "b", calib={"probabilities": {"a": 0.8, "b": 0.2}})]
    rows += [_row(9, {"a": 0.6, "b": 0.4}, "a", calib={"probabilities": {"a": 0.5, "b": 0.5}})]
    records = _write(tmp_path / "r.jsonl", rows)
    assert _tune(records, tmp_path / "raw.json").threshold == 0.6  # 1 of 10 counted at 0.6
    calibrated = _tune(records, tmp_path / "cal.json", "--use-calib")
    assert (calibrated.threshold, _metrics(calibrated)) == (0.9, (8, 0, 0.0))
    _assert_parity(records, capsys, use_calib=True)
    _assert_parity(records, capsys, use_calib=False)


def test_a_tie_made_by_a_fit_is_escalated_by_the_gate_too(tmp_path, capsys) -> None:
    # Temperature 10 maps raw 0.5000000000000001 / 0.4999999999999999 (a unique top) to exactly
    # 0.5 / 0.5. `s1 apply` writes it, tune leaves it out, and the gate, calibrating the live raw
    # answer with the same fit, escalates it.
    fit = Fit(question_id=CHOICE, model_served="m", primitive="choice", method="temperature", split="cal", n=2,
              classes=["a", "b"], params={"temperature": 10.0}, nll_before=0.7, nll_after=0.6,
              fitted_at="2026-11-13T00:00:00+00:00", source_sha256="", warnings=[], items=[["cal", "c1"], ["cal", "c2"]])
    fits = tmp_path / "fits.json"
    save_fits([fit], fits)
    rows = _rows((8, {"a": 0.99, "b": 0.01}, "a"), (1, {"a": 0.95, "b": 0.05}, "b"),
                 (1, {"a": 0.5000000000000001, "b": 0.4999999999999999}, "a"))
    raw = _write(tmp_path / "r.jsonl", rows)
    applied = tmp_path / "applied.jsonl"
    assert cli_main(["apply", raw, "--fit", str(fits), "-o", str(applied)]) == 0
    capsys.readouterr()
    last = json.loads(applied.read_text(encoding="utf-8").splitlines()[-1])["calib"]["probabilities"]
    assert last["a"] == last["b"] == 0.5
    group = _group(str(applied), use_calib=True)
    assert top_tied(group.P).tolist() == [False] * 9 + [True]
    policy = _tune(str(applied), tmp_path / "p.json", "--use-calib")
    assert policy.fit_sha256 == fit.fit_sha256
    accepted, errors, _ = _metrics(policy)
    assert accepted == 8 and errors == 0  # the wrong one at 0.95 breaks 0.1 (1 of 9); the tie is out
    gate = DecisionGate([policy], [fit])
    decisions = [gate.evaluate_choice(question_id=CHOICE, model_served="m", probs_raw=r["probabilities"]) for r in rows]
    assert [d.escalated for d in decisions] == [False] * 8 + [True, True]
    _assert_parity(str(applied), capsys, use_calib=True)


# ---- the curve ----


def test_the_curve_counts_a_tie_only_in_n(tmp_path, capsys) -> None:
    rows = _rows((8, {"a": 0.9, "b": 0.1}, "a"), (1, {"a": 0.8, "b": 0.2}, "b"), (1, {"a": 0.5, "b": 0.5}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    curve = _assert_parity(records, capsys)
    assert [(p["threshold"], p["accepted"], p["errors"], p["coverage"]) for p in curve] == [(0.9, 8, 0, 0.8), (0.8, 9, 1, 0.9)]
    # At 0.8 one item, the tie, is delegated: `threshold_curve`'s formula with the default error
    # cost 0.10 and delegate accuracy 0.98.
    last = curve[-1]
    assert last["delegate_per_1k_usd"] == (1 / 10) * COST_SYS2 * 1000.0
    assert last["error_per_1k_usd"] == ((1 + 1 * (1.0 - 0.98)) / 10) * 0.10 * 1000.0
    assert last["cp_upper_95"] == clopper_pearson_upper(1, 9, 0.05)


def test_always_escalated_counts_in_n_only() -> None:
    conf, correct = [0.9, 0.9, 0.8], [True, True, False]
    assert threshold_curve(conf, correct) == threshold_curve(conf, correct, always_escalated=0)
    # The default output, pinned: default costs c_s1 0.00004, c_delegate 0.004, c_error 0.10,
    # delegate accuracy 0.98.
    expected = [
        {"threshold": 0.9, "accepted": 2, "coverage": 2 / 3, "errors": 0, "error_rate": 0.0, "cp_upper_95": clopper_pearson_upper(0, 2, 0.05),
         "s1_per_1k_usd": 0.04, "delegate_per_1k_usd": 4 / 3, "error_per_1k_usd": 0.02 / 3 * 100},
        {"threshold": 0.8, "accepted": 3, "coverage": 1.0, "errors": 1, "error_rate": 1 / 3, "cp_upper_95": clopper_pearson_upper(1, 3, 0.05),
         "s1_per_1k_usd": 0.04, "delegate_per_1k_usd": 0.0, "error_per_1k_usd": 100 / 3},
    ]
    got = threshold_curve(conf, correct)
    assert len(got) == 2
    for point, want in zip(got, expected):
        assert {k: point[k] for k in ("threshold", "accepted", "errors")} == {k: want[k] for k in ("threshold", "accepted", "errors")}
        for k in ("coverage", "error_rate", "cp_upper_95", "s1_per_1k_usd", "delegate_per_1k_usd", "error_per_1k_usd"):
            assert point[k] == pytest.approx(want[k], rel=1e-12), k
        assert point["cost_per_1k_usd"] == pytest.approx(want["s1_per_1k_usd"] + want["delegate_per_1k_usd"] + want["error_per_1k_usd"], rel=1e-12)
    plain, with_two = threshold_curve(conf, correct), threshold_curve(conf, correct, always_escalated=2)
    for a, b in zip(plain, with_two):
        assert (a["threshold"], a["accepted"], a["errors"], a["error_rate"], a["cp_upper_95"]) == \
               (b["threshold"], b["accepted"], b["errors"], b["error_rate"], b["cp_upper_95"])
        assert b["coverage"] == b["accepted"] / 5
    assert threshold_curve([], [], always_escalated=3) == []
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError, match="always_escalated"):
            threshold_curve(conf, correct, always_escalated=bad)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def test_the_demo_curves_have_parity(ws, tmp_path, capsys) -> None:
    for line_prim in ("choice", "score"):
        only = tmp_path / f"{line_prim}.jsonl"
        only.write_text("".join(line + "\n" for line in Path(ws["records_categorical"]).read_text(encoding="utf-8").splitlines()
                                if json.loads(line)["primitive"] == line_prim), encoding="utf-8")
        _assert_parity(str(only), capsys)
        _assert_parity(str(only), capsys, use_calib=True)


# ---- the web simulator ----


def test_the_simulator_agrees_with_tune_and_curve(tmp_path, capsys) -> None:
    rows = _rows((8, {"a": 0.9, "b": 0.1}, "a"), (1, {"a": 0.8, "b": 0.2}, "b"), (1, {"a": 0.5, "b": 0.5}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    policy = _tune(records, tmp_path / "p.json")
    capsys.readouterr()
    r = TestClient(create_app(records_path=records)).get(
        "/api/simulate-policy", params={"error_budget": 0.1, "use_calib": False, "cost_s1": COST_S1, "cost_sys2": COST_SYS2})
    assert r.status_code == 200
    assert r.json()["tuned_threshold"]["threshold"] == policy.threshold == 0.9
    assert r.json()["threshold_curve"] == _curve(records, capsys)


# ---- the demo ----


def test_the_demo_policies_leave_ties_out(tmp_path) -> None:
    rows = _rows((8, {"a": 0.9, "b": 0.1}, "a"), (1, {"a": 0.8, "b": 0.2}, "b"), (1, {"a": 0.5, "b": 0.5}, "a"))
    group = _group(_write(tmp_path / "r.jsonl", rows))
    fit = Fit(question_id=CHOICE, model_served="m", primitive="choice", method="temperature", split="cal", n=2,
              classes=["a", "b"], params={"temperature": 1.0}, nll_before=0.7, nll_after=0.6,
              fitted_at="2026-11-13T00:00:00+00:00", source_sha256="", warnings=[], items=[["cal", "c1"], ["cal", "c2"]])
    (policy,) = _categorical_policies([group], [fit])
    assert policy.threshold == 0.9 and _metrics(policy) == (8, 0, 0.0)
