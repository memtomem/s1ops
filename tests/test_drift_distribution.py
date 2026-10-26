"""`s1 drift` compares one distribution on both sides: raw by default, calibrated on request (#20).

Falling back to raw per record mixed calibrated reference records with raw current ones, so PSI
measured the calibration map rather than drift.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, read_records
from s1ops.contracts.schemas import PolicyEntry
from s1ops.demo import generate_workspace
from s1ops.monitor.drift import DriftInputError, audit_drift, jensen_shannon, psi
from s1ops.web.app import create_app

OPTIONS = {"choice": ("A", "B", "C"), "score": ("0", "1", "2")}


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _row(primitive: str, item: int, raw, calib, *, qid: str = "q", served: str | None = "m") -> dict:
    row = {
        "schema": "s1-record/1", "item_id": f"i{item}", "question_id": qid, "primitive": primitive,
        "model_requested": "m", "model_served": served,
    }
    if served is None:
        return row
    if primitive == "noul":
        row["p"] = raw
        if calib is not None:
            row["calib"] = {"p": calib}
        return row
    top = max(raw, key=raw.get)
    row.update({"choice" if primitive == "choice" else "score": top if primitive == "choice" else int(top),
                "confidence": raw[top], "probabilities": raw})
    if calib is not None:
        row["calib"] = {"probabilities": calib}
    return row


def _dist(rng: random.Random, keys: tuple[str, ...], lean: float) -> dict[str, float]:
    w = [rng.random() + (lean if i == 0 else 0.0) for i in range(len(keys))]
    total = sum(w)
    vals = [round(x / total, 4) for x in w[:-1]]
    return dict(zip(keys, [*vals, round(1.0 - sum(vals), 4)]))


def _sharpen(dist: dict[str, float]) -> dict[str, float]:
    sq = {k: v * v for k, v in dist.items()}
    total = sum(sq.values())
    keys = list(sq)
    vals = [round(sq[k] / total, 4) for k in keys[:-1]]
    return dict(zip(keys, [*vals, round(1.0 - sum(vals), 4)]))


def _rows(primitive: str, seed: int, lean: float, n: int = 60) -> list[dict]:
    rng = random.Random(f"{primitive}:{seed}")
    rows = []
    for i in range(n):
        if primitive == "noul":
            p = round(min(0.97, max(0.03, rng.betavariate(2 + lean, 2))), 4)
            rows.append(_row(primitive, i, p, round(p * p, 4)))
        else:
            raw = _dist(rng, OPTIONS[primitive], lean)
            rows.append(_row(primitive, i, raw, _sharpen(raw)))
    return rows


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _distribution(row: dict, calibrated: bool) -> list[float]:
    if row["primitive"] == "noul":
        p = row["calib"]["p"] if calibrated else row["p"]
        return [1.0 - p, p]
    dist = row["calib"]["probabilities"] if calibrated else row["probabilities"]
    total = math.fsum(dist.values())
    keys = OPTIONS[row["primitive"]] if row["primitive"] == "choice" else sorted(dist, key=int)
    return [dist[k] / total for k in keys]


def _entropy(d: list[float]) -> float:
    return -sum(p * math.log2(p) for p in d if p > 0)


def _escalation_rate(rows: list[dict], pol: PolicyEntry) -> float:
    def escalated(d: list[float]) -> bool:
        top = sorted(d, reverse=True)
        if pol.primitive == "noul":
            return max(top) < pol.threshold
        return top[0] < pol.threshold or top[0] == top[1]

    esc = [escalated(_distribution(r, pol.calibrated)) for r in rows]
    return round(sum(esc) / len(esc), 6)


def _expected(ref: list[dict], cur: list[dict], calibrated: bool) -> dict:
    def score(d: list[float], primitive: str) -> float:
        return d[1] if primitive == "noul" else max(d)

    rd = [_distribution(r, calibrated) for r in ref]
    cd = [_distribution(r, calibrated) for r in cur]
    rs = [score(d, r["primitive"]) for d, r in zip(rd, ref)]
    cs = [score(d, r["primitive"]) for d, r in zip(cd, cur)]

    def quantiles(s: list[float]) -> dict:
        q10, q50, q90 = np.quantile(np.asarray(s), [0.10, 0.50, 0.90])
        return {"p10": round(float(q10), 6), "p50": round(float(q50), 6), "p90": round(float(q90), 6),
                "mean": round(float(np.asarray(s).mean()), 6)}

    return {
        "psi": round(psi(rs, cs), 6),
        "js_divergence": round(jensen_shannon(rs, cs), 6),
        "quantiles_ref": quantiles(rs),
        "quantiles_cur": quantiles(cs),
        "mean_entropy_ref": round(sum(_entropy(d) for d in rd) / len(rd), 6),
        "mean_entropy_cur": round(sum(_entropy(d) for d in cd) / len(cd), 6),
    }


@pytest.mark.parametrize("primitive", ["noul", "choice", "score"])
@pytest.mark.parametrize("policy_calibrated", [False, True], ids=["raw-policy", "calibrated-policy"])
def test_each_mode_compares_its_own_distribution_on_both_sides(tmp_path: Path, primitive: str, policy_calibrated: bool, capsys) -> None:
    ref, cur = _rows(primitive, 1, 0.0), _rows(primitive, 2, 1.5)
    ref_path, cur_path = _write(tmp_path / "ref.jsonl", ref), _write(tmp_path / "cur.jsonl", cur)
    pol = PolicyEntry(question_id="q", model_served="m", primitive=primitive, mode="error_budget", split="eval",
                      n=1, threshold=0.75 if primitive == "noul" else 0.6, calibrated=policy_calibrated)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "s1-policy/1", "policies": [pol.to_dict()]}))
    expected_escalation = (_escalation_rate(ref, pol), _escalation_rate(cur, pol))
    # The policy's own distribution decides; the other one gives different rates here.
    other = dataclasses.replace(pol, calibrated=not policy_calibrated)
    assert (_escalation_rate(ref, other), _escalation_rate(cur, other)) != expected_escalation
    reports = {}
    for mode, flags in (("raw", []), ("calibrated", ["--calibrated"])):
        assert cli_main(["drift", ref_path, cur_path, "--policy", str(policy), *flags]) == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["distribution"] == mode
        (q,) = rep["questions"]
        assert {k: q[k] for k in _expected(ref, cur, False)} == _expected(ref, cur, mode == "calibrated")
        # The comparison mode does not change what the policy routes.
        assert (q["escalation_rate_ref"], q["escalation_rate_cur"]) == expected_escalation
        reports[mode] = q
    # The fixture is built so that every compared metric differs between the two modes.
    for key in _expected(ref, cur, False):
        assert reports["raw"][key] != reports["calibrated"][key], key


def test_the_demo_compares_raw_probabilities_by_default(ws, capsys) -> None:
    args = ["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"]]
    assert cli_main(args) == 0
    default = json.loads(capsys.readouterr().out)
    assert cli_main([*args, "--raw-only"]) == 0
    assert json.loads(capsys.readouterr().out) == default
    assert default["distribution"] == "raw"
    (q,) = default["questions"]
    # Measured on main ef8953c: 4.17284 mixed calibrated reference with partly raw canary records.
    assert (q["psi"], q["mean_entropy_ref"], q["mean_entropy_cur"]) == (6.649894, 0.367971, 0.751814)


def test_records_parsed_either_way_give_the_report_paths_give(ws) -> None:
    by_path = audit_drift(ws["records"], ws["records_canary"])
    for use_calib in (False, True):
        refs = read_records(ws["records"], use_calib=use_calib)
        curs = read_records(ws["records_canary"], use_calib=use_calib)
        assert audit_drift(refs, curs) == by_path
    assert audit_drift(read_records(ws["records"], use_calib=True), read_records(ws["records_canary"])) == by_path


@pytest.mark.parametrize("side", ["reference", "current"])
def test_calibrated_refuses_a_record_without_calib_even_in_an_unshared_question(tmp_path: Path, side: str) -> None:
    shared = _rows("noul", 1, 0.0, n=4)
    lone = _row("noul", 9, 0.7, None, qid="only-here")
    rows = {"reference": [*shared, lone], "current": list(shared)}
    if side == "current":
        rows = {"reference": list(shared), "current": [*shared, lone]}
    ref = _write(tmp_path / "ref.jsonl", rows["reference"])
    cur = _write(tmp_path / "cur.jsonl", rows["current"])
    # Raw mode compares the shared question only and does not need calib.
    assert [q["question_id"] for q in audit_drift(ref, cur)["questions"]] == ["q"]
    with pytest.raises(DriftInputError) as exc:
        audit_drift(ref, cur, use_calib=True)
    assert (exc.value.side, exc.value.line) == (side, 5)
    assert str(exc.value).startswith(f"{side} records: line 5: a calibrated comparison")


@pytest.mark.parametrize("calib", [{}, {"method": "temperature"}, {"probabilities": {"A": 0.5, "B": 0.5}}],
                         ids=["empty", "no-values", "wrong-key"])
def test_calibrated_refuses_a_calib_without_the_primitives_values(tmp_path: Path, calib: dict) -> None:
    rows = _rows("noul", 1, 0.0, n=3)
    rows[1] = {**rows[1], "calib": calib}
    path = _write(tmp_path / "r.jsonl", rows)
    with pytest.raises(DriftInputError) as exc:
        audit_drift(path, path, use_calib=True)
    assert (exc.value.side, exc.value.line) == ("reference", 2)


def test_calibrated_does_not_need_calib_on_failed_records(tmp_path: Path, capsys) -> None:
    rows = [*_rows("noul", 1, 0.0, n=4), _row("noul", 9, None, None, served=None)]
    path = _write(tmp_path / "r.jsonl", rows)
    assert cli_main(["drift", path, path, "--calibrated"]) == 0
    (q,) = json.loads(capsys.readouterr().out)["questions"]
    assert (q["n_ref"], q["failure_rate_ref"]) == (5, 0.2)


def test_a_malformed_stored_calib_is_refused_in_raw_mode_from_records_too(tmp_path: Path) -> None:
    rows = _rows("noul", 1, 0.0, n=3)
    rows[1] = {**rows[1], "calib": {"p": "nan"}}
    path = _write(tmp_path / "r.jsonl", rows)
    good = _write(tmp_path / "good.jsonl", _rows("noul", 1, 0.0, n=3))
    records = read_records(path)  # use_calib=False does not read calib
    with pytest.raises(RecordError, match=r"^line 2: calib\.p") as exc:
        audit_drift(read_records(good), records)
    assert exc.value.line == 2


def test_calibrated_and_raw_only_cannot_be_combined(ws, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["drift", ws["records"], ws["records"], "--calibrated", "--raw-only"])
    assert exc.value.code == 1
    assert capsys.readouterr().err.splitlines()[-1] == "s1: argument --raw-only: not allowed with argument --calibrated"


def test_web_drift_is_the_cli_default_report(ws, tmp_path: Path, capsys) -> None:
    client = TestClient(create_app(workspace_dir=Path(ws["records"]).parent))
    assert cli_main(["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"]]) == 0
    assert client.get("/api/drift").json() == json.loads(capsys.readouterr().out)

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "records.jsonl").write_text(Path(ws["records"]).read_text())
    rep = TestClient(create_app(workspace_dir=empty)).get("/api/drift").json()
    assert (rep["distribution"], rep["questions"]) == ("raw", [])


def test_web_drift_names_the_file_with_a_malformed_calib(ws, tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "records.jsonl").write_text(Path(ws["records"]).read_text())
    rows = [json.loads(line) for line in Path(ws["records_canary"]).read_text().splitlines()]
    n = next(i for i, r in enumerate(rows) if "calib" in r)
    rows[n]["calib"]["p"] = "nan"
    _write(bad / "records_canary.jsonl", rows)
    resp = TestClient(create_app(workspace_dir=bad), raise_server_exceptions=False).get("/api/drift")
    assert resp.status_code == 400
    assert resp.json()["detail"] == f"{bad / 'records_canary.jsonl'}: line {n + 1}: calib.p must be a number, got 'nan'"
