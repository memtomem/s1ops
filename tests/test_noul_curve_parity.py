"""`s1 policy curve` and the web simulator count a noul answer as the routing rule does: a curve
point is what a noul `error_budget` policy at its threshold routes, `p == 0.5` included (#120)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records, select
from s1ops.contracts.schemas import PolicyEntry, load_policies
from s1ops.demo import generate_workspace
from s1ops.improve.policy import clopper_pearson_upper
from s1ops.runtime.routing import routed_metrics
from s1ops.web.app import create_app

QID = "q-0000000000000000"
COST_S1, COST_SYS2 = 0.0001, 0.01


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def _write(path: Path, rows: list[tuple]) -> str:
    """`(p, label)` or `(p, label, calib_p)` rows as noul records."""
    with path.open("w", encoding="utf-8") as fh:
        for n, (p, label, *calib) in enumerate(rows):
            row = {"schema": "s1-record/1", "dataset": "d", "item_id": f"i{n}", "question_id": QID, "primitive": "noul",
                   "model_requested": "m", "model_served": "m", "p": p, "label": label}
            if calib:
                row["calib"] = {"p": calib[0]}
            fh.write(json.dumps(row) + "\n")
    return str(path)


def _curve(records: str, capsys, *extra: str) -> list[dict]:
    assert cli_main(["policy", "curve", records, "--cost-s1", str(COST_S1), "--cost-sys2", str(COST_SYS2), *extra]) == 0
    (group,) = [g for g in json.loads(capsys.readouterr().out)["groups"] if g["primitive"] == "noul"]
    return group["curve"]


def _noul_group(records: str, use_calib: bool):
    (group,) = [g for g in select(read_records(records, use_calib=use_calib), allow_model_mismatch=True).groups if g.primitive == "noul"]
    return group


def _routed(group, threshold: float | None, calibrated: bool) -> tuple[int, int]:
    policy = PolicyEntry(question_id=group.question_id, model_served=group.model_served, primitive="noul", mode="error_budget",
                         split="eval", n=group.n, threshold=threshold, error_budget=0.1, calibrated=calibrated)
    m = routed_metrics(policy, group)
    return m["accepted"], m["errors"]


def _assert_parity(records: str, capsys, calibrated: bool = False) -> list[dict]:
    """Every curve point has the counts a policy at its threshold routes."""
    curve = _curve(records, capsys, *(("--use-calib",) if calibrated else ()))
    group = _noul_group(records, calibrated)
    assert curve
    for point in curve:
        assert (point["accepted"], point["errors"]) == _routed(group, point["threshold"], calibrated), point
        assert point["coverage"] == point["accepted"] / group.n
        assert point["error_rate"] == point["errors"] / point["accepted"]
    return curve


def _web(records: str, **params) -> dict:
    r = TestClient(create_app(records_path=records)).get("/api/simulate-policy", params={"cost_s1": COST_S1, "cost_sys2": COST_SYS2, **params})
    assert r.status_code == 200, r.text
    return r.json()


# ---- the curve ----


def test_a_tie_at_one_half_is_counted_as_routed(tmp_path, capsys) -> None:
    # Routing answers true at 0.5, and both labels there are false: 2 of 10 wrong, not 0.
    records = _write(tmp_path / "r.jsonl", [(0.9, True)] * 8 + [(0.5, False)] * 2)
    curve = _assert_parity(records, capsys)
    point = curve[-1]
    assert (point["threshold"], point["accepted"], point["errors"], point["error_rate"]) == (0.5, 10, 2, 0.2)
    assert point["cp_upper_95"] == clopper_pearson_upper(2, 10, 0.05)
    # `threshold_curve`'s cost model at the CLI's costs, the default error cost 0.10 and delegate
    # accuracy 0.98; nothing is delegated at 0.5.
    assert point["s1_per_1k_usd"] == COST_S1 * 1000.0
    assert point["delegate_per_1k_usd"] == 0.0
    assert point["error_per_1k_usd"] == (2 / 10) * 0.10 * 1000.0
    assert point["cost_per_1k_usd"] == point["s1_per_1k_usd"] + point["error_per_1k_usd"]


def test_both_labels_at_one_half_and_p_beside_one_minus_p(tmp_path, capsys) -> None:
    # 0.8 and 0.2 share confidence 0.8 and answer opposite ways. At 0.5 one label is true and one
    # false, so either reading of the tie counts one error: this pins parity, not the tie.
    assert 1 - 0.2 == 0.8
    rows = [(0.95, True)] * 6 + [(0.8, True), (0.8, False), (0.2, False), (0.2, True)] + [(0.5, True), (0.5, False)]
    records = _write(tmp_path / "r.jsonl", rows)
    curve = _assert_parity(records, capsys)
    assert [(p["threshold"], p["accepted"], p["errors"]) for p in curve] == [(0.95, 6, 0), (0.8, 10, 2), (0.5, 12, 3)]


def test_calibrated_values_at_one_half_are_counted_as_routed(tmp_path, capsys) -> None:
    # Raw 0.7 and 0.3 are far from 0.5; calibrated to exactly 0.5, routing answers true, so both
    # false labels are errors. Raw, 0.3 answers false and is right: one error.
    rows = [(0.95, True, 0.9)] * 8 + [(0.7, False, 0.5), (0.3, False, 0.5), (0.7, True, 0.5)]
    records = _write(tmp_path / "r.jsonl", rows)
    calibrated = _assert_parity(records, capsys, calibrated=True)
    assert [(p["threshold"], p["accepted"], p["errors"]) for p in calibrated] == [(0.9, 8, 0), (0.5, 11, 2)]
    raw = _assert_parity(records, capsys, calibrated=False)
    assert [(p["threshold"], p["accepted"], p["errors"]) for p in raw] == [(0.95, 8, 0), (0.7, 11, 1)]


def test_the_demo_curve_has_parity(ws, capsys) -> None:
    _assert_parity(ws["records"], capsys)
    _assert_parity(ws["records"], capsys, calibrated=True)


def test_choice_and_score_curves_do_not_change(ws, capsys) -> None:
    from s1ops.eval.metrics import top_label
    from s1ops.improve.policy import threshold_curve

    assert cli_main(["policy", "curve", ws["records_categorical"], "--cost-s1", str(COST_S1), "--cost-sys2", str(COST_SYS2)]) == 0
    groups = json.loads(capsys.readouterr().out)["groups"]
    by_key = {g.key: g for g in select(read_records(ws["records_categorical"]), allow_model_mismatch=True).groups}
    assert {g["primitive"] for g in groups} == {"choice", "score"}
    for g in groups:
        grp = by_key[(g["question_id"], g["model_served"], g["primitive"])]
        assert g["curve"] == threshold_curve(*top_label(grp.P, grp.y), c_s1=COST_S1, c_delegate=COST_SYS2)


# ---- the web simulator ----


@pytest.mark.parametrize(
    "rows, threshold",
    [
        pytest.param([(0.9, True)] * 8 + [(0.5, False)] * 2, 0.9, id="false-at-one-half-raises-it"),
        pytest.param([(0.9, True)] * 8 + [(0.5, True)] * 2, 0.5, id="true-at-one-half-lowers-it"),
    ],
)
def test_the_simulator_threshold_is_the_tuned_one(tmp_path, capsys, rows, threshold) -> None:
    records = _write(tmp_path / "r.jsonl", rows)
    out = tmp_path / "p.json"
    assert cli_main(["policy", "tune", records, "-o", str(out), "--noul-routing", "threshold", "--error-budget", "0.1"]) == 0
    capsys.readouterr()
    (policy,) = load_policies(out)
    web = _web(records, error_budget=0.1, use_calib=False)
    assert web["tuned_threshold"]["threshold"] == policy.threshold == threshold
    assert web["threshold_curve"] == _curve(records, capsys)


@pytest.mark.parametrize("calibrated", [False, True], ids=["raw", "calibrated"])
def test_the_demo_simulator_agrees_with_tune_and_curve(ws, tmp_path, capsys, calibrated) -> None:
    extra = ("--use-calib",) if calibrated else ()
    out = tmp_path / "p.json"
    assert cli_main(["policy", "tune", ws["records"], "-o", str(out), "--noul-routing", "threshold", "--error-budget", "0.1", *extra]) == 0
    capsys.readouterr()
    (policy,) = load_policies(out)
    web = _web(ws["records"], error_budget=0.1, use_calib=calibrated)
    assert web["tuned_threshold"]["threshold"] == policy.threshold
    assert web["threshold_curve"] == _curve(ws["records"], capsys, *extra)


def test_the_simulator_counts_calibrated_one_half_as_routed(tmp_path, capsys) -> None:
    rows = [(0.95, True, 0.9)] * 8 + [(0.7, False, 0.5), (0.3, False, 0.5), (0.7, True, 0.5)]
    records = _write(tmp_path / "r.jsonl", rows)
    web = _web(records, error_budget=0.15, use_calib=True)
    # All three calibrate to 0.5. Routed, they answer true: 2 of 11 wrong (0.18) breaks 0.15, so the
    # threshold stays at 0.9. Counted by argmax (false at 0.5), 1 of 11 (0.09) would take 0.5.
    assert web["threshold_curve"][-1]["errors"] == 2
    assert web["tuned_threshold"]["threshold"] == 0.9
    assert web["threshold_curve"] == _curve(records, capsys, "--use-calib")
