"""`s1 policy tune --noul-routing threshold`: a noul group routed by the `--error-budget` threshold,
so the budget bounds `accepted_error` on the split, where the default F1 band does not (#111)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import load_policies
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import Fit, save_fits
from s1ops.improve.policy import clopper_pearson_upper
from s1ops.runtime.gate import DecisionGate

QID = "q-0000000000000000"


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def _row(n: int, p: float, label: bool, *, calib_p: float | None = None, qid: str = QID, model: str = "m") -> dict:
    row = {"schema": "s1-record/1", "dataset": "d", "item_id": f"i{n}", "question_id": qid, "primitive": "noul",
           "model_requested": model, "model_served": model, "p": p, "label": label}
    if calib_p is not None:
        row["calib"] = {"p": calib_p}
    return row


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _rows(*groups: tuple[int, float, bool]) -> list[dict]:
    """`(count, p, label)` groups as rows with distinct item ids."""
    flat = [(p, label) for count, p, label in groups for _ in range(count)]
    return [_row(n, p, label) for n, (p, label) in enumerate(flat)]


def _tune(records: str, out: Path, *extra: str) -> int:
    return cli_main(["policy", "tune", records, "-o", str(out), "--noul-routing", "threshold", *extra])


def _metrics(policy) -> tuple:
    m = policy.metrics_on_split
    return m["accepted"], m["errors"], m["accepted_error"]


def _error(capsys) -> str:
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("s1: ") and err.count("\n") == 1
    return err.rstrip("\n")


def _gate_agrees(policy, rows: list[dict], fits: list | None = None) -> None:
    """The gate decides each row, from its raw `p`, as the threshold rule does on the value it routes
    (`max(q, 1 - q) >= threshold` accepts, answering `q >= 0.5`), and its totals are the counted ones."""
    gate = DecisionGate([policy], fits)
    accepted = errors = 0
    for row in rows:
        d = gate.evaluate_noul(question_id=policy.question_id, model_served=policy.model_served, p_raw=row["p"])
        q = d.p_effective
        assert d.p_calibrated == (q if policy.calibrated else None)
        takes = policy.threshold is not None and max(q, 1 - q) >= policy.threshold
        assert (d.escalated, d.predicted) == (not takes, q >= 0.5)
        assert d.action == ("ESCALATE" if not takes else "ALLOW" if q >= 0.5 else "BLOCK")
        if not d.escalated:
            accepted += 1
            errors += d.predicted != row["label"]
    assert (accepted, errors) == _metrics(policy)[:2]


# ---- the bound ----


@pytest.mark.parametrize("calibrated", [False, True], ids=["raw", "calibrated"])
def test_the_demo_threshold_meets_the_budget_the_band_breaks(ws, tmp_path, capsys, calibrated) -> None:
    extra = ("--use-calib",) if calibrated else ()
    band, threshold = tmp_path / "band.json", tmp_path / "threshold.json"
    assert cli_main(["policy", "tune", ws["records"], "--error-budget", "0.1", "-o", str(band), *extra]) == 0
    assert _tune(ws["records"], threshold, "--error-budget", "0.1", *extra) == 0
    (b,) = load_policies(band)
    (t,) = load_policies(threshold)
    assert b.metrics_on_split["accepted_error"] > 0.1
    assert (t.mode, t.band_lo, t.band_hi, t.calibrated) == ("error_budget", None, None, calibrated)
    assert t.metrics_on_split["selection"] == "error_budget"
    accepted, errors, rate = _metrics(t)
    assert 0 < accepted < t.n and errors <= 0.1 * accepted and rate == errors / accepted


def test_a_tie_at_one_half_is_counted_as_the_rule_answers(tmp_path, capsys) -> None:
    # The rule answers `p >= 0.5`, true; the label is false. Counted as the argmax answer (false),
    # both 0.5 items look right, the threshold drops to 0.5, and 2 of 10 accepted answers are wrong.
    rows = _rows((8, 0.9, True), (2, 0.5, False))
    records = _write(tmp_path / "r.jsonl", rows)
    out = tmp_path / "p.json"
    assert _tune(records, out, "--error-budget", "0.1") == 0
    (policy,) = load_policies(out)
    assert policy.threshold == 0.9
    assert _metrics(policy) == (8, 0, 0.0)
    _gate_agrees(policy, rows)


def test_tie_groups_at_the_threshold_are_taken_or_left_whole(tmp_path, capsys) -> None:
    # Confidence 0.9 holds p 0.9 and p 0.1 (1 - 0.1 == 0.9 exactly), two of them wrong: 2 of 20
    # meets 0.1, so all ten are accepted. Adding the 0.8 group, 3 of 25, breaks it: all five escalate.
    assert 1 - 0.1 == 0.9 and 1 - 0.2 == 0.8
    rows = _rows(
        (10, 0.95, True),
        (5, 0.9, True), (1, 0.9, False), (3, 0.1, False), (1, 0.1, True),
        (4, 0.8, True), (1, 0.2, True),
    )
    records = _write(tmp_path / "r.jsonl", rows)
    out = tmp_path / "p.json"
    assert _tune(records, out, "--error-budget", "0.1") == 0
    (policy,) = load_policies(out)
    assert policy.threshold == 0.9
    assert _metrics(policy) == (20, 2, 0.1)
    _gate_agrees(policy, rows)


def test_calibrated_values_that_collapse_into_one_tie_are_one_group(tmp_path, capsys) -> None:
    # The fit maps raw 0.99 to 0.95 and raw 0.905-0.94 to 0.9, flat segments of an isotonic map. Raw,
    # the three right answers at 0.92-0.94 and one wrong one at 0.91 come in one by one: 1 of 14
    # meets 0.1, so the raw threshold is 0.91. Calibrated, those five are one tie with two wrong,
    # 2 of 15 breaks 0.1, and the whole tie escalates: the threshold is 0.95.
    fit = Fit(question_id=QID, model_served="m", primitive="noul", method="isotonic", split="cal", n=2,
              classes=[False, True], params={"x": [0.0, 0.9, 0.945, 0.95, 1.0], "y": [0.0, 0.9, 0.9, 0.95, 0.95]},
              nll_before=0.7, nll_after=0.6, fitted_at="2026-11-13T00:00:00+00:00", source_sha256="", warnings=[],
              items=[["cal", "c1"], ["cal", "c2"]])
    fits = tmp_path / "fits.json"
    save_fits([fit], fits)
    rows = _rows((10, 0.99, True), (1, 0.94, True), (1, 0.93, True), (1, 0.92, True), (1, 0.91, False), (1, 0.905, False))
    raw = _write(tmp_path / "r.jsonl", rows)
    applied = tmp_path / "applied.jsonl"
    assert cli_main(["apply", raw, "--fit", str(fits), "-o", str(applied)]) == 0
    assert sorted({json.loads(line)["calib"]["p"] for line in applied.read_text(encoding="utf-8").splitlines()}) == [0.9, 0.95]

    raw_out, cal_out = tmp_path / "raw.json", tmp_path / "cal.json"
    assert _tune(raw, raw_out, "--error-budget", "0.1") == 0
    assert _tune(str(applied), cal_out, "--error-budget", "0.1", "--use-calib") == 0
    (by_raw,) = load_policies(raw_out)
    (by_cal,) = load_policies(cal_out)
    assert (by_raw.threshold, _metrics(by_raw)[:2]) == (0.91, (14, 1))
    assert (by_cal.calibrated, by_cal.fit_sha256, by_cal.threshold) == (True, fit.fit_sha256, 0.95)
    assert _metrics(by_cal) == (10, 0, 0.0)
    _gate_agrees(by_cal, rows, [fit])


def test_the_cp_bound_holds_where_the_point_estimate_would_accept_more(tmp_path, capsys) -> None:
    # 3 wrong of 33 meets 0.1 as a point estimate, so 0.6 is the point threshold; its 95 % upper
    # bound does not, while 0 of 30 does, so under the bound the threshold stays at 0.99.
    rows = _rows((30, 0.99, True), (3, 0.6, False))
    records = _write(tmp_path / "r.jsonl", rows)
    point, cp = tmp_path / "point.json", tmp_path / "cp.json"
    assert _tune(records, point, "--error-budget", "0.1") == 0
    assert _tune(records, cp, "--error-budget", "0.1", "--use-cp-upper") == 0
    (by_point,) = load_policies(point)
    (by_cp,) = load_policies(cp)
    assert (by_point.threshold, _metrics(by_point)[:2]) == (0.6, (33, 3))
    assert clopper_pearson_upper(3, 33, 0.05) > 0.1
    accepted, errors, _ = _metrics(by_cp)
    assert (by_cp.threshold, accepted, errors) == (0.99, 30, 0)
    assert clopper_pearson_upper(errors, accepted, 0.05) <= 0.1
    assert by_cp.metrics_on_split["use_cp_upper"] is True
    _gate_agrees(by_cp, rows)


# ---- no threshold meets the budget ----


def test_no_threshold_escalates_every_item(tmp_path, capsys) -> None:
    rows = _rows((2, 0.9, False), (2, 0.2, True))
    records = _write(tmp_path / "r.jsonl", rows)
    out = tmp_path / "p.json"
    assert _tune(records, out, "--error-budget", "0.1") == 0
    (policy,) = load_policies(out)
    assert (policy.mode, policy.threshold) == ("error_budget", None)
    assert _metrics(policy) == (0, 0, None)
    gate = DecisionGate([policy])
    assert all(gate.evaluate_noul(question_id=QID, model_served="m", p_raw=r["p"]).action == "ESCALATE" for r in rows)


def test_the_demo_under_the_cp_bound_has_no_threshold(ws, tmp_path, capsys) -> None:
    out = tmp_path / "p.json"
    assert _tune(ws["records"], out, "--error-budget", "0.1", "--use-cp-upper") == 0
    (policy,) = load_policies(out)
    assert policy.threshold is None and _metrics(policy) == (0, 0, None)


# ---- what stays as it was ----


def _without_created_at(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    for p in doc["policies"]:
        p.pop("created_at", None)
    doc.pop("created_at", None)
    return doc


@pytest.mark.parametrize("records", ["records", "records_categorical"])
def test_band_is_the_default_and_writes_the_same_file(ws, tmp_path, capsys, records) -> None:
    default, band = tmp_path / "default.json", tmp_path / "band.json"
    assert cli_main(["policy", "tune", ws[records], "-o", str(default)]) == 0
    assert cli_main(["policy", "tune", ws[records], "-o", str(band), "--noul-routing", "band"]) == 0
    assert _without_created_at(default) == _without_created_at(band)


def test_choice_and_score_policies_do_not_change(ws, tmp_path, capsys) -> None:
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(Path(ws["records"]).read_text(encoding="utf-8") + Path(ws["records_categorical"]).read_text(encoding="utf-8"), encoding="utf-8")
    default, threshold = tmp_path / "default.json", tmp_path / "threshold.json"
    assert cli_main(["policy", "tune", str(mixed), "-o", str(default)]) == 0
    assert _tune(str(mixed), threshold) == 0
    before = {(p["question_id"], p["primitive"]): p for p in _without_created_at(default)["policies"]}
    after = {(p["question_id"], p["primitive"]): p for p in _without_created_at(threshold)["policies"]}
    assert before.keys() == after.keys()
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {k for k in before if k[1] == "noul"} and changed


# ---- refusals ----


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--max-escalation", "0.2"), "s1: --max-escalation sizes the F1 band, which --noul-routing threshold replaces; pass one of them"),
        (("--max-false-pass", "0.05"), "s1: --noul-routing threshold and --max-false-pass each replace the F1 band; pass one of them"),
    ],
)
def test_options_that_size_or_replace_the_band_are_refused(ws, tmp_path, capsys, extra, message) -> None:
    out = tmp_path / "p.json"
    assert _tune(ws["records"], out, *extra) == 1
    assert _error(capsys) == message
    assert not out.exists()


def test_a_categorical_file_has_no_noul_group_to_route(ws, tmp_path, capsys) -> None:
    out = tmp_path / "p.json"
    assert _tune(ws["records_categorical"], out) == 1
    assert _error(capsys) == "s1: --noul-routing applies to noul groups, and none is selected"
    assert not out.exists()


def test_a_filter_that_leaves_no_noul_group_is_refused(ws, tmp_path, capsys) -> None:
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(Path(ws["records"]).read_text(encoding="utf-8") + Path(ws["records_categorical"]).read_text(encoding="utf-8"), encoding="utf-8")
    choice_qid = next(json.loads(line)["question_id"] for line in Path(ws["records_categorical"]).read_text(encoding="utf-8").splitlines()
                      if json.loads(line)["primitive"] == "choice")
    out = tmp_path / "p.json"
    assert _tune(str(mixed), out, "--question", choice_qid) == 1
    assert _error(capsys) == "s1: --noul-routing applies to noul groups, and none is selected"
    assert not out.exists()


def test_an_unknown_value_is_a_usage_error(ws, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["policy", "tune", ws["records"], "-o", str(tmp_path / "p.json"), "--noul-routing", "cutoff"])
    assert exc.value.code == 1
    err = _error(capsys)  # how the choices are quoted differs across Python versions
    assert err.startswith("s1: argument --noul-routing: invalid choice: 'cutoff' (choose from ")
    assert err.endswith("threshold'); see s1 policy tune -h") or err.endswith("threshold); see s1 policy tune -h")


def test_a_trained_item_is_still_refused_first(tmp_path, capsys) -> None:
    rows = _rows((8, 0.9, True), (2, 0.1, False))
    records = _write(tmp_path / "r.jsonl", rows)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "schema": "s1-training-manifest/1", "checkpoint": "m", "split": "train", "source_sha256": "0" * 64,
        "created_at": "2026-11-13T00:00:00+00:00", "inherited_items": [],
        "items": [{"dataset": "d", "item_id": "i0", "question_id": QID, "label": True}],
    }), encoding="utf-8")
    out = tmp_path / "p.json"
    assert _tune(records, out, "--training-manifest", str(manifest), "--max-escalation", "0.2") == 2
    assert not out.exists()


def test_a_record_without_calib_is_still_refused(tmp_path, capsys) -> None:
    records = _write(tmp_path / "r.jsonl", _rows((8, 0.9, True), (2, 0.1, False)))
    out = tmp_path / "p.json"
    assert _tune(records, out, "--use-calib") == 1
    assert "has no calibrated probabilities" in _error(capsys)
    assert not out.exists()
