"""`s1 policy tune --max-false-pass`: a noul cutoff chosen by how many label-false items pass.

The oracle below is a plain loop over every candidate, written apart from the vectorized helper,
so a CLI branch that writes another cutoff (say 1.0, which meets any cap) fails the comparison
even when routing and feasibility alone would look right."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import load_policies
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import load_fits
from s1ops.improve.policy import FalsePassCut, clopper_pearson_upper, select_false_pass_cutoff
from s1ops.runtime.gate import DecisionGate

QID, MODEL = "q-0123456789abcdef", "kev-test"


def oracle(p, label, rate, use_cp_upper=False, alpha=0.05):
    p, label = [float(x) for x in p], [bool(x) for x in label]
    n_false = sum(not y for y in label)
    for c in sorted(set(p) | {1.0}):
        fp = sum(1 for x, y in zip(p, label) if not y and x >= c)
        # The reported rate against the cap, as the selector compares it (tests/test_budget_boundary.py).
        ok = clopper_pearson_upper(fp, n_false, alpha) <= rate if use_cp_upper else fp / n_false <= rate
        if ok:
            tp = sum(1 for x, y in zip(p, label) if y and x >= c)
            return FalsePassCut(cutoff=c, n_true=len(p) - n_false, n_false=n_false, true_pass=tp, false_pass=fp)
    return None


def _cut(false_p, true_p, rate, **kw):
    p = list(false_p) + list(true_p)
    label = np.array([False] * len(false_p) + [True] * len(true_p))
    return select_false_pass_cutoff(p, label, rate, **kw)


# --- the selection rule ---------------------------------------------------------------------------


def test_hand_computed_cutoff_and_counts() -> None:
    false_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    true_p = [0.55, 0.75, 0.85, 0.97, 0.99]
    # At most 2 of 10 false items may pass: 0.9 and 0.95 pass from any cutoff above 0.8, and the
    # smallest observed value above 0.8 is the true item at 0.85.
    assert _cut(false_p, true_p, 0.2) == FalsePassCut(cutoff=0.85, n_true=5, n_false=10, true_pass=3, false_pass=2)


def test_an_item_at_the_cutoff_passes() -> None:
    # A false item tied with a true one at 0.6: at cutoff 0.6 the false one passes, so a cap of 0
    # moves the cutoff past both, to 0.9.
    assert _cut([0.3, 0.6], [0.6, 0.9], 0.0) == FalsePassCut(cutoff=0.9, n_true=2, n_false=2, true_pass=1, false_pass=0)


def test_cap_zero_takes_one_when_nothing_lies_above_the_false_items() -> None:
    cut = _cut([0.2, 0.9], [0.5], 0.0)
    assert cut == FalsePassCut(cutoff=1.0, n_true=1, n_false=2, true_pass=0, false_pass=0)
    assert cut.true_pass_rate == 0.0


def test_a_false_item_at_one_cannot_be_blocked() -> None:
    assert _cut([0.2, 1.0], [0.5], 0.0) is None


def test_cp_boundary_148_vs_149() -> None:
    assert _cut([0.1] * 148, [0.9], 0.02, use_cp_upper=True) is None
    assert _cut([0.1] * 149, [0.9], 0.02, use_cp_upper=True) == FalsePassCut(cutoff=0.9, n_true=1, n_false=149, true_pass=1, false_pass=0)


def test_cp_with_false_passes_and_a_non_default_alpha() -> None:
    rng = np.random.default_rng(7)
    false_p = np.round(rng.beta(3, 6, 600), 3)
    true_p = np.round(rng.beta(6, 3, 60), 3)
    p = np.r_[false_p, true_p]
    label = np.r_[np.zeros(600, bool), np.ones(60, bool)]
    for alpha in (0.05, 0.10):
        cut = select_false_pass_cutoff(p, label, 0.05, use_cp_upper=True, alpha=alpha)
        assert cut == oracle(p, label, 0.05, True, alpha)
        assert cut.false_pass > 0
    looser = select_false_pass_cutoff(p, label, 0.05, use_cp_upper=True, alpha=0.10)
    stricter = select_false_pass_cutoff(p, label, 0.05, use_cp_upper=True, alpha=0.05)
    assert looser.cutoff < stricter.cutoff  # alpha reaches the bound


def test_no_true_items_reports_recall_as_null() -> None:
    cut = _cut([0.1, 0.4], [], 0.0)
    assert cut.cutoff == 1.0 and cut.true_pass_rate is None and cut.false_pass_rate == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_false_pass": 1.0},
        {"max_false_pass": -0.01},
        {"max_false_pass": float("nan")},
        {"max_false_pass": 0.02, "use_cp_upper": True, "alpha": 0.0},
        {"max_false_pass": 0.02, "use_cp_upper": True, "alpha": 1.0},
        {"max_false_pass": 0.02, "use_cp_upper": True, "alpha": 1.5},
        {"max_false_pass": 0.02, "use_cp_upper": True, "alpha": float("nan")},
    ],
)
def test_invalid_arguments_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        select_false_pass_cutoff([0.1, 0.9], np.array([False, True]), **kwargs)


def test_labels_must_be_boolean_and_include_a_false_item() -> None:
    with pytest.raises(ValueError, match="boolean"):
        select_false_pass_cutoff([0.1, 0.9], [0, 1], 0.1)
    with pytest.raises(ValueError, match="boolean"):
        select_false_pass_cutoff([0.1, 0.9], [0.0, 1.0], 0.1)
    with pytest.raises(ValueError, match="label-false"):
        select_false_pass_cutoff([0.1, 0.9], np.array([True, True]), 0.1)


@pytest.mark.parametrize(("rate", "alpha"), [(0.0, 0.05), (5e-324, 0.05), (0.02, 1e-20)])
def test_unreachable_cp_caps_return_none(rate, alpha) -> None:
    assert _cut([0.1] * 2000, [0.9], rate, use_cp_upper=True, alpha=alpha) is None


def test_a_denormal_cap_without_cp_is_cap_zero() -> None:
    assert _cut([0.1, 0.4], [0.7], 5e-324) == _cut([0.1, 0.4], [0.7], 0.0)


def test_agrees_with_the_oracle_on_random_draws() -> None:
    rng = random.Random(34)
    for _ in range(200):
        n = rng.randint(1, 60)
        p = [round(rng.random(), rng.choice((1, 2, 6))) for _ in range(n)]
        label = [rng.random() < 0.3 for _ in range(n)]
        if all(label):
            label[0] = False
        rate = rng.choice((0.0, 0.02, 0.1, rng.random() * 0.99))
        use_cp = rng.random() < 0.4
        alpha = rng.choice((0.05, 0.1, 0.2))
        got = select_false_pass_cutoff(p, np.array(label), rate, use_cp_upper=use_cp, alpha=alpha)
        assert got == oracle(p, label, rate, use_cp, alpha), (p, label, rate, use_cp, alpha)


# --- through the CLI ------------------------------------------------------------------------------


def _records(rng, n, split, pos_rate=0.1):
    rows = []
    for i in range(n):
        y = rng.random() < pos_rate
        p = rng.betavariate(6, 4) if y else rng.betavariate(4, 6)
        rows.append(
            {"schema": "jev-record/1", "item_id": f"{split}-{i}", "question_id": QID, "primitive": "noul",
             "model_requested": MODEL, "model_served": MODEL, "dataset": "synthetic", "split": split,
             "label": y, "ts": "2026-10-27T00:00:00+00:00", "p": round(p, 4)}
        )
    return rows


def _write(path: Path, rows) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


@pytest.fixture
def pipeline(tmp_path: Path):
    rng = random.Random(0)
    cal = _write(tmp_path / "cal.jsonl", _records(rng, 300, "cal"))
    ev = _write(tmp_path / "eval.jsonl", _records(rng, 400, "eval"))
    fits = str(tmp_path / "fits.json")
    applied = str(tmp_path / "eval_cal.jsonl")
    assert cli_main(["fit", cal, "-o", fits, "--method", "platt"]) == 0
    assert cli_main(["apply", ev, "--fit", fits, "-o", applied]) == 0
    return {"tmp": tmp_path, "eval": ev, "applied": applied, "fits": fits}


def _tune(capsys, records, out, *extra) -> list:
    assert cli_main(["policy", "tune", records, "-o", str(out), *extra]) == 0
    capsys.readouterr()
    return load_policies(out)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("use_cp", [False, True])
def test_calibrated_cli_cutoff_matches_the_oracle_and_routes(pipeline, capsys, use_cp) -> None:
    extra = ["--use-calib", "--max-false-pass", "0.05"] + (["--use-cp-upper", "--cp-alpha", "0.1"] if use_cp else [])
    (pol,) = _tune(capsys, pipeline["applied"], pipeline["tmp"] / "policy.json", *extra)
    rows = _rows(pipeline["applied"])
    p = [r["calib"]["p"] for r in rows]
    label = [r["label"] for r in rows]
    want = oracle(p, label, 0.05, use_cp, 0.1)
    assert want is not None and want.cutoff < 1.0 and want.true_pass > 0
    assert (pol.mode, pol.band_lo, pol.band_hi, pol.threshold, pol.error_budget) == ("escalation_band", want.cutoff, want.cutoff, None, None)
    assert pol.calibrated and pol.fit_sha256 is not None
    m = pol.metrics_on_split
    assert (m["n_true"], m["n_false"], m["true_pass"], m["false_pass"]) == (want.n_true, want.n_false, want.true_pass, want.false_pass)
    assert (m["selection"], m["max_false_pass"], m["use_cp_upper"]) == ("max_false_pass", 0.05, use_cp)
    assert m["false_pass_rate"] == want.false_pass / want.n_false <= 0.05
    gate = DecisionGate([pol], load_fits(pipeline["fits"]))
    passed = {True: 0, False: 0}
    for r in rows:
        d = gate.evaluate_noul(question_id=QID, model_served=MODEL, p_raw=r["p"])
        assert d.action in ("ALLOW", "BLOCK")  # lo == hi never escalates
        assert (d.action != "BLOCK") == (r["calib"]["p"] >= want.cutoff), r
        passed[r["label"]] += d.action != "BLOCK"
    assert (passed[True], passed[False]) == (want.true_pass, want.false_pass)


def test_raw_cli_cutoff_matches_the_oracle(pipeline, capsys) -> None:
    (pol,) = _tune(capsys, pipeline["eval"], pipeline["tmp"] / "policy.json", "--max-false-pass", "0.02")
    assert pol.metrics_on_split["cp_alpha"] is None
    rows = _rows(pipeline["eval"])
    want = oracle([r["p"] for r in rows], [r["label"] for r in rows], 0.02)
    assert (pol.band_lo, pol.band_hi, pol.calibrated, pol.fit_sha256) == (want.cutoff, want.cutoff, False, None)
    assert pol.metrics_on_split["false_pass"] == want.false_pass
    gate = DecisionGate([pol], [])
    for r in rows:
        d = gate.evaluate_noul(question_id=QID, model_served=MODEL, p_raw=r["p"])
        assert (d.action != "BLOCK") == (r["p"] >= want.cutoff)


def test_choice_and_score_groups_keep_the_error_budget(tmp_path: Path, capsys) -> None:
    ws = generate_workspace(tmp_path / "ws")
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(Path(ws["records"]).read_text() + Path(ws["records_categorical"]).read_text())
    policies = _tune(capsys, str(mixed), tmp_path / "policy.json", "--use-calib", "--max-false-pass", "0.05")
    by_primitive = {pol.primitive: pol for pol in policies}
    assert set(by_primitive) == {"noul", "choice", "score"}
    assert by_primitive["noul"].metrics_on_split["selection"] == "max_false_pass"
    for prim in ("choice", "score"):
        assert by_primitive[prim].mode == "error_budget" and by_primitive[prim].error_budget == 0.10
        assert by_primitive[prim].metrics_on_split["selection"] == "error_budget"


def test_without_the_flag_the_noul_band_is_unchanged(pipeline, capsys) -> None:
    (pol,) = _tune(capsys, pipeline["applied"], pipeline["tmp"] / "policy.json", "--use-calib")
    assert pol.metrics_on_split["selection"] == "escalation_band" and pol.error_budget == 0.10 and pol.threshold is not None


# --- CLI errors -----------------------------------------------------------------------------------


def _error(capsys) -> str:
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("s1: ") and err.count("\n") == 1 and "Traceback" not in err
    return err


@pytest.mark.parametrize("rate", ["1", "1.5", "-0.1", "nan", "inf"])
def test_cli_rejects_a_rate_outside_zero_one(pipeline, capsys, rate) -> None:
    out = pipeline["tmp"] / "policy.json"
    assert cli_main(["policy", "tune", pipeline["eval"], "-o", str(out), "--max-false-pass", rate]) == 1
    assert _error(capsys) == f"s1: --max-false-pass must be a number in [0, 1), got {float(rate)!r}\n"
    assert not out.exists()


def test_a_rate_of_zero_is_accepted(pipeline, capsys) -> None:
    (pol,) = _tune(capsys, pipeline["eval"], pipeline["tmp"] / "policy.json", "--max-false-pass", "0")
    assert pol.metrics_on_split["max_false_pass"] == 0.0


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-escalation", "0.35"],  # the combination refusal used to win: the range was checked per group
        ["--question", "q-nope"],  # and so did no matching group
    ],
)
def test_the_rate_is_checked_before_the_records_and_the_other_flags(pipeline, capsys, extra) -> None:
    out = pipeline["tmp"] / "policy.json"
    assert cli_main(["policy", "tune", pipeline["eval"], "-o", str(out), "--max-false-pass", "1", *extra]) == 1
    assert _error(capsys) == "s1: --max-false-pass must be a number in [0, 1), got 1.0\n"
    assert not out.exists()


def test_a_bad_rate_on_a_categorical_file_reports_the_rate(tmp_path: Path, capsys) -> None:
    # Before, `applies to noul groups` won and the rate was never checked.
    ws = generate_workspace(tmp_path / "ws")
    out = tmp_path / "policy.json"
    for rate in ("1", "1.5", "-0.1", "nan"):
        assert cli_main(["policy", "tune", ws["records_categorical"], "-o", str(out), "--max-false-pass", rate]) == 1
        assert _error(capsys) == f"s1: --max-false-pass must be a number in [0, 1), got {float(rate)!r}\n"
        assert not out.exists()


def _flat(n_false: int, n: int = 10, p: float = 0.9) -> list[dict]:
    return [
        {"schema": "jev-record/1", "item_id": f"i{i}", "question_id": QID, "primitive": "noul",
         "model_requested": MODEL, "model_served": MODEL, "dataset": "d", "split": "eval",
         "label": i >= n_false, "ts": "2026-10-27T00:00:00+00:00", "p": p}
        for i in range(n)
    ]


def test_a_valid_rate_with_no_matching_group_reports_no_groups(tmp_path: Path, capsys) -> None:
    records = _write(tmp_path / "records.jsonl", _flat(5))
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", records, "-o", str(out), "--max-false-pass", "0.1", "--question", "q-nope"]) == 1
    assert _error(capsys) == (
        f"s1: no labeled groups matching --question q-nope in {records} "
        "(1 group(s) before filters; skipped 0 without a response, 0 unlabeled, 0 in other splits)\n"
    )
    assert not out.exists()


_NO_FALSE = f"{QID} ({MODEL}): no label-false items: nothing bounds the false-pass rate"
_NO_CUT = (
    f"{QID} ({MODEL}): no cutoff keeps the Clopper-Pearson upper bound (alpha 0.05) at or under 0.01 with 5 "
    "label-false items; more label-false items, a larger --max-false-pass or a larger --cp-alpha may help"
)


@pytest.mark.parametrize(
    ("n_false", "extra", "message"),
    [(0, ["--max-false-pass", "0.1"], _NO_FALSE), (5, ["--max-false-pass", "0.01", "--use-cp-upper"], _NO_CUT)],
)
# A newline in the path is written as the two characters backslash and n, so the line stays one line.
@pytest.mark.parametrize(("name", "shown"), [("records.jsonl", "records.jsonl"), ("a\nb.jsonl", "a\\nb.jsonl")])
def test_a_group_refusal_starts_with_the_records_file(tmp_path: Path, capsys, n_false, extra, message, name, shown) -> None:
    records = _write(tmp_path / name, _flat(n_false))
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", records, "-o", str(out), *extra]) == 1
    assert _error(capsys) == f"s1: {tmp_path}/{shown}: {message}\n"
    assert not out.exists()


def test_cli_rejects_max_escalation_with_max_false_pass(pipeline, capsys) -> None:
    out = pipeline["tmp"] / "policy.json"
    assert cli_main(["policy", "tune", pipeline["eval"], "-o", str(out), "--max-false-pass", "0.02", "--max-escalation", "0.35"]) == 1
    assert "--max-escalation" in _error(capsys)
    assert not out.exists()


def test_cli_rejects_the_flag_when_no_noul_group_is_selected(tmp_path: Path, capsys) -> None:
    ws = generate_workspace(tmp_path / "ws")
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(Path(ws["records"]).read_text() + Path(ws["records_categorical"]).read_text())
    choice_qid = next(r["question_id"] for r in _rows(ws["records_categorical"]) if r["primitive"] == "choice")
    out = tmp_path / "policy.json"
    for records, extra in ((ws["records_categorical"], []), (str(mixed), ["--question", choice_qid])):
        assert cli_main(["policy", "tune", records, "-o", str(out), "--max-false-pass", "0.02", *extra]) == 1
        assert "--max-false-pass applies to noul groups, and none is selected" in _error(capsys)
        assert not out.exists()


@pytest.mark.parametrize(("rate", "alpha"), [("0.02", "0.05"), ("0", "0.05"), ("0.02", "1e-20")])
def test_cli_reports_an_unreachable_cp_cap(tmp_path: Path, capsys, rate, alpha) -> None:
    rows = _records(random.Random(1), 60, "eval")
    records = _write(tmp_path / "eval.jsonl", rows)
    out = tmp_path / "policy.json"
    argv = ["policy", "tune", records, "-o", str(out), "--max-false-pass", rate, "--use-cp-upper", "--cp-alpha", alpha]
    assert cli_main(argv) == 1
    err = _error(capsys)
    assert err.startswith(f"s1: {records}: {QID} ({MODEL}): no cutoff keeps ")
    assert f"{sum(not r['label'] for r in rows)} label-false" in err
    assert not out.exists()


@pytest.mark.parametrize("alpha", ["nan", "inf", "-inf", "2"])
def test_an_unused_cp_alpha_still_writes_a_loadable_policy(pipeline, capsys, alpha) -> None:
    (pol,) = _tune(capsys, pipeline["eval"], pipeline["tmp"] / "policy.json", "--max-false-pass", "0.02", f"--cp-alpha={alpha}")
    assert pol.metrics_on_split["cp_alpha"] is None and pol.metrics_on_split["use_cp_upper"] is False


@pytest.mark.parametrize("alpha", ["0", "1", "nan"])
def test_cli_rejects_an_invalid_cp_alpha(pipeline, capsys, alpha) -> None:
    out = pipeline["tmp"] / "policy.json"
    argv = ["policy", "tune", pipeline["eval"], "-o", str(out), "--max-false-pass", "0.02", "--use-cp-upper", "--cp-alpha", alpha]
    assert cli_main(argv) == 1
    assert "alpha" in _error(capsys)
    assert not out.exists()
