"""Choice / score gating and the shared routing rule (`runtime/routing.py`).

The gate is checked against real `s1 fit` -> `s1 apply` -> `s1 policy tune` output, so a policy
routes at runtime exactly as it was tuned; drift and both exporters route through the same rule.
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from s1ops.bridges.langfuse import _predict_info, build_langfuse_events
from s1ops.bridges.openinference import build_openinference_spans
from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, parse_record, read_records, select
from s1ops.contracts.schemas import PolicyEntry, load_policies
from s1ops.eval.metrics import top_label
from s1ops.monitor.drift import _escalation_rate
from s1ops.improve.calibrate import FitError, load_fits
from s1ops.runtime.gate import DecisionGate, ModelServedMismatchError, PolicyIntegrityError, _calibrated, _normalized
from s1ops.runtime.routing import ROUTING_RULE, route

QID = {"choice": "q-choice00000000a", "score": "q-score000000000b"}
CLASSES = {"choice": ["A", "B", "C"], "score": [0, 1, 2, 3]}


def _rows(rng: np.random.Generator, primitive: str, split: str, n: int, order: list[int]) -> list[dict]:
    """Overconfident answers rounded to two decimals, so ties, repeats and non-unit sums occur."""
    classes = CLASSES[primitive]
    rows = []
    for i in range(n):
        y = int(rng.integers(len(classes)))
        logits = rng.normal(0.0, 1.0, len(classes))
        logits[y] += 1.2
        p = np.exp(2.5 * logits)
        p = np.round(p / p.sum(), 2)
        if abs(p.sum() - 1.0) > 0.015:
            p[int(p.argmax())] += round(1.0 - float(p.sum()), 2)
        dist = {str(classes[j]): float(p[j]) for j in order}
        top = max(dist, key=dist.get)
        row = {
            "schema": "s1-record/1",
            "dataset": "synthetic",
            "item_id": f"{split}-{i}",
            "question_id": QID[primitive],
            "primitive": primitive,
            "model_requested": "m1",
            "model_served": "m1",
            "split": split,
            "confidence": max(dist.values()),
            "probabilities": dist,
            "label": classes[y],
        }
        if primitive == "choice":
            row["choice"] = top
        else:
            row["score"] = int(top)
        rows.append(row)
    return rows


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


@pytest.fixture(params=["choice", "score"])
def tuned(request, tmp_path: Path) -> dict:
    """Train, eval and runtime use three different key orders on purpose."""
    primitive = request.param
    k = len(CLASSES[primitive])
    rng = np.random.default_rng(7)
    train = _write(tmp_path / "train.jsonl", _rows(rng, primitive, "cal", 300, list(range(k))))
    eval_rows = _rows(rng, primitive, "eval", 400, list(range(k))[::-1])
    evals = _write(tmp_path / "eval.jsonl", eval_rows)
    out = {"primitive": primitive, "eval_rows": eval_rows, "dir": tmp_path}
    out["fits"] = str(tmp_path / "fits.json")
    out["applied"] = str(tmp_path / "applied.jsonl")
    out["policy_raw"] = str(tmp_path / "policy_raw.json")
    out["policy_cal"] = str(tmp_path / "policy_cal.json")
    assert cli_main(["fit", train, "-o", out["fits"]]) == 0
    assert cli_main(["apply", evals, "--fit", out["fits"], "-o", out["applied"]]) == 0
    assert cli_main(["policy", "tune", evals, "-o", out["policy_raw"], "--error-budget", "0.2"]) == 0
    assert cli_main(["policy", "tune", out["applied"], "-o", out["policy_cal"], "--error-budget", "0.2", "--use-calib"]) == 0
    return out


def _runtime_probs(row: dict, primitive: str) -> dict:
    """The answer as a live call would hand it over, in yet another key order."""
    items = list(row["probabilities"].items())
    items = items[1:] + items[:1]
    return {(int(k) if primitive == "score" else k): v for k, v in items}


def _evaluate(gate: DecisionGate, primitive: str, probs: dict, **kw):
    fn = gate.evaluate_choice if primitive == "choice" else gate.evaluate_score
    return fn(question_id=QID[primitive], model_served="m1", probs_raw=probs, **kw)


@pytest.mark.parametrize("calibrated", [False, True])
def test_gate_routes_every_record_as_the_tuner_measured_it(tuned, calibrated: bool) -> None:
    primitive = tuned["primitive"]
    policy_path = tuned["policy_cal" if calibrated else "policy_raw"]
    (pol,) = load_policies(policy_path)
    assert pol.primitive == primitive and pol.calibrated is calibrated
    assert pol.threshold is not None

    tuned_on = tuned["applied"] if calibrated else str(tuned["dir"] / "eval.jsonl")
    (grp,) = select(read_records(tuned_on, use_calib=calibrated), allow_model_mismatch=True).groups
    conf, _ = top_label(grp.P, grp.y)
    tied = (grp.P == grp.P.max(axis=1, keepdims=True)).sum(axis=1) > 1
    by_item = {item_id: i for i, (_, item_id) in enumerate(grp.items)}

    gate = DecisionGate.from_files(policy_path, tuned["fits"])
    (fit,) = load_fits(tuned["fits"])
    at_threshold = ties = 0
    for row in tuned["eval_rows"]:
        i = by_item[row["item_id"]]
        probs = _runtime_probs(row, primitive)
        # Bit for bit, not rounded: a threshold is compared exactly.
        classes = list(probs) if primitive == "choice" else sorted(probs)
        routed = _normalized(classes, [probs[c] for c in classes])
        if calibrated:
            routed = _calibrated(fit, classes, routed)
        assert routed == [grp.P[i][grp.classes.index(c)] for c in classes], row["item_id"]
        d = _evaluate(gate, primitive, probs)
        if tied[i]:
            ties += 1
            assert d.action == "ESCALATE"
            continue
        assert d.escalated is bool(conf[i] < pol.threshold), row["item_id"]
        assert d.p_effective == round(float(conf[i]), 6)
        at_threshold += bool(conf[i] == pol.threshold)
        if not d.escalated:
            assert d.action == f"SELECT:{d.predicted}"
            assert str(d.predicted) == str(grp.classes[int(grp.P[i].argmax())])
    # The threshold is one of the tuning confidences, so some answers sit exactly on it.
    assert at_threshold > 0
    assert ties > 0


@pytest.mark.parametrize("calibrated", [False, True])
def test_the_gate_compares_the_tuned_threshold_exactly(tuned, calibrated: bool) -> None:
    """Move the real threshold one float either side of an answer that sits exactly on it."""
    primitive = tuned["primitive"]
    policy_path = tuned["policy_cal" if calibrated else "policy_raw"]
    (pol,) = load_policies(policy_path)
    tuned_on = tuned["applied"] if calibrated else str(tuned["dir"] / "eval.jsonl")
    (grp,) = select(read_records(tuned_on, use_calib=calibrated), allow_model_mismatch=True).groups
    conf, _ = top_label(grp.P, grp.y)
    i = int(np.flatnonzero(conf == pol.threshold)[0])
    row = next(r for r in tuned["eval_rows"] if r["item_id"] == grp.items[i][1])
    probs = _runtime_probs(row, primitive)
    (fit,) = load_fits(tuned["fits"])
    for threshold, escalated in ((pol.threshold, False), (float(np.nextafter(pol.threshold, 2)), True), (float(np.nextafter(pol.threshold, -1)), False)):
        gate = DecisionGate([dataclasses.replace(pol, threshold=threshold)], [fit])
        assert _evaluate(gate, primitive, probs).escalated is escalated, threshold


def test_raw_policy_routes_the_same_with_or_without_a_fit(tuned) -> None:
    primitive = tuned["primitive"]
    with_fit = DecisionGate.from_files(tuned["policy_raw"], tuned["fits"])
    without = DecisionGate.from_files(tuned["policy_raw"])
    for row in tuned["eval_rows"][:80]:
        probs = _runtime_probs(row, primitive)
        a, b = _evaluate(with_fit, primitive, probs), _evaluate(without, primitive, probs)
        assert (a.action, a.predicted, a.p_effective) == (b.action, b.predicted, b.p_effective)
        assert a.p_calibrated is not None and b.p_calibrated is None


def test_calibrated_policy_without_its_fit_is_refused(tuned, capsys) -> None:
    primitive = tuned["primitive"]
    gate = DecisionGate.from_files(tuned["policy_cal"])
    probs = _runtime_probs(tuned["eval_rows"][0], primitive)
    with pytest.raises(ValueError, match="calibrated"):
        _evaluate(gate, primitive, probs)
    rc = cli_main(["gate", tuned["policy_cal"], "--question", QID[primitive], "--model", "m1", "--probs", json.dumps({str(k): v for k, v in probs.items()})])
    assert rc == 1
    assert "pass the fits.json" in capsys.readouterr().err


def test_cli_gate_probs_matches_the_library(tuned, capsys) -> None:
    primitive = tuned["primitive"]
    gate = DecisionGate.from_files(tuned["policy_cal"], tuned["fits"])
    probs = _runtime_probs(tuned["eval_rows"][3], primitive)
    expected = _evaluate(gate, primitive, probs).as_dict()
    capsys.readouterr()
    rc = cli_main([
        "gate", tuned["policy_cal"], "--fit", tuned["fits"], "--question", QID[primitive], "--model", "m1",
        "--probs", json.dumps({str(k): v for k, v in probs.items()}),
    ])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(expected))
    if primitive == "score":
        weighted = sum(int(k) * p for k, p in expected["probabilities"].items())
        assert expected["expected_score"] == pytest.approx(weighted, abs=1e-5)
    else:
        assert expected["expected_score"] is None


def test_model_mismatch_and_integrity_on_categorical(tuned) -> None:
    primitive = tuned["primitive"]
    gate = DecisionGate.from_files(tuned["policy_raw"])
    probs = _runtime_probs(tuned["eval_rows"][0], primitive)
    fn = gate.evaluate_choice if primitive == "choice" else gate.evaluate_score
    with pytest.raises(ModelServedMismatchError):
        fn(question_id=QID[primitive], model_served="m2", probs_raw=probs)
    with pytest.raises(PolicyIntegrityError):
        fn(question_id=QID[primitive], model_served="m1", probs_raw=probs, expected_policy_sha256="deadbeef00000000")


def _policy(primitive: str, **kw) -> PolicyEntry:
    base = {"question_id": "q", "model_served": "m", "primitive": primitive, "mode": "error_budget", "split": "eval", "n": 1}
    return PolicyEntry(**{**base, **kw})


@pytest.mark.parametrize("primitive, classes", [("choice", ["A", "B", "C"]), ("score", [0, 1, 2])])
def test_threshold_boundary_and_its_float_neighbours(primitive: str, classes: list) -> None:
    t = 0.6
    pol = _policy(primitive, threshold=t)
    on = route(pol, classes, [t, 0.3, 1 - t - 0.3])
    below = route(pol, classes, [float(np.nextafter(t, 0)), 0.3, 0.1])
    above = route(pol, classes, [float(np.nextafter(t, 1)), 0.3, 0.1])
    assert on.action == f"SELECT:{classes[0]}" and above.action == f"SELECT:{classes[0]}"
    assert below.action == "ESCALATE"


@pytest.mark.parametrize("primitive, classes", [("choice", ["A", "B", "C"]), ("score", [0, 1, 2])])
def test_a_tie_at_the_top_escalates_even_at_threshold_zero(primitive: str, classes: list) -> None:
    routed = route(_policy(primitive, threshold=0.0), classes, [0.4, 0.2, 0.4])
    assert (routed.action, routed.escalated, routed.predicted) == ("ESCALATE", True, classes[0])


@pytest.mark.parametrize("primitive", ["choice", "score"])
def test_null_threshold_always_escalates(primitive: str) -> None:
    classes = CLASSES[primitive][:2]
    assert route(_policy(primitive, threshold=None), classes, [1.0, 0.0]).action == "ESCALATE"


def test_a_two_level_score_is_never_routed_as_noul() -> None:
    routed = route(_policy("score", threshold=0.6), [0, 1], [0.1, 0.9])
    assert (routed.action, routed.predicted) == ("SELECT:1", 1)
    with pytest.raises(ValueError, match="noul classes"):
        route(_policy("noul", threshold=0.6), [0, 1], [0.1, 0.9])
    with pytest.raises(ValueError, match="non-negative integer levels"):
        route(_policy("score", threshold=0.6), [False, True], [0.1, 0.9])


@pytest.mark.parametrize(
    "primitive, classes, message",
    [
        ("choice", ["A", 1], "option names"),
        ("choice", ["A", ""], "option names"),
        ("score", [0, -1], "non-negative integer levels"),
        ("score", ["0", "1"], "non-negative integer levels"),
        ("choice", ["A", "A"], "distinct"),
    ],
)
def test_categorical_classes_are_checked(primitive: str, classes: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        route(_policy(primitive, threshold=0.5), classes, [0.9, 0.1])


def test_categorical_band_policy_is_refused() -> None:
    pol = _policy("choice", mode="escalation_band", band_lo=0.2, band_hi=0.8, threshold=0.5)
    with pytest.raises(ValueError, match="error_budget only"):
        route(pol, ["A", "B"], [0.9, 0.1])


@pytest.mark.parametrize(
    "band, p, expected",
    [
        ((0.1, 0.3), 0.4, ("ALLOW", True)),
        ((0.7, 0.9), 0.6, ("BLOCK", False)),
        ((0.3, 0.8), 0.45, ("ESCALATE", False)),
        ((0.3, 0.8), 0.55, ("ESCALATE", True)),
    ],
)
def test_noul_off_centre_bands_keep_their_predictions(band, p, expected) -> None:
    pol = _policy("noul", mode="escalation_band", band_lo=band[0], band_hi=band[1], threshold=0.8)
    routed = route(pol, [False, True], [1 - p, p])
    assert (routed.action, routed.predicted) == expected


def _gate(primitive: str, **kw) -> DecisionGate:
    return DecisionGate([_policy(primitive, threshold=0.5, **kw)])


@pytest.mark.parametrize(
    "probs, message",
    [
        ({"A": float("nan"), "B": 0.5}, "finite"),
        ({"A": True, "B": 0.0}, "must be a number"),
        ({"A": 0.6, "B": 0.5}, "sum to 1.1"),
        ({"A": 1.0}, "at least two"),
        ({"A": "0.5", "B": 0.5}, "must be a number"),
        ({"A": 1.5, "B": -0.5}, r"\[0, 1\]"),
        ({"": 0.5, "B": 0.5}, "non-empty strings"),
    ],
)
def test_choice_inputs_are_validated(probs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _gate("choice").evaluate_choice(question_id="q", model_served="m", probs_raw=probs)


@pytest.mark.parametrize(
    "probs, message",
    [
        ({"x": 0.5, "1": 0.5}, "non-negative integers"),
        ({-1: 0.5, 1: 0.5}, "non-negative integers"),
        ({True: 0.5, 2: 0.5}, "non-negative integers"),
        ({0: 0.2, "0": 0.3, 1: 0.5}, "repeats level 0"),
        ({"0": 0.2, "00": 0.3, "1": 0.5}, "repeats level 0"),
        ({0: 0.1, 10**400: 0.9}, "too large"),
        ({"0": 0.1, "1" + "0" * 400: 0.9}, "too large"),
    ],
)
def test_score_levels_are_validated(probs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _gate("score").evaluate_score(question_id="q", model_served="m", probs_raw=probs)


def test_non_unit_sums_are_renormalized_as_the_parser_does() -> None:
    gate = _gate("choice")
    for probs in ({"A": 0.5, "B": 0.49}, {"A": 0.51, "B": 0.509}):
        record = parse_record({
            "schema": "s1-record/1", "item_id": "i", "question_id": "q", "primitive": "choice",
            "model_requested": "m", "model_served": "m", "choice": "A", "confidence": 0.5, "probabilities": probs,
        })
        d = gate.evaluate_choice(question_id="q", model_served="m", probs_raw=probs)
        assert d.p_effective == round(record.probs[0], 6)
    with pytest.raises(ValueError, match="sum"):
        gate.evaluate_choice(question_id="q", model_served="m", probs_raw={"A": 0.51, "B": 0.51})


def test_answer_classes_must_match_the_fit(tuned) -> None:
    primitive = tuned["primitive"]
    gate = DecisionGate.from_files(tuned["policy_cal"], tuned["fits"])
    probs = _runtime_probs(tuned["eval_rows"][0], primitive)
    probs.pop(next(iter(probs)))
    total = sum(probs.values())
    with pytest.raises(ValueError, match="differ from the fit"):
        _evaluate(gate, primitive, {k: v / total for k, v in probs.items()})


def test_wrong_primitive_is_exit_1_not_a_model_mismatch(tuned, capsys) -> None:
    primitive = tuned["primitive"]
    gate = DecisionGate.from_files(tuned["policy_raw"])
    with pytest.raises(ValueError, match=f"for primitive {primitive}, not noul"):
        gate.evaluate_noul(question_id=QID[primitive], model_served="m1", p_raw=0.9)
    rc = cli_main(["gate", tuned["policy_raw"], "--question", QID[primitive], "--model", "m1", "--p", "0.9"])
    assert rc == 1
    assert "pass --probs, not --p" in capsys.readouterr().err
    probs = json.dumps({str(k): v for k, v in _runtime_probs(tuned["eval_rows"][0], primitive).items()})
    rc = cli_main(["gate", tuned["policy_raw"], "--question", QID[primitive], "--model", "m2", "--probs", probs])
    assert rc == 3


@pytest.mark.parametrize(
    "args, message",
    [
        (["--p", "0.5", "--probs", "{}"], "not allowed with argument"),
        ([], "one of the arguments --p --probs is required"),
    ],
)
def test_cli_gate_needs_exactly_one_answer_flag(tuned, args: list, message: str, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["gate", tuned["policy_raw"], "--question", QID[tuned["primitive"]], "--model", "m1", *args])
    assert exc.value.code == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "args, message",
    [
        (["--probs", "[0.5, 0.5]"], "must be a JSON object"),
        (["--probs", '{"A": NaN, "B": 0.5}'], "non-finite"),
        (["--probs", '{"A": 0.5, "A": 0.5}'], "repeats"),
        (["--probs", "{nope"], "not valid JSON"),
    ],
)
def test_cli_gate_probs_errors(tuned, args: list, message: str, capsys) -> None:
    rc = cli_main(["gate", tuned["policy_raw"], "--question", QID[tuned["primitive"]], "--model", "m1", *args])
    assert rc == 1
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err


def test_cli_gate_unknown_question_with_probs(tuned, capsys) -> None:
    rc = cli_main(["gate", tuned["policy_raw"], "--question", "q-0000000000000000", "--model", "m1", "--probs", '{"A": 0.5, "B": 0.5}'])
    assert rc == 1
    assert "No policy found" in capsys.readouterr().err


def test_cli_gate_noul_question_with_probs(tmp_path: Path, capsys) -> None:
    from s1ops.demo import generate_workspace

    ws = generate_workspace(tmp_path / "ws")
    qid = json.loads(Path(ws["policy"]).read_text())["policies"][0]["question_id"]
    rc = cli_main(["gate", ws["policy"], "--question", qid, "--model", "jev-1.13.0", "--probs", '{"A": 0.5, "B": 0.5}'])
    assert rc == 1
    assert "pass --p, not --probs" in capsys.readouterr().err


@pytest.mark.parametrize(
    "p, action, predicted, p_calibrated",
    [(0.98, "ALLOW", True, 0.873792), (0.5, "ESCALATE", True, 0.5), (0.02, "BLOCK", False, 0.126208)],
)
def test_noul_decision_gains_only_null_fields(tmp_path: Path, p, action, predicted, p_calibrated) -> None:
    from s1ops.demo import generate_workspace

    ws = generate_workspace(tmp_path / "ws")
    gate = DecisionGate.from_files(ws["policy"], ws["fits"])
    qid = json.loads(Path(ws["policy"]).read_text())["policies"][0]["question_id"]
    # The values the gate on main returned for the seeded demo, plus the two new null fields.
    assert gate.evaluate_noul(question_id=qid, model_served="jev-1.13.0", p_raw=p).as_dict() == {
        "question_id": "q-30076be2728be682",
        "model_served": "jev-1.13.0",
        "primitive": "noul",
        "action": action,
        "escalated": action == "ESCALATE",
        "predicted": predicted,
        "p_raw": p,
        "p_calibrated": p_calibrated,
        "p_effective": p_calibrated,
        "policy_sha256": "d411d78ef9968964",
        "probabilities": None,
        "expected_score": None,
    }


def _record(primitive: str, **answer) -> object:
    row = {"schema": "s1-record/1", "item_id": "i", "question_id": "q", "primitive": primitive, "model_requested": "m", "model_served": "m"}
    return parse_record({**row, **answer})


def test_exporters_route_noul_error_budget_like_the_gate() -> None:
    pol = _policy("noul", threshold=0.8)
    r = _record("noul", p=0.6, label=True)
    info = _predict_info(r, pol)
    # Before: ALLOW if p >= threshold else BLOCK, predicted p >= threshold -> BLOCK / False.
    assert (info["policy_action"], info["predicted"]) == ("ESCALATE", True)
    gate = DecisionGate([pol]).evaluate_noul(question_id="q", model_served="m", p_raw=0.6)
    assert (gate.action, gate.predicted) == (info["policy_action"], info["predicted"])
    assert _predict_info(r, _policy("noul", threshold=None))["policy_action"] == "ESCALATE"


def test_exporters_noul_band_predicted_inside_the_band() -> None:
    pol = _policy("noul", mode="escalation_band", band_lo=0.3, band_hi=0.8, threshold=0.9)
    info = _predict_info(_record("noul", p=0.6), pol)
    # Before: predicted was p >= band_hi, so False inside the band.
    assert (info["policy_action"], info["predicted"]) == ("ESCALATE", True)


def test_exporters_route_categorical_ties_and_null_threshold() -> None:
    tie = _record("choice", choice="B", confidence=0.45, probabilities={"A": 0.45, "B": 0.45, "C": 0.1})
    info = _predict_info(tie, _policy("choice", threshold=0.0))
    # Before: SELECT:B, the lexicographically largest tied label.
    assert (info["policy_action"], info["predicted"]) == ("ESCALATE", "A")
    clear = _record("choice", choice="A", confidence=0.9, probabilities={"A": 0.9, "B": 0.1})
    assert _predict_info(clear, _policy("choice", threshold=None))["policy_action"] == "ESCALATE"
    score = _record("score", score=10, confidence=0.6, probabilities={"2": 0.4, "10": 0.6})
    info = _predict_info(score, _policy("score", threshold=0.5))
    assert (info["policy_action"], info["predicted"]) == ("SELECT:10", "10")
    assert info["p_effective"] == 0.6
    # Labels are exported as the record wrote them, with or without a policy.
    padded = _record("score", score=1, confidence=0.9, probabilities={"00": 0.1, "01": 0.9})
    assert _predict_info(padded, None)["predicted"] == "01"
    info = _predict_info(padded, _policy("score", threshold=0.8))
    assert (info["policy_action"], info["predicted"]) == ("SELECT:01", "01")


@pytest.mark.parametrize("probs", [{"0": 0.2, "00": 0.3, "1": 0.5}, {"0": 1, "00": 0, "1": 0}])
def test_exporters_and_drift_agree_on_colliding_score_keys(probs: dict) -> None:
    # The parser folds "0" and "00" into one level (out of scope here); exporters must route
    # whatever it produced, as drift does, rather than renormalize their own copy.
    r = _record("score", score=1, confidence=0.5, probabilities=probs)
    pol = _policy("score", threshold=0.6)
    escalated = _escalation_rate([r], pol) == 1.0
    assert (_predict_info(r, pol)["policy_action"] == "ESCALATE") is escalated


def test_cli_gate_refuses_a_huge_score_level_in_one_line(tuned, capsys) -> None:
    if tuned["primitive"] != "score":
        pytest.skip("score only")
    probs = '{"0": 0.1, "1' + "0" * 400 + '": 0.9}'
    rc = cli_main(["gate", tuned["policy_raw"], "--question", QID["score"], "--model", "m1", "--probs", probs])
    assert rc == 1
    err = capsys.readouterr().err
    assert "too large" in err and "Traceback" not in err


def test_exported_probabilities_belong_to_the_exported_label() -> None:
    r = _record(
        "choice", choice="A", confidence=0.9,
        probabilities={"A": 0.9, "B": 0.1}, calib={"probabilities": {"A": 0.5, "B": 0.5}},
    )
    info = _predict_info(r, _policy("choice", threshold=0.4))
    assert (info["policy_action"], info["predicted"]) == ("ESCALATE", "A")
    assert (info["p_raw"], info["p_calib"], info["p_effective"]) == (0.9, 0.5, 0.5)


@pytest.mark.parametrize("command", [["export", "langfuse"], ["export", "phoenix"], ["drift"]])
def test_calibrated_keys_that_differ_from_the_raw_ones_are_one_line(tmp_path: Path, command: list, capsys) -> None:
    row = {
        "schema": "s1-record/1", "item_id": "i", "question_id": "q", "primitive": "choice",
        "model_requested": "m", "model_served": "m", "choice": "A", "confidence": 0.8,
        "probabilities": {"A": 0.8, "B": 0.2}, "calib": {"probabilities": {"A": 1.0}},
    }
    records = _write(tmp_path / "r.jsonl", [row])
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "s1-policy/1", "policies": [_policy("choice", question_id="q", threshold=0.5).to_dict()]}))
    if command == ["drift"]:
        args = ["drift", records, records]
    else:
        args = [*command, records, "-o", str(tmp_path / "out"), "--policy", str(policy)]
    assert cli_main(args) == 1
    err = capsys.readouterr().err
    assert "calib.probabilities keys" in err and "line 1" in err and "Traceback" not in err


def test_exporters_route_on_the_renormalized_top_probability() -> None:
    # 0.495 / 0.49 sums to 0.985; renormalized, A is 0.5025..., above a 0.5 threshold.
    r = _record("choice", choice="A", confidence=0.495, probabilities={"A": 0.495, "B": 0.49})
    info = _predict_info(r, _policy("choice", threshold=0.5))
    assert info["policy_action"] == "SELECT:A"
    assert info["p_effective"] == 0.495


def test_exports_mark_policy_routed_records_with_the_rule() -> None:
    r = _record("choice", choice="A", confidence=0.9, probabilities={"A": 0.9, "B": 0.1}, label="A")
    pol = _policy("choice", threshold=0.5)
    (span,) = build_openinference_spans([r], policies=[pol])
    assert span["attributes"]["s1.routing_rule"] == ROUTING_RULE
    events = build_langfuse_events([r], policies=[pol])
    observations = [e for e in json.loads(json.dumps(events, default=str)).get("observations", [])]
    assert observations and all(o["metadata"]["routing_rule"] == ROUTING_RULE for o in observations)
    (bare,) = build_openinference_spans([r])
    assert "s1.routing_rule" not in bare["attributes"]
    (obs,) = build_langfuse_events([r])["observations"]
    assert "routing_rule" not in obs["metadata"]


def test_drift_escalation_rate_uses_the_shared_rule() -> None:
    noul = [_record("noul", p=p) for p in (0.6, 0.95, 0.05)]
    # Before: escalated when p < threshold, so 0.6 and 0.05 -> 2/3. Now max(p, 1-p) < 0.8: only 0.6.
    assert _escalation_rate(noul, _policy("noul", threshold=0.8)) == round(1 / 3, 6)
    assert _escalation_rate(noul, _policy("noul", threshold=None)) == 1.0
    tie = _record("choice", choice="A", confidence=0.5, probabilities={"A": 0.5, "B": 0.5})
    clear = _record("choice", choice="A", confidence=0.9, probabilities={"A": 0.9, "B": 0.1})
    assert _escalation_rate([tie, clear], _policy("choice", threshold=0.0)) == 0.5


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> dict:
    """The seeded demo's calibrated band policy and fit, plus the same policy as an error budget."""
    from s1ops.demo import generate_workspace

    ws = generate_workspace(tmp_path_factory.mktemp("demo") / "ws")
    (band,) = load_policies(ws["policy"])
    return {
        "ws": ws,
        "qid": band.question_id,
        "band": band,
        "budget": dataclasses.replace(band, mode="error_budget", threshold=0.8),
        "fits": load_fits(ws["fits"]),
    }


@pytest.mark.parametrize("mode", ["band", "budget"])
@pytest.mark.parametrize("with_fit", [False, True])
@pytest.mark.parametrize(
    "p",
    [
        float("nan"), float("inf"), float("-inf"), 1.5, -0.2, True, np.bool_(True), None, "0.9", 1 + 0j,
        Decimal("0.5"), pytest.param(10**400, id="10**400"), pytest.param(Fraction(10**400, 1), id="Fraction(10**400)"),
        pytest.param(10**5000, id="10**5000"), pytest.param(Fraction(10**5000, 1), id="Fraction(10**5000)"),
        # Out of range, but each rounds into it as a float.
        pytest.param(Fraction(2**54 + 1, 2**54), id="Fraction(1+2**-54)"), pytest.param(Fraction(-1, 10**400), id="Fraction(-1e-400)"),
    ],
)
def test_noul_p_raw_that_is_not_a_probability_is_refused(demo, mode: str, with_fit: bool, p, monkeypatch) -> None:
    def transform(*args, **kwargs):
        raise AssertionError("calibrated before p_raw was checked")

    monkeypatch.setattr("s1ops.runtime.gate.transform", transform)
    gate = DecisionGate([demo[mode]], demo["fits"] if with_fit else None)
    with pytest.raises(ValueError, match=r"^p_raw must be a (number|finite number in \[0, 1\]), got ") as exc:
        gate.evaluate_noul(question_id=demo["qid"], model_served="jev-1.13.0", p_raw=p)
    assert not isinstance(exc.value, FitError)


@pytest.mark.parametrize(
    "p", [np.float16(0.5), np.float32(0.9), np.float64(0.9), np.int64(0), np.int64(1), Fraction(1, 2), 0, 1, 0.0, 1.0]
)
def test_noul_p_raw_takes_any_real_number_type(demo, p) -> None:
    gate = DecisionGate([demo["band"]], demo["fits"])
    d = gate.evaluate_noul(question_id=demo["qid"], model_served="jev-1.13.0", p_raw=p).as_dict()
    assert d == gate.evaluate_noul(question_id=demo["qid"], model_served="jev-1.13.0", p_raw=float(p)).as_dict()
    assert all(type(d[k]) is float for k in ("p_raw", "p_calibrated", "p_effective"))
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize("primitive", ["choice", "score"])
def test_categorical_probabilities_take_any_real_number_type(primitive: str) -> None:
    gate = _gate(primitive)
    fn = gate.evaluate_choice if primitive == "choice" else gate.evaluate_score
    classes = ["A", "B", "C"] if primitive == "choice" else [0, 1, 2]
    exotic = dict(zip(classes, [np.float32(0.25), Fraction(1, 4), np.float64(0.5)]))
    d = fn(question_id="q", model_served="m", probs_raw=exotic).as_dict()
    assert d == fn(question_id="q", model_served="m", probs_raw={c: float(v) for c, v in exotic.items()}).as_dict()
    assert all(type(v) is float for v in [d["p_raw"], d["p_effective"], *d["probabilities"].values()])
    assert type(d["expected_score"]) is float if primitive == "score" else d["expected_score"] is None
    json.dumps(d, allow_nan=False)


def test_the_policy_is_looked_up_before_p_raw_is_checked(demo, capsys) -> None:
    with pytest.raises(ModelServedMismatchError):
        DecisionGate([demo["band"]]).evaluate_noul(question_id=demo["qid"], model_served="other", p_raw=float("nan"))
    assert cli_main(["gate", demo["ws"]["policy"], "--question", demo["qid"], "--model", "other", "--p", "nan"]) == 3


@pytest.mark.parametrize("p", ["nan", "inf", "1.5", "-0.2"])
@pytest.mark.parametrize("with_fit", [False, True])
def test_cli_gate_refuses_p_outside_a_probability_in_one_line(demo, p: str, with_fit: bool, capsys) -> None:
    fit = ["--fit", demo["ws"]["fits"]] if with_fit else []
    rc = cli_main(["gate", demo["ws"]["policy"], *fit, "--question", demo["qid"], "--model", "jev-1.13.0", "--p", p])
    out, err = capsys.readouterr()
    assert rc == 1 and out == ""
    assert err.startswith("s1: p_raw must be a finite number in [0, 1]") and err.count("\n") == 1


@pytest.mark.parametrize(
    "calib, p_calib, action, predicted, brier",
    [
        ({"p": 0.9, "method": "platt", "in_fit_sample": False}, 0.9, "ALLOW", True, 0.01),
        ({"p": 1}, 1.0, "ALLOW", True, 0.0),
        ({"p": 0}, 0.0, "BLOCK", False, 1.0),
        ({"method": "platt"}, None, "ESCALATE", False, 0.49),
    ],
)
def test_exporters_route_the_calibrated_noul_p(calib: dict, p_calib, action: str, predicted: bool, brier: float) -> None:
    # Raw p 0.3 escalates under a 0.8 budget, so routing the raw value cannot pass for a calibrated one.
    r = _record("noul", p=0.3, label=True, calib=calib)
    p_eff = 0.3 if p_calib is None else p_calib
    common = {"p_raw": 0.3, "p_calib": p_calib, "p_effective": p_eff, "brier": brier}
    assert _predict_info(r, _policy("noul", threshold=0.8)) == {
        **common, "predicted": predicted, "policy_action": action, "routing_rule": ROUTING_RULE,
    }
    assert _predict_info(r, None) == {
        **common, "predicted": p_eff >= 0.5, "policy_action": "ALLOW" if p_eff >= 0.5 else "BLOCK", "routing_rule": None,
    }


@pytest.mark.parametrize("calib_p", ["nan", 1.5, -0.1, None, True])
def test_exporters_refuse_a_calibrated_noul_p_that_is_not_a_probability(calib_p) -> None:
    r = _record("noul", p=0.6, calib={"p": calib_p})
    for pol in (None, _policy("noul", threshold=0.8)):
        with pytest.raises(RecordError, match="calib.p"):
            _predict_info(r, pol)


@pytest.mark.parametrize("command", [["export", "langfuse"], ["export", "phoenix"]])
def test_cli_export_refuses_a_calibrated_noul_p_in_one_line(tmp_path: Path, command: list, capsys) -> None:
    row = {
        "schema": "s1-record/1", "item_id": "i", "question_id": "q", "primitive": "noul",
        "model_requested": "m", "model_served": "m", "p": 0.6, "calib": {"p": "nan"},
    }
    records = _write(tmp_path / "r.jsonl", [row])
    assert cli_main([*command, records, "-o", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert "calib.p" in err and "line 1" in err and "Traceback" not in err
