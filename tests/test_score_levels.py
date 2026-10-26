"""One score-level rule (`records.score_level`) for the parser, the gate, compare and the exporters.

A key is a non-negative int or a string of 1 to 20 ASCII digits naming a level in `0..2**53`, and no
two keys of one answer may name the same level.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from s1ops.bridges.langfuse import _predict_info, build_langfuse_events
from s1ops.bridges.openinference import build_openinference_spans
from s1ops.cli import main as cli_main
from s1ops.contracts.records import MAX_SCORE_LEVEL, RecordError, parse_record, score_level
from s1ops.contracts.schemas import PolicyEntry
from s1ops.eval.compare import CompareInputError, compare_records
from s1ops.runtime.gate import DecisionGate

TOO_LARGE = [
    MAX_SCORE_LEVEL + 1,
    str(MAX_SCORE_LEVEL + 1),
    "0" * 20 + "1",
    10**5000,
    "1" + "0" * 5000,
]
NOT_A_LEVEL = ["", "٣", "+1", " 1", "1.0", "-1", -1, True, 1.0, None]


@pytest.fixture
def min_int_digits():
    """The smallest `int()` digit limit Python allows, so a conversion past it would fail loudly."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    yield
    sys.set_int_max_str_digits(old)


def _score_row(probs: dict, **extra) -> dict:
    row = {
        "schema": "s1-record/1", "item_id": "i", "question_id": "q", "primitive": "score",
        "model_requested": "m", "model_served": "m", "score": 1, "confidence": max(probs.values()),
        "probabilities": probs,
    }
    return {**row, **extra}


def _policy(**kw) -> PolicyEntry:
    base = {"question_id": "q", "model_served": "m", "primitive": "score", "mode": "error_budget", "split": "eval", "n": 1, "threshold": 0.5}
    return PolicyEntry(**{**base, **kw})


def test_the_largest_level_is_2_53() -> None:
    assert MAX_SCORE_LEVEL == 2**53 == 9007199254740992
    assert score_level(9007199254740992) == score_level("9007199254740992") == 2**53
    for key in (9007199254740993, "9007199254740993"):
        with pytest.raises(ValueError, match="too large"):
            score_level(key)


@pytest.mark.parametrize(
    "key, level",
    [(0, 0), (1, 1), (MAX_SCORE_LEVEL, MAX_SCORE_LEVEL), ("0", 0), ("000", 0), ("01", 1),
     (str(MAX_SCORE_LEVEL), MAX_SCORE_LEVEL), ("0" * 19 + "1", 1)],
)
def test_levels_up_to_2_53_are_accepted(key, level: int, min_int_digits) -> None:
    assert score_level(key) == level
    assert float(level) == level  # exact


@pytest.mark.parametrize("key", TOO_LARGE, ids=lambda k: f"{type(k).__name__}:{len(str(k)) if not isinstance(k, int) else k.bit_length()}")
def test_larger_levels_are_refused_in_a_short_message(key, min_int_digits) -> None:
    with pytest.raises(ValueError, match="too large") as exc:
        score_level(key)
    assert len(str(exc.value)) < 160


class _NoRepr(int):
    def __repr__(self) -> str:
        raise AssertionError("a large int was converted to decimal for a message")


@pytest.mark.parametrize("value", [_NoRepr(2**64 + 1), _NoRepr(10**5000)], ids=["2**64+1", "10**5000"])
def test_large_ints_are_not_converted_to_decimal_for_a_message(value: int) -> None:
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)  # no limit: only the size check stops a slow conversion
    try:
        with pytest.raises(ValueError, match=r"score level <int of \d+ bits> is too large"):
            score_level(value)
        with pytest.raises(ValueError, match=r"p_raw must be a finite number in \[0, 1\], got <int of \d+ bits>"):
            DecisionGate([_policy(primitive="noul")]).evaluate_noul(question_id="q", model_served="m", p_raw=value)
    finally:
        sys.set_int_max_str_digits(old)


@pytest.mark.parametrize("key", NOT_A_LEVEL, ids=repr)
def test_other_keys_are_not_levels(key) -> None:
    with pytest.raises(ValueError, match="non-negative integers or their decimal strings"):
        score_level(key)


@pytest.mark.parametrize(
    "probs, keys",
    [({"0": 0.2, "00": 0.3, "1": 0.5}, ("'0'", "'00'")), ({"01": 0.4, "2": 0.2, "1": 0.4}, ("'01'", "'1'"))],
)
@pytest.mark.parametrize("use_calib", [False, True])
def test_the_parser_refuses_two_keys_for_one_level(probs: dict, keys: tuple, use_calib: bool) -> None:
    calib = {"calib": {"probabilities": dict(reversed(list(probs.items())))}} if use_calib else {}
    with pytest.raises(RecordError, match=rf"^line 7: score keys {keys[0]} and {keys[1]} both name level|^line 7: score keys {keys[1]} and {keys[0]} both name level"):
        parse_record(_score_row(probs, **calib), 7, use_calib=use_calib)


@pytest.mark.parametrize("key", TOO_LARGE[1:3] + ["1" + "0" * 5000], ids=["2**53+1", "21 digits", "5000 digits"])
def test_the_parser_refuses_a_large_level_with_its_line(key: str, min_int_digits) -> None:
    with pytest.raises(RecordError, match=r"^line 3: score level .* is too large"):
        parse_record(_score_row({"0": 0.5, key: 0.5}), 3)


@pytest.mark.parametrize("probs", [{0: 0.5, "1": 0.5}, {"0": 0.5, 1: 0.5}])
def test_the_parser_takes_string_keys_only(probs: dict) -> None:
    # JSON object keys are strings; an int key in a hand-built row is refused, as before.
    with pytest.raises(RecordError, match=r"^line 4: score level keys must be strings of digits, got [01]$"):
        parse_record(_score_row(probs), 4)


def test_padded_calibrated_keys_parse_as_before() -> None:
    raw = {"10": 0.2, "01": 0.3, "00": 0.5}
    cal = {"00": 0.1, "10": 0.61, "01": 0.3}
    r = parse_record(_score_row(raw, calib={"probabilities": cal}), use_calib=True)
    total = math.fsum(cal.values())
    assert r.classes == (0, 1, 10)
    assert r.probs == (0.1 / total, 0.3 / total, 0.61 / total)
    r = parse_record(_score_row(raw, calib={"probabilities": cal}))
    assert r.classes == (0, 1, 10) and r.probs == (0.5, 0.3, 0.2)


def test_the_gate_takes_the_same_levels() -> None:
    gate = DecisionGate([_policy()])
    d = gate.evaluate_score(question_id="q", model_served="m", probs_raw={0: 0.25, MAX_SCORE_LEVEL: 0.75})
    assert d.predicted == MAX_SCORE_LEVEL and d.expected_score == 0.75 * MAX_SCORE_LEVEL
    assert gate.evaluate_score(question_id="q", model_served="m", probs_raw={"0": 0.25, "0" * 19 + "1": 0.75}).predicted == 1
    for key in TOO_LARGE:
        with pytest.raises(ValueError, match="too large"):
            gate.evaluate_score(question_id="q", model_served="m", probs_raw={0: 0.5, key: 0.5})


@pytest.mark.parametrize(
    "probs, message",
    [
        ({"0": 0.2, "00": 0.3, "1": 0.5}, "both name level 0"),
        ({"0": 0.5, str(MAX_SCORE_LEVEL + 1): 0.5}, "too large"),
        ({"0": 0.5, "1" + "0" * 5000: 0.5}, "too large"),
        ({"0": 0.5, "x": 0.5}, "non-negative integers"),
    ],
)
def test_compare_refuses_the_same_keys_from_dicts(probs: dict, message: str, min_int_digits) -> None:
    good = _score_row({"0": 0.4, "1": 0.6})
    with pytest.raises(CompareInputError, match=message):
        compare_records([good], [_score_row(probs)])


def test_compare_predicts_padded_levels() -> None:
    rep = compare_records([_score_row({"00": 0.4, "01": 0.6})], [_score_row({"0": 0.7, "1": 0.3})])
    ((flipped,),) = [q["flipped_items"] for q in rep["questions"]]
    assert (flipped["primary_pred"], flipped["shadow_pred"]) == (1, 0)


def test_exporters_keep_long_padded_labels_under_a_policy() -> None:
    long_one = "0" * 19 + "1"
    r = parse_record(_score_row({"0": 0.1, long_one: 0.9}, label=1))
    pol = _policy(threshold=0.5)
    info = _predict_info(r, pol)
    assert (info["policy_action"], info["predicted"], info["p_effective"]) == (f"SELECT:{long_one}", long_one, 0.9)
    (span,) = build_openinference_spans([r], policies=[pol])
    assert span["attributes"]["s1.policy_action"] == f"SELECT:{long_one}"
    (obs,) = build_langfuse_events([r], policies=[pol])["observations"]
    assert obs["output"]["predicted"] == long_one


def test_audit_reports_finite_numbers_at_the_largest_level(tmp_path: Path, capsys) -> None:
    rows = [
        _score_row({"0": 1.0, str(MAX_SCORE_LEVEL): 0.0}, item_id=f"i{i}", label=MAX_SCORE_LEVEL)
        for i in range(3)
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cli_main(["audit", str(path), "--boot", "0"]) == 0
    json.loads(capsys.readouterr().out, parse_constant=lambda c: pytest.fail(f"non-finite {c} in audit output"))


@pytest.mark.parametrize("probs", [{"0": 0.2, "00": 0.3, "1": 0.5}, {"0": 0.5, str(MAX_SCORE_LEVEL + 1): 0.5}], ids=["collision", "2**53+1"])
@pytest.mark.parametrize("command", ["audit", "fit", "compare", "drift", "export langfuse", "export phoenix"])
def test_commands_refuse_bad_score_keys_in_one_line(tmp_path: Path, command: str, probs: dict, capsys) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(_score_row(probs, split="cal", label=0)) + "\n")
    r = str(path)
    args = {
        "audit": ["audit", r],
        "fit": ["fit", r, "-o", str(tmp_path / "fits.json")],
        "compare": ["compare", r, r],
        "drift": ["drift", r, r],
        "export langfuse": ["export", "langfuse", r, "-o", str(tmp_path / "out")],
        "export phoenix": ["export", "phoenix", r, "-o", str(tmp_path / "out")],
    }[command]
    assert cli_main(args) == 1
    err = capsys.readouterr().err
    assert err.startswith("s1: ") and err.count("\n") == 1
    assert "line 1" in err and ("both name level 0" in err or "too large" in err)


def test_cli_gate_refuses_a_5000_digit_level_in_our_words(tmp_path: Path, capsys) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "s1-policy/1", "policies": [_policy().to_dict()]}))
    probs = '{"0": 0.5, "1' + "0" * 5000 + '": 0.5}'
    assert cli_main(["gate", str(policy), "--question", "q", "--model", "m", "--probs", probs]) == 1
    err = capsys.readouterr().err
    assert err.startswith("s1: score level '1000") and "too large" in err and len(err) < 200
