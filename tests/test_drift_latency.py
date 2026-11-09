"""`s1 drift` reports null for a latency statistic it cannot compute.

`latency_ms` is not validated, and finite values overflow too: a sum past the float range, or an
interpolation between opposite extremes. Refusing them would need a ceiling nothing specifies.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import parse_record
from s1ops.demo import generate_workspace
from s1ops.monitor.drift import _latency_summary
from s1ops.web.app import create_app

HUGE = 10**400
ENVELOPE = {"schema": "s1-record/1", "question_id": "q", "primitive": "noul", "model_requested": "m", "model_served": "m", "p": 0.5}


def _summary(values: list) -> dict:
    records = [parse_record({**ENVELOPE, "item_id": f"i{n}", "latency_ms": v}, n + 1) for n, v in enumerate(values)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return _latency_summary(records)


@pytest.mark.parametrize(
    "values, p50, p95, mean",
    [
        pytest.param([1e308] * 4, 1e308, 1e308, None, id="sum-overflows"),
        pytest.param([1e308], 1e308, 1e308, 1e308, id="one-large-value"),
        pytest.param([1.7e308, -1.7e308], None, None, 0.0, id="interpolation-overflows"),
        pytest.param([1.7e308, -1.7e308] * 2, None, 1.7e308, 0.0, id="median-overflows"),
        pytest.param([HUGE, 1.0], None, None, None, id="huge-integer"),
        pytest.param([-HUGE, 1.0], None, None, None, id="huge-negative-integer"),
        pytest.param([-HUGE, 10, 20, 30], 15.0, 28.5, None, id="huge-negative-among-ordinary"),
        pytest.param([10, 20, 30, HUGE], 25.0, None, None, id="huge-among-ordinary"),
        pytest.param([HUGE, -HUGE], None, None, None, id="huge-of-both-signs"),
        pytest.param([math.inf, 1.0], None, None, None, id="decoded-inf"),
        pytest.param([math.nan, 1.0], None, None, None, id="decoded-nan"),
    ],
)
def test_a_statistic_that_is_not_finite_is_null(values: list, p50, p95, mean) -> None:
    assert _summary(values) == {"p50_ms": p50, "p95_ms": p95, "mean_ms": mean}


@pytest.mark.parametrize(
    "values, p50, p95, mean",
    [
        pytest.param([10.0, 20.0, 30.0, 40.0], 25.0, 38.5, 25.0, id="floats"),
        pytest.param([10, 20, 30, 40], 25.0, 38.5, 25.0, id="integers"),
        pytest.param([2**53 + 1, 2**53 + 1], 9007199254740992.0, 9007199254740992.0, 9007199254740992.0, id="past-2-53"),
        pytest.param([-5.0, 5.0], 0.0, 4.5, 0.0, id="negative"),
        pytest.param([0.123456, 0.123456], 0.12, 0.12, 0.12, id="rounded"),
        pytest.param([1e300, 1e300], 1e300, 1e300, 1e300, id="large-and-finite"),
    ],
)
def test_ordinary_values_are_summarized_as_they_were(values: list, p50: float, p95: float, mean: float) -> None:
    summary = _summary(values)
    assert [summary[k].hex() for k in ("p50_ms", "p95_ms", "mean_ms")] == [p50.hex(), p95.hex(), mean.hex()]


@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param(["fast", True, None, 5.0], 5.0, id="a-number-among-others"),
        pytest.param(["fast", True, None], None, id="no-number"),
    ],
)
def test_what_is_not_a_number_is_left_out(values: list, expected) -> None:
    assert _summary(values) == {"p50_ms": expected, "p95_ms": expected, "mean_ms": expected}


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _one_question(source: str, values: list) -> tuple[str, str]:
    """`source` rewritten as `len(values)` answered rows of its first question, holding `values`."""
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("model_served") is not None]
    qid = rows[0]["question_id"]
    rows = [r for r in rows if r["question_id"] == qid][: len(values)]
    assert len(rows) == len(values)
    Path(source).write_text("".join(json.dumps({**r, "latency_ms": v}) + "\n" for r, v in zip(rows, values)))
    return qid, source


def _no_constant(name: str):
    pytest.fail(f"{name} in the report")


CASES = [
    pytest.param([1e308] * 4, {"p50_ms": 1e308, "p95_ms": 1e308, "mean_ms": None}, id="sum-overflows"),
    pytest.param([1.7e308, -1.7e308], {"p50_ms": None, "p95_ms": None, "mean_ms": 0.0}, id="interpolation-overflows"),
    pytest.param([HUGE, 1.0], {"p50_ms": None, "p95_ms": None, "mean_ms": None}, id="huge-integer"),
]


@pytest.mark.parametrize("values, expected", CASES)
@pytest.mark.parametrize("side", ["latency_cur", "latency_ref"])
def test_the_cli_prints_valid_json_with_null_and_no_warning(ws, capsys, values: list, expected: dict, side: str) -> None:
    reference, current = ws["records_train"], ws["records_canary"]
    qid, _ = _one_question(current if side == "latency_cur" else reference, values)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert cli_main(["drift", reference, current]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out, parse_constant=_no_constant)
    assert next(q for q in report["questions"] if q["question_id"] == qid)[side] == expected


@pytest.mark.parametrize("values, expected", CASES)
def test_the_web_answers_200_with_null(ws, values: list, expected: dict) -> None:
    qid, _ = _one_question(ws["records_canary"], values)
    response = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get("/api/drift")
    assert response.status_code == 200
    assert next(q for q in response.json()["questions"] if q["question_id"] == qid)["latency_cur"] == expected
