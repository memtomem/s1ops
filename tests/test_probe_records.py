"""A probe run leaves only records `read_records` takes, and nothing a backend returns can set what it should not.

A states row's `label` is checked before anything is reserved. A response is read back as a record before it is
settled, and only the answer keys are copied from it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import question_id
from s1ops.contracts.records import RecordError, parse_record, read_records, read_row
from s1ops.ledger.journal import Ledger, LedgerUnusable
from s1ops.lint.loader import parse_question
from s1ops.monitor import runner
from s1ops.monitor.runner import BackendError, BackendResponse, DeterministicMockBackend, check_labels, probe_tasks, run_batch

NOUL = {"type": "noul", "instructions": "Is it about money?"}
CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}
SCORE = {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]}
ONE_STATE = [{"item_id": "i", "state": "s"}]
COST = 1e-7
RAW = "RAW STATE TEXT"


class Custom:
    """A third-party backend that answers what it is told to, counting its calls."""

    def __init__(self, *, error: BackendError | None = None, **response) -> None:
        self.error = error
        self.response = {"model_served": "m", "answer": {"p": 0.6}, "cost_usd": COST, "latency_ms": 1.0, **response}
        self.calls = 0

    def infer(self, state, question, *, model_requested):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return BackendResponse(**self.response)


def _run(tmp_path: Path, backend, *, states=ONE_STATE, questions=None, **options) -> tuple[list[dict], list[dict]]:
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        records = run_batch(states, questions or {"q": NOUL}, backend=backend, ledger=ledger, model_requested="m", **options)
    return records, [json.loads(line) for line in journal.read_text().splitlines()]


def _probe_files(tmp_path: Path, questions: dict, label_literal: str, *, rows: int = 1) -> tuple[list[str], Path, Path, str]:
    """`s1 probe` argv over the questions, with the raw label literal on the last of `rows` states."""
    q_path, s_path, out, ledger = tmp_path / "q.json", tmp_path / "s.jsonl", tmp_path / "o.jsonl", tmp_path / "l.jsonl"
    q_path.write_text(json.dumps(questions))
    lines = [json.dumps({"item_id": f"i{n}", "state": "hello"}) for n in range(1, rows)]
    lines.append('{"item_id": "last", "state": "hello", "label": %s}' % label_literal)
    s_path.write_text("\n".join(lines) + "\n")
    return ["probe", str(q_path), "--states", str(s_path), "-o", str(out), "--ledger", str(ledger), "--no-expand-probes"], out, ledger, str(s_path)


REFUSED = [
    (NOUL, '"true"', "noul label must be a JSON boolean, got 'true'"),
    (NOUL, "1", "noul label must be a JSON boolean, got 1"),
    (NOUL, "1e400", "noul label must be a JSON boolean, got inf"),
    (NOUL, "NaN", "noul label must be a JSON boolean, got nan"),
    (CHOICE, '"z"', "choice label 'z' is not one of the options ['a', 'b']"),
    (CHOICE, "true", "choice label must be a string, got True"),
    (CHOICE, "1", "choice label must be a string, got 1"),
    (SCORE, "3", "score label 3 is not one of the levels [0, 1, 2]"),
    (SCORE, "-1", "score label -1 is not one of the levels [0, 1, 2]"),
    (SCORE, '"2"', "score label must be an integer level, got '2'"),
    (SCORE, "true", "score label must be an integer level, got True"),
    (SCORE, "2.0", "score label must be an integer level, got 2.0"),
]
ACCEPTED = [
    (NOUL, "true", True),
    (NOUL, "false", False),
    (NOUL, "null", None),
    (CHOICE, '"a"', "a"),
    (CHOICE, "null", None),
    (SCORE, "0", 0),
    (SCORE, "2", 2),
]


@pytest.mark.parametrize("question, literal, message", REFUSED)
def test_probe_refuses_a_label_before_anything_exists(tmp_path: Path, capsys, question, literal, message) -> None:
    argv, out, ledger, states = _probe_files(tmp_path, {"q": question}, literal)
    assert cli_main(argv) == 1
    assert capsys.readouterr().err == f"s1: {states}: row 1: question 'q': {message}\n"
    assert not out.exists() and not ledger.exists()


@pytest.mark.parametrize("question, literal, value", ACCEPTED)
def test_probe_accepts_a_label_the_reader_accepts(tmp_path: Path, question, literal, value) -> None:
    argv, out, _, _ = _probe_files(tmp_path, {"q": question}, literal)
    assert cli_main(argv) == 0
    assert [r.label for r in read_records(out)] == [value]


@pytest.mark.parametrize("question, literal, message", [c for c in REFUSED if c[1] not in ("1e400", "NaN")])
def test_the_reader_says_the_same_about_the_mock_s_answer(question, literal, message) -> None:
    """The check takes its options from the question and the reader from the response: the two must agree."""
    answer = DeterministicMockBackend().infer("s", parse_question("q", question, frozenset()).question, model_requested="m").answer
    envelope = {"schema": "s1-record/1", "item_id": "i", "question_id": "q", "primitive": question["type"], "model_requested": "m", "model_served": "m"}
    with pytest.raises(RecordError) as caught:
        read_row(json.dumps({**envelope, **answer, "label": json.loads(literal)}), 1, None)
    assert caught.value.reason == message


def test_one_label_cannot_suit_a_noul_and_a_choice(tmp_path: Path, capsys) -> None:
    argv, out, _, states = _probe_files(tmp_path, {"n": NOUL, "c": CHOICE}, "true")
    assert cli_main(argv) == 1
    assert capsys.readouterr().err == f"s1: {states}: row 1: question 'c': choice label must be a string, got True\n"
    assert not out.exists()


def test_a_bad_label_on_a_later_row_stops_the_run_before_any_call(tmp_path: Path) -> None:
    backend = Custom()
    states = [{"item_id": "a", "state": "s", "label": True}, {"item_id": "b", "state": "s", "label": "yes"}]
    with pytest.raises(ValueError, match=r"^row 2: question 'q': noul label must be a JSON boolean, got 'yes'$"):
        _run(tmp_path, backend, states=states)
    assert backend.calls == 0
    assert not (tmp_path / "journal.jsonl").exists()


def test_the_cli_reports_the_row_it_found(tmp_path: Path, capsys) -> None:
    argv, _, _, states = _probe_files(tmp_path, {"q": NOUL}, '"yes"', rows=3)
    assert cli_main(argv) == 1
    assert capsys.readouterr().err == f"s1: {states}: row 3: question 'q': noul label must be a JSON boolean, got 'yes'\n"


def test_only_the_noul_of_a_probe_carries_the_label(tmp_path: Path) -> None:
    states = [{"item_id": "i", "state": "s", "label": True}]
    records, _ = _run(tmp_path, DeterministicMockBackend(), states=states, probe_mode=True)
    assert {r["probe_role"]: r.get("label", "absent") for r in records} == {"noul": True, "twin": "absent", "inverse": "absent"}


def test_a_probe_label_is_checked_against_the_noul_alone() -> None:
    tasks = probe_tasks({"q": NOUL}, probe_mode=True)
    check_labels([{"item_id": "i", "state": "s", "label": True}], tasks)
    with pytest.raises(ValueError, match=r"^row 1: question 'q': noul label must be a JSON boolean, got 'yes'$"):
        check_labels([{"item_id": "i", "state": "s", "label": "yes"}], tasks)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"primitive": "noul", "p": 0.5, "label": "true"}, "noul label must be a JSON boolean, got 'true'"),
        ({"primitive": "noul", "p": 0.5, "label": "x" * 50}, "noul label must be a JSON boolean, got " + repr("x" * 40) + "... (50 characters)"),
        (
            {"primitive": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.5, "b": 0.5}, "label": 1},
            "choice label must be a string, got 1",
        ),
        (
            {"primitive": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.5, "b": 0.5}, "label": "z"},
            "choice label 'z' is not one of the options ['a', 'b']",
        ),
        (
            {"primitive": "score", "score": 0.5, "confidence": 0.5, "probabilities": {"0": 0.5, "1": 0.5}, "label": True},
            "score label must be an integer level, got True",
        ),
        (
            {"primitive": "score", "score": 0.5, "confidence": 0.5, "probabilities": {"0": 0.5, "1": 0.5}, "label": 2},
            "score label 2 is not one of the levels [0, 1]",
        ),
    ],
)
def test_the_parser_keeps_its_messages(extra: dict, message: str) -> None:
    row = {"schema": "s1-record/1", "item_id": "i", "question_id": "q", "model_requested": "m", "model_served": "m", **extra}
    with pytest.raises(RecordError) as caught:
        parse_record(row, 3)
    assert caught.value.reason == message


NOT_DUMPABLE = object()


@pytest.mark.parametrize(
    "response",
    [
        pytest.param({"model_served": ""}, id="served-empty"),
        pytest.param({"model_served": 7}, id="served-int"),
        pytest.param({"model_served": math.nan}, id="served-nan"),
        pytest.param({"model_served": {"x"}}, id="served-set"),
        pytest.param({"latency_ms": math.nan}, id="latency-nan"),
        pytest.param({"latency_ms": math.inf}, id="latency-inf"),
        pytest.param({"answer": {}}, id="answer-empty"),
        pytest.param({"answer": 5}, id="answer-not-a-mapping"),
        pytest.param({"answer": {"p": 1.5}}, id="p-out-of-range"),
        pytest.param({"answer": {"p": NOT_DUMPABLE}}, id="p-not-dumpable"),
    ],
)
def test_an_unreadable_response_is_recorded_as_invalid_and_settled_once(tmp_path: Path, response: dict) -> None:
    backend = Custom(**response)
    records, journal = _run(tmp_path, backend, max_retries=2)
    assert backend.calls == 1
    assert len(records) == 1
    assert (records[0]["model_served"], records[0]["error_code"]) == (None, "invalid_response")
    assert "latency_ms" not in records[0] and "p" not in records[0]
    read_row(json.dumps(records[0], sort_keys=True), 1, None)
    settled = [e for e in journal if e["event"] == "settle"]
    assert [(e["outcome"], e["usd"]) for e in settled] == [("measured", COST)]
    assert "model_served" not in settled[0]
    assert len([e for e in journal if e["event"] == "reserve"]) == 1


def test_an_unreadable_response_without_a_cost_is_charged_its_reservation(tmp_path: Path) -> None:
    records, journal = _run(tmp_path, Custom(model_served="", cost_usd=None))
    assert records[0]["error_code"] == "invalid_response"
    settled = [e for e in journal if e["event"] == "settle"]
    reserved = [e for e in journal if e["event"] == "reserve"]
    assert [(e["outcome"], e["usd"]) for e in settled] == [("unknown", reserved[0]["worst_usd"])]


@pytest.mark.parametrize("response", [{"model_served": "", "cost_usd": math.nan}, {"cost_usd": math.nan}, {"cost_usd": None}], ids=["unreadable", "readable-nan", "readable-none"])
def test_a_cost_the_ledger_refuses_stops_the_run_with_no_record(tmp_path: Path, response: dict) -> None:
    seen: list[dict] = []
    with pytest.raises(ValueError, match="^usd must be a finite non-negative number$"):
        _run(tmp_path, Custom(**response), on_record=seen.append)
    assert seen == []
    events = [json.loads(line)["event"] for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert events == ["reserve"]


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"state": RAW}, id="state"),
        pytest.param({"item_id": "someone-else"}, id="item_id"),
        pytest.param({"question_id": ""}, id="question_id"),
        pytest.param({"model_requested": ""}, id="model_requested"),
        pytest.param({"dataset": "another"}, id="dataset"),
        pytest.param({"split": "train"}, id="split"),
        pytest.param({"label": True}, id="label"),
        pytest.param({"calib": {"p": 0.9}}, id="calib"),
        pytest.param({"probe_role": "noul"}, id="probe_role"),
        pytest.param({"extra": math.inf}, id="unknown-non-finite"),
    ],
)
def test_other_answer_keys_are_dropped_and_the_paid_answer_kept(tmp_path: Path, extra: dict) -> None:
    records, journal = _run(tmp_path, Custom(answer={"p": 0.6, **extra}))
    record = records[0]
    assert record["p"] == 0.6 and record["model_served"] == "m" and "error_code" not in record
    assert (record["item_id"], record["model_requested"], record["dataset"]) == ("i", "m", "default")
    assert record["question_id"] == question_id(parse_question("q", NOUL, frozenset()).question)
    assert set(record) == {"schema", "dataset", "item_id", "state_sha256", "question_id", "primitive", "model_requested", "model_served", "latency_ms", "p"}
    read_row(json.dumps(record, sort_keys=True), 1, None)
    assert [(e["outcome"], e["usd"]) for e in journal if e["event"] == "settle"] == [("measured", COST)]


def test_raw_state_from_a_backend_never_reaches_the_written_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("s1ops.cli._probe_backend", lambda args: Custom(answer={"p": 0.6, "state": RAW}))
    argv, out, ledger, _ = _probe_files(tmp_path, {"q": NOUL}, "true")
    assert cli_main(argv) == 0
    assert RAW.encode() not in out.read_bytes()
    assert RAW.encode() not in ledger.read_bytes()
    assert [r.label for r in read_records(out)] == [True]


@pytest.mark.parametrize("code", [math.nan, math.inf, {"x"}], ids=["nan", "inf", "set"])
def test_a_failure_record_with_an_unreadable_code_says_invalid_response(tmp_path: Path, code) -> None:
    records, journal = _run(tmp_path, Custom(error=BackendError(code, sent=True, usage_usd=COST)))
    assert (records[0]["model_served"], records[0]["error_code"]) == (None, "invalid_response")
    read_row(json.dumps(records[0], sort_keys=True), 1, None)
    assert [(e["outcome"], e["usd"]) for e in journal if e["event"] == "settle"] == [("measured", COST)]


def test_a_failure_record_keeps_a_code_the_reader_takes(tmp_path: Path) -> None:
    records, _ = _run(tmp_path, Custom(error=BackendError("timeout", sent=None)))
    assert records[0]["error_code"] == "timeout"


def test_an_integer_code_is_left_as_the_reader_takes_it(tmp_path: Path) -> None:
    records, _ = _run(tmp_path, Custom(error=BackendError(7, sent=None)))
    assert records[0]["error_code"] == 7
    read_row(json.dumps(records[0], sort_keys=True), 1, None)


def test_an_overrun_on_an_unreadable_response_records_it_and_then_stops(tmp_path: Path) -> None:
    backend = Custom(model_served="", cost_usd=0.5)
    seen: list[dict] = []
    states = [{"item_id": "a", "state": "s"}, {"item_id": "b", "state": "s"}]
    with pytest.raises(LedgerUnusable):
        _run(tmp_path, backend, states=states, on_record=seen.append, max_retries=2)
    assert backend.calls == 1
    assert [(r["item_id"], r["error_code"]) for r in seen] == [("a", "invalid_response")]


def test_a_retry_that_succeeds_records_its_own_answer(tmp_path: Path) -> None:
    class FailsOnce(Custom):
        def infer(self, state, question, *, model_requested):
            self.calls += 1
            if self.calls == 1:
                raise BackendError("timeout", sent=None, retryable=True)
            return BackendResponse(**self.response)

    backend = FailsOnce()
    records, journal = _run(tmp_path, backend, max_retries=1, sleep=lambda _: None)
    assert backend.calls == 2
    assert records[0]["p"] == 0.6 and "error_code" not in records[0]
    assert [e["outcome"] for e in journal if e["event"] == "settle"] == ["unknown", "measured"]


def test_an_interrupt_while_reading_back_is_charged_its_reservation_once(tmp_path: Path, monkeypatch) -> None:
    def interrupted(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_readable", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, Custom())
    events = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [e.get("outcome") for e in events if e["event"] == "settle"] == ["unknown"]


def _expected(question: dict, state: dict, *, tail: list[tuple[str, object]] = (), probe: dict | None = None) -> list[tuple[str, object]]:
    """A mock answer's record, key by key, in the order `run_batch` has always built it."""
    model = parse_question("q", probe or question, frozenset()).question
    answer = DeterministicMockBackend().infer(state["state"], model, model_requested="m")
    head: list[tuple[str, object]] = [
        ("schema", "s1-record/1"),
        ("dataset", state.get("dataset", "default")),
        ("item_id", state["item_id"]),
        ("state_sha256", hashlib.sha256(state["state"].encode()).hexdigest()),
        ("question_id", question_id(model)),
        ("primitive", model.type),
        ("model_requested", "m"),
    ]
    if state.get("split") is not None:
        head.append(("split", state["split"]))
    return [*head, ("model_served", "m"), ("latency_ms", answer.latency_ms), *answer.answer.items(), *tail]


@pytest.mark.parametrize(
    "question, state, tail",
    [
        pytest.param(NOUL, {"item_id": "i", "state": "s"}, [], id="noul"),
        pytest.param(NOUL, {"item_id": "i", "state": "s", "label": False}, [("label", False)], id="noul-labeled"),
        pytest.param(NOUL, {"item_id": "i", "state": "s", "label": None}, [("label", None)], id="noul-null-label"),
        pytest.param(NOUL, {"dataset": "d", "item_id": "i", "state": "s", "split": "train"}, [], id="noul-split"),
        pytest.param(CHOICE, {"item_id": "i", "state": "s", "label": "a"}, [("label", "a")], id="choice"),
        pytest.param(SCORE, {"item_id": "i", "state": "s", "label": 1}, [("label", 1)], id="score"),
    ],
)
def test_a_good_answer_is_the_same_record_key_by_key(tmp_path: Path, question: dict, state: dict, tail: list) -> None:
    records, _ = _run(tmp_path, DeterministicMockBackend(), states=[state], questions={"q": question})
    assert list(records[0].items()) == _expected(question, state, tail=tail)


def test_each_probe_role_is_the_same_record_key_by_key(tmp_path: Path) -> None:
    state = {"item_id": "i", "state": "s", "label": True}
    records, _ = _run(tmp_path, DeterministicMockBackend(), states=[state], probe_mode=True)
    by_role = {r["probe_role"]: r for r in records}
    tasks = {t.role: t for t in probe_tasks({"q": NOUL}, probe_mode=True)}
    for role, tail in (
        ("noul", [("label", True), ("probe_role", "noul"), ("probe_question", "q")]),
        ("twin", [("probe_role", "twin"), ("probe_question", "q")]),
        ("inverse", [("probe_role", "inverse"), ("probe_question", "q"), ("probe_inverse", "generated")]),
    ):
        wire = tasks[role].question.wire_dict()
        assert list(by_role[role].items()) == _expected(NOUL, state, tail=tail, probe=wire), role


def test_an_answer_keeps_the_order_the_backend_gave_it(tmp_path: Path) -> None:
    answer = {"confidence": 0.6, "extra": 1, "probabilities": {"b": 0.4, "a": 0.6}, "choice": "a"}
    records, _ = _run(tmp_path, Custom(answer=answer), questions={"q": CHOICE})
    assert list(records[0])[-3:] == ["confidence", "probabilities", "choice"]


def test_a_failed_request_is_the_same_record_key_by_key(tmp_path: Path) -> None:
    state = {"dataset": "d", "item_id": "i", "state": "s", "split": "train", "label": True}
    records, _ = _run(tmp_path, Custom(error=BackendError("timeout", sent=None)), states=[state])
    assert list(records[0].items()) == [
        ("schema", "s1-record/1"),
        ("dataset", "d"),
        ("item_id", "i"),
        ("state_sha256", hashlib.sha256(b"s").hexdigest()),
        ("question_id", question_id(parse_question("q", NOUL, frozenset()).question)),
        ("primitive", "noul"),
        ("model_requested", "m"),
        ("split", "train"),
        ("model_served", None),
        ("error_code", "timeout"),
        ("label", True),
    ]
