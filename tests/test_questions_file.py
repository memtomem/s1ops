"""`s1 lint check`, `s1 probe` and `s1 labelaudit --questions` read a questions file one way."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import CliError, _question_from_file
from s1ops.cli import main as cli_main
from s1ops.contracts.questions import NoulQuestion, parse_wire_question, question_id
from s1ops.demo import generate_workspace
from s1ops.ledger.journal import Ledger
from s1ops.lint.lint import KNOWN_RULES
from s1ops.lint.loader import MAX_NESTING, load_questions
from s1ops.monitor.runner import DeterministicMockBackend, probe_tasks, run_batch
from s1ops.web.app import create_app

A = {"type": "noul", "instructions": "Is this about billing?"}
B = {"type": "noul", "instructions": "Is this about refunds?"}
QID_A, QID_B = question_id(parse_wire_question(A)), question_id(parse_wire_question(B))


@pytest.fixture(autouse=True)
def _default_digit_limit():
    """`PYTHONINTMAXSTRDIGITS` can move the limit the 5,000-digit case relies on."""
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    yield
    sys.set_int_max_str_digits(before)


def _nested(levels: int) -> str:
    """A document nesting `levels` containers: the file's object, the question's, then `instructions`."""
    inner = levels - 2
    return '{"q": {"type": "noul", "instructions": ' + '{"x": ' * inner + '"y"' + "}" * inner + "}}"


def _write(tmp_path: Path, doc) -> Path:
    path = tmp_path / "q.json"
    if isinstance(doc, bytes):
        path.write_bytes(doc)
    else:
        path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return path


def _probe(tmp_path: Path, q: Path, capsys) -> tuple[int, Path, list[str]]:
    states = tmp_path / "s.jsonl"
    states.write_text(json.dumps({"dataset": "d", "item_id": "i", "state": "s"}) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    rc = cli_main(["probe", str(q), "--states", str(states), "-o", str(run / "out.jsonl"), "--ledger", str(run / "j.jsonl"), "--no-expand-probes"])
    return rc, run, capsys.readouterr().err.strip().splitlines()


def _lint_ids(q: Path) -> dict[str, str]:
    loaded = load_questions(q, KNOWN_RULES)
    assert loaded.errors == [] and all(spec.errors == [] for spec in loaded.specs)
    return {spec.name: spec.qid for spec in loaded.specs}


ACCEPTED = [
    pytest.param({"a": A}, {"a": QID_A}, id="plain"),
    pytest.param({"questions": {"a": A, "b": B}}, {"a": QID_A, "b": QID_B}, id="wrapper"),
    pytest.param({"questions": A}, {"questions": QID_A}, id="a-question-named-questions"),
    pytest.param({"questions": {"questions": A}}, {"questions": QID_A}, id="wrapper-around-a-question-named-questions"),
    pytest.param({"questions": A, "b": B}, {"questions": QID_A, "b": QID_B}, id="named-questions-beside-another"),
    pytest.param({"questions": {"type": A, "b": B}}, {"type": QID_A, "b": QID_B}, id="wrapper-with-a-question-named-type"),
]


@pytest.mark.parametrize("doc, expected", ACCEPTED)
def test_the_three_commands_see_the_same_questions(tmp_path, capsys, doc, expected) -> None:
    q = _write(tmp_path, doc)
    assert _lint_ids(q) == expected
    assert cli_main(["lint", "check", str(q), "--format", "json"]) == 0
    capsys.readouterr()
    rc, run, err = _probe(tmp_path, q, capsys)
    assert (rc, err) == (0, [])
    rows = [json.loads(line) for line in (run / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(row["question_id"] for row in rows) == sorted(expected.values())
    for qid in expected.values():
        assert question_id(_question_from_file(str(q), qid)) == qid


REFUSED = [
    pytest.param({"ok": A, "broken": {}}, "question 'broken'", id="empty-entry"),
    pytest.param({"ok": A, "broken": {"instructions": "x?"}}, "question 'broken'", id="no-type"),
    pytest.param({"ok": A, "broken": None}, "question 'broken': a question must be a JSON object", id="null-entry"),
    pytest.param({"ok": A, "broken": []}, "question 'broken': a question must be a JSON object", id="list-entry"),
    pytest.param({"ok": A, "broken": "x"}, "question 'broken': a question must be a JSON object", id="string-entry"),
    pytest.param({"ok": A, "broken": 7}, "question 'broken': a question must be a JSON object", id="number-entry"),
    pytest.param({"ok": A, "broken": True}, "question 'broken': a question must be a JSON object", id="bool-entry"),
    pytest.param(f'{{"a": {json.dumps(A)}, "a": {json.dumps(B)}}}', "question name 'a' appears more than once", id="repeated-name"),
    pytest.param(f'{{"questions": {{"a": {json.dumps(A)}, "a": {json.dumps(B)}}}}}', "question name 'a' appears more than once", id="repeated-name-in-wrapper"),
    pytest.param(f'{{"questions": {{"a": {json.dumps(A)}}}, "questions": {{"a": {json.dumps(A)}}}}}', "question name 'questions' appears more than once", id="repeated-wrapper"),
    pytest.param({}, "the file has no questions", id="empty"),
    pytest.param({"questions": {}}, "the file has no questions", id="empty-wrapper"),
    pytest.param({"questions": {"a": A}, "version": 1}, "question 'questions'", id="wrapper-with-sibling"),
    pytest.param({"questions": {"type": None}}, "question 'type': a question must be a JSON object", id="wrapper-type-null"),
    pytest.param({"questions": []}, "question 'questions': a question must be a JSON object", id="wrapper-list"),
    pytest.param([A], "the file must be a JSON object mapping question names to questions", id="top-level-array"),
    pytest.param(_nested(MAX_NESTING + 1), f"nested more than {MAX_NESTING} levels", id="one-level-too-deep"),
    pytest.param(_nested(990), f"nested more than {MAX_NESTING} levels", id="near-the-interpreter-limit"),
    pytest.param(_nested(200_000), f"nested more than {MAX_NESTING} levels", id="past-the-decoder-limit"),
    pytest.param('{"a": ' + "1" * 5000 + "}", "not valid JSON", id="integer-past-the-digit-limit"),
    pytest.param('{"a": ', "not valid JSON", id="not-json"),
    pytest.param(b'{"a": "\xff"}', "not valid UTF-8", id="not-utf8"),
]


@pytest.mark.parametrize("doc, expected", REFUSED)
def test_probe_refuses_before_anything_exists(tmp_path, capsys, doc, expected) -> None:
    q = _write(tmp_path, doc)
    rc, run, err = _probe(tmp_path, q, capsys)
    assert rc == 1 and len(err) == 1
    assert err[0].startswith(f"s1: {q}: ") and expected in err[0]
    assert not run.exists()  # so no journal, `.lock` or output either


@pytest.mark.parametrize("doc, expected", REFUSED)
def test_labelaudit_refuses_the_same_files(tmp_path, doc, expected) -> None:
    q = _write(tmp_path, doc)
    with pytest.raises(CliError) as refused:
        _question_from_file(str(q), QID_A)
    assert str(refused.value).startswith(f"{q}: ") and expected in str(refused.value)


@pytest.mark.parametrize("doc, expected", REFUSED)
def test_lint_reports_the_same_files_as_a_finding(tmp_path, capsys, doc, expected) -> None:
    q = _write(tmp_path, doc)
    assert cli_main(["lint", "check", str(q), "--format", "json"]) == 1
    findings = json.loads(capsys.readouterr().out)["findings"]
    assert {f["rule"] for f in findings if f["severity"] == "error"} == {"JL000"}
    reported = [f"question {f['question']!r}: {f['message']}" if f.get("question") else f["message"] for f in findings]
    assert any(expected in line for line in reported)


def test_nesting_at_the_limit_is_read_by_all_three(tmp_path, capsys) -> None:
    q = _write(tmp_path, _nested(MAX_NESTING))
    (qid,) = _lint_ids(q).values()
    assert cli_main(["lint", "check", str(q), "--format", "json"]) in (0, 1)  # a report, never a traceback
    capsys.readouterr()
    assert question_id(_question_from_file(str(q), qid)) == qid
    rc, _, err = _probe(tmp_path, q, capsys)
    assert (rc, err) == (0, [])


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file without permission")
def test_a_read_failure_names_the_file_once(tmp_path, capsys) -> None:
    q = _write(tmp_path, {"a": A})
    q.chmod(0)
    try:
        rc, run, err = _probe(tmp_path, q, capsys)
        with pytest.raises(PermissionError):
            _question_from_file(str(q), QID_A)
        assert cli_main(["lint", "check", str(q), "--format", "json"]) == 1
        report = capsys.readouterr().out
    finally:
        q.chmod(0o644)
    assert rc == 1 and err == [f"s1: {q}: Permission denied"]
    assert not run.exists()
    assert "JL000" in report and "cannot read" in report


def test_a_fifo_is_read_once(tmp_path, capsys) -> None:
    fifo = tmp_path / "q.fifo"
    os.mkfifo(fifo)
    # A daemon, so a command that fails before opening the FIFO cannot keep the test process alive.
    writer = threading.Thread(target=lambda: fifo.write_text(json.dumps({"a": A}), encoding="utf-8"), daemon=True)
    writer.start()
    try:
        rc, run, err = _probe(tmp_path, fifo, capsys)
    finally:
        writer.join(timeout=5)
    assert (rc, err) == (0, [])
    assert json.loads((run / "out.jsonl").read_text(encoding="utf-8"))["question_id"] == QID_A


@pytest.mark.parametrize("value", [None, [], "x", 7, True, object()], ids=["none", "list", "str", "int", "bool", "object"])
def test_only_a_question_model_skips_the_parser(value) -> None:
    with pytest.raises(ValueError, match="question 'q': a question must be a JSON object"):
        probe_tasks({"q": value}, probe_mode=False)


def test_run_batch_still_takes_models_and_dicts(tmp_path) -> None:
    with Ledger(tmp_path / "j.jsonl", cap_usd=1.0) as ledger:
        rows = run_batch(
            [{"item_id": "i", "state": "s"}],
            {"model": NoulQuestion(instructions="Is this about billing?"), "dict": dict(B)},
            backend=DeterministicMockBackend(),
            ledger=ledger,
            model_requested="m",
        )
    assert sorted(row["question_id"] for row in rows) == sorted([QID_A, QID_B])


def test_the_demo_questions_file_is_read_the_same(tmp_path, capsys) -> None:
    ws = generate_workspace(tmp_path / "ws")
    q = Path(ws["questions"])
    loaded = load_questions(q, KNOWN_RULES)
    assert loaded.errors == []
    for spec in loaded.specs:
        assert question_id(_question_from_file(str(q), spec.qid)) == spec.qid
    expected = {spec.qid for spec in loaded.specs}
    assert {question_id(t.question) for t in probe_tasks(json.loads(q.read_text(encoding="utf-8")), probe_mode=False)} == expected


def test_the_web_lint_tab_takes_a_wrapper(tmp_path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    plain = json.loads(Path(ws["questions"]).read_text(encoding="utf-8"))
    client = TestClient(create_app(workspace_dir=Path(ws["records"]).parent))
    before = client.get("/api/lint").json()
    Path(ws["questions"]).write_text(json.dumps({"questions": plain}), encoding="utf-8")
    after = client.get("/api/lint").json()
    assert after == before
