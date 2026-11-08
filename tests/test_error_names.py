"""`s1 labelaudit` errors name the file they are about; a usage error is one line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import question_id
from s1ops.demo import generate_workspace
from s1ops.improve.labelaudit import LabelAuditInputError, SheetInputError


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def _sheet(ws: dict[str, str], tmp_path: Path, *extra: str, states: str | None = None) -> list[str]:
    return ["labelaudit", "sheet", ws["records"], "--states", states or ws["states_local"], "--fit", ws["fits"],
            "-o", str(tmp_path / "sheet.md"), "-k", str(tmp_path / "key.json"), "--top", "6", "--controls", "6", *extra]


def _one_line(capsys) -> str:
    captured = capsys.readouterr()
    lines = captured.err.strip().splitlines()
    assert captured.out == "" and len(lines) == 1
    return lines[0]


def test_too_few_states_names_the_states_file(ws, tmp_path, capsys) -> None:
    states = tmp_path / "states.jsonl"
    states.write_text(Path(ws["states_local"]).read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert cli_main(_sheet(ws, tmp_path, states=str(states))) == 1
    line = _one_line(capsys)
    assert line.startswith(f"s1: {states}: no state for ") and "the states file must hold" in line


def test_too_few_items_names_the_records_file(ws, tmp_path, capsys) -> None:
    assert cli_main(_sheet(ws, tmp_path, "--top", "5000")) == 1
    line = _one_line(capsys)
    assert line.startswith(f"s1: {ws['records']}: need at least 5006 distinct items, got ")


def test_a_question_that_does_not_fit_the_records_names_the_questions_file(ws, tmp_path, capsys) -> None:
    choice = {"type": "choice", "instructions": "What is the topic?", "criteria": {"a": "first", "b": "second"}}
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"topic": choice}), encoding="utf-8")
    records = tmp_path / "records.jsonl"
    rows = [json.loads(line) for line in Path(ws["records"]).read_text(encoding="utf-8").splitlines()]
    records.write_text("".join(json.dumps({**row, "question_id": question_id(choice)}) + "\n" for row in rows), encoding="utf-8")
    args = ["labelaudit", "sheet", str(records), "--states", ws["states_local"], "--questions", str(questions),
            "-o", str(tmp_path / "sheet.md"), "-k", str(tmp_path / "key.json"), "--top", "6", "--controls", "6"]
    assert cli_main(args) == 1
    line = _one_line(capsys)
    assert line.startswith(f"s1: {questions}: question {question_id(choice)}: ") and "choice question" in line


@pytest.mark.parametrize("command", ["rank", "sheet"])
def test_a_fit_refusal_names_both_files(ws, tmp_path, capsys, command) -> None:
    fits = tmp_path / "fits.json"
    doc = json.loads(Path(ws["fits"]).read_text(encoding="utf-8"))
    doc["fits"] = []
    fits.write_text(json.dumps(doc), encoding="utf-8")
    args = ["labelaudit", "rank", ws["records"], "--fit", str(fits)] if command == "rank" else _sheet(ws, tmp_path, "--fit", str(fits))
    assert cli_main(args) == 1
    line = _one_line(capsys)
    assert line.startswith(f"s1: {ws['records']} with --fit {fits}: ") and "no fit for question_id" in line


@pytest.mark.parametrize("command", ["rank", "sheet"])
def test_a_fit_for_another_model_names_both_files(ws, tmp_path, capsys, command) -> None:
    fits = tmp_path / "fits.json"
    doc = json.loads(Path(ws["fits"]).read_text(encoding="utf-8"))
    for fit in doc["fits"]:
        fit["model_served"] = "another-model"
    fits.write_text(json.dumps(doc), encoding="utf-8")
    args = ["labelaudit", "rank", ws["records"], "--fit", str(fits)] if command == "rank" else _sheet(ws, tmp_path, "--fit", str(fits))
    assert cli_main(args) == 1
    assert _one_line(capsys).startswith(f"s1: {ws['records']} with --fit {fits}: ")


def _made_sheet(ws, tmp_path, capsys) -> tuple[Path, Path, str]:
    assert cli_main(_sheet(ws, tmp_path)) == 0
    capsys.readouterr()
    sheet = tmp_path / "sheet.md"
    return sheet, tmp_path / "key.json", sheet.read_text(encoding="utf-8")


def test_a_blank_sheet_names_the_sheet(ws, tmp_path, capsys) -> None:
    sheet, key, _ = _made_sheet(ws, tmp_path, capsys)
    assert cli_main(["labelaudit", "estimate", str(sheet), "--key", str(key)]) == 1
    line = _one_line(capsys)
    assert line.startswith(f"s1: {sheet}: item ")


def test_an_invalid_answer_names_the_sheet(ws, tmp_path, capsys) -> None:
    sheet, key, form = _made_sheet(ws, tmp_path, capsys)
    sheet.write_text(form.replace("\n답: \n", "\n답: nope\n", 1), encoding="utf-8")
    assert cli_main(["labelaudit", "estimate", str(sheet), "--key", str(key)]) == 1
    line = _one_line(capsys)
    assert line.startswith(f"s1: {sheet}: item ") and "nope" in line


@pytest.mark.parametrize(
    "extra, error",
    [
        pytest.param(("--top", "5000"), SheetInputError, id="too-few-items"),
        pytest.param(("--states", "SMALL"), SheetInputError, id="too-few-states"),
        pytest.param(("--fit", "EMPTY"), LabelAuditInputError, id="fit-refused"),
    ],
)
def test_debug_reraises_the_library_error(ws, tmp_path, monkeypatch, capsys, extra, error) -> None:
    monkeypatch.setenv("S1_DEBUG", "1")
    small = tmp_path / "small.jsonl"
    small.write_text(Path(ws["states_local"]).read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({**json.loads(Path(ws["fits"]).read_text(encoding="utf-8")), "fits": []}), encoding="utf-8")
    values = {"SMALL": str(small), "EMPTY": str(empty)}
    with pytest.raises(error):
        cli_main(_sheet(ws, tmp_path, extra[0], values.get(extra[1], extra[1])))


def test_debug_reraises_a_rank_fit_refusal_and_an_answer_error(ws, tmp_path, monkeypatch, capsys) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({**json.loads(Path(ws["fits"]).read_text(encoding="utf-8")), "fits": []}), encoding="utf-8")
    sheet, key, _ = _made_sheet(ws, tmp_path, capsys)
    monkeypatch.setenv("S1_DEBUG", "1")
    with pytest.raises(LabelAuditInputError):
        cli_main(["labelaudit", "rank", ws["records"], "--fit", str(empty)])
    with pytest.raises(ValueError, match="^item "):
        cli_main(["labelaudit", "estimate", str(sheet), "--key", str(key)])


@pytest.mark.parametrize("command", ["rank", "sheet"])
def test_debug_keeps_the_cause_of_a_fit_refusal(ws, tmp_path, monkeypatch, capsys, command) -> None:
    def refuse(*args, **kwargs):
        try:
            raise KeyError("underneath")
        except KeyError as e:
            raise LabelAuditInputError("refused") from e

    monkeypatch.setenv("S1_DEBUG", "1")
    monkeypatch.setattr("s1ops.cli.rank_with_skips", refuse)
    monkeypatch.setattr("s1ops.cli.build_sheet", refuse)
    args = ["labelaudit", "rank", ws["records"], "--fit", ws["fits"]] if command == "rank" else _sheet(ws, tmp_path)
    with pytest.raises(LabelAuditInputError) as refused:
        cli_main(args)
    assert isinstance(refused.value.__cause__, KeyError)
    assert not refused.value.__suppress_context__ or refused.value.__cause__ is not None


@pytest.mark.parametrize(
    "argv, expected",
    [
        pytest.param([], "s1: the following arguments are required: cmd; see s1 -h", id="no-subcommand"),
        pytest.param(["labelaudit"], "s1: the following arguments are required: la_cmd; see s1 labelaudit -h", id="nested-no-subcommand"),
        pytest.param(["labelaudit", "rank"], "s1: the following arguments are required: records; see s1 labelaudit rank -h", id="nested"),
        pytest.param(["policy", "tune"], "s1: the following arguments are required: records, -o/--output; see s1 policy tune -h", id="policy-tune"),
        pytest.param(["nope"], None, id="unknown-subcommand"),
        pytest.param(["labelaudit", "rank", "r.jsonl", "--top", "nope"], "s1: argument --top: invalid int value: 'nope'; see s1 labelaudit rank -h", id="bad-type"),
    ],
)
def test_a_usage_error_is_one_line_pointing_at_help(capsys, argv, expected) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(argv)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    if expected is None:
        assert lines[0].startswith("s1: argument") and lines[0].endswith("; see s1 -h")
    else:
        assert lines == [expected]


def test_help_still_prints_usage_and_exits_0(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["fit", "-h"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("usage: s1 fit ")
