"""`s1 probe` refuses a states file that holds no rows before it makes the backend or opens the ledger or output:
such a run used to exit 0 with an empty output, and only the next command noticed (#121)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import s1ops.cli as cli
from s1ops.cli import main as cli_main
from s1ops.contracts.questions import NoulQuestion
from s1ops.ledger.journal import lock_path


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("\n\n", id="blank-lines"),
        pytest.param("  \n\t\n", id="whitespace"),
        pytest.param("[]\n", id="json-array"),
    ],
)
def test_a_states_file_without_rows_is_refused_before_the_backend_ledger_or_output(tmp_path: Path, capsys, monkeypatch, text: str) -> None:
    def no_backend(args):
        raise AssertionError("the backend was made")

    monkeypatch.setattr(cli, "_probe_backend", no_backend)
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"q": NoulQuestion(instructions="Is it spam?").wire_dict()}), encoding="utf-8")
    states = tmp_path / "states.jsonl"
    states.write_text(text, encoding="utf-8")
    out, ledger = tmp_path / "out.jsonl", tmp_path / "ledger.jsonl"
    stale = tmp_path / "stale.jsonl"
    stale.write_text("previous\n", encoding="utf-8")

    for output in (out, stale):
        rc = cli_main(["probe", str(questions), "--states", str(states), "-o", str(output), "--ledger", str(ledger)])
        captured = capsys.readouterr()
        assert (rc, captured.out, captured.err) == (1, "", f"s1: {states}: holds no state rows, so the probe would send nothing\n")
    assert not out.exists() and not ledger.exists() and not lock_path(ledger).exists()
    assert stale.read_text(encoding="utf-8") == "previous\n"


def test_a_questions_error_still_comes_first(tmp_path: Path, capsys) -> None:
    questions = tmp_path / "q.json"
    questions.write_text("{}", encoding="utf-8")
    states = tmp_path / "states.jsonl"
    states.write_text("", encoding="utf-8")
    rc = cli_main(["probe", str(questions), "--states", str(states), "-o", str(tmp_path / "o.jsonl"), "--ledger", str(tmp_path / "l.jsonl")])
    err = capsys.readouterr().err
    assert rc == 1 and err.startswith(f"s1: {questions}") and "state rows" not in err
