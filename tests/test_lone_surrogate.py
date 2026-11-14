"""A lone surrogate is refused anywhere in a record row, when it is read.

JSON can escape half of a UTF-16 pair on its own and Python keeps it in a `str`, but no UTF-8 file can
hold it. `s1 fit` and `s1 policy tune` used to open their output, write part of it and stop at the first
such id or option name, leaving a file that was not JSON; both exporters failed on a line that named no
file. `s1 probe` wrote such an id into its records, and stopped on such a question wording only after the
questions before it had been paid for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts import records
from s1ops.contracts.records import RecordError, parse_record, read_records, read_row
from s1ops.demo import generate_workspace
from s1ops.ledger.journal import Ledger, lock_path
from s1ops.monitor.runner import BackendError, BackendResponse, probe_tasks, run_batch
from s1ops.web.app import create_app

# Built from code points: this file holds no surrogate and no escape of one.
HIGH, LOW, UNDECODABLE_BYTE = chr(0xD800), chr(0xDC00), chr(0xDC80)
BACKSLASH = chr(92)
WHY = "holds a lone surrogate (U+D800), which is not a character"
IN_A_ROW = f"a string {WHY}"

NOUL = {"type": "noul", "instructions": "Is it about money?"}
OTHER = {"type": "noul", "instructions": "Is it urgent?"}
CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write(path: Path, rows: list[dict]) -> str:
    """As a writer with `ensure_ascii` leaves it: the surrogate is six ASCII characters in the file."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert path.read_bytes().isascii()
    return str(path)


def _with_text(source: str | Path, path: Path, literal: str) -> str:
    """`source` with `"note": literal` added to its first row as text, so the escape's case is the test's."""
    lines = Path(source).read_text().splitlines()
    lines[0] = f'{lines[0].rstrip()[:-1]}, "note": {literal}}}'
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _set(field: str, value: object):
    return lambda row: row.__setitem__(field, value)


def _rename_first_option(row: dict) -> None:
    first = next(iter(row["probabilities"]))
    row["probabilities"] = {(k + HIGH if k == first else k): v for k, v in row["probabilities"].items()}
    for field in ("choice", "label"):
        if row.get(field) == first:
            row[field] = first + HIGH


# --- the reader --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        _set("item_id", "x" + HIGH),
        _set("dataset", HIGH),
        _set("model_requested", "m" + HIGH),
        _set("model_served", "m" + HIGH),
        _set("split", HIGH + "eval"),
        _set("note", "free text " + HIGH),
        _set("extra", [1, {"a": ["x", HIGH]}]),
        _set("extra", {"a": {"b": {"k" + HIGH: 1}}}),
        lambda row: row.update(model_served=None, error_code="timeout" + HIGH),
    ],
    ids=["item_id", "dataset", "model_requested", "model_served", "split", "unread-field", "in-a-list", "a-key", "no-answer"],
)
def test_a_records_file_with_one_is_refused_wherever_it_is(ws, tmp_path: Path, change) -> None:
    rows = _rows(ws["records"])[:3]
    change(rows[1])
    bad = _write(tmp_path / "bad.jsonl", rows)
    with pytest.raises(RecordError) as e:
        read_records(bad)
    assert str(e.value) == f"{bad}: line 2: {IN_A_ROW}"
    assert (e.value.line, e.value.path) == (2, bad)


def test_a_choice_option_name_with_one_is_refused(ws, tmp_path: Path) -> None:
    rows = [r for r in _rows(ws["records_categorical"]) if r["primitive"] == "choice"][:2]
    _rename_first_option(rows[0])
    bad = _write(tmp_path / "bad.jsonl", rows)
    with pytest.raises(RecordError, match="line 1: a string holds a lone surrogate"):
        read_records(bad)


@pytest.mark.parametrize(
    "escape, code_point",
    [("ud800", "U+D800"), ("uD800", "U+D800"), ("uDBFF", "U+DBFF"), ("udc00", "U+DC00"), ("uDfFf", "U+DFFF"), ("udc80", "U+DC80")],
)
def test_the_escape_is_found_in_either_case_in_a_field_nothing_validates(ws, tmp_path: Path, escape: str, code_point: str) -> None:
    """In `note`, so the check on validated strings cannot stand in for the walk over the row."""
    bad = _with_text(ws["records"], tmp_path / "bad.jsonl", f'["ok", {{"k": "x{BACKSLASH}{escape}"}}]')
    with pytest.raises(RecordError) as e:
        read_records(bad)
    # U+DC80 written as an escape is ASCII text in a valid UTF-8 file: not "not valid UTF-8".
    assert str(e.value) == f"{bad}: line 1: a string holds a lone surrogate ({code_point}), which is not a character"


def test_an_escaped_pair_is_one_character_and_is_accepted(ws, tmp_path: Path) -> None:
    path = _with_text(ws["records"], tmp_path / "pair.jsonl", f'"{BACKSLASH}ud83d{BACKSLASH}ude00"')
    assert read_records(path)[0].raw["note"] == chr(0x1F600)


def test_an_escaped_backslash_before_the_same_letters_is_text(ws, tmp_path: Path) -> None:
    path = _with_text(ws["records"], tmp_path / "text.jsonl", f'"{BACKSLASH}{BACKSLASH}ud800"')
    assert read_records(path)[0].raw["note"] == BACKSLASH + "ud800"


def test_a_raw_one_in_a_line_passed_to_read_row_is_refused(ws) -> None:
    row = {**_rows(ws["records"])[0], "note": "x" + LOW}
    with pytest.raises(RecordError) as e:
        read_row(json.dumps(row, ensure_ascii=False), 3, "given.jsonl")
    assert str(e.value) == "given.jsonl: line 3: the line holds a lone surrogate (U+DC00), which is not a character"


@pytest.mark.parametrize("note", [UNDECODABLE_BYTE, HIGH + UNDECODABLE_BYTE], ids=["alone", "after-another"])
def test_text_from_bytes_that_were_not_utf8_keeps_its_message(ws, note: str) -> None:
    """`surrogateescape` puts U+DC80..U+DCFF where a file's bytes were not UTF-8."""
    row = {**_rows(ws["records"])[0], "note": note}
    with pytest.raises(RecordError) as e:
        read_row(json.dumps(row, ensure_ascii=False), 1, None)
    assert str(e.value) == "line 1: not valid UTF-8"


def test_a_deeply_nested_row_is_walked_without_recursion() -> None:
    deep: list = [HIGH]
    for _ in range(100_000):
        deep = [deep]
    assert records._row_surrogate_problem({"deep": deep}) == IN_A_ROW
    clean: list = ["x"]
    for _ in range(100_000):
        clean = [clean]
    assert records._row_surrogate_problem({"deep": clean}) is None


def test_a_row_nested_as_deep_as_the_decoder_takes_is_refused_by_its_content(ws, tmp_path: Path) -> None:
    depth = 200
    bad = _with_text(ws["records"], tmp_path / "deep.jsonl", "[" * depth + f'"{BACKSLASH}ud800"' + "]" * depth)
    with pytest.raises(RecordError, match="line 1: a string holds a lone surrogate"):
        read_records(bad)


# --- a decoded row -----------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["item_id", "question_id", "model_requested", "model_served", "split", "dataset"])
def test_parse_record_refuses_one_in_a_field_it_validates(ws, field: str) -> None:
    row = {**_rows(ws["records"])[0], field: "x" + HIGH}
    with pytest.raises(RecordError) as e:
        parse_record(row, 7)
    assert str(e.value) == f"line 7: {field} {WHY}"


def test_parse_record_still_takes_one_in_a_field_it_does_not_read(ws) -> None:
    """A decoded row has only its validated fields checked, as with a non-finite number."""
    row = {**_rows(ws["records"])[0], "note": HIGH}
    assert parse_record(row).raw["note"] == HIGH


# --- commands: one line naming the file, and the output left alone ----------------------------


def _bad_copy(ws, tmp_path: Path, name: str, change=_set("item_id", "x" + HIGH)) -> str:
    rows = _rows(ws[name])
    change(rows[0])
    return _write(tmp_path / f"{name}.jsonl", rows)


@pytest.mark.parametrize(
    "name, argv",
    [
        ("records_train", ["fit", "{bad}", "-o", "{out}", "--method", "temperature"]),
        ("records", ["policy", "tune", "{bad}", "-o", "{out}", "--error-budget", "0.10"]),
        ("records", ["export", "phoenix", "{bad}", "-o", "{out}"]),
        ("records", ["export", "langfuse", "{bad}", "-o", "{out}"]),
        ("records_canary", ["apply", "{bad}", "--fit", "{fits}", "-o", "{out}", "--pass-through-unfitted"]),
    ],
    ids=["fit", "policy-tune", "export-phoenix", "export-langfuse", "apply"],
)
def test_a_command_names_the_file_and_leaves_its_output_as_it_was(ws, tmp_path: Path, capsys, name: str, argv: list[str]) -> None:
    bad = _bad_copy(ws, tmp_path, name)
    out = tmp_path / "out"
    out.write_bytes(b"what was here before\n")
    filled = [a.format(bad=bad, out=out, fits=ws["fits"]) for a in argv]
    assert cli_main(filled) == 1
    assert capsys.readouterr().err == f"s1: {bad}: line 1: {IN_A_ROW}\n"
    assert out.read_bytes() == b"what was here before\n"


def test_fit_no_longer_truncates_fits_json_on_an_option_name(ws, tmp_path: Path, capsys) -> None:
    rows = [r for r in _rows(ws["records_categorical_train"]) if r["primitive"] == "choice"]
    for row in rows:
        _rename_first_option(row)
    bad = _write(tmp_path / "train.jsonl", rows)
    out = tmp_path / "fits.json"
    before = Path(ws["fits_categorical"]).read_bytes()
    out.write_bytes(before)
    assert cli_main(["fit", bad, "-o", str(out), "--method", "temperature"]) == 1
    assert capsys.readouterr().err == f"s1: {bad}: line 1: {IN_A_ROW}\n"
    assert out.read_bytes() == before


@pytest.mark.parametrize("argv", [["audit", "{bad}"], ["drift", "{bad}", "{other}"], ["drift", "{other}", "{bad}"]], ids=["audit", "drift-reference", "drift-current"])
def test_commands_that_took_such_a_record_now_refuse_it(ws, tmp_path: Path, capsys, argv: list[str]) -> None:
    bad = _bad_copy(ws, tmp_path, "records")
    assert cli_main([a.format(bad=bad, other=ws["records_canary"]) for a in argv]) == 1
    assert capsys.readouterr().err == f"s1: {bad}: line 1: {IN_A_ROW}\n"


def test_the_web_answers_400_naming_the_file(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    path = Path(ws["records"])
    rows = _rows(path)
    rows[0]["item_id"] = "x" + HIGH
    _write(path, rows)
    response = TestClient(create_app(workspace_dir=path.parent)).get("/api/audit?n_boot=20")
    assert response.status_code == 400
    assert response.json()["detail"] == f"{path}: line 1: {IN_A_ROW}"


# --- s1 probe: refused before anything is sent -------------------------------------------------


class Counting:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls, self.fail = 0, fail

    def infer(self, state, question, *, model_requested):
        self.calls += 1
        if self.fail:
            raise BackendError("timeout", sent=True)
        answer = {"p": 0.6} if question.type == "noul" else {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.7}
        return BackendResponse(model_served="m", answer=answer, cost_usd=1e-7, latency_ms=1.0)


def _batch(
    tmp_path: Path, states: list[dict], questions: dict, *, model: str = "m", fail: bool = False, **options
) -> tuple[Counting, list[dict], list[str]]:
    """`run_batch`, returning the backend, the records, and the journal's lines."""
    backend, journal = Counting(fail=fail), tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        try:
            out = run_batch(states, questions, backend=backend, ledger=ledger, model_requested=model, **options)
        finally:
            lines = journal.read_text().splitlines() if journal.exists() else []
            _batch.last = (backend, lines)  # what a refused run left
    return backend, out, lines


def _refused(tmp_path: Path, message: str, states: list[dict], questions: dict, **options) -> None:
    with pytest.raises(ValueError) as e:
        _batch(tmp_path, states, questions, **options)
    backend, lines = _batch.last
    assert str(e.value) == message
    assert backend.calls == 0 and lines == []


GOOD = {"item_id": "first", "state": "s"}


@pytest.mark.parametrize("field", ["item_id", "dataset", "split"])
@pytest.mark.parametrize("probe_mode", [False, True])
def test_a_states_row_after_a_good_one_stops_the_run_before_any_call(tmp_path: Path, field: str, probe_mode: bool) -> None:
    bad = {"item_id": "second", "state": "s", field: "x" + HIGH}
    _refused(tmp_path, f"row 2: '{field}' {WHY}", [GOOD, bad], {"q": NOUL}, probe_mode=probe_mode)


@pytest.mark.parametrize("names", [("q", "b" + HIGH), ("b" + HIGH, "q")], ids=["after-a-good-one", "first"])
@pytest.mark.parametrize("probe_mode", [False, True])
def test_a_question_name_with_one_stops_the_run_before_any_call(tmp_path: Path, names: tuple[str, str], probe_mode: bool) -> None:
    questions = dict(zip(names, (NOUL, OTHER)))
    bad = "b" + HIGH
    _refused(tmp_path, f"question {bad!r}: the name {WHY}", [GOOD], questions, probe_mode=probe_mode)


@pytest.mark.parametrize(
    "instructions",
    ["Is it urgent" + HIGH, [{"type": "text", "text": "Is it urgent" + HIGH}]],
    ids=["string", "parts"],
)
def test_a_wording_with_one_stops_the_run_before_the_question_ahead_of_it_is_paid(tmp_path: Path, instructions) -> None:
    """`question_id` cannot hash it; it used to be computed when the question's turn came."""
    questions = {"q": NOUL, "second": {"type": "noul", "instructions": instructions}}
    with pytest.raises(ValueError) as e:
        _batch(tmp_path, [GOOD], questions)
    backend, lines = _batch.last
    assert str(e.value) == f"question 'second' (primary): the wording {WHY}"
    assert backend.calls == 0 and lines == []


def test_in_probe_mode_the_question_itself_is_the_first_task_refused() -> None:
    with pytest.raises(ValueError, match=r"^question 'second' \(noul\): the wording holds a lone surrogate"):
        probe_tasks({"q": NOUL, "second": {"type": "noul", "instructions": "Is it urgent" + HIGH}}, probe_mode=True)


def test_a_supplied_inverse_with_one_is_refused_as_the_inverse_of_its_question(tmp_path: Path) -> None:
    """The question hashes; the wording of the inverse it supplies does not."""
    supplied = {**OTHER, "x-s1-lint": {"inverse": {"instructions": "Is it not urgent" + HIGH}}}
    with pytest.raises(ValueError) as e:
        _batch(tmp_path, [GOOD], {"q": NOUL, "second": supplied}, probe_mode=True)
    backend, lines = _batch.last
    assert str(e.value) == f"question 'second' (inverse): the wording {WHY}"
    assert backend.calls == 0 and lines == []


def test_a_model_name_with_one_stops_the_run_before_any_call(tmp_path: Path) -> None:
    message = "model_requested holds a lone surrogate (U+DC80), which is not a character"
    _refused(tmp_path, message, [GOOD], {"q": NOUL}, model="m" + UNDECODABLE_BYTE)


def test_no_questions_is_still_an_empty_run(tmp_path: Path) -> None:
    backend, out, lines = _batch(tmp_path, [GOOD], {})
    assert (out, backend.calls, lines) == ([], 0, [])


def test_every_string_a_probe_record_holds_has_a_check_before_or_after_the_call(tmp_path: Path) -> None:
    """A new string field copied from the run's inputs needs its own refusal: the reader takes no
    lone surrogate, and a record the reader refuses is paid for and lost."""
    state = {"item_id": "i", "dataset": "d", "split": "eval", "state": "s"}
    _, probes, _ = _batch(tmp_path / "a", [{**state, "label": True}], {"q": NOUL}, probe_mode=True)
    _, choices, _ = _batch(tmp_path / "b", [{**state, "label": "a"}], {"c": CHOICE})
    _, failures, _ = _batch(tmp_path / "c", [{**state, "label": True}], {"q": NOUL}, probe_mode=True, fail=True)
    assert [r["model_served"] for r in failures] == [None] * 3
    strings = {key for record in probes + choices + failures for key, value in record.items() if isinstance(value, str)}
    assert strings == {
        "schema", "primitive", "probe_role", "probe_inverse",  # constants
        "state_sha256", "question_id",  # digests
        "item_id", "dataset", "split",  # check_states
        "model_requested",  # run_batch, and --model
        "probe_question",  # probe_tasks
        "label",  # check_labels: one of the question's options
        "model_served", "choice", "error_code",  # the backend's: read back before the record is kept
    }


def _probe_argv(tmp_path: Path, questions: dict, states: list[dict]) -> tuple[list[str], Path, Path, Path, Path]:
    q_path, s_path, out, ledger = tmp_path / "q.json", tmp_path / "s.jsonl", tmp_path / "o.jsonl", tmp_path / "l.jsonl"
    q_path.write_text(json.dumps(questions))
    _write(s_path, states)
    out.write_bytes(b"an earlier run\n")
    argv = ["probe", str(q_path), "--states", str(s_path), "-o", str(out), "--ledger", str(ledger), "--cap-usd", "1"]
    return argv, q_path, s_path, out, ledger


def _nothing_was_opened(out: Path, ledger: Path) -> bool:
    return out.read_bytes() == b"an earlier run\n" and not ledger.exists() and not lock_path(ledger).exists()


def test_the_cli_names_the_states_file_and_opens_nothing(tmp_path: Path, capsys) -> None:
    argv, _, s_path, out, ledger = _probe_argv(tmp_path, {"q": NOUL}, [GOOD, {"item_id": "x" + HIGH, "state": "s"}])
    assert cli_main(argv) == 1
    assert capsys.readouterr().err == f"s1: {s_path}: row 2: 'item_id' {WHY}\n"
    assert _nothing_was_opened(out, ledger)


@pytest.mark.parametrize(
    "second, reason",
    [
        (("b" + HIGH, OTHER), "question {name}: the name " + WHY),
        (("second", {"type": "noul", "instructions": "Is it urgent" + HIGH}), "question 'second' (noul): the wording " + WHY),
    ],
    ids=["name", "wording"],
)
def test_the_cli_names_the_questions_file_and_opens_nothing(tmp_path: Path, capsys, second, reason: str) -> None:
    argv, q_path, _, out, ledger = _probe_argv(tmp_path, {"q": NOUL, second[0]: second[1]}, [GOOD])
    assert cli_main(argv) == 1
    # The name is shown as its `repr`, whose backslash the error line keeps.
    assert capsys.readouterr().err == f"s1: {q_path}: {reason.format(name=repr(second[0]))}\n"
    assert _nothing_was_opened(out, ledger)


def test_the_cli_refuses_a_model_argument_that_was_not_utf8(tmp_path: Path, capsys) -> None:
    argv, _, _, out, ledger = _probe_argv(tmp_path, {"q": NOUL}, [GOOD])
    assert cli_main([*argv, "--model", "m" + UNDECODABLE_BYTE]) == 1
    assert capsys.readouterr().err == "s1: --model holds a lone surrogate (U+DC80), which is not a character\n"
    assert _nothing_was_opened(out, ledger)


def test_the_mock_probe_without_a_model_still_runs(tmp_path: Path, capsys) -> None:
    argv, _, _, out, _ = _probe_argv(tmp_path, {"q": NOUL}, [GOOD])
    assert cli_main(argv) == 0
    assert len(read_records(out)) == 3
