"""A JSONL-or-array input (`--states`, `--features`, judge grades) is read once, so a JSON array
given through process substitution works as it does from a path (#139); and every decoding failure
in it, or in another JSON input the CLI reads (`labelaudit estimate --key`), is one error line naming
the file: an integer past the decoder's digit limit and deep nesting escaped without it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from s1ops.cli import CliError, _load_jsonl_dicts
from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from tests.test_source_digest import _through_fifo

DEEP = "[" * 200_000 + "]" * 200_000
HUGE = "1" * 5000


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


def _as_array(source: str, path: Path) -> Path:
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return path


def _run(capsys, argv: list[str]) -> tuple[int, str, str]:
    rc = cli_main(argv)
    out, err = capsys.readouterr()
    return rc, out, err


def _fifo_run(tmp_path: Path, name: str, source: Path, run):
    where = tmp_path / name
    where.mkdir()
    return _through_fifo(where, str(source), run)


# --- one read: each command, a JSON array through a FIFO -------------------------------------


def test_slice_features(ws, tmp_path, capsys) -> None:
    features = _as_array(ws["features"], tmp_path / "features.json")
    argv = ["slice", "mine", ws["records"], "--features"]
    expected = _run(capsys, [*argv, str(features)])
    assert expected[0] == 0 and json.loads(expected[1])["groups"]
    assert _fifo_run(tmp_path, "f", features, lambda fifo: _run(capsys, [*argv, str(fifo)])) == expected


def test_judge_grades(ws, tmp_path, capsys) -> None:
    grades = _as_array(ws["judge_grades"], tmp_path / "grades.json")
    expected = _run(capsys, ["judge", "audit", str(grades)])
    assert expected[0] == 0
    assert _fifo_run(tmp_path, "g", grades, lambda fifo: _run(capsys, ["judge", "audit", str(fifo)])) == expected


def test_labelaudit_sheet_states(ws, tmp_path, capsys) -> None:
    states = _as_array(ws["states_local"], tmp_path / "states.json")

    def sheet(states_path: Path, out: Path) -> tuple[int, str, str, str, str]:
        rc, stdout, err = _run(capsys, [
            "labelaudit", "sheet", ws["records"], "--states", str(states_path), "--fit", ws["fits"],
            "-o", str(out / "sheet.md"), "-k", str(out / "key.json"), "--top", "3", "--controls", "3",
        ])
        return rc, stdout.replace(str(out), "OUT"), err, (out / "sheet.md").read_text(), (out / "key.json").read_text()

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    expected = sheet(states, tmp_path / "a")
    assert expected[0] == 0
    assert _fifo_run(tmp_path, "s", states, lambda fifo: sheet(fifo, tmp_path / "b")) == expected


def test_probe_states(ws, tmp_path, capsys) -> None:
    states = _as_array(ws["states_local"], tmp_path / "states.json")

    def probe(states_path: Path, name: str) -> tuple[int, int, list[tuple]]:
        out = tmp_path / f"{name}.jsonl"
        rc, stdout, err = _run(capsys, ["probe", ws["questions"], "--states", str(states_path), "-o", str(out), "--ledger", str(tmp_path / f"{name}-ledger.jsonl")])
        assert err == ""
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        return rc, json.loads(stdout)["n_records"], sorted((r["question_id"], r["item_id"], r.get("p")) for r in rows)

    expected = probe(states, "path")
    assert expected[0] == 0 and expected[1] > 0
    assert _fifo_run(tmp_path, "p", states, lambda fifo: probe(fifo, "fifo")) == expected


# --- the decoded text equals the file's ------------------------------------------------------


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_any_newline_gives_the_same_rows(tmp_path, newline) -> None:
    path = tmp_path / "rows.json"
    path.write_bytes(newline.join(['[', ' {"item_id": "a", "state": "x"},', ' {"item_id": "b", "state": "y"}', ']', '']).encode())
    assert _load_jsonl_dicts(path) == [{"item_id": "a", "state": "x"}, {"item_id": "b", "state": "y"}]


def test_a_syntax_error_reads_as_before(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_bytes(b'[\r\n {"item_id": "a",}\r\n]\r\n')
    with pytest.raises(json.JSONDecodeError) as old:
        json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(CliError) as e:
        _load_jsonl_dicts(path)
    assert str(e.value) == f"{path}: not valid JSON: {old.value}"


def test_a_bom_is_still_refused(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_bytes(b'\xef\xbb\xbf[{"item_id": "a", "state": "x"}]\n')
    with pytest.raises(CliError, match=r": line 1: not valid JSON: Unexpected UTF-8 BOM"):
        _load_jsonl_dicts(path)


def test_an_escaped_surrogate_is_not_a_bad_byte(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('[{"item_id": "a", "state": "\\udc80"}]\n', encoding="ascii")
    assert _load_jsonl_dicts(path) == [{"item_id": "a", "state": "\udc80"}]


def test_a_bad_byte_is_named_by_its_physical_line(tmp_path) -> None:
    """U+2028 in a string is not a line break: the bad byte below is on line 2."""
    path = tmp_path / "rows.json"
    path.write_bytes('[{"item_id": "a", "state": "x\u2028y"},\n'.encode() + b'{"item_id": "b", "state": "\xff"}]\n')
    with pytest.raises(CliError) as e:
        _load_jsonl_dicts(path)
    assert str(e.value) == f"{path}: line 2: not valid UTF-8"


# --- decoder failures are one line naming the file -------------------------------------------


@pytest.fixture
def default_digit_limit():
    """`PYTHONINTMAXSTRDIGITS` can move the limit the 5,000-digit cases rely on."""
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    yield
    sys.set_int_max_str_digits(before)


@pytest.mark.usefixtures("default_digit_limit")
@pytest.mark.parametrize(
    "text, where, reason",
    [
        pytest.param(DEEP + "\n", "", "nested too deeply", id="array-deep"),
        pytest.param(f'[{{"n": {HUGE}}}]\n', "", "Exceeds the limit", id="array-huge-int"),
        pytest.param('\n\n{"extra": ' + DEEP + "}\n", ": line 3", "nested too deeply", id="jsonl-deep"),
        pytest.param(f'{{"item_id": "a"}}\n{{"n": {HUGE}}}\n', ": line 2", "Exceeds the limit", id="jsonl-huge-int"),
    ],
)
def test_a_decoder_failure_names_the_file(tmp_path, capsys, text, where, reason) -> None:
    path = tmp_path / "grades.json"
    path.write_text(text, encoding="utf-8")
    rc, out, err = _run(capsys, ["judge", "audit", str(path)])
    assert (rc, out) == (1, "") and err.count("\n") == 1
    assert err.startswith(f"s1: {path}{where}: not valid JSON: ") and reason in err


def test_labelaudit_estimate_key_nested_too_deeply(ws, tmp_path, capsys) -> None:
    sheet, key = tmp_path / "sheet.md", tmp_path / "key.json"
    rc, _, _ = _run(capsys, [
        "labelaudit", "sheet", ws["records"], "--states", ws["states_local"], "--fit", ws["fits"],
        "-o", str(sheet), "-k", str(key), "--top", "3", "--controls", "3",
    ])
    assert rc == 0
    key.write_text(DEEP, encoding="utf-8")
    assert _run(capsys, ["labelaudit", "estimate", str(sheet), "--key", str(key)]) == (1, "", f"s1: {key}: not valid JSON: nested too deeply\n")
