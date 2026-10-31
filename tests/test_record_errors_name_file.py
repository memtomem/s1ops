"""Record and CLI input errors name the file: a broken line says which file and which line, an
input that cannot be inspected says why, and `<(...)` process substitution keeps working."""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, parse_record, read_records
from s1ops.demo import generate_workspace


def _error_text(fn, *args) -> str:
    try:
        fn(*args)
    except ValueError as e:
        return str(e)
    raise AssertionError("expected a ValueError")


JSON_ERROR = _error_text(json.loads, "{\n")  # a broken line keeps its newline
UTF8_ERROR = _error_text(b"\xff{".decode, "utf-8")


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _one_line(capsys) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1, err
    return lines[0]


def _broken_copy(src: str, dest: Path, line_no: int, data: bytes) -> Path:
    lines = Path(src).read_bytes().split(b"\n")
    lines[line_no - 1] = data
    dest.write_bytes(b"\n".join(lines))
    return dest


# --- read_records ----------------------------------------------------------------------------


def test_record_error_carries_line_reason_and_path(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text("{\n")
    with pytest.raises(RecordError) as e:
        read_records(path)
    assert (e.value.line, e.value.reason, e.value.path) == (1, f"not valid JSON: {JSON_ERROR}", path)
    assert str(e.value) == f"{path}: line 1: not valid JSON: {JSON_ERROR}"


def test_parse_record_alone_stays_pathless() -> None:
    with pytest.raises(RecordError) as e:
        parse_record({"schema": "nope"}, 7)
    assert e.value.path is None
    assert str(e.value).startswith("line 7: ")


def test_invalid_utf8_names_its_line(ws, tmp_path: Path) -> None:
    path = _broken_copy(ws["records"], tmp_path / "late.jsonl", 100, b"\xff{}")
    with pytest.raises(RecordError) as e:
        read_records(path)
    assert str(e.value) == f"{path}: line 100: not valid UTF-8"


def test_nesting_too_deep_is_a_record_error(tmp_path: Path) -> None:
    path = tmp_path / "deep.jsonl"
    path.write_text("[" * 200_000 + "]" * 200_000 + "\n")
    with pytest.raises(RecordError) as e:
        read_records(path)
    assert str(e.value) == f"{path}: line 1: not valid JSON: nested too deeply"


def test_line_endings_and_numbers_match_text_mode(ws, tmp_path: Path) -> None:
    rows = Path(ws["records"]).read_text().splitlines()[:6]
    plain = tmp_path / "plain.jsonl"
    plain.write_text("\n".join(rows) + "\n")
    mixed = tmp_path / "mixed.jsonl"
    # \n, \r\n, lone \r, a blank line, and an unterminated last record
    mixed.write_bytes(
        rows[0].encode() + b"\n" + rows[1].encode() + b"\r\n" + rows[2].encode() + b"\r"
        + b"\r\n" + rows[3].encode() + b"\r" + rows[4].encode() + b"\n" + rows[5].encode()
    )
    expected = [r.item_id for r in read_records(plain)]
    got = read_records(mixed)
    assert [r.item_id for r in got] == expected
    assert [r.line for r in got] == [1, 2, 3, 5, 6, 7]


def test_invalid_utf8_after_a_lone_cr_names_the_right_line(ws, tmp_path: Path) -> None:
    rows = Path(ws["records"]).read_text().splitlines()[:2]
    path = tmp_path / "cr.jsonl"
    path.write_bytes(rows[0].encode() + b"\r" + rows[1].encode() + b"\r" + b"\xff{}\n")
    with pytest.raises(RecordError) as e:
        read_records(path)
    assert str(e.value) == f"{path}: line 3: not valid UTF-8"


def test_a_json_surrogate_escape_is_still_accepted(ws, tmp_path: Path) -> None:
    row = json.loads(Path(ws["records"]).read_text().splitlines()[0])
    row["dataset"] = "\udc80"
    path = tmp_path / "escape.jsonl"
    path.write_text(json.dumps(row) + "\n")  # ensure_ascii writes the escape as "\\udc80"
    assert b"\\udc80" in path.read_bytes()
    assert read_records(path)[0].dataset == "\udc80"


# --- CLI: drift and compare say which input is broken ----------------------------------------


@pytest.mark.parametrize("side", ["reference", "current"])
def test_drift_names_the_broken_side(ws, tmp_path: Path, capsys, side: str) -> None:
    bad = _broken_copy(ws["records"], tmp_path / f"{side}.jsonl", 1, b"{")
    ref = str(bad) if side == "reference" else ws["records"]
    cur = str(bad) if side == "current" else ws["records_canary"]
    assert cli_main(["drift", ref, cur]) == 1
    assert _one_line(capsys) == f"s1: {bad}: line 1: not valid JSON: {JSON_ERROR}"


@pytest.mark.parametrize("side", ["primary", "shadow"])
def test_compare_names_the_broken_side(ws, tmp_path: Path, capsys, side: str) -> None:
    bad = _broken_copy(ws["records"], tmp_path / f"{side}.jsonl", 1, b"\xff")
    primary = str(bad) if side == "primary" else ws["records"]
    shadow = str(bad) if side == "shadow" else ws["records_kev"]
    assert cli_main(["compare", primary, shadow]) == 1
    assert _one_line(capsys) == f"s1: {bad}: line 1: not valid UTF-8"


@pytest.mark.parametrize("side", ["reference", "current"])
def test_drift_names_the_file_of_a_bad_stored_calib(ws, tmp_path: Path, capsys, side: str) -> None:
    lines = Path(ws["records"]).read_text().splitlines()
    row = json.loads(lines[0])
    row["calib"] = {**row.get("calib", {}), "p": "nan"}
    lines[0] = json.dumps(row)
    bad = tmp_path / f"{side}-calib.jsonl"
    bad.write_text("\n".join(lines) + "\n")
    expected = f"s1: {bad}: {_error_text(lambda: parse_record(row, 1, use_calib=True))}"
    ref = str(bad) if side == "reference" else ws["records"]
    cur = str(bad) if side == "current" else ws["records_canary"]
    assert cli_main(["drift", ref, cur]) == 1
    assert _one_line(capsys) == expected


# --- CLI: JSON helpers -----------------------------------------------------------------------


def test_jsonl_helper_names_the_non_utf8_line(ws, tmp_path: Path, capsys) -> None:
    bad = _broken_copy(ws["judge_grades"], tmp_path / "grades.jsonl", 2, b"\xff{}")
    assert cli_main(["judge", "audit", str(bad)]) == 1
    assert _one_line(capsys) == f"s1: {bad}: line 2: not valid UTF-8"


@pytest.fixture
def sheet_and_key(ws, tmp_path: Path) -> tuple[Path, Path]:
    sheet, key = tmp_path / "sheet.md", tmp_path / "key.json"
    rc = cli_main([
        "labelaudit", "sheet", ws["records"], "--states", ws["states_local"], "--fit", ws["fits"],
        "-o", str(sheet), "-k", str(key), "--top", "3", "--controls", "3",
    ])
    assert rc == 0
    return sheet, key


def test_labelaudit_estimate_names_a_non_utf8_sheet(sheet_and_key, capsys) -> None:
    sheet, key = sheet_and_key
    capsys.readouterr()
    sheet.write_bytes(b"\xff{")
    assert cli_main(["labelaudit", "estimate", str(sheet), "--key", str(key)]) == 1
    assert _one_line(capsys) == f"s1: {sheet}: not valid UTF-8: {UTF8_ERROR}"


def test_labelaudit_estimate_names_a_non_utf8_key(sheet_and_key, capsys) -> None:
    sheet, key = sheet_and_key
    capsys.readouterr()
    key.write_bytes(b"\xff{")
    assert cli_main(["labelaudit", "estimate", str(sheet), "--key", str(key)]) == 1
    assert _one_line(capsys) == f"s1: {key}: not valid UTF-8: {UTF8_ERROR}"


# --- CLI: inputs that cannot be inspected ----------------------------------------------------


def _gate(policy: Path) -> list[str]:
    return ["gate", str(policy), "--question", "q", "--model", "m", "--p", "0.5"]


def test_a_symlink_loop_is_not_reported_missing(tmp_path: Path, capsys) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.symlink_to(b)
    b.symlink_to(a)
    assert cli_main(_gate(a)) == 1
    assert _one_line(capsys) == f"s1: policy file cannot be read: {a}: {os.strerror(errno.ELOOP)}"


def test_a_dangling_symlink_is_named_as_one(tmp_path: Path, capsys) -> None:
    link = tmp_path / "policy.json"
    link.symlink_to(tmp_path / "gone.json")
    assert cli_main(_gate(link)) == 1
    assert _one_line(capsys) == f"s1: policy file is a dangling symbolic link: {link}"


def test_a_missing_input_still_names_its_producer(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "policy.json"
    assert cli_main(_gate(missing)) == 1
    assert _one_line(capsys) == f"s1: policy file not found: {missing} (run `s1 policy tune` first)"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root searches any directory")
def test_an_input_under_a_non_searchable_parent_cannot_be_read(ws, tmp_path: Path, capsys) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    policy = locked / "policy.json"
    shutil.copy(ws["policy"], policy)
    os.chmod(locked, 0o600)
    try:
        rc = cli_main(_gate(policy))
    finally:
        os.chmod(locked, 0o700)
    assert rc == 1
    assert _one_line(capsys) == f"s1: policy file cannot be read: {policy}: {os.strerror(errno.EACCES)}"


def test_a_permission_error_from_stat_cannot_be_read(tmp_path: Path, capsys, monkeypatch) -> None:
    import s1ops.cli as cli

    target = tmp_path / "policy.json"
    real_stat = os.stat

    def denied(path, *args, **kwargs):
        if Path(path) == target:
            raise PermissionError(13, os.strerror(13), str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(cli.os, "stat", denied)
    assert cli_main(_gate(target)) == 1
    assert _one_line(capsys) == f"s1: policy file cannot be read: {target}: {os.strerror(13)}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="process substitution needs bash")
def test_process_substitution_still_works(ws) -> None:
    run = f'"{sys.executable}" -c "import sys; from s1ops.cli import main; sys.exit(main(sys.argv[1:]))"'
    for cmd in (
        f'{run} audit <(cat "{ws["records"]}")',
        f'{run} gate <(cat "{ws["policy"]}") --fit <(cat "{ws["fits"]}") '
        "--question q-30076be2728be682 --model jev-1.13.0 --p 0.52",
    ):
        done = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("separator", ["\x0b", "\x1c", "\u2028"])
def test_jsonl_helper_counts_physical_lines(tmp_path: Path, capsys, separator: str) -> None:
    bad = tmp_path / "grades.jsonl"
    bad.write_bytes(separator.encode() + b"\n" + b"\xff\n")
    assert cli_main(["judge", "audit", str(bad)]) == 1
    assert _one_line(capsys) == f"s1: {bad}: line 2: not valid UTF-8"


def test_jsonl_helper_keeps_a_raw_line_separator_inside_a_string(tmp_path: Path) -> None:
    from s1ops.cli import _load_jsonl_dicts

    path = tmp_path / "states.jsonl"
    row = {"dataset": "d", "item_id": "i", "state": "a\u2028b\x1cc"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert _load_jsonl_dicts(path) == [row]
