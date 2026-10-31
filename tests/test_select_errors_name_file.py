"""Errors found after reading still name the records file (#61).

`select`'s duplicate-item and class-mismatch checks, and the re-parse that drift, routing and both
exporters run on `Record.raw`, see records rather than lines of a file. `Record.path` carries the
file `read_records` read, so those errors start with it as read errors do.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.bridges.langfuse import _predict_info
from s1ops.cli import main as cli_main
from s1ops.contracts.records import Record, RecordError, check_classes, parse_record, read_records, reparse, select
from s1ops.contracts.schemas import PolicyEntry
from s1ops.demo import generate_workspace
from s1ops.monitor.drift import audit_drift
from s1ops.runtime.routing import route_record
from s1ops.web.app import create_app

CALIB_ERROR = "calib.p must be a number, got 'nan'"


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _first_row(path: str, primitive: str) -> dict:
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["primitive"] == primitive and row.get("label") is not None and row["model_served"] == row["model_requested"]:
            return row
    raise AssertionError(f"no labeled {primitive} row in {path}")


@pytest.fixture(scope="module")
def noul(ws) -> dict:
    return _first_row(ws["records_train"], "noul")


@pytest.fixture(scope="module")
def choice(ws) -> dict:
    return _first_row(ws["records_categorical_train"], "choice")


def _extra_option(row: dict) -> dict:
    """The same question answered with one more option: classes that cannot come from one wording."""
    probs = {k: v * 0.9 for k, v in row["probabilities"].items()}
    return {**row, "item_id": row["item_id"] + "-x", "probabilities": {**probs, "zz-extra": 0.1}}


def _write(path: Path, *rows: dict) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _key(row: dict) -> tuple[str, str, str]:
    return (row["question_id"], row["model_served"], row["primitive"])


def _duplicate(row: dict, where: str) -> str:
    return f"item ({row['dataset']!r}, {row['item_id']!r}) (dataset, item_id) appears twice in group {_key(row)} ({where})"


def _classes(row: dict, where: str) -> str:
    extra = sorted([*row["probabilities"], "zz-extra"])
    return (
        f"{row['question_id']} ({row['model_served']}): answer classes {extra} differ from "
        f"{sorted(row['probabilities'])} {where} for the same question; the question_id cannot be trusted"
    )


def _classes_from(first: dict, then: dict, where: str) -> str:
    """The class-mismatch reason when `then`'s options differ from `first`'s."""
    return (
        f"{first['question_id']} ({first['model_served']}): answer classes {sorted(then['probabilities'])} differ from "
        f"{sorted(first['probabilities'])} {where} for the same question; the question_id cannot be trusted"
    )


def _error(fn, *args, **kwargs) -> RecordError:
    with pytest.raises(RecordError) as e:
        fn(*args, **kwargs)
    return e.value


def _one_line(capsys) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1, err
    return lines[0]


# --- CLI -------------------------------------------------------------------------------------

SELECT_COMMANDS = [
    ["audit", "{records}"],
    ["fit", "{records}", "-o", "{out}"],
    ["policy", "tune", "{records}", "-o", "{out}"],
    ["policy", "curve", "{records}"],
]


@pytest.mark.parametrize("argv", SELECT_COMMANDS, ids=lambda a: " ".join(a[:2]).replace(" {records}", ""))
def test_cli_names_the_file_for_a_duplicate_item(tmp_path: Path, capsys, noul: dict, argv: list[str]) -> None:
    records = _write(tmp_path / "dup.jsonl", noul, noul)
    assert cli_main([a.format(records=records, out=tmp_path / "out.json") for a in argv]) == 1
    assert _one_line(capsys) == f"s1: {records}: line 2: {_duplicate(noul, 'first on line 1')}"
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("argv", SELECT_COMMANDS, ids=lambda a: " ".join(a[:2]).replace(" {records}", ""))
def test_cli_names_the_file_for_a_class_mismatch(tmp_path: Path, capsys, choice: dict, argv: list[str]) -> None:
    records = _write(tmp_path / "classes.jsonl", choice, _extra_option(choice))
    assert cli_main([a.format(records=records, out=tmp_path / "out.json") for a in argv]) == 1
    assert _one_line(capsys) == f"s1: {records}: line 2: {_classes(choice, '(first on line 1)')}"


@pytest.mark.parametrize("with_policy", [False, True], ids=["raw", "policy"])
@pytest.mark.parametrize("fmt", ["langfuse", "phoenix"])
@pytest.mark.parametrize("primitive", ["noul", "choice"])
def test_cli_export_names_the_file_for_a_bad_stored_calib(tmp_path: Path, capsys, ws, noul, choice, fmt, primitive, with_policy) -> None:
    # The exporters read without `use_calib` and re-parse `raw` with it (`bridges/langfuse.py`).
    if primitive == "noul":
        bad, reason = {**noul, "calib": {"p": "nan"}}, CALIB_ERROR
    else:
        probs = dict(choice["probabilities"])
        bad = {**choice, "calib": {"probabilities": {**probs, "zz-extra": 0.0}}}
        reason = f"calib.probabilities keys {sorted([*probs, 'zz-extra'])} differ from probabilities keys {sorted(probs)}"
    # One line: with `--policy`, a record without `calib` ahead of it would be refused by routing first.
    records = _write(tmp_path / "r.jsonl", bad)
    policy = ["--policy", ws["policy" if primitive == "noul" else "policy_categorical"]] if with_policy else []
    assert cli_main(["export", fmt, str(records), "-o", str(tmp_path / "out"), *policy]) == 1
    assert _one_line(capsys) == f"s1: {records}: line 1: {reason}"


# --- select, SDK -----------------------------------------------------------------------------


def test_a_duplicate_without_paths_keeps_the_0_4_0_message(noul: dict) -> None:
    e = _error(select, [parse_record(noul, 1), parse_record(noul, 2)])
    assert str(e) == f"line 2: {_duplicate(noul, 'first on line 1')}"
    assert (e.line, e.path) == (2, None)


def test_a_duplicate_names_the_file_of_both_rows(tmp_path: Path, noul: dict) -> None:
    a = _write(tmp_path / "a.jsonl", noul)
    b = _write(tmp_path / "b.jsonl", {**noul, "item_id": "z"}, noul)
    e = _error(select, read_records(a) + read_records(b))
    assert str(e) == f"{b}: line 2: {_duplicate(noul, f'first on line 1 of {a}')}"
    assert (e.line, e.path, e.reason) == (2, b, _duplicate(noul, f"first on line 1 of {a}"))
    e = _error(select, read_records(b) + read_records(a))
    assert str(e) == f"{a}: line 1: {_duplicate(noul, f'first on line 2 of {b}')}"


def test_a_duplicate_between_a_file_and_a_record_with_no_path(tmp_path: Path, noul: dict) -> None:
    a = _write(tmp_path / "a.jsonl", {**noul, "item_id": "z"}, noul)
    (_, from_file) = read_records(a)
    e = _error(select, [parse_record(noul, 7), from_file])
    assert str(e) == f"{a}: line 2: {_duplicate(noul, 'first on line 7 of a record with no path')}"
    e = _error(select, [from_file, parse_record(noul, 7)])
    assert str(e) == f"line 7: {_duplicate(noul, f'first on line 2 of {a}')}"
    assert e.path is None


def test_one_spelling_as_str_and_path_is_one_file(tmp_path: Path, noul: dict) -> None:
    a = _write(tmp_path / "a.jsonl", noul, noul)
    (first,) = read_records(a)[:1]
    second = read_records(str(a))[1]
    assert (type(first.path), type(second.path)) == (type(a), str)
    e = _error(select, [first, second])
    assert str(e) == f"{a}: line 2: {_duplicate(noul, 'first on line 1')}"


def test_a_class_mismatch_names_where_the_first_classes_came_from(tmp_path: Path, choice: dict) -> None:
    other = _extra_option(choice)
    e = _error(select, [parse_record(choice, 1), parse_record(other, 2)])
    assert str(e) == f"line 2: {_classes(choice, '(first on line 1)')}"
    a = _write(tmp_path / "a.jsonl", choice)
    b = _write(tmp_path / "b.jsonl", other)
    e = _error(select, read_records(a) + read_records(b))
    assert str(e) == f"{b}: line 1: {_classes(choice, f'(first on line 1 of {a})')}"
    assert e.path == b
    e = _error(select, read_records(b) + read_records(a))
    assert str(e) == f"{a}: line 1: {_classes_from(other, choice, f'(first on line 1 of {b})')}"
    e = _error(select, [parse_record(choice, 4), *read_records(b)])
    assert str(e) == f"{b}: line 1: {_classes(choice, '(first on line 4 of a record with no path)')}"
    e = _error(select, [*read_records(a), parse_record(other, 9)])
    assert str(e) == f"line 9: {_classes(choice, f'(first on line 1 of {a})')}"
    assert e.path is None
    both = _write(tmp_path / "both.jsonl", choice, other)
    e = _error(select, [read_records(both)[0], read_records(str(both))[1]])
    assert str(e) == f"{both}: line 2: {_classes(choice, '(first on line 1)')}"


def test_check_classes_keeps_its_wording_and_gains_the_path(tmp_path: Path, choice: dict) -> None:
    other = _extra_option(choice)
    expected = tuple(choice["probabilities"])
    e = _error(check_classes, _key(choice), expected, parse_record(other, 3))
    assert str(e) == f"line 3: {_classes(choice, 'seen earlier')}"
    (rec,) = read_records(_write(tmp_path / "b.jsonl", other))
    e = _error(check_classes, _key(choice), expected, rec)
    assert str(e) == f"{tmp_path / 'b.jsonl'}: line 1: {_classes(choice, 'seen earlier')}"


# --- re-parse --------------------------------------------------------------------------------


def _bad_calib_file(tmp_path: Path, noul: dict) -> Path:
    return _write(tmp_path / "bad.jsonl", noul, {**noul, "item_id": "other", "calib": {"p": "nan"}})


def test_drift_names_the_file_of_a_record_it_re_parses(tmp_path: Path, noul: dict) -> None:
    bad = _bad_calib_file(tmp_path, noul)
    good = _write(tmp_path / "good.jsonl", noul)
    e = _error(audit_drift, read_records(good), read_records(bad))
    assert str(e) == f"{bad}: line 2: {CALIB_ERROR}"
    assert (e.line, e.path, e.reason) == (2, bad, CALIB_ERROR)


def test_route_record_names_the_file_of_a_record_it_re_parses(tmp_path: Path, noul: dict) -> None:
    # The exporters re-parse in `_predict_info` before routing, so routing needs its own test.
    (_, rec) = read_records(_bad_calib_file(tmp_path, noul))
    pol = PolicyEntry(
        question_id=noul["question_id"], model_served=noul["model_served"], primitive="noul",
        mode="error_budget", split="eval", n=1, threshold=0.5, calibrated=True,
    )
    e = _error(route_record, pol, rec)
    assert str(e) == f"{tmp_path / 'bad.jsonl'}: line 2: {CALIB_ERROR}"


def test_predict_info_names_the_file(tmp_path: Path, noul: dict) -> None:
    (_, rec) = read_records(_bad_calib_file(tmp_path, noul))
    assert str(_error(_predict_info, rec)) == f"{tmp_path / 'bad.jsonl'}: line 2: {CALIB_ERROR}"


def test_reparse_keeps_the_path_on_its_result(tmp_path: Path, noul: dict) -> None:
    path = _write(tmp_path / "a.jsonl", {**noul, "calib": {"p": 0.25}})
    (rec,) = read_records(path)
    for use_calib in (False, True):
        again = reparse(rec, use_calib=use_calib)
        assert again.path == path
        assert again == parse_record(rec.raw, rec.line, use_calib=use_calib)
    # Drift's raw comparison returns re-parsed records; their path must survive (drift.py's raw branch).
    from s1ops.monitor.drift import _compared

    (out,) = _compared([rec], False, "reference")
    assert out.path == path
    (out,) = _compared([rec], True, "reference")
    assert out.path == path and out.probs[1] == 0.25


# --- Record ----------------------------------------------------------------------------------


def test_record_path_is_keyword_only_and_outside_equality(tmp_path: Path, noul: dict) -> None:
    path = _write(tmp_path / "a.jsonl", noul)
    (from_file,) = read_records(path)
    bare = parse_record(noul, 1)
    assert (from_file.path, bare.path) == (path, None)
    assert from_file == bare and hash(from_file) == hash(bare) and repr(from_file) == repr(bare)
    assert "path=" not in repr(from_file)
    assert Record.__match_args__ == (
        "line", "item_id", "question_id", "primitive", "model_requested", "model_served",
        "split", "dataset", "label", "classes", "probs", "raw",
    )
    assert [f.name for f in dataclasses.fields(Record)][-1] == "path"
    assert dataclasses.replace(bare, path="x.jsonl").path == "x.jsonl"


# --- web -------------------------------------------------------------------------------------


def test_web_names_the_records_file_once(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    rec_file = Path(ws["records"])
    lines = rec_file.read_text().splitlines()
    rec_file.write_text("\n".join([*lines, lines[0]]) + "\n")
    first = json.loads(lines[0])
    r = TestClient(create_app(workspace_dir=rec_file.parent)).get("/api/audit?n_boot=20")
    assert r.status_code == 400
    assert r.json()["detail"] == f"{rec_file}: line {len(lines) + 1}: {_duplicate(first, 'first on line 1')}"
