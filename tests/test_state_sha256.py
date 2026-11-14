"""A record's `state_sha256` is a SHA-256 digest or it is not there.

Both exporters copy the field into their output and `s1 apply` copies it through. Nothing looked at it,
so a records file that held item text there, or an object, exported with exit 0 and that value in the
Phoenix span and the Langfuse observation: the one place a records file could carry what `safe-v1` keeps
out of every export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, parse_record, read_records, reparse
from s1ops.demo import generate_workspace
from s1ops.web.app import create_app

DIGEST = "0123456789abcdef" * 4
SECRET = "customer wrote: my card is 4111 1111"
RULE = "state_sha256 must be the 64 lowercase hex characters of a SHA-256 digest, got "

REFUSED = [
    (SECRET, "a string of 36 characters"),
    ({"state": SECRET}, "a value of type dict"),
    ([DIGEST], "a value of type list"),
    (12345, "a value of type int"),
    (True, "a value of type bool"),
    ("", "a string of 0 characters"),
    (DIGEST.upper(), "64 characters that are not all in 0-9 and a-f"),
    (DIGEST[:-1] + "g", "64 characters that are not all in 0-9 and a-f"),
    (DIGEST[:-1], "a string of 63 characters"),
    (DIGEST + "0", "a string of 65 characters"),
    (DIGEST + "\n", "a string of 65 characters"),
    ("sha256:" + DIGEST, "a string of 71 characters"),
]
IDS = ["text", "object", "list", "number", "bool", "empty", "upper-case", "not-hex", "short", "long", "newline", "prefixed"]


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def _no_answer(row: dict) -> dict:
    kept = {k: v for k, v in row.items() if k not in ("p", "calib", "label")}
    return {**kept, "model_served": None, "error_code": "timeout"}


# --- the reader --------------------------------------------------------------------------------


@pytest.mark.parametrize("value, got", REFUSED, ids=IDS)
@pytest.mark.parametrize("answered", [True, False], ids=["answered", "no-answer"])
def test_a_records_file_is_refused_without_showing_what_it_held(ws, tmp_path: Path, value, got: str, answered: bool) -> None:
    rows = _rows(ws["records"])[:3]
    rows[1] = {**(rows[1] if answered else _no_answer(rows[1])), "state_sha256": value}
    bad = _write(tmp_path / "bad.jsonl", rows)
    with pytest.raises(RecordError) as e:
        read_records(bad)
    assert str(e.value) == f"{bad}: line 2: {RULE}{got}"
    assert SECRET not in str(e.value) and DIGEST[:16] not in str(e.value).lower()


@pytest.mark.parametrize("change", [lambda row: row.pop("state_sha256"), lambda row: row.update(state_sha256=None), lambda row: row.update(state_sha256=DIGEST)], ids=["absent", "null", "digest"])
@pytest.mark.parametrize("answered", [True, False], ids=["answered", "no-answer"])
def test_a_missing_or_null_field_and_a_digest_are_accepted(ws, tmp_path: Path, change, answered: bool) -> None:
    rows = _rows(ws["records"])[:2]
    rows[0] = rows[0] if answered else _no_answer(rows[0])
    change(rows[0])
    records = read_records(_write(tmp_path / "ok.jsonl", rows))
    assert records[0].raw.get("state_sha256") in (None, DIGEST)


def test_every_demo_record_and_every_probe_record_already_holds_a_digest(ws) -> None:
    names = ("records_train", "records", "records_kev", "records_canary", "probe_records", "records_categorical_train", "records_categorical")
    held = [r.raw.get("state_sha256") for name in names for r in read_records(ws[name])]
    assert held and all(isinstance(v, str) and len(v) == 64 for v in held)


# --- a decoded row, and a record read again ----------------------------------------------------


@pytest.mark.parametrize("options", [{}, {"require_answer": False}, {"use_calib": True}], ids=["default", "no-answer-needed", "use-calib"])
@pytest.mark.parametrize("answered", [True, False], ids=["answered", "no-answer"])
def test_parse_record_refuses_it_however_the_row_is_read(ws, options: dict, answered: bool) -> None:
    row = _rows(ws["records"])[0]
    row = {**(row if answered else _no_answer(row)), "state_sha256": SECRET}
    with pytest.raises(RecordError) as e:
        parse_record(row, 5, **options)
    assert str(e.value) == f"line 5: {RULE}a string of 36 characters"


def test_reparse_refuses_a_value_changed_after_reading_and_keeps_the_path(ws) -> None:
    record = read_records(ws["records"])[3]
    record.raw["state_sha256"] = {"state": SECRET}
    with pytest.raises(RecordError) as e:
        reparse(record, use_calib=True)
    assert str(e.value) == f"{ws['records']}: line 4: {RULE}a value of type dict"
    assert e.value.path == ws["records"]


# --- commands ----------------------------------------------------------------------------------


def _bad_copy(ws, tmp_path: Path, name: str, value) -> str:
    rows = _rows(ws[name])
    rows[0]["state_sha256"] = value
    return _write(tmp_path / f"{name}.jsonl", rows)


@pytest.mark.parametrize("value, got", [(SECRET, "a string of 36 characters"), ({"state": SECRET}, "a value of type dict")], ids=["text", "object"])
@pytest.mark.parametrize(
    "name, argv",
    [
        ("records", ["export", "phoenix", "{bad}", "-o", "{out}"]),
        ("records", ["export", "langfuse", "{bad}", "-o", "{out}"]),
        ("records_canary", ["apply", "{bad}", "--fit", "{fits}", "-o", "{out}", "--pass-through-unfitted"]),
    ],
    ids=["export-phoenix", "export-langfuse", "apply"],
)
def test_a_command_that_would_copy_it_out_refuses_and_writes_nothing(ws, tmp_path: Path, capsys, name: str, argv: list[str], value, got: str) -> None:
    bad = _bad_copy(ws, tmp_path, name, value)
    out = tmp_path / "out"
    assert cli_main([a.format(bad=bad, out=out, fits=ws["fits"]) for a in argv]) == 1
    captured = capsys.readouterr()
    assert captured.err == f"s1: {bad}: line 1: {RULE}{got}\n"
    assert SECRET not in captured.out + captured.err
    assert not out.exists()


def test_a_digest_is_still_exported_as_it_is(ws, tmp_path: Path, capsys) -> None:
    rows = _rows(ws["records"])[:1]
    rows[0]["state_sha256"] = DIGEST
    source = _write(tmp_path / "ok.jsonl", rows)
    assert cli_main(["export", "phoenix", source, "-o", str(tmp_path / "spans.jsonl")]) == 0
    assert cli_main(["export", "langfuse", source, "-o", str(tmp_path / "bundle.json")]) == 0
    span = json.loads((tmp_path / "spans.jsonl").read_text().splitlines()[0])
    bundle = json.loads((tmp_path / "bundle.json").read_text())
    assert span["attributes"]["s1.state_sha256"] == DIGEST
    assert bundle["observations"][0]["input"]["state_sha256"] == DIGEST


def test_the_web_answers_400_without_the_value(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    path = Path(ws["records"])
    rows = _rows(path)
    rows[0]["state_sha256"] = SECRET
    _write(path, rows)
    response = TestClient(create_app(workspace_dir=path.parent)).get("/api/audit?n_boot=20")
    assert response.status_code == 400
    assert response.json()["detail"] == f"{path}: line 1: {RULE}a string of 36 characters"
    assert SECRET not in response.text
