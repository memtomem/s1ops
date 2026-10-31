"""A malformed or unreadable input file names itself: the policy and fit loaders, the CLI lines
that report them, and the web workbench's 400s (a file that exists but cannot be read is not a 500)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import s1ops.lint.lint as lint_module
import s1ops.web.app as web_app
from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import load_policies
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import FitError, load_fits
from s1ops.lint.lint import audit_probe, audit_probe_records
from s1ops.contracts.records import read_records
from s1ops.web.app import create_app


def _error_text(fn, *args) -> str:
    try:
        fn(*args)
    except ValueError as e:
        return str(e)
    raise AssertionError("expected a ValueError")


# Python's default `int()` digit limit, which the "digits" case is past; the module pins it, so an
# interpreter run with the limit raised or disabled (PYTHONINTMAXSTRDIGITS) still sees the case.
DIGIT_LIMIT = 4300


def _digit_limit_error() -> str:
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(DIGIT_LIMIT)
    try:
        return _error_text(json.loads, "1" * 5000)
    finally:
        sys.set_int_max_str_digits(old)


@pytest.fixture(autouse=True)
def _pinned_digit_limit():
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(DIGIT_LIMIT)
    yield
    sys.set_int_max_str_digits(old)


# The exact texts Python gives for these inputs, so expected lines stay exact on every version.
JSON_ERROR = _error_text(json.loads, "{")
UTF8_ERROR = _error_text(b"\xff{".decode, "utf-8")
NOT_JSON = {
    "syntax": (b"{", f"not valid JSON: {JSON_ERROR}"),
    "utf8": (b"\xff{", f"not valid UTF-8: {UTF8_ERROR}"),
    "nan": (b'{"schema": NaN}', "non-finite number NaN is not valid JSON"),
    "digits": (b"1" * 5000, f"not valid JSON: {_digit_limit_error()}"),
    "deep": (b"[" * 200_000 + b"0" + b"]" * 200_000, "not valid JSON: nested too deeply"),
}


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _client(ws: dict[str, str], **kwargs) -> TestClient:
    return TestClient(create_app(workspace_dir=Path(ws["records"]).parent, **kwargs), raise_server_exceptions=False)


# --- loaders ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(NOT_JSON))
def test_load_policies_names_the_file(tmp_path: Path, case: str) -> None:
    data, why = NOT_JSON[case]
    path = tmp_path / "policy.json"
    path.write_bytes(data)
    with pytest.raises(ValueError) as e:
        load_policies(path)
    assert type(e.value) is ValueError
    assert str(e.value) == f"{path}: {why}"


@pytest.mark.parametrize("case", sorted(NOT_JSON))
def test_load_fits_names_the_file(tmp_path: Path, case: str) -> None:
    data, why = NOT_JSON[case]
    path = tmp_path / "fits.json"
    path.write_bytes(data)
    with pytest.raises(FitError) as e:
        load_fits(path)
    assert str(e.value) == f"{path}: {why}"


# --- CLI -------------------------------------------------------------------------------------


def _one_line(capsys) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1, err
    return lines[0]


@pytest.mark.parametrize("case", sorted(NOT_JSON))
@pytest.mark.parametrize("broken", ["policy", "fits"])
def test_gate_says_which_of_its_two_inputs_is_broken(ws, tmp_path: Path, capsys, case: str, broken: str) -> None:
    data, why = NOT_JSON[case]
    bad = tmp_path / f"bad-{broken}.json"
    bad.write_bytes(data)
    policy = str(bad) if broken == "policy" else ws["policy"]
    fits = str(bad) if broken == "fits" else ws["fits"]
    rc = cli_main(["gate", policy, "--fit", fits, "--question", "q", "--model", "m", "--p", "0.5"])
    assert rc == 1
    assert _one_line(capsys) == f"s1: {bad}: {why}"


def test_lint_check_reports_non_utf8_questions_as_jl000(tmp_path: Path, capsys) -> None:
    path = tmp_path / "questions.json"
    path.write_bytes(b"\xff{")
    assert cli_main(["lint", "check", str(path), "--format", "json"]) == 1
    out, err = capsys.readouterr()
    assert err == ""
    report = json.loads(out)
    assert [(f["rule"], f["message"]) for f in report["findings"]] == [("JL000", f"not valid UTF-8: {UTF8_ERROR}")]


# --- web: a file that exists but cannot be read ----------------------------------------------

# Every workspace file a web endpoint reads through `_load_or_400`, with one endpoint that reads it.
UNREADABLE = [
    ("records", "/api/overview"),
    ("records", "/api/groups"),
    ("fits", "/api/overview"),
    ("fits", "/api/audit?n_boot=20"),
    ("fits", "/api/labelaudit"),
    ("policy", "/api/drift"),
    ("records_canary", "/api/drift"),
    ("features", "/api/slices"),
    ("records_kev", "/api/compare"),
    ("judge_grades", "/api/compare"),
    ("probe_records", "/api/lint"),
]


@pytest.mark.parametrize("key, url", UNREADABLE)
def test_web_directory_in_place_of_a_file_is_400_naming_it(ws, key: str, url: str) -> None:
    path = Path(ws[key])
    path.unlink()
    path.mkdir()
    r = _client(ws).get(url)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{path}: cannot read: Is a directory"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")
@pytest.mark.parametrize("key, url", [("records", "/api/overview"), ("fits", "/api/audit?n_boot=20"), ("policy", "/api/drift")])
def test_web_file_without_read_permission_is_400_naming_it(ws, key: str, url: str) -> None:
    path = Path(ws[key])
    os.chmod(path, 0)
    try:
        r = _client(ws).get(url)
    finally:
        os.chmod(path, 0o644)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{path}: cannot read: Permission denied"


def test_web_permission_error_from_a_loader_is_400(ws, monkeypatch) -> None:
    def denied(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(web_app, "load_fits", denied)
    r = _client(ws).get("/api/overview")
    assert r.status_code == 400
    assert r.json()["detail"] == f"{ws['fits']}: cannot read: Permission denied"


def test_web_os_error_inside_a_computation_stays_500(ws, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise OSError("bug in compare_records")

    monkeypatch.setattr(web_app, "compare_records", broken)
    assert _client(ws).get("/api/compare").status_code == 500


# --- web: the path appears once --------------------------------------------------------------


def test_web_names_a_loader_message_path_once(ws) -> None:
    Path(ws["policy"]).write_text('{"schema": "x"}')
    Path(ws["fits"]).write_bytes(b"{")
    client = _client(ws)
    assert client.get("/api/drift").json()["detail"] == f"{ws['policy']}: not a s1-policy/1 file"
    assert client.get("/api/overview").json()["detail"] == f"{ws['fits']}: not valid JSON: {JSON_ERROR}"


def test_web_names_the_path_once_for_a_relative_workspace(ws, monkeypatch) -> None:
    root = Path(ws["records"]).parent
    monkeypatch.chdir(root.parent)
    Path(ws["fits"]).write_bytes(b"{")
    client = TestClient(create_app(workspace_dir=Path(root.name)), raise_server_exceptions=False)
    assert client.get("/api/overview").json()["detail"] == f"{Path(root.name) / 'fits.json'}: not valid JSON: {JSON_ERROR}"


def test_web_names_an_explicit_fits_path_once(ws, tmp_path: Path) -> None:
    override = tmp_path / "elsewhere" / "fits.json"
    override.parent.mkdir()
    override.write_bytes(b"\xff{")
    r = _client(ws, fits_path=override).get("/api/overview")
    assert r.json()["detail"] == f"{override}: not valid UTF-8: {UTF8_ERROR}"


def test_web_record_errors_keep_their_single_prefix(ws) -> None:
    Path(ws["records"]).write_text("{not json\n")
    detail = _client(ws).get("/api/audit?n_boot=20").json()["detail"]
    assert detail == f"{ws['records']}: line 1: not valid JSON: {_error_text(json.loads, '{not json')}"


# --- web: lint -------------------------------------------------------------------------------


def test_web_lint_reports_non_utf8_questions_as_jl000(ws) -> None:
    Path(ws["questions"]).write_bytes(b"\xff{")
    r = _client(ws).get("/api/lint")
    assert r.status_code == 200
    assert [(f["rule"], f["message"]) for f in r.json()["findings"]] == [("JL000", f"not valid UTF-8: {UTF8_ERROR}")]


def test_web_lint_reads_the_probe_file_once(ws, monkeypatch) -> None:
    reads: list[Path] = []

    def counting(path, *args, **kwargs):
        reads.append(Path(path))
        return read_records(path, *args, **kwargs)

    # One shared counter over both module aliases: the web's and the one audit_probe_records uses.
    monkeypatch.setattr(web_app, "read_records", counting)
    monkeypatch.setattr(lint_module, "read_records", counting)
    r = _client(ws).get("/api/lint")
    assert r.status_code == 200
    assert reads.count(Path(ws["probe_records"])) == 1


def test_audit_probe_records_equals_audit_probe_of_its_records(ws) -> None:
    path = ws["probe_records"]
    expected = audit_probe(read_records(path))
    assert expected  # the demo probe file pairs probes, so the comparison is not vacuous
    assert audit_probe_records(path) == expected
