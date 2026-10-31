"""The web workbench inspects workspace files without a 500 or a false "missing": a file it cannot
inspect, a dangling or looping link, and a FIFO are each a 400 naming the file; a file that is
really absent is still optional."""

from __future__ import annotations

import errno
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import s1ops.web.app as web_app
from s1ops.demo import generate_workspace
from s1ops.web.app import create_app

NOT_ROOT = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root searches any directory")


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _client(ws: dict[str, str], **kwargs) -> TestClient:
    return TestClient(create_app(workspace_dir=Path(ws["records"]).parent, **kwargs), raise_server_exceptions=False)


# Every workspace file a request inspects, with one endpoint that inspects it.
INSPECTED = [
    ("records", "/api/overview"),
    ("records", "/api/groups"),
    ("records", "/api/audit?n_boot=20"),
    ("fits", "/api/overview"),
    ("fits", "/api/labelaudit"),
    ("policy", "/api/drift"),
    ("records_canary", "/api/drift"),
    ("features", "/api/slices"),
    ("records_kev", "/api/compare"),
    ("judge_grades", "/api/compare"),
    ("probe_records", "/api/lint"),
    ("questions", "/api/lint"),
]


def _replace_with_loop(path: Path) -> None:
    path.unlink()
    other = path.with_name(path.name + ".other")
    path.symlink_to(other)
    other.symlink_to(path)


@pytest.mark.parametrize("key, url", INSPECTED)
def test_a_symlink_loop_is_400_naming_the_file(ws, key: str, url: str) -> None:
    path = Path(ws[key])
    _replace_with_loop(path)
    r = _client(ws).get(url)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{path}: cannot read: {os.strerror(errno.ELOOP)}"


@pytest.mark.parametrize("key, url", INSPECTED)
def test_a_dangling_link_is_400_naming_the_file(ws, key: str, url: str) -> None:
    path = Path(ws[key])
    path.unlink()
    path.symlink_to(path.with_name("gone"))
    r = _client(ws).get(url)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{path}: cannot read: dangling symbolic link"


def _release(fifo: Path) -> None:
    """Let a reader blocked on `fifo` see EOF, without ever blocking: a non-blocking open for
    writing fails at once (ENXIO) while no reader is there."""
    try:
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    os.close(fd)


def _get_with_fifo_guard(client: TestClient, url: str, fifo: Path):
    out: dict = {}
    worker = threading.Thread(target=lambda: out.setdefault("r", client.get(url)), daemon=True)
    worker.start()
    worker.join(timeout=20)
    if worker.is_alive():  # a regression reads the FIFO: release every read, bounded, then fail
        deadline = time.monotonic() + 20
        while worker.is_alive() and time.monotonic() < deadline:
            _release(fifo)
            worker.join(timeout=0.2)
        pytest.fail(f"{url} blocked reading the FIFO {fifo}")
    return out["r"]


@pytest.mark.parametrize("key, url", INSPECTED)
def test_a_fifo_is_400_and_never_read(ws, key: str, url: str) -> None:
    path = Path(ws[key])
    path.unlink()
    os.mkfifo(path)
    r = _get_with_fifo_guard(_client(ws), url, path)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{path}: not a regular file"


@NOT_ROOT
@pytest.mark.parametrize("url", ["/api/overview", "/api/groups", "/api/audit?n_boot=20", "/api/compare", "/api/drift"])
def test_records_under_a_non_searchable_parent_are_400(ws, tmp_path: Path, url: str) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    records = locked / "records.jsonl"
    shutil.copy(ws["records"], records)
    client = _client(ws, records_path=records)
    os.chmod(locked, 0o600)
    try:
        r = client.get(url)
    finally:
        os.chmod(locked, 0o700)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{records}: cannot read: {os.strerror(errno.EACCES)}"


def _deny(monkeypatch, target: Path, call: str) -> None:
    real = getattr(os, call)

    def denied(path, *args, **kwargs):
        if Path(path) == target:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(web_app.os, call, denied)


@pytest.mark.parametrize("call", ["lstat", "stat"])
@pytest.mark.parametrize("url", ["/api/compare", "/api/drift"])
def test_uninspectable_primary_records_are_not_an_empty_200(ws, monkeypatch, call: str, url: str) -> None:
    client = _client(ws)
    _deny(monkeypatch, Path(ws["records"]), call)
    r = client.get(url)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{ws['records']}: cannot read: {os.strerror(errno.EACCES)}"


def test_an_uninspectable_file_is_not_replaced_by_the_example_copy(ws, monkeypatch) -> None:
    base = Path(ws["records"]).parent
    (base / "examples" / "data").mkdir(parents=True)
    shutil.copy(ws["records"], base / "examples" / "data" / "records.jsonl")
    # Both calls, as for a parent without search permission: `Path.exists()` goes through os.stat.
    _deny(monkeypatch, base / "records.jsonl", "lstat")
    _deny(monkeypatch, base / "records.jsonl", "stat")
    r = TestClient(create_app(workspace_dir=base), raise_server_exceptions=False).get("/api/overview")
    assert r.status_code == 400
    assert r.json()["detail"] == f"{base / 'records.jsonl'}: cannot read: {os.strerror(errno.EACCES)}"


def test_a_missing_file_still_falls_back_to_the_example_copy(ws) -> None:
    base = Path(ws["records"]).parent
    (base / "examples" / "data").mkdir(parents=True)
    shutil.move(ws["fits"], base / "examples" / "data" / "fits.json")
    r = TestClient(create_app(workspace_dir=base)).get("/api/overview")
    assert r.status_code == 200
    assert r.json()["files"]["fits"] == str(base / "examples" / "data" / "fits.json")


def test_create_app_starts_with_uninspectable_files(ws) -> None:
    _replace_with_loop(Path(ws["fits"]))
    Path(ws["judge_grades"]).unlink()
    os.mkfifo(ws["judge_grades"])
    create_app(workspace_dir=Path(ws["records"]).parent)  # no exception, no blocking read


@pytest.mark.parametrize("key, url", [("judge_grades", "/api/compare"), ("policy", "/api/drift"), ("features", "/api/slices")])
def test_a_really_missing_optional_file_is_still_a_200(ws, key: str, url: str) -> None:
    Path(ws[key]).unlink()
    client = _client(ws)
    assert client.get(url).status_code == 200
    assert Path(ws[key]).name in client.get("/api/overview").json()["missing_files"]


def test_a_questions_directory_stays_a_lint_finding(ws) -> None:
    path = Path(ws["questions"])
    path.unlink()
    path.mkdir()
    r = _client(ws).get("/api/lint")
    assert r.status_code == 200
    assert [(f["rule"], f["message"]) for f in r.json()["findings"]] == [
        ("JL000", f"cannot read {path}: {os.strerror(errno.EISDIR)}")
    ]


def test_a_raw_line_separator_inside_a_judge_comment_is_one_row(ws) -> None:
    # str.splitlines would cut this row at U+2028 and report invalid JSON.
    import json

    path = Path(ws["judge_grades"])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["note"] = "a\u2028b"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert "\u2028" in path.read_text(encoding="utf-8")
    assert _client(ws).get("/api/compare").status_code == 200


@pytest.mark.parametrize("call", ["lstat", "stat"])
def test_an_uninspectable_example_copy_is_400_not_missing(ws, monkeypatch, call: str) -> None:
    # No records.jsonl in the workspace, and the examples/data copy cannot be inspected.
    base = Path(ws["records"]).parent
    (base / "examples" / "data").mkdir(parents=True)
    alt = base / "examples" / "data" / "records.jsonl"
    shutil.move(ws["records"], alt)
    _deny(monkeypatch, alt, call)
    r = TestClient(create_app(workspace_dir=base), raise_server_exceptions=False).get("/api/overview")
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{alt}: cannot read: {os.strerror(errno.EACCES)}"


@NOT_ROOT
def test_an_example_copy_under_a_non_searchable_directory_is_400(ws) -> None:
    base = Path(ws["records"]).parent
    examples = base / "examples" / "data"
    examples.mkdir(parents=True)
    alt = examples / "records.jsonl"
    shutil.move(ws["records"], alt)
    os.chmod(examples, 0o600)
    try:
        r = TestClient(create_app(workspace_dir=base), raise_server_exceptions=False).get("/api/overview")
    finally:
        os.chmod(examples, 0o700)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{alt}: cannot read: {os.strerror(errno.EACCES)}"
