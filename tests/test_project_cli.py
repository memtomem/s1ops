"""`s1 project`: init, add, list, rename and remove over a registry file; each refusal is one `s1:` line
and exit 1, and a refused command leaves a registry as it was."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from s1ops import registry
from s1ops.cli import main as cli_main

pytestmark = pytest.mark.filterwarnings("error::ResourceWarning", "error::pytest.PytestUnraisableExceptionWarning")


def _ok(capsys, *argv: str) -> dict:
    assert cli_main(list(argv)) == 0
    out, err = capsys.readouterr()
    assert err == ""
    return json.loads(out)


def _error_line(capsys, *argv: str) -> str:
    assert cli_main(list(argv)) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("s1: ") and err.endswith("\n") and err.count("\n") == 1, err
    return err[len("s1: "):-1]


@pytest.fixture
def db(tmp_path: Path, capsys) -> str:
    path = str(tmp_path / "projects.db")
    assert _ok(capsys, "project", "init", path) == {"created": path, "projects": 0}
    return path


@pytest.fixture
def ws(tmp_path: Path) -> str:
    (tmp_path / "ws-a").mkdir()
    return str(tmp_path / "ws-a")


def test_the_commands_end_to_end(db: str, ws: str, tmp_path: Path, capsys) -> None:
    added = _ok(capsys, "project", "add", "alpha", "--workspace", ws, "--name", "알파 티켓", "--db", db)
    assert added == {"added": {"id": 1, "slug": "alpha", "name": "알파 티켓", "workspace_path": ws}}
    _ok(capsys, "project", "add", "beta", "--workspace", ws, "--db", db)
    assert _ok(capsys, "project", "rename", "beta", "Beta", "--db", db)["renamed"]["name"] == "Beta"
    removed = _ok(capsys, "project", "remove", "alpha", "--db", db)
    assert removed == {"removed": added["added"], "workspace_kept": True}
    assert Path(ws).is_dir()
    assert _ok(capsys, "project", "list", "--db", db) == {
        "projects": [{"id": 2, "slug": "beta", "name": "Beta", "workspace_path": ws}]
    }


def test_a_missing_or_unreadable_registry_names_the_command_that_makes_one(tmp_path: Path, ws: str, capsys) -> None:
    missing = str(tmp_path / "missing.db")
    assert _error_line(capsys, "project", "list", "--db", missing) == (
        f"project registry not found: {missing} (run `s1 project init` first)"
    )
    assert not Path(missing).exists()
    assert _error_line(capsys, "project", "add", "a", "--workspace", ws, "--db", str(tmp_path)) == (
        f"project registry is a directory: {tmp_path}"
    )
    fifo = tmp_path / "fifo.db"
    os.mkfifo(fifo)
    assert _error_line(capsys, "project", "list", "--db", str(fifo)) == f"project registry is not a regular file: {fifo}"


def test_a_refused_command_leaves_the_registry_as_it_was(db: str, ws: str, tmp_path: Path, capsys) -> None:
    _ok(capsys, "project", "add", "alpha", "--workspace", ws, "--db", db)
    before = Path(db).read_bytes()
    nowhere = str(tmp_path / "nowhere")
    refused = {
        ("project", "init", db): f"project registry already exists: {db}",
        ("project", "add", "alpha", "--workspace", ws, "--db", db): f"project 'alpha' already exists in {db}",
        ("project", "add", "Alpha", "--workspace", ws, "--db", db): f"a project slug is {registry.SLUG_RULE}",
        ("project", "add", "beta", "--workspace", nowhere, "--db", db): (
            f"workspace directory cannot be read: {nowhere}: No such file or directory"
        ),
        ("project", "rename", "gamma", "G", "--db", db): f"no project 'gamma' in {db}",
        ("project", "remove", "gamma", "--db", db): f"no project 'gamma' in {db}",
        ("project", "rename", "alpha", " padded", "--db", db): "a project name has no leading or trailing space",
    }
    for argv, line in refused.items():
        assert _error_line(capsys, *argv) == line, argv
    assert Path(db).read_bytes() == before


def test_a_hostile_name_stays_on_one_line(db: str, ws: str, capsys) -> None:
    # The refusal names the rule; the CLI line escapes anything a message might interpolate.
    for name, point in (("two\nlines", "000A"), ("\x1b[31mred", "001B"), ("rtl\u202eoverride", "202E")):
        assert _error_line(capsys, "project", "add", "alpha", "--workspace", ws, "--name", name, "--db", db) == (
            f"a project name holds a control, direction or line-separator character (U+{point})"
        )
    assert _error_line(capsys, "project", "add", "alpha\n", "--workspace", ws, "--db", db) == (
        f"a project slug is {registry.SLUG_RULE}"
    )
    assert _ok(capsys, "project", "list", "--db", db) == {"projects": []}


def test_a_foreign_database_is_one_line(tmp_path: Path, capsys) -> None:
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"plain text, not a database " * 40)
    assert _error_line(capsys, "project", "list", "--db", str(junk)) == (
        f"project registry cannot be used: {junk}: file is not a database"
    )


def test_a_registry_path_starting_with_two_slashes_round_trips(tmp_path: Path, ws: str, capsys) -> None:
    # A file URI would read "//first-dir" as an authority; the path is the same file to the filesystem.
    path = "/" + str(tmp_path / "r.db")
    _ok(capsys, "project", "init", path)
    _ok(capsys, "project", "add", "alpha", "--workspace", ws, "--db", path)
    assert [p["slug"] for p in _ok(capsys, "project", "list", "--db", path)["projects"]] == ["alpha"]


def test_debug_mode_re_raises_a_registry_error(db: str, capsys, monkeypatch) -> None:
    # As for any library error: S1_DEBUG=1 gives the traceback instead of the line.
    monkeypatch.setenv("S1_DEBUG", "1")
    with pytest.raises(registry.RegistryError):
        cli_main(["project", "remove", "gamma", "--db", db])
