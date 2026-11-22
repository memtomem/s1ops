"""The project registry (`s1ops.registry`): a SQLite file holding slug, name and workspace directory per
project. It refuses a file that is not exactly what `init_registry` writes, a bad argument before the file
is touched, and turns every SQLite error into a `RegistryError`."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path

import pytest

from s1ops import registry
from s1ops.registry import Project, RegistryError

# An unclosed connection is a ResourceWarning on 3.13+; every call must close its own.
pytestmark = pytest.mark.filterwarnings("error::ResourceWarning", "error::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "projects.db"
    registry.init_registry(path)
    return path


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    d = tmp_path / "ws-a"
    d.mkdir()
    return d


def _refused(kind: str, fn, *args, **kwargs) -> str:
    with pytest.raises(RegistryError) as info:
        fn(*args, **kwargs)
    assert info.value.kind == kind, str(info.value)
    return str(info.value)


def _raw(db: Path, *sql: str) -> None:
    with contextlib.closing(sqlite3.connect(db, isolation_level=None)) as conn:
        for statement in sql:
            conn.execute(statement)


# --- init ---------------------------------------------------------------------------------------


def test_init_writes_an_empty_registry_and_nothing_else(tmp_path: Path) -> None:
    path = tmp_path / "r.db"
    registry.init_registry(path)
    assert registry.check_registry(path) == 0 and registry.list_projects(path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.db"]
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA application_id").fetchone() == (registry.APPLICATION_ID,)
        assert conn.execute("PRAGMA user_version").fetchone() == (registry.SCHEMA_VERSION,)
        assert conn.execute("SELECT name, seq FROM sqlite_sequence").fetchall() == [("projects", 0)]


def test_init_refuses_an_existing_path_and_leaves_it_as_it_was(tmp_path: Path, db: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_bytes(b"keep me")
    for path in (db, other):
        before = path.read_bytes()
        assert "already exists" in _refused("exists", registry.init_registry, path)
        assert path.read_bytes() == before
    dangling = tmp_path / "link.db"
    dangling.symlink_to(tmp_path / "nowhere.db")
    _refused("exists", registry.init_registry, dangling)
    assert not (tmp_path / "nowhere.db").exists()


def test_a_failed_init_leaves_nothing_and_can_be_run_again(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "r.db"

    def no_link(src, dst):
        raise PermissionError(13, "Operation not permitted", dst)

    monkeypatch.setattr(registry.os, "link", no_link)
    with pytest.raises(PermissionError):
        registry.init_registry(path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    registry.init_registry(path)
    assert registry.list_projects(path) == []


def test_a_file_name_needing_uri_escapes_opens(tmp_path: Path, ws: Path) -> None:
    path = tmp_path / "a b?c#d%20e.db"
    registry.init_registry(path)
    registry.add_project(path, "alpha", str(ws))
    assert [p.slug for p in registry.list_projects(path)] == ["alpha"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a b?c#d%20e.db", "ws-a"]


# --- operations ---------------------------------------------------------------------------------


def test_add_list_rename_remove(db: Path, ws: Path, tmp_path: Path) -> None:
    other = tmp_path / "ws-b"
    other.mkdir()
    a = registry.add_project(db, "alpha", str(ws), "알파 티켓")
    b = registry.add_project(db, "beta", str(other))
    assert a == Project(1, "alpha", "알파 티켓", str(ws)) and b == Project(2, "beta", "beta", str(other))
    assert registry.list_projects(db) == [a, b]
    assert registry.get_project(db, slug="beta") == b == registry.get_project(db, project_id=2)
    renamed = registry.rename_project(db, "Beta tickets", slug="beta")
    assert renamed == Project(2, "beta", "Beta tickets", str(other))
    assert registry.rename_project(db, "Alpha", project_id=1) == Project(1, "alpha", "Alpha", str(ws))
    assert registry.remove_project(db, slug="alpha") == Project(1, "alpha", "Alpha", str(ws))
    assert registry.list_projects(db) == [renamed]
    assert ws.is_dir()


def test_a_relative_workspace_gets_the_working_directory_in_front(db: Path, ws: Path, tmp_path: Path, monkeypatch) -> None:
    link = tmp_path / "current"
    link.symlink_to(ws)
    monkeypatch.chdir(tmp_path)
    assert registry.add_project(db, "alpha", "ws-a").workspace_path == os.path.join(os.getcwd(), "ws-a")
    assert registry.add_project(db, "beta", "current").workspace_path == os.path.join(os.getcwd(), "current")


def test_an_id_is_never_reused(db: Path, ws: Path) -> None:
    # Removing the newest registration and adding again would reuse its id without AUTOINCREMENT.
    registry.add_project(db, "alpha", str(ws))
    registry.add_project(db, "beta", str(ws))
    registry.remove_project(db, slug="beta")
    assert registry.add_project(db, "beta", str(ws)).id == 3
    registry.remove_project(db, project_id=1)
    assert registry.add_project(db, "alpha", str(ws)).id == 4
    _refused("missing", registry.get_project, db, project_id=1)


def test_missing_and_existing_projects(db: Path, ws: Path) -> None:
    registry.add_project(db, "alpha", str(ws))
    assert "no project 'beta'" in _refused("missing", registry.rename_project, db, "x", slug="beta")
    assert "no project #9" in _refused("missing", registry.remove_project, db, project_id=9)
    assert "already exists" in _refused("exists", registry.add_project, db, "alpha", str(ws))
    for bad in (0, -1, True, "1", 1.0):
        _refused("invalid", registry.get_project, db, project_id=bad)
    with pytest.raises(TypeError):
        registry.get_project(db)
    with pytest.raises(TypeError):
        registry.get_project(db, slug="alpha", project_id=1)


@pytest.mark.parametrize("slug", ["", "Alpha", "-a", "a_b", "a b", "alpha\n", "a" * 64, "é", "a/b", 7, None])
def test_a_bad_slug_is_refused_before_the_file_is_touched(db: Path, ws: Path, slug) -> None:
    before = db.read_bytes()
    _refused("invalid", registry.add_project, db, slug, str(ws))
    assert db.read_bytes() == before


@pytest.mark.parametrize("name", ["", " x", "x ", "a\nb", "a\x1bb", "a\x85b", "a\u202eb", "a\u2066b", "a\u2028b", "a\u061cb", "a\ud800b", "x" * 81, 3])
def test_a_bad_name_is_refused_before_the_file_is_touched(db: Path, ws: Path, name) -> None:
    registry.add_project(db, "alpha", str(ws))
    before = db.read_bytes()
    _refused("invalid", registry.add_project, db, "beta", str(ws), name)
    _refused("invalid", registry.rename_project, db, name, slug="alpha")
    assert db.read_bytes() == before


def test_the_character_rule_does_not_follow_the_unicode_version(db: Path, ws: Path) -> None:
    # U+1FAE9 is printable to 3.14 and unassigned (so not printable) to 3.11; a row one supported Python
    # writes must read on every other, so the rule is a fixed set of code points, not isprintable().
    # Everything outside the refused set is kept as given, joiners and invisible characters included.
    accepted = (
        "face \U0001fae9", "unassigned \U000e0080 \U0010fffd", "한글 이름", "coder \U0001f469\u200d\U0001f4bb",
        "zwnj \u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645", "cgj a\u034fb", "nonchar \ufdd0", "shy a\u00adb", "bom a\ufeffb",
    )
    for name in accepted:
        project = registry.add_project(db, f"p{len(registry.list_projects(db))}", str(ws), name)
        assert project.name == name
    assert [p.name for p in registry.list_projects(db)] == list(accepted)


def test_a_name_message_names_a_surrogate_without_showing_it(db: Path, ws: Path) -> None:
    message = _refused("invalid", registry.add_project, db, "beta", str(ws), "a\udc80")
    assert "U+DC80" in message and "\udc80" not in message
    assert registry.add_project(db, "beta", str(ws), "x" * 80).name == "x" * 80


def test_a_bad_workspace_is_refused_before_the_file_is_touched(db: Path, tmp_path: Path) -> None:
    afile = tmp_path / "file.txt"
    afile.write_text("x")
    before = db.read_bytes()
    cases = {
        "": "non-empty", "a\x00b": "NUL", "a\udcffb": "U+DCFF", "a\nb": "U+000A", "a\u202eb": "U+202E", "a\u2029b": "U+2029",
        str(tmp_path / "nope"): "cannot be read", str(afile): "not a directory",
    }
    for path, said in cases.items():
        assert said in _refused("invalid", registry.add_project, db, "alpha", path)
    with pytest.raises(RegistryError):
        registry.add_project(db, "alpha", None)  # type: ignore[arg-type]
    assert db.read_bytes() == before


def test_the_stored_workspace_path_is_checked_not_only_the_argument(db: Path, tmp_path: Path, monkeypatch) -> None:
    # `.` is a fine argument, but the absolute path it becomes holds the working directory's name.
    odd = tmp_path / "two\nlines"
    odd.mkdir()
    monkeypatch.chdir(odd)
    before = db.read_bytes()
    assert "U+000A" in _refused("invalid", registry.add_project, db, "alpha", ".")
    assert db.read_bytes() == before
    assert registry.list_projects(db) == []


def test_a_bad_selector_is_refused_before_the_file_is_opened(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    for call in (
        lambda: registry.get_project(missing, slug="Bad Slug"),
        lambda: registry.rename_project(missing, "x", project_id=0),
        lambda: registry.remove_project(missing, project_id=True),
    ):
        _refused("invalid", call)
    with pytest.raises(TypeError):
        registry.remove_project(missing)


def _link_dotdot(tmp_path: Path) -> tuple[Path, Path]:
    """a/link -> b/child, so `a/link/..` is b to the filesystem and a to text that folds `..`."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "child").mkdir(parents=True)
    (tmp_path / "a" / "link").symlink_to(tmp_path / "b" / "child")
    return tmp_path / "a", tmp_path / "b"


def test_a_path_through_a_link_and_dotdot_is_the_file_the_filesystem_finds(tmp_path: Path, ws: Path) -> None:
    a, b = _link_dotdot(tmp_path)
    via = a / "link" / ".." / "r.db"
    registry.init_registry(via)
    assert (b / "r.db").is_file() and not (a / "r.db").exists()
    registry.init_registry(a / "r.db")
    registry.add_project(via, "alpha", str(ws))
    assert [p.slug for p in registry.list_projects(b / "r.db")] == ["alpha"]
    assert registry.list_projects(a / "r.db") == []
    assert "already exists" in _refused("exists", registry.init_registry, via)


def test_a_workspace_through_a_link_and_dotdot_is_stored_as_given(db: Path, tmp_path: Path, monkeypatch) -> None:
    # Both a and b exist; the filesystem reads `a/link/..` as b, so that is what the stored path must name.
    a, b = _link_dotdot(tmp_path)
    (b / "marker").write_text("b")
    stored = registry.add_project(db, "alpha", str(a / "link" / "..")).workspace_path
    assert stored == str(a / "link" / "..") and (Path(stored) / "marker").is_file()
    monkeypatch.chdir(a)
    assert registry.add_project(db, "beta", "link/..").workspace_path == os.path.join(os.getcwd(), "link/..")


def test_a_registry_path_the_filesystem_refuses_is_refused(db: Path, tmp_path: Path) -> None:
    assert "cannot be read" in _refused("unusable", registry.list_projects, str(db) + "/")
    with pytest.raises(FileNotFoundError):
        registry.init_registry(tmp_path / "absent" / ".." / "r.db")
    assert not (tmp_path / "r.db").exists() and not (tmp_path / "absent").exists()
    for bad in (tmp_path / "x" / "..", str(tmp_path) + "/", str(tmp_path / "new.db") + "/"):
        _refused("invalid", registry.init_registry, bad)
    assert not (tmp_path / "new.db").exists()


def test_init_through_a_link_and_dotdot_uses_the_physical_parent(tmp_path: Path) -> None:
    # Only b/only-here exists; `a/link/../only-here` reaches it, `a/only-here` (the text folded) does not.
    a, b = _link_dotdot(tmp_path)
    (b / "only-here").mkdir()
    registry.init_registry(a / "link" / ".." / "only-here" / "r.db")
    assert sorted(p.name for p in (b / "only-here").iterdir()) == ["r.db"]


def test_the_hard_link_refusal_mentions_an_interrupted_init(db: Path, ws: Path) -> None:
    # Killed between linking the finished file into place and removing its temporary name.
    orphan = db.parent / ".s1-registry-0123456789abcdef.tmp"
    os.link(db, orphan)
    message = _refused("unusable", registry.list_projects, db)
    assert "2 hard links" in message and ".s1-registry-*.tmp name beside it" in message
    # The message names no file: a scan cannot tell a leftover from a backup someone linked on purpose.
    assert str(orphan) not in message
    orphan.unlink()
    assert registry.list_projects(db) == []


def test_a_hard_linked_registry_is_refused(db: Path, tmp_path: Path, ws: Path) -> None:
    other = tmp_path / "second-name.db"
    os.link(db, other)
    before = db.read_bytes()
    for path in (db, other):
        assert "2 hard links" in _refused("unusable", registry.add_project, path, "alpha", str(ws))
    assert db.read_bytes() == before
    other.unlink()
    assert registry.add_project(db, "alpha", str(ws)).id == 1


# --- files that are not this registry ------------------------------------------------------------


def _schema_variant(tmp_path: Path, name: str, *sql: str, app_id: int = registry.APPLICATION_ID, version: int = 1) -> Path:
    path = tmp_path / name
    _raw(path, f"PRAGMA application_id = {app_id}", f"PRAGMA user_version = {version}", *sql)
    return path


def test_a_file_that_is_not_exactly_a_registry_is_refused_and_left_alone(tmp_path: Path, ws: Path) -> None:
    cols = "slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL, workspace_path TEXT NOT NULL"
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database at all, just bytes" * 40)
    variants = {
        "not a database": junk,
        "another app": _schema_variant(tmp_path, "other.db", registry._SCHEMA, app_id=7),
        "no stamp": _schema_variant(tmp_path, "nostamp.db", registry._SCHEMA, app_id=0, version=0),
        "newer": _schema_variant(tmp_path, "newer.db", registry._SCHEMA, version=2),
        "no table": _schema_variant(tmp_path, "notable.db"),
        "no AUTOINCREMENT": _schema_variant(tmp_path, "noauto.db", f"CREATE TABLE projects(id INTEGER PRIMARY KEY, {cols})"),
        "no UNIQUE": _schema_variant(
            tmp_path, "nounique.db",
            "CREATE TABLE projects(id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL, name TEXT NOT NULL, workspace_path TEXT NOT NULL)",
        ),
        "a trigger": _schema_variant(
            tmp_path, "trigger.db", registry._SCHEMA,
            "CREATE TRIGGER t AFTER DELETE ON projects BEGIN SELECT 1; END",
        ),
        "an extra table": _schema_variant(tmp_path, "extra.db", registry._SCHEMA, "CREATE TABLE notes(x)"),
    }
    said = {"newer": "newer s1ops", "another app": "not a project registry", "no table": "schema"}
    for name, path in variants.items():
        before = path.read_bytes()
        for call in (
            lambda: registry.list_projects(path),
            lambda: registry.check_registry(path),
            lambda: registry.add_project(path, "alpha", str(ws)),
            lambda: registry.remove_project(path, slug="alpha"),
        ):
            message = _refused("unusable", call)
            assert str(path) in message, (name, message)
            assert said.get(name, "") in message, (name, message)
        assert path.read_bytes() == before, name


def test_a_path_that_is_not_a_regular_file_is_refused_and_nothing_is_created(tmp_path: Path) -> None:
    assert "not found" in _refused("unusable", registry.list_projects, tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()
    assert "not a regular file" in _refused("unusable", registry.list_projects, tmp_path)
    fifo = tmp_path / "fifo.db"
    os.mkfifo(fifo)
    assert "not a regular file" in _refused("unusable", registry.list_projects, fifo)


def test_a_hand_edited_row_is_refused_not_a_type_error(db: Path, ws: Path) -> None:
    registry.add_project(db, "alpha", str(ws))
    for statement in (
        "UPDATE projects SET name = x'00ff'",                     # a BLOB (an integer would be stored as text)
        "UPDATE projects SET workspace_path = 'relative/dir'",    # not absolute
        "UPDATE projects SET slug = 'Not A Slug'",
        "UPDATE projects SET name = 'a' || char(10) || 'b'",      # a newline
    ):
        _raw(db, "UPDATE projects SET slug = 'alpha', name = 'alpha'", f"UPDATE projects SET workspace_path = '{ws}'", statement)
        message = _refused("unusable", registry.list_projects, db)
        assert "row 1" in message and str(db) in message, (statement, message)
        _refused("unusable", registry.get_project, db, slug="alpha")


def test_an_id_counter_missing_below_an_id_or_not_an_integer_is_refused(db: Path, ws: Path) -> None:
    # A counter lowered only to the largest id still there cannot be told from a true one; what the file
    # shows (a non-integer counter, one below an id it holds, or none at all) is refused.
    registry.add_project(db, "alpha", str(ws))
    registry.add_project(db, "beta", str(ws))
    for statement in (
        "UPDATE sqlite_sequence SET seq = 'two'",
        "UPDATE sqlite_sequence SET seq = 1",
        "DELETE FROM sqlite_sequence",
        "INSERT INTO sqlite_sequence VALUES ('other', 9)",
    ):
        _raw(db, "DELETE FROM sqlite_sequence", "INSERT INTO sqlite_sequence VALUES ('projects', 2)", statement)
        before = db.read_bytes()
        assert "id counter" in _refused("unusable", registry.add_project, db, "gamma", str(ws)), statement
        assert db.read_bytes() == before
    _raw(db, "DELETE FROM sqlite_sequence", "INSERT INTO sqlite_sequence VALUES ('projects', 2)")
    assert registry.add_project(db, "gamma", str(ws)).id == 3


def test_a_counter_deleted_after_the_last_project_went_is_refused(db: Path, ws: Path) -> None:
    # Without the counter row a registry whose projects were all removed would hand id 1 out again.
    registry.add_project(db, "alpha", str(ws))
    registry.remove_project(db, slug="alpha")
    _raw(db, "DELETE FROM sqlite_sequence")
    before = db.read_bytes()
    assert "id counter" in _refused("unusable", registry.add_project, db, "beta", str(ws))
    assert db.read_bytes() == before


def test_text_that_is_not_utf8_is_refused(db: Path, ws: Path) -> None:
    registry.add_project(db, "alpha", str(ws))
    _raw(db, "UPDATE projects SET name = CAST(x'61ff62' AS TEXT)")
    assert str(db) in _refused("unusable", registry.list_projects, db)


# --- concurrency --------------------------------------------------------------------------------


def test_a_writer_held_past_the_timeout_is_busy_and_changes_nothing(db: Path, ws: Path, monkeypatch) -> None:
    monkeypatch.setattr(registry, "BUSY_TIMEOUT_S", 0.05)
    registry.add_project(db, "alpha", str(ws))
    with contextlib.closing(sqlite3.connect(db, isolation_level=None)) as holder:
        holder.execute("BEGIN IMMEDIATE")
        assert "is busy" in _refused("busy", registry.add_project, db, "beta", str(ws))
        _refused("busy", registry.rename_project, db, "x", slug="alpha")
        _refused("busy", registry.remove_project, db, slug="alpha")
        # A reader is not blocked by a writer that has not committed (rollback journal, RESERVED lock).
        assert [p.slug for p in registry.list_projects(db)] == ["alpha"]
        holder.execute("ROLLBACK")
    assert registry.list_projects(db) == [Project(1, "alpha", "alpha", str(ws))]
    assert registry.add_project(db, "beta", str(ws)).id == 2


def test_a_write_holds_the_write_lock_from_its_first_read(db: Path, ws: Path, monkeypatch) -> None:
    # Its check ("is the slug free?") and its change commit together: no other writer can start between them.
    registry.add_project(db, "alpha", str(ws))
    seen = []
    real = registry._check_schema

    def probe(conn, path):
        # Probed before the schema is read: the lock must be taken by BEGIN, before the first read.
        with contextlib.closing(sqlite3.connect(db, isolation_level=None, timeout=0)) as other:
            try:
                other.execute("BEGIN IMMEDIATE")
                other.execute("ROLLBACK")
                seen.append("free")
            except sqlite3.OperationalError:
                seen.append("held")
        real(conn, path)

    monkeypatch.setattr(registry, "_check_schema", probe)
    registry.rename_project(db, "x", slug="alpha")
    registry.add_project(db, "beta", str(ws))
    registry.remove_project(db, slug="beta")
    registry.list_projects(db)
    assert seen == ["held", "held", "held", "free"]


def test_a_call_keeps_the_locks_another_connection_in_this_process_holds(db: Path, ws: Path) -> None:
    # POSIX locks belong to the process: closing any descriptor on the file drops all of them. Only SQLite,
    # which defers such a close, may open the file here, so a lock held elsewhere in the process survives.
    import subprocess
    import sys

    probe = (
        "import sqlite3, sys\n"
        "c = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)\n"
        "try:\n"
        "    c.execute('BEGIN IMMEDIATE'); c.execute('ROLLBACK'); print('free')\n"
        "except sqlite3.OperationalError:\n"
        "    print('held')\n"
        "c.close()\n"
    )
    with contextlib.closing(sqlite3.connect(db, isolation_level=None)) as holder:
        holder.execute("BEGIN IMMEDIATE")
        registry.list_projects(db)
        registry.check_registry(db)
        with pytest.raises(RegistryError):
            registry.list_projects(db / "nope")
        out = subprocess.run([sys.executable, "-c", probe, str(db)], capture_output=True, text=True, check=True).stdout
        assert out.strip() == "held"
        holder.execute("ROLLBACK")


def test_a_failed_write_rolls_back(db: Path, ws: Path, monkeypatch) -> None:
    registry.add_project(db, "alpha", str(ws))
    real = registry._one

    def fail_after_check(conn, path, slug, project_id):
        real(conn, path, slug, project_id)
        conn.execute("UPDATE projects SET name = 'changed'")
        raise RuntimeError("boom")

    monkeypatch.setattr(registry, "_one", fail_after_check)
    with pytest.raises(RuntimeError):
        registry.rename_project(db, "x", slug="alpha")
    monkeypatch.undo()
    assert registry.get_project(db, slug="alpha").name == "alpha"
