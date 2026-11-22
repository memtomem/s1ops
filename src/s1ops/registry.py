"""A registry of projects: each a slug, a display name and one workspace directory, kept in a SQLite
file that `s1 project` reads and writes (the workbench is to read it too, as `s1 web --db`). It holds no records, labels or
state text, only where each workspace is.

Every call opens its own connection to an existing file (`mode=rw`, so a mistyped path never creates
one) and runs in one transaction: `BEGIN IMMEDIATE` for a write, whose checks and change commit
together, a plain `BEGIN` for a read. A file is used only when it holds exactly what `init_registry`
writes (its application id, its schema version, and its schema objects, compared as SQLite stores
them), so a foreign database, a newer one, or one whose table lost a constraint is refused through a
read-only connection, before any connection that could change it is opened (a read-write one would roll
back a crashed writer's journal or checkpoint a WAL file). Every `sqlite3.Error` leaves as a
`RegistryError`.

A project's `id` is its registration number: SQLite's `AUTOINCREMENT` never reuses one, and nothing
changes a registration's directory, so an id names one directory for as long as it exists, and a slug
removed and registered again gets a new id.

Paths are never normalized: a relative one only gets the working directory in front (`os.path.join`,
which keeps `..` and symbolic links), so the path that is checked is the path that is opened or stored,
and the filesystem decides what `link/..` or a trailing `/` means, as it does for every other command.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import sqlite3
import stat
import secrets
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from s1ops.contracts.records import surrogate_problem

APPLICATION_ID = 0x53316F70  # "S1op"
SCHEMA_VERSION = 1
_SCHEMA = (
    "CREATE TABLE projects(id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL UNIQUE, "
    "name TEXT NOT NULL, workspace_path TEXT NOT NULL)"
)
BUSY_TIMEOUT_S = 5.0
NAME_MAX = 80
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,62}", re.ASCII)
# The characters a stored name or path may not hold: those that break or rearrange a line where the value
# is shown (a terminal, the CLI's error line, the page) -- C0 and C1 controls with DEL, the bidirectional
# controls (U+061C, U+200E-U+200F, U+202A-U+202E, U+2066-U+2069), the line and paragraph separators
# (U+2028-U+2029) -- and lone surrogates, which no UTF-8 file can hold. Everything else is accepted,
# joiners included (an emoji sequence uses U+200D). A fixed set of code points rather than
# `str.isprintable()`, whose answer follows the interpreter's Unicode version: a row one supported Python
# writes must read on every other.
_FORBIDDEN = re.compile(
    "[\x00-\x1f\x7f-\x9f\u061c\u200e\u200f\u2028-\u202e\u2066-\u2069\ud800-\udfff]"
)
_TMP_PREFIX, _TMP_SUFFIX = ".s1-registry-", ".tmp"
SLUG_RULE = "1-63 characters of a-z, 0-9 and '-', starting with a letter or digit"


class RegistryError(ValueError):
    """A refused registry operation. `kind` is `invalid` (an argument), `exists` (a slug or file already
    there), `missing` (no such project), `busy` (a lock it needs stayed taken past the timeout) or
    `unusable` (the file is not a registry this version can use)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Project:
    id: int
    slug: str
    name: str
    workspace_path: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "slug": self.slug, "name": self.name, "workspace_path": self.workspace_path}


# --- argument rules -----------------------------------------------------------------------------


def slug_problem(slug: object) -> str | None:
    if not isinstance(slug, str) or _SLUG.fullmatch(slug) is None:
        return f"a project slug is {SLUG_RULE}"
    return None


def name_problem(name: object) -> str | None:
    if not isinstance(name, str):
        return "a project name is text"
    problem = surrogate_problem("the project name", name)
    if problem:
        return problem
    if not 1 <= len(name) <= NAME_MAX:
        return f"a project name is 1-{NAME_MAX} characters"
    if name != name.strip():
        return "a project name has no leading or trailing space"
    return _forbidden_problem("a project name", name)


def _forbidden_problem(what: str, text: str) -> str | None:
    found = _FORBIDDEN.search(text)
    if found is None:
        return None
    return f"{what} holds a control, direction or line-separator character (U+{ord(found.group()):04X})"


def path_problem(path: object) -> str | None:
    """Why `path` cannot be stored, checked before the filesystem or SQLite sees it: `os.stat` raises
    `ValueError` on a NUL, and a lone surrogate (an undecodable byte under `surrogateescape`) passes
    `stat` and then fails to bind."""
    if not isinstance(path, str) or path == "":
        return "a workspace path is a non-empty text"
    if "\x00" in path:
        return "a workspace path holds no NUL character"
    problem = surrogate_problem("the workspace path", path)
    if problem:
        return problem
    return _forbidden_problem("a workspace path", path)


def _absolute(path: str) -> str:
    """`path` with the working directory in front when it is relative; nothing else changes (no `..`
    folding, no link resolution), so it names the same file the filesystem finds at `path`."""
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def _refuse(problem: str | None) -> None:
    if problem:
        raise RegistryError("invalid", problem)


def _workspace_dir(path: str) -> str:
    """`path` made absolute without normalizing it, after checking it names a directory now."""
    _refuse(path_problem(path))
    full = _absolute(path)
    # The stored path is checked too: a relative one takes the working directory's name, which may hold anything.
    _refuse(path_problem(full))
    try:
        st = os.stat(full)
    except OSError as e:
        raise RegistryError("invalid", f"workspace directory cannot be read: {full}: {e.strerror or e}") from None
    if not stat.S_ISDIR(st.st_mode):
        raise RegistryError("invalid", f"workspace is not a directory: {full}")
    return full


# --- the file -----------------------------------------------------------------------------------


@functools.cache
def _expected_schema() -> frozenset[tuple[str, str, str | None]]:
    """The schema objects `_SCHEMA` makes, as this SQLite stores them (the table, its unique index and
    the `AUTOINCREMENT` counter table)."""
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        conn.execute(_SCHEMA)
        return frozenset(conn.execute("SELECT type, name, sql FROM sqlite_master"))


def _error(db: str | os.PathLike[str], e: sqlite3.Error) -> RegistryError:
    name = getattr(e, "sqlite_errorname", "")
    if name.startswith(("SQLITE_BUSY", "SQLITE_LOCKED")):
        return RegistryError("busy", f"project registry is busy: {db} (another process is using it; try again)")
    return RegistryError("unusable", f"project registry cannot be used: {db}: {e}")


def _check_file(db: str | os.PathLike[str]) -> str:
    full = _absolute(os.fspath(db))
    try:
        st = os.stat(full)
    except FileNotFoundError:
        raise RegistryError("unusable", f"project registry not found: {db} (run `s1 project init` first)") from None
    except OSError as e:
        raise RegistryError("unusable", f"project registry cannot be read: {db}: {e.strerror or e}") from None
    if not stat.S_ISREG(st.st_mode):
        raise RegistryError("unusable", f"project registry is not a regular file: {db}")
    if st.st_nlink > 1:
        # SQLite keeps its rollback journal beside the name it opened, so a write interrupted under one
        # name would not be rolled back when the file is next opened under another (the ledger's rule too).
        # No scan for the other name: only SQLite may open and close this file in-process (closing any other
        # descriptor on it drops the process's POSIX locks), and a scan cannot tell a leftover from a backup.
        raise RegistryError(
            "unusable",
            f"project registry has {st.st_nlink} hard links: {db} (give it one name; a `s1 project init` "
            f"interrupted before it finished leaves a {_TMP_PREFIX}*{_TMP_SUFFIX} name beside it)",
        )
    return full


def _check_schema(conn: sqlite3.Connection, db: str | os.PathLike[str]) -> None:
    (app_id,) = conn.execute("PRAGMA application_id").fetchone()
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    if app_id != APPLICATION_ID:
        raise RegistryError("unusable", f"not a project registry: {db} (make one with `s1 project init`)")
    if version != SCHEMA_VERSION:
        newer = " (written by a newer s1ops)" if isinstance(version, int) and version > SCHEMA_VERSION else ""
        raise RegistryError("unusable", f"project registry has schema version {version}, not {SCHEMA_VERSION}: {db}{newer}")
    if frozenset(conn.execute("SELECT type, name, sql FROM sqlite_master")) != _expected_schema():
        raise RegistryError("unusable", f"project registry's schema is not the one `s1 project init` writes: {db}")
    # AUTOINCREMENT's counter: `init_registry` writes it (at 0), so it is always one row for `projects` holding
    # an integer at least the largest id. SQLite takes the larger of it and the largest id, so a missing,
    # lowered or non-integer counter would hand out an id a removed project had, and that project's URL would
    # open another directory. (A counter lowered only to the largest id still there cannot be told from a
    # true one.)
    counters = conn.execute("SELECT name, seq FROM sqlite_sequence").fetchall()
    (top,) = conn.execute("SELECT max(id) FROM projects").fetchone()
    ok = (
        len(counters) == 1 and counters[0][0] == "projects" and type(counters[0][1]) is int
        and counters[0][1] >= (top if type(top) is int else 0)
    )
    if not ok:
        raise RegistryError("unusable", f"project registry's id counter is not one `s1 project` keeps: {db}")


def _uri(full: str, mode: str) -> str:
    # "file://" and an empty authority: the path is absolute, and one that starts with "//" would otherwise
    # be read as an authority. Quoted from its bytes, so a name that is not UTF-8 (held as surrogate escapes)
    # opens the file `os.stat` found.
    return "file://" + urllib.parse.quote_from_bytes(os.fsencode(full)) + f"?mode={mode}"


def _identify(full: str, db: str | os.PathLike[str]) -> None:
    """Refuse a file that is not a registry through a read-only connection, before any connection that could
    change it: a read-write one would roll back a crashed writer's journal or checkpoint a WAL file on
    close, and both would change a file this refuses. A read-only connection does neither (SQLite refuses
    the open instead, which is a refusal here too)."""
    try:
        conn = sqlite3.connect(_uri(full, "ro"), uri=True, isolation_level=None, timeout=BUSY_TIMEOUT_S)
    except sqlite3.Error as e:
        raise _error(db, e) from None
    with contextlib.closing(conn):
        try:
            conn.execute("BEGIN")
            _check_schema(conn, db)
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            raise _error(db, e) from None


@contextlib.contextmanager
def _transaction(db: str | os.PathLike[str], *, write: bool) -> Iterator[sqlite3.Connection]:
    full = _check_file(db)
    _identify(full, db)
    try:
        conn = sqlite3.connect(_uri(full, "rw"), uri=True, isolation_level=None, timeout=BUSY_TIMEOUT_S)
    except sqlite3.Error as e:
        raise _error(db, e) from None
    with contextlib.closing(conn):
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            # Again inside this transaction: the file may have been replaced since it was identified.
            _check_schema(conn, db)
            yield conn
            conn.execute("COMMIT")
        except BaseException as exc:
            if conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
            if isinstance(exc, sqlite3.Error):
                raise _error(db, exc) from None
            raise


def _row(db: str | os.PathLike[str], row: tuple[Any, ...]) -> Project:
    """A stored row, re-checked: the file is the user's, and a hand-inserted value of another type or
    shape must be refused here rather than fail later as a `TypeError`."""
    pid, slug, name, path = row
    problem = None
    if type(pid) is not int or pid < 1:
        problem = "an id that is not a positive integer"
    elif slug_problem(slug) or name_problem(name) or path_problem(path) or not os.path.isabs(path):
        problem = "a slug, name or workspace path `s1 project add` would refuse"
    if problem:
        shown = pid if type(pid) is int else type(pid).__name__
        raise RegistryError("unusable", f"project registry row {shown} holds {problem}: {db}")
    return Project(pid, slug, name, path)


def _select(conn: sqlite3.Connection, db: str | os.PathLike[str]) -> list[Project]:
    """Every project, each row re-checked: one rule for every operation, so a hand-edited row is refused
    whichever project a call names."""
    rows = conn.execute("SELECT id, slug, name, workspace_path FROM projects ORDER BY id")
    return [_row(db, r) for r in rows]


def _check_selector(slug: str | None, project_id: int | None) -> None:
    """Checked before the file is opened, so a missing file or a lock never hides a bad selector."""
    if (slug is None) == (project_id is None):
        raise TypeError("name a project by exactly one of slug= and project_id=")
    if slug is not None:
        _refuse(slug_problem(slug))
    elif type(project_id) is not int or project_id < 1:
        raise RegistryError("invalid", "a project id is a positive integer")


def _one(conn: sqlite3.Connection, db: str | os.PathLike[str], slug: str | None, project_id: int | None) -> Project:
    if slug is not None:
        found = [p for p in _select(conn, db) if p.slug == slug]
        what = f"project {slug!r}"
    else:
        found = [p for p in _select(conn, db) if p.id == project_id]
        what = f"project #{project_id}"
    if not found:
        raise RegistryError("missing", f"no {what} in {db}")
    return found[0]


# --- operations ---------------------------------------------------------------------------------


def init_registry(db: str | os.PathLike[str]) -> None:
    """Create an empty registry at `db`, which must not exist. The schema and its stamps are written
    to a temporary file beside it in one transaction and then linked into place, so no reader ever
    sees a half-made file at `db`. A failure that unwinds removes the temporary file; a process killed
    before the link leaves a `.s1-registry-*.tmp` beside `db`, never a file at `db`, and one killed between
    the link and the removal leaves both names on one file, which is refused until the temporary one is
    removed (as it is by everyone for the moment between the two)."""
    final = _absolute(os.fspath(db))
    if os.path.basename(final) in ("", ".", ".."):
        raise RegistryError("invalid", f"project registry path names a directory, not a file: {db}")
    if os.path.lexists(final):
        raise RegistryError("exists", f"project registry already exists: {db}")
    # Not `tempfile.mkstemp`, which makes its directory absolute by folding `..` as text.
    directory = os.path.dirname(final)
    while True:
        tmp = os.path.join(directory, f"{_TMP_PREFIX}{secrets.token_hex(8)}{_TMP_SUFFIX}")
        try:
            os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            break
        except FileExistsError:
            continue
    try:
        try:
            with contextlib.closing(sqlite3.connect(tmp, isolation_level=None)) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute(_SCHEMA)
                conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('projects', 0)")
                conn.execute("COMMIT")
        except sqlite3.Error as e:
            raise _error(db, e) from None
        try:
            os.link(tmp, final)
        except FileExistsError:
            raise RegistryError("exists", f"project registry already exists: {db}") from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def check_registry(db: str | os.PathLike[str]) -> int:
    """The number of projects in `db`, after checking it is a usable registry whose rows all read."""
    with _transaction(db, write=False) as conn:
        return len(_select(conn, db))


def list_projects(db: str | os.PathLike[str]) -> list[Project]:
    with _transaction(db, write=False) as conn:
        return _select(conn, db)


def get_project(db: str | os.PathLike[str], *, slug: str | None = None, project_id: int | None = None) -> Project:
    _check_selector(slug, project_id)
    with _transaction(db, write=False) as conn:
        return _one(conn, db, slug, project_id)


def add_project(db: str | os.PathLike[str], slug: str, workspace: str, name: str | None = None) -> Project:
    """Register `workspace` (made absolute, which must be a directory now) as `slug`, named `name`
    (the slug when not given)."""
    _refuse(slug_problem(slug))
    name = slug if name is None else name
    _refuse(name_problem(name))
    path = _workspace_dir(workspace)
    with _transaction(db, write=True) as conn:
        if any(p.slug == slug for p in _select(conn, db)):
            raise RegistryError("exists", f"project {slug!r} already exists in {db}")
        cur = conn.execute("INSERT INTO projects(slug, name, workspace_path) VALUES (?, ?, ?)", (slug, name, path))
        return Project(int(cur.lastrowid), slug, name, path)


def rename_project(
    db: str | os.PathLike[str], name: str, *, slug: str | None = None, project_id: int | None = None
) -> Project:
    """Change a project's display name; its slug, id and directory stay."""
    _refuse(name_problem(name))
    _check_selector(slug, project_id)
    with _transaction(db, write=True) as conn:
        found = _one(conn, db, slug, project_id)
        conn.execute("UPDATE projects SET name = ? WHERE id = ?", (name, found.id))
        return Project(found.id, found.slug, name, found.workspace_path)


def remove_project(db: str | os.PathLike[str], *, slug: str | None = None, project_id: int | None = None) -> Project:
    """Remove a project's registration. Its directory is not touched."""
    _check_selector(slug, project_id)
    with _transaction(db, write=True) as conn:
        found = _one(conn, db, slug, project_id)
        conn.execute("DELETE FROM projects WHERE id = ?", (found.id,))
        return found
