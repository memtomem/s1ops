"""`s1 web --db` writing its registry: a project added, renamed and removed through the registry's own
routes; the rule that a write names no other site; the body, read as `application/json` only, counted
as it arrives and quoted by none of this file's refusals; and that a write waiting for the file's lock
holds no one else up."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops import registry
from s1ops.cli import main as cli_main
from s1ops.web.app import STATIC_DIR, create_app
from s1ops.web.projects import (
    ADD_SHAPE,
    BODY_MAX,
    BODY_NOT_OBJECT,
    BODY_TYPE,
    PATH_NOT_ABSOLUTE,
    REGISTRY_REFUSED,
    RENAME_SHAPE,
    WRITE_REFUSED,
    create_projects_app,
)
from tests.test_web_render import _all_functions, _node
from tests.test_web_sheet import LOOPBACK

HOST = "127.0.0.1:8765"
ORIGIN = "http://" + HOST
JSON = (("content-type", "application/json"),)
OWN = (("origin", ORIGIN), ("sec-fetch-site", "same-origin"))
FRAMES = {"content-security-policy": "frame-ancestors 'none'", "x-frame-options": "DENY"}
LIST, ONE = "/api/registry/projects", "/api/registry/projects/1"
WRITES = ("add_project", "rename_project", "remove_project")
BACKSLASH = chr(92)


class _Registry(dict):
    """The fixture's parts. Its `client` is made when first asked for: the oldest FastAPI supported has
    no test client that can be a loopback one, and the tests that send hand-made scopes run there too."""

    def __missing__(self, key: str) -> TestClient:
        if key != "client":
            raise KeyError(key)
        self[key] = TestClient(self["app"], **LOOPBACK)
        return self[key]


@pytest.fixture
def reg(tmp_path: Path) -> dict:
    """A registry of #1 alpha (a directory holding a file) and #2 beta, and a directory not yet registered."""
    alpha, beta, new, db = (tmp_path / name for name in ("alpha", "beta", "new", "projects.db"))
    for directory in (alpha, beta, new):
        directory.mkdir()
    (alpha / "keep.txt").write_text("kept")
    registry.init_registry(db)
    registry.add_project(db, "alpha", str(alpha), "Alpha tickets")
    registry.add_project(db, "beta", str(beta))
    return _Registry(db=db, alpha=alpha, beta=beta, new=new, app=create_projects_app(db))


@pytest.fixture
def wrote(monkeypatch) -> list[str]:
    """The registry's writes, by name, as they are called."""
    calls: list[str] = []
    for name in WRITES:
        real = getattr(registry, name)
        monkeypatch.setattr(registry, name, lambda *a, _name=name, _real=real, **k: (calls.append(_name), _real(*a, **k))[1])
    return calls


@pytest.fixture
def opened(monkeypatch) -> list[bool]:
    """Every opening of the registry file, a read as well as a write (`write`, as the registry asked for
    it): the three writes above are not the only calls that open it. A test empties the list after its
    own reads and before the request it is about."""
    calls: list[bool] = []
    real = registry._transaction

    @contextlib.contextmanager
    def recorded(db, *, write: bool):
        calls.append(write)
        with real(db, write=write) as conn:
            yield conn

    monkeypatch.setattr(registry, "_transaction", recorded)
    return calls


def _rows(db: Path) -> list[dict]:
    return [p.as_dict() for p in registry.list_projects(db)]


async def _call(app, method: str, path: str, *, headers=(), chunks: tuple[bytes, ...] | None = (), host: str = HOST,
                scheme: str = "http", client=("127.0.0.1", 50000), then: str = "end") -> dict:
    """One request as a hand-made scope. `chunks` are handed over one per `receive`; `None` means the
    body must not be asked for at all (the first `receive` fails the test). After the last chunk the
    body ends, or with `then="disconnect"` the client goes away instead."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": scheme, "path": path,
        "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(b"host", host.encode()), *((k.encode(), v.encode()) for k, v in headers)],
        "client": client, "server": ("127.0.0.1", 8765),
    }
    got: dict = {"body": b"", "receives": 0, "starts": 0}
    left = list(chunks or ())

    async def receive() -> dict:
        assert chunks is not None, "the body was asked for before the request was refused"
        got["receives"] += 1
        if left:
            chunk = left.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(left) or then == "disconnect"}
        return {"type": "http.disconnect"} if then == "disconnect" or got["receives"] > max(1, len(chunks)) else {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            got["starts"] += 1
            got["status"] = message["status"]
            got["headers"] = {k.decode().lower(): v.decode() for k, v in message["headers"]}
        else:
            got["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return got


def _send(app, method: str, path: str, **kwargs) -> dict:
    return asyncio.run(_call(app, method, path, **kwargs))


def _detail(got: dict) -> str:
    return json.loads(got["body"])["detail"]


def _body(**fields) -> tuple[bytes]:
    return (json.dumps(fields).encode(),)


# --- adding, renaming, removing ----------------------------------------------------------------------


def test_a_post_registers_a_directory_as_s1_project_add_does(reg) -> None:
    client = reg["client"]
    r = client.post(LIST, json={"slug": "gamma", "workspace_path": str(reg["new"]), "name": "Gamma, new"})
    assert r.status_code == 201
    assert r.json() == {"project": {"id": "3", "slug": "gamma", "name": "Gamma, new", "workspace_path": str(reg["new"])}}
    assert _rows(reg["db"])[2] == {"id": 3, "slug": "gamma", "name": "Gamma, new", "workspace_path": str(reg["new"])}
    assert client.get(LIST).json()["projects"][2] == r.json()["project"]
    # Without a name it is named by its slug, as on the command line.
    r = client.post(LIST, json={"slug": "delta", "workspace_path": str(reg["new"])})
    assert (r.status_code, r.json()["project"]["name"], r.json()["project"]["id"]) == (201, "delta", "4")


def test_a_patch_changes_the_display_name_only(reg) -> None:
    r = reg["client"].patch(ONE, json={"name": "Alpha, renamed"})
    assert r.status_code == 200
    assert r.json() == {"project": {"id": "1", "slug": "alpha", "name": "Alpha, renamed", "workspace_path": str(reg["alpha"])}}
    assert _rows(reg["db"]) == [
        {"id": 1, "slug": "alpha", "name": "Alpha, renamed", "workspace_path": str(reg["alpha"])},
        {"id": 2, "slug": "beta", "name": "beta", "workspace_path": str(reg["beta"])},
    ]


def test_a_delete_removes_the_registration_and_leaves_the_directory(reg) -> None:
    client = reg["client"]
    r = client.delete(ONE)
    assert r.status_code == 200
    assert r.json() == {"project": {"id": "1", "slug": "alpha", "name": "Alpha tickets", "workspace_path": str(reg["alpha"])}}
    assert [p["slug"] for p in _rows(reg["db"])] == ["beta"]
    assert (reg["alpha"] / "keep.txt").read_text() == "kept"


def test_a_removed_number_answers_404_from_then_on_and_never_names_the_new_registration(reg) -> None:
    client = reg["client"]
    assert client.delete(ONE).status_code == 200
    again = client.post(LIST, json={"slug": "alpha", "workspace_path": str(reg["alpha"]), "name": "Alpha again"})
    assert (again.status_code, again.json()["project"]["id"]) == (201, "3")
    before = _rows(reg["db"])
    for r in (client.patch(ONE, json={"name": "stale"}), client.delete(ONE)):
        assert r.status_code == 404 and r.json()["detail"] == f"no project #1 in {reg['db']}"
    assert _rows(reg["db"]) == before
    for url in ("/api/registry/projects/9", "/api/registry/projects/9223372036854775808"):
        assert client.patch(url, json={"name": "x"}).status_code == 404 and client.delete(url).status_code == 404


def test_an_id_past_what_a_browser_number_holds_is_written_as_its_own_string(reg) -> None:
    big = "9007199254740993"
    with contextlib.closing(sqlite3.connect(reg["db"], isolation_level=None)) as conn:
        conn.execute(f"UPDATE projects SET id = {big} WHERE id = 2")
        conn.execute(f"UPDATE sqlite_sequence SET seq = {big}")
    client = reg["client"]
    r = client.patch(f"/api/registry/projects/{big}", json={"name": "Big"})
    assert (r.status_code, r.json()["project"]["id"], r.json()["project"]["name"]) == (200, big, "Big")
    # Its neighbour, which a float cannot tell from it, is no project.
    assert client.delete("/api/registry/projects/9007199254740992").status_code == 404
    r = client.delete(f"/api/registry/projects/{big}")
    assert (r.status_code, r.json()["project"]["id"]) == (200, big)
    assert [p["slug"] for p in _rows(reg["db"])] == ["alpha"]


def test_the_list_is_served_as_the_same_bytes_as_before_it_shared_its_route_with_the_post(reg) -> None:
    name = "Caf" + chr(0xE9) + " " + chr(0xD55C) + chr(0xAE00) + " \\ \" <b>"
    registry.rename_project(reg["db"], name, project_id=2)
    expected = {"registry": {"path": str(reg["db"])}, "projects": [
        {"id": "1", "slug": "alpha", "name": "Alpha tickets", "workspace_path": str(reg["alpha"])},
        {"id": "2", "slug": "beta", "name": name, "workspace_path": str(reg["beta"])},
    ]}
    got = _send(reg["app"], "GET", LIST)
    # Key order, no spaces, characters as themselves: what the route answered when it was a function returning a dict.
    assert got["body"] == json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert got["headers"]["content-type"] == "application/json"


# --- a write names no other site ---------------------------------------------------------------------

FOREIGN = [
    pytest.param((("origin", "http://127.0.0.1:9000"),), id="another-port"),
    pytest.param((("origin", "http://localhost:8765"),), id="localhost-for-127"),
    pytest.param((("origin", "https://" + HOST),), id="another-scheme"),
    pytest.param((("origin", "null"),), id="origin-null"),
    pytest.param((("origin", ORIGIN + "/"),), id="origin-with-a-path"),
    pytest.param((("origin", ORIGIN), ("origin", "http://evil.example")), id="two-origins-first-right"),
    pytest.param((("origin", ORIGIN), ("origin", ORIGIN)), id="the-right-origin-twice"),
    pytest.param((("origin", ORIGIN + ", http://evil.example"),), id="a-list-in-one-origin"),
    pytest.param((("sec-fetch-site", "same-site"),), id="same-site"),
    pytest.param((("sec-fetch-site", "cross-site"),), id="cross-site"),
    pytest.param((("sec-fetch-site", "none"),), id="site-none"),
    pytest.param((("sec-fetch-site", "same-origin"), ("sec-fetch-site", "same-origin")), id="same-origin-twice"),
    pytest.param((("origin", ORIGIN), ("sec-fetch-site", "cross-site")), id="right-origin-cross-site"),
    pytest.param((("origin", "http://evil.example"), ("sec-fetch-site", "same-origin")), id="wrong-origin-same-origin"),
]


@pytest.mark.parametrize("headers", FOREIGN)
@pytest.mark.parametrize(("method", "path"), [
    ("POST", LIST), ("PATCH", ONE), ("DELETE", ONE), ("HEAD", LIST), ("OPTIONS", LIST), ("OPTIONS", ONE), ("PUT", LIST),
    ("POST", "/p/1/"), ("DELETE", "/p/1/api/overview"), ("POST", "/p/"), ("POST", "/"),
])
def test_a_request_naming_another_site_is_refused_before_anything_is_read(reg, wrote, opened, method, path, headers) -> None:
    before = _rows(reg["db"])
    opened.clear()
    got = _send(reg["app"], method, path, headers=(*JSON, *headers), chunks=None)
    assert (got["status"], _detail(got)) == (403, WRITE_REFUSED)
    assert {k: got["headers"].get(k) for k in FRAMES} == FRAMES
    assert not any(name.startswith("access-control") for name in got["headers"])
    assert wrote == [] and opened == []  # neither a write nor a read of the registry
    assert _rows(reg["db"]) == before


@pytest.mark.parametrize("headers", [OWN, (), (("origin", ORIGIN),), (("sec-fetch-site", "same-origin"),)], ids=["page", "neither", "origin", "site"])
def test_the_page_and_a_local_client_pass_the_guard(reg, headers) -> None:
    app = reg["app"]
    got = _send(app, "POST", LIST, headers=(*JSON, *headers), chunks=_body(slug="gamma", workspace_path=str(reg["new"])))
    assert got["status"] == 201
    assert _send(app, "PATCH", ONE, headers=(*JSON, *headers), chunks=_body(name="Renamed"))["status"] == 200
    assert _send(app, "DELETE", ONE, headers=headers)["status"] == 200
    assert [p["slug"] for p in _rows(reg["db"])] == ["beta", "gamma"]
    # What was a 405 for this page stays one.
    assert _send(app, "POST", "/p/2/", headers=headers)["status"] == 405


def test_the_origin_is_compared_with_the_requests_own_scheme_and_host(reg) -> None:
    app = reg["app"]
    assert _send(app, "DELETE", ONE, host="localhost:8765", headers=(("origin", "http://localhost:8765"),))["status"] == 200
    assert _send(app, "DELETE", "/api/registry/projects/2", scheme="https", headers=(("origin", "https://" + HOST),))["status"] == 200
    assert _rows(reg["db"]) == []
    # A default port is left out of both by a browser.
    assert _send(app, "DELETE", ONE, host="localhost", headers=(("origin", "http://localhost"),))["status"] == 404
    assert _send(app, "DELETE", ONE, host="localhost", headers=(("origin", "http://localhost:80"),))["status"] == 403
    assert _send(app, "DELETE", ONE, scheme="https", headers=(("origin", ORIGIN),))["status"] == 403


def test_a_get_is_answered_whatever_site_it_names(reg) -> None:
    got = _send(reg["app"], "GET", LIST, headers=(("origin", "http://evil.example"), ("sec-fetch-site", "cross-site")))
    assert got["status"] == 200 and len(json.loads(got["body"])["projects"]) == 2
    assert not any(name.startswith("access-control") for name in got["headers"])


def test_this_computer_only_comes_before_the_write_rule(reg, wrote, opened) -> None:
    for kwargs in ({"host": "evil.example:8765", "headers": (*JSON, ("origin", "http://evil.example:8765"))},
                   {"client": ("10.0.0.5", 50000), "headers": (*JSON, *OWN)},
                   {"headers": (*JSON, *OWN, ("x-forwarded-for", "127.0.0.1"))}):
        got = _send(reg["app"], "POST", LIST, chunks=None, **kwargs)
        assert (got["status"], _detail(got)) == (403, REGISTRY_REFUSED), kwargs
    assert wrote == [] and opened == []


def test_no_method_of_the_registrys_routes_answers_with_a_permission_for_another_site(reg) -> None:
    for method, path, allow in (("OPTIONS", LIST, {"GET", "POST"}), ("HEAD", LIST, {"GET", "POST"}), ("PUT", LIST, {"GET", "POST"}),
                                ("OPTIONS", ONE, {"PATCH", "DELETE"}), ("GET", ONE, {"PATCH", "DELETE"}), ("POST", ONE, {"PATCH", "DELETE"})):
        got = _send(reg["app"], method, path, headers=(("access-control-request-method", "POST"),))
        assert got["status"] == 405, (method, path)
        # Every method the path has: a route for each would name only the first.
        assert set(got["headers"]["allow"].split(", ")) == allow, (method, path)
        assert not any(name.startswith("access-control") for name in got["headers"])


# --- the body ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("types", [
    (), (("content-type", "text/plain"),), (("content-type", "application/json; charset=utf-8"),),
    (("content-type", "application/json, text/plain"),), (("content-type", "application/json"), ("content-type", "application/json")),
    (("content-type", "application/json"), ("content-type", "text/plain")), (("content-type", "Application/JSON"),),
    (("content-type", "application/x-www-form-urlencoded"),), (("content-type", "text/plain;charset=UTF-8"),),
], ids=["none", "text", "charset", "list", "twice", "two", "case", "form", "no-cors"])
@pytest.mark.parametrize(("method", "path"), [("POST", LIST), ("PATCH", ONE)])
def test_a_body_that_is_not_typed_as_json_is_refused_unread(reg, wrote, opened, method, path, types) -> None:
    before = _rows(reg["db"])
    opened.clear()
    got = _send(reg["app"], method, path, headers=(*types, *OWN), chunks=None)
    assert (got["status"], _detail(got)) == (415, BODY_TYPE)
    assert {k: got["headers"].get(k) for k in FRAMES} == FRAMES
    assert wrote == [] and opened == []
    assert _rows(reg["db"]) == before


def _padded(n: int, **fields) -> bytes:
    base = json.dumps(fields).encode()
    return base[:-1] + b" " * (n - len(base)) + b"}"


def test_the_body_is_counted_as_it_arrives_and_not_read_past_the_limit(reg, wrote) -> None:
    app, new = reg["app"], str(reg["new"])
    assert BODY_MAX == 65536  # the number README gives (64 KiB); the sizes below are of it, not of whatever the constant says
    # Exactly the limit is taken, in one piece and in many.
    whole = _padded(BODY_MAX, slug="gamma", workspace_path=new)
    assert len(whole) == BODY_MAX
    assert _send(app, "POST", LIST, headers=JSON, chunks=(whole,))["status"] == 201
    pieces = _padded(BODY_MAX, slug="delta", workspace_path=new)
    assert _send(app, "POST", LIST, headers=JSON, chunks=tuple(pieces[i:i + 1024] for i in range(0, BODY_MAX, 1024)))["status"] == 201
    before, calls = _rows(reg["db"]), list(wrote)
    # One byte more is not, however it arrives; a body that never ends is left after the chunk that passed the limit.
    over = _padded(BODY_MAX + 1, slug="epsilon", workspace_path=new)
    got = _send(app, "POST", LIST, headers=JSON, chunks=(over,))
    assert (got["status"], _detail(got)) == (413, "the body is larger than 65536 bytes")
    endless = (b"x" * 1024,) * 200
    for method, path in (("POST", LIST), ("PATCH", ONE)):
        got = _send(app, method, path, headers=JSON, chunks=endless)
        assert (got["status"], got["receives"]) == (413, 65), (method, got)
    assert _send(app, "POST", LIST, headers=(*JSON, ("content-length", str(len(over)))), chunks=(over,))["status"] == 413  # a declared length too
    assert wrote == calls and _rows(reg["db"]) == before


NOT_AN_OBJECT = [
    pytest.param(b"", id="empty"),
    pytest.param(b"\xff\xfe{}", id="not-utf8"),
    pytest.param(b'{"name": "\xff"}', id="a-value-not-utf8"),
    pytest.param(b"\xef\xbb\xbf{}", id="byte-order-mark"),
    pytest.param(b"[]", id="list"),
    pytest.param(b'"x"', id="string"),
    pytest.param(b"null", id="null"),
    pytest.param(b"1", id="number"),
    pytest.param(b"{", id="cut"),
    pytest.param(b"[" * 60000, id="nested-lists"),
    pytest.param(b'{"a":' * 12000, id="nested-objects"),
    pytest.param(b'{"name": ' + b"9" * 5000 + b"}", id="a-number-too-long-to-read"),
    pytest.param(b'{"name": "a", "name": "b"}', id="a-key-twice"),
    pytest.param(b'{"slug": "x", "slug": "x", "workspace_path": "/"}', id="a-key-twice-same-value"),
]


@pytest.mark.parametrize("body", NOT_AN_OBJECT)
@pytest.mark.parametrize(("method", "path"), [("POST", LIST), ("PATCH", ONE)])
def test_a_body_that_is_not_one_json_object_is_a_400(reg, wrote, method, path, body) -> None:
    before = _rows(reg["db"])
    got = _send(reg["app"], method, path, headers=JSON, chunks=(body,))
    assert (got["status"], _detail(got)) == (400, BODY_NOT_OBJECT)
    assert wrote == [] and _rows(reg["db"]) == before


def test_an_object_of_other_keys_is_a_400_that_quotes_nothing_of_it(reg, wrote) -> None:
    app, new, before = reg["app"], str(reg["new"]), _rows(reg["db"])
    lone = ('{"' + BACKSLASH + 'ud800": 1, "slug": "x", "workspace_path": "/"}').encode()
    for body in (b"{}", b'{"slug": "gamma"}', b'{"workspace_path": "/"}', lone,
                 json.dumps({"slug": "gamma", "workspace_path": new, "note": "SECRET"}).encode(),
                 json.dumps({"slug": "gamma", "workspace_path": new, "id": 7}).encode()):
        got = _send(app, "POST", LIST, headers=JSON, chunks=(body,))
        assert (got["status"], _detail(got)) == (400, ADD_SHAPE), body
    for body in (b"{}", b'{"name": "x", "slug": "y"}', b'{"slug": "y"}', lone, ('{"name": "x", "' + BACKSLASH + 'udfff": 1}').encode()):
        got = _send(app, "PATCH", ONE, headers=JSON, chunks=(body,))
        assert (got["status"], _detail(got)) == (400, RENAME_SHAPE), body
    assert wrote == [] and _rows(reg["db"]) == before


def test_a_field_the_registrys_rules_refuse_is_a_400_in_their_words(reg, wrote) -> None:
    app, new, before = reg["app"], str(reg["new"]), _rows(reg["db"])
    surrogate = json.loads('"a' + BACKSLASH + 'ud800"')

    def add(**fields) -> tuple[int, str]:
        got = _send(app, "POST", LIST, headers=JSON, chunks=(json.dumps(fields).encode(),))
        return got["status"], _detail(got)

    # Refused here, before the registry is called: the path, and a name that was given.
    assert add(slug="gamma", workspace_path="relative/dir") == (400, PATH_NOT_ABSOLUTE)
    assert add(slug="gamma", workspace_path=".") == (400, PATH_NOT_ABSOLUTE)
    assert add(slug="gamma", workspace_path=["/"]) == (400, "a workspace path is a non-empty text")
    assert add(slug="gamma", workspace_path="") == (400, "a workspace path is a non-empty text")
    assert add(slug="gamma", workspace_path=None) == (400, "a workspace path is a non-empty text")
    assert add(slug="gamma", workspace_path=new + "\x00") == (400, "a workspace path holds no NUL character")
    assert add(slug="gamma", workspace_path=new, name=None) == (400, "a project name is text")
    assert add(slug="gamma", workspace_path=new, name=7) == (400, "a project name is text")
    assert add(slug="gamma", workspace_path=new, name=float("nan")) == (400, "a project name is text")
    assert add(slug="gamma", workspace_path=new, name="") == (400, "a project name is 1-80 characters")
    assert add(slug="gamma", workspace_path=new, name=" x") == (400, "a project name has no leading or trailing space")
    assert add(slug="gamma", workspace_path=new, name="a" + chr(0x202E) + "b")[1].endswith("(U+202E)")
    status, detail = add(slug="gamma", workspace_path=new, name=surrogate)
    assert status == 400 and "U+D800" in detail
    assert wrote == []
    # Refused by the registry's call: the slug, and a directory that is not one.
    assert add(slug=1, workspace_path=new)[0] == 400
    assert add(slug="Gamma", workspace_path=new)[0] == 400
    assert add(slug=None, workspace_path=new)[0] == 400
    assert add(slug="gamma", workspace_path=str(reg["alpha"] / "keep.txt")) == (400, f"workspace is not a directory: {reg['alpha'] / 'keep.txt'}")
    assert add(slug="gamma", workspace_path=new + "/missing")[1].startswith(f"workspace directory cannot be read: {new}/missing: ")
    assert wrote == ["add_project"] * 5
    for name, said in ((None, "a project name is text"), (["x"], "a project name is text"), ("", "a project name is 1-80 characters"),
                       ("x" * 81, "a project name is 1-80 characters")):
        got = _send(app, "PATCH", ONE, headers=JSON, chunks=(json.dumps({"name": name}).encode(),))
        assert (got["status"], _detail(got)) == (400, said), name
    assert wrote == ["add_project"] * 5 + ["rename_project"] * 4
    assert _rows(reg["db"]) == before


def test_a_name_given_as_null_registers_nothing_where_no_name_takes_the_slug(reg) -> None:
    new = str(reg["new"])
    r = reg["client"].post(LIST, json={"slug": "gamma", "workspace_path": new, "name": None})
    assert (r.status_code, r.json()["detail"]) == (400, "a project name is text")
    assert [p["slug"] for p in _rows(reg["db"])] == ["alpha", "beta"]
    r = reg["client"].post(LIST, json={"slug": "gamma", "workspace_path": new})
    assert (r.status_code, r.json()["project"]["name"]) == (201, "gamma")


def test_a_directory_the_filesystem_encoding_cannot_hold_is_refused_not_a_failure(reg, monkeypatch, capsys) -> None:
    real, new, before = os.stat, str(reg["new"]), _rows(reg["db"])

    def stat(path, *args, **kwargs):
        # Only this directory: `os` is one module, and the test runner stats its own files through it.
        if path == new:
            raise UnicodeEncodeError("ascii", "x", 0, 1, "ordinal not in range(128)")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    r = reg["client"].post(LIST, json={"slug": "gamma", "workspace_path": new})
    assert r.status_code == 400
    assert r.json()["detail"] == f"workspace directory cannot be read: {new}: 'ascii' codec can't encode character '\\x78' in position 0: ordinal not in range(128)"
    assert cli_main(["project", "add", "gamma", "--workspace", new, "--db", str(reg["db"])]) == 1
    out = capsys.readouterr()
    assert out.out == "" and out.err.count("\n") == 1 and out.err.startswith(f"s1: workspace directory cannot be read: {new}: ")
    assert _rows(reg["db"]) == before


@pytest.mark.parametrize(("method", "path"), [("POST", LIST), ("PATCH", ONE)])
def test_a_client_that_goes_away_mid_body_changes_nothing_and_fails_nothing(reg, wrote, method, path) -> None:
    before = _rows(reg["db"])
    got = _send(reg["app"], method, path, headers=JSON, chunks=(b'{"name": "x"',), then="disconnect")
    # An answer no one receives; the point is that the handler ended without raising.
    assert got["status"] == 400 and got["body"] == b"" and got["starts"] == 1
    assert wrote == [] and _rows(reg["db"]) == before


def test_a_delete_reads_no_body_and_asks_for_no_type(reg) -> None:
    assert _send(reg["app"], "DELETE", ONE, headers=(("content-type", "text/plain"), ("content-length", "5")), chunks=None)["status"] == 200
    assert [p["slug"] for p in _rows(reg["db"])] == ["beta"]


# --- statuses ----------------------------------------------------------------------------------------


def test_a_taken_slug_is_a_409(reg, wrote) -> None:
    before = _rows(reg["db"])
    r = reg["client"].post(LIST, json={"slug": "alpha", "workspace_path": str(reg["new"])})
    assert (r.status_code, r.json()["detail"]) == (409, f"project 'alpha' already exists in {reg['db']}")
    assert {k: r.headers.get(k) for k in FRAMES} == FRAMES
    assert wrote == ["add_project"] and _rows(reg["db"]) == before


@pytest.mark.parametrize("key", ["0", "01", "1x", "-1", "1.0", "%201", "12345678901234567890"])
@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
def test_a_key_that_is_no_number_is_a_404_with_nothing_read_or_opened(reg, wrote, method, key) -> None:
    reg["db"].unlink()  # an opened registry would answer 400
    got = _send(reg["app"], method, "/api/registry/projects/" + key.replace("%20", " "), headers=JSON, chunks=None)
    assert (got["status"], _detail(got)) == (404, "Not Found")
    assert wrote == []


def test_a_registry_that_cannot_be_used_or_is_busy_says_so(reg, wrote, monkeypatch, tmp_path) -> None:
    client, new = reg["client"], str(reg["new"])
    monkeypatch.setattr(registry, "BUSY_TIMEOUT_S", 0.05)
    before = _rows(reg["db"])
    with contextlib.closing(sqlite3.connect(reg["db"], isolation_level=None)) as holder:
        holder.execute("BEGIN IMMEDIATE")
        for r in (client.post(LIST, json={"slug": "gamma", "workspace_path": new}), client.patch(ONE, json={"name": "x"}), client.delete(ONE)):
            assert r.status_code == 503 and r.json()["detail"].startswith(f"project registry is busy: {reg['db']}")
        holder.execute("ROLLBACK")
    assert _rows(reg["db"]) == before
    foreign = tmp_path / "foreign.db"
    with contextlib.closing(sqlite3.connect(foreign)) as conn:
        conn.execute("CREATE TABLE t(x)")
    other = TestClient(create_projects_app(foreign), **LOOPBACK)
    for r in (other.post(LIST, json={"slug": "gamma", "workspace_path": new}), other.patch(ONE, json={"name": "x"}), other.delete(ONE)):
        assert r.status_code == 400 and r.json()["detail"].startswith("not a project registry: ")
    assert wrote == ["add_project", "rename_project", "remove_project"] * 2


@pytest.mark.parametrize(("method", "path", "body", "name"), [
    ("POST", LIST, {"slug": "gamma", "workspace_path": "NEW"}, "add_project"), ("PATCH", ONE, {"name": "Renamed"}, "rename_project"),
    ("DELETE", ONE, None, "remove_project"),
], ids=["add", "rename", "remove"])
def test_a_write_inside_the_registry_holds_no_other_request_up(reg, monkeypatch, method, path, body, name) -> None:
    """A registry call can wait seconds for the file's lock. Here the call is held at a barrier of the
    test's own, so nothing depends on SQLite's timing: the list is answered while the write is held
    there, and only then is the write let through. A handler that called the registry on the event loop
    would sit at the barrier with the loop, and the list could not be answered until it gave up."""
    entered, release, left, expired = (threading.Event() for _ in range(4))
    real = getattr(registry, name)

    def held(*args, **kwargs):
        entered.set()
        if not release.wait(10):  # bounded: a write on the event loop ends the test as a failure, not as a hang
            expired.set()
        try:
            return real(*args, **kwargs)
        finally:
            left.set()

    monkeypatch.setattr(registry, name, held)
    app, before = reg["app"], _rows(reg["db"])
    chunks = () if body is None else (json.dumps(body).replace("NEW", str(reg["new"])).encode(),)

    async def both() -> tuple[dict, bool, dict]:
        write = asyncio.create_task(_call(app, method, path, headers=JSON, chunks=chunks))
        try:
            while not entered.is_set() and not write.done():
                await asyncio.sleep(0.005)
            listed = await _call(app, "GET", LIST)
            held_meanwhile = entered.is_set() and not release.is_set() and not left.is_set() and not write.done()
        finally:
            release.set()
        return listed, held_meanwhile, await write

    listed, held_meanwhile, written = asyncio.run(both())
    # The barrier was opened by the test. Had it given up by itself (the write on the event loop, or a
    # list that took ten seconds), "held meanwhile" below would be about a call that was let through.
    assert not expired.is_set()
    assert listed["status"] == 200 and len(json.loads(listed["body"])["projects"]) == 2
    assert held_meanwhile  # the write was inside its registry call, and had not left it, when the list was answered
    assert written["status"] == (201 if method == "POST" else 200) and _rows(reg["db"]) != before


def test_a_workspaces_own_workbench_has_no_registry_route(tmp_path) -> None:
    client = TestClient(create_app(workspace_dir=tmp_path))
    for method, url in (("GET", LIST), ("POST", LIST), ("PATCH", ONE), ("DELETE", ONE)):
        r = client.request(method, url, json={"slug": "x", "workspace_path": str(tmp_path)} if method == "POST" else None)
        assert r.status_code in (404, 405), (method, url)


# --- the page: sending a change ----------------------------------------------------------------------

PAGE = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
HOSTILE = "\"'><img src=x onerror=alert(1)>"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")
ALPHA = {"id": "3", "slug": "alpha", "name": "Alpha tickets", "workspace_path": "/w/alpha"}
LISTED = {"registry": {"path": "/r/projects.db"}, "projects": [
    {"id": "2", "slug": "beta", "name": "Beta", "workspace_path": "/w/beta"}, ALPHA]}


def _functions(exclude: tuple[str, ...] = ()) -> str:
    """The page's functions, through one name a test can replace with a wrong page's."""
    return _all_functions(exclude=exclude)


def _js(body: str) -> object:
    return _node(f"{_functions()}\n{body}")


@needs_node
def test_a_change_is_one_request_of_what_was_typed() -> None:
    out = _js("""
process.stdout.write(JSON.stringify([
  projectWriteRequest("project-add", {slug: " a ", path: "rel/dir ", name: " N "}),
  projectWriteRequest("project-add", {slug: "", path: "", name: ""}),
  projectWriteRequest("project-rename", {id: "3", name: ""}),
  projectWriteRequest("project-remove", {id: "9223372036854775807", name: "ignored"}),
  ["03", "3/..", "", undefined, 3, "1e3", "12345678901234567890"].map(id => projectWriteRequest("project-remove", {id})),
  projectWriteRequest("project-open", {id: "3"}),
]));""")
    assert out == [
        # As typed: the server says what is wrong with a space or a relative path.
        {"method": "POST", "url": "/api/registry/projects", "body": {"slug": " a ", "workspace_path": "rel/dir ", "name": " N "}},
        # An empty name box is no name, not an empty one.
        {"method": "POST", "url": "/api/registry/projects", "body": {"slug": "", "workspace_path": ""}},
        {"method": "PATCH", "url": "/api/registry/projects/3", "body": {"name": ""}},
        {"method": "DELETE", "url": "/api/registry/projects/9223372036854775807"},
        [None] * 7,  # a number that is not a key sends nothing
        None,
    ]


@needs_node
def test_the_outcome_is_worded_from_what_the_server_answered() -> None:
    hostile = {"id": "7", "slug": "s", "name": HOSTILE, "workspace_path": "/w/" + HOSTILE}
    out = _js(f"""
const p = {json.dumps(hostile)};
const e = (status, detail) => Object.assign(new Error(detail || "x"), status === null ? {{}} : {{status}}, detail === undefined ? {{}} : {{detail}});
process.stdout.write(JSON.stringify({{
  texts: ["project-add", "project-rename", "project-remove"].map(act => projectResultText(act, p)),
  outcomes: [e(409, "taken"), e(400, "bad"), e(404, "none"), e(413, "big"), e(403, "site"), e(503, "busy"),
             e(503), e(409), e(500, "boom"), e(502, "gateway"), e(null), e(null, "said"), e(200, "odd"), null, undefined].map(projectOutcome),
  refused: projectFailureText(e(409, {json.dumps(HOSTILE)})),
  unconfirmed: [e(null), e(500, "boom"), e(503)].map(projectFailureText),
  projects: [p, {{...p, id: 7}}, {{...p, id: "07"}}, {{...p, name: null}}, {{id: "7", slug: "s", name: "n"}}, {{}}, null, "x", [p]].map(isProject),
}}));""")
    who = f"project #7: {HOSTILE} (s)"
    assert out["texts"] == [f"Added {who}.", f"Renamed {who}.", f"Removed {who} from the registry. The workspace directory stays: /w/{HOSTILE}"]
    # Refused: a 4xx or a 503 that carries the server's own sentence. Everything else may have been done.
    assert out["outcomes"] == ["refused"] * 6 + ["unconfirmed"] * 9
    assert out["refused"] == "Nothing was changed: " + HOSTILE
    assert out["unconfirmed"] == ["No usable answer from the server: the registry may or may not have been changed. Refresh to see it before trying again."] * 3
    assert out["projects"] == [True] + [False] * 8


@needs_node
def test_the_sender_types_its_body_and_tells_an_answer_it_cannot_use_from_a_refusal() -> None:
    out = _js(f"""
const asked = [];
let answer = null, over = null;
const fetch = async (url, init) => {{ if (over) return over(init); asked.push([url, init]); return answer; }};
const sendJsonOver = async (fn, ...args) => {{ over = fn; try {{ return await sendJson(...args); }} finally {{ over = null; }} }};
let location = {{pathname: "/p/"}};
const res = (status, body) => ({{ok: status >= 200 && status < 300, status, statusText: "Status " + status,
  json: async () => {{ if (body === undefined) throw new SyntaxError("not JSON"); return body; }}}});
const tried = async (...args) => {{ try {{ return ["ok", await sendJson(...args)]; }}
  catch (err) {{ return ["error", err.message, err.status === undefined ? null : err.status, err.detail === undefined ? null : err.detail, projectOutcome(err)]; }} }};
(async () => {{
  const out = {{}};
  const good = {json.dumps(ALPHA)};
  answer = res(201, {{project: good}});
  out.added = await tried("POST", "/api/registry/projects", {{slug: "alpha"}});
  location = {{pathname: "/p/3/"}};  // a registry address is the registry's from any page
  out.removed = await tried("DELETE", "/api/registry/projects/3");
  out.asked = [...asked];
  location = {{pathname: "/p/"}};
  answer = res(409, {{detail: "taken"}}); out.refused = await tried("POST", "/api/registry/projects", {{}});
  answer = res(503, undefined); out.html503 = await tried("PATCH", "/api/registry/projects/3", {{name: "x"}});
  answer = res(503, {{}}); out.empty503 = await tried("PATCH", "/api/registry/projects/3", {{name: "x"}});
  answer = res(400, {{detail: [{{loc: ["body"]}}]}}); out.listDetail = await tried("PATCH", "/api/registry/projects/3", {{name: "x"}});
  answer = res(500, {{detail: "boom"}}); out.failed = await tried("DELETE", "/api/registry/projects/3");
  answer = res(201, {{project: {{}}}}); out.emptyProject = await tried("POST", "/api/registry/projects", {{}});
  answer = res(201, {{}}); out.noProject = await tried("POST", "/api/registry/projects", {{}});
  answer = res(201, undefined); out.notJson = await tried("POST", "/api/registry/projects", {{}});
  answer = res(200, {{project: {{...good, id: 3}}}}); out.numberId = await tried("DELETE", "/api/registry/projects/3");
  answer = null;  // what a browser does with `redirect: "error"` when the server redirects
  const refuse = init => init.redirect === "error" ? Promise.reject(new TypeError("Failed to fetch")) : Promise.resolve(res(404, {{detail: "said by another address"}}));
  out.redirected = await (async () => {{ try {{ return ["ok", await sendJsonOver(refuse, "PATCH", "/api/registry/projects/3", {{name: "x"}})]; }}
    catch (err) {{ return ["error", err.message, err.status === undefined ? null : err.status, err.detail === undefined ? null : err.detail, projectOutcome(err)]; }} }})();
  process.stdout.write(JSON.stringify(out));
}})();""")
    assert out["added"] == ["ok", ALPHA] and out["removed"] == ["ok", ALPHA]
    # No redirect is followed: another address's answer would say nothing about this request.
    assert out["asked"] == [
        ["/api/registry/projects", {"method": "POST", "redirect": "error", "headers": {"Content-Type": "application/json"}, "body": '{"slug":"alpha"}'}],
        ["/api/registry/projects/3", {"method": "DELETE", "redirect": "error"}],  # no body, so no type
    ]
    assert out["redirected"] == ["error", "Failed to fetch", None, None, "unconfirmed"]
    assert out["refused"] == ["error", "taken", 409, "taken", "refused"]
    # A status alone is not the server's sentence: the message is the status line, and nothing is claimed.
    assert out["html503"] == ["error", "503 Status 503", 503, None, "unconfirmed"]
    assert out["empty503"] == ["error", "503 Status 503", 503, None, "unconfirmed"]
    assert out["listDetail"][2:] == [400, None, "unconfirmed"]
    assert out["failed"] == ["error", "boom", 500, "boom", "unconfirmed"]
    # A 2xx that is not the project is no success, and has no status to be read as a refusal.
    for case in ("emptyProject", "noProject", "notJson", "numberId"):
        assert out[case] == ["error", "the answer does not hold the project", None, None, "unconfirmed"], case


def _write_run(steps: str, payload: dict = LISTED) -> dict:
    """The registry's page in node: the list loaded and kept, a stand-in DOM that knows what holds the
    focus and where each control is, and `sendJson` held until a step answers it."""
    js = f"""
let STATE = null;
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [], groupsState = "loading", groupsError = "";
const els = {{}};
const out = {{listWrites: [], events: []}};
const body = {{id: "body"}};
const row = {{id: "row", inside: "projects-container"}};          // a row's button
const addButton = {{id: "add-button", inside: "form-project-add"}};
const submits = [0, 1, 2].map(i => ({{id: "submit" + i, attrs: {{}}, setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }}}}));
const el = id => (els[id] ??= {{id, html: "", textContent: "", value: "", open: false, dataset: {{}}, attrs: {{}}, opener: null,
  set innerHTML(v) {{
    this.html = v;
    if (id !== "projects-container") return;
    out.listWrites.push(v.includes("skeleton") ? "loading" : (v.match(/data-act="project-open"/g) || []).length + " rows");
    if (document.activeElement && document.activeElement.inside === id) document.activeElement = body;  // what held the focus is gone
  }},
  get innerHTML() {{ return this.html; }},
  classList: {{toggle() {{}}}}, querySelector: () => null,
  querySelectorAll: sel => sel === ".project-submit" ? submits : [],
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }},
  // As measured in Chrome: showModal moves the focus into the dialog, close gives it back to what had it.
  showModal() {{ this.open = true; this.opener = document.activeElement; document.activeElement = {{id: id + "-control", inside: id}}; out.events.push("showModal " + id); }},
  close() {{ this.open = false; document.activeElement = this.opener; out.events.push("close " + id); }},
  contains(x) {{ return Boolean(x) && x.inside === id; }},
  focus() {{ document.activeElement = this; out.events.push("focus " + id); }}}});
const document = {{getElementById: el, querySelectorAll: () => [], activeElement: body, body}};
const pending = [], sent = [];
let fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
let sendJson = (method, url, body) => new Promise((resolve, reject) => sent.push({{method, url, body, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const payload = {json.dumps(payload)};
const result = () => [el("projects-result").dataset.kind || "", el("projects-result").textContent];
const at = () => document.activeElement.id;
const writing = () => [el("tab-projects").dataset.writing || "", submits.map(b => b.attrs["aria-disabled"] || "")];
const refusal = (status, detail) => Object.assign(new Error(detail), {{status, detail}});
const boxes = ["inp-project-slug", "inp-project-path", "inp-project-name"];
const type = (...values) => boxes.forEach((id, i) => {{ el(id).value = values[i]; }});
const typed = () => boxes.map(id => el(id).value);
{_functions(("fetchJson", "sendJson"))}
STATE = stateFromLocation("", "/p/");
(async () => {{
  render();
  pending.shift().resolve(payload);
  await tick(); await tick();
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


@needs_node
def test_a_rows_button_opens_its_dialog_with_the_listed_projects_words() -> None:
    hostile = {"registry": {"path": "/r/p.db"}, "projects": [{"id": "3", "slug": "alpha", "name": HOSTILE, "workspace_path": "/w/" + HOSTILE}]}
    out = _write_run("""
  out.atStart = [...out.listWrites];
  document.activeElement = row;
  openProjectDialog("rename", "3");
  const rename = el("dlg-project-rename");
  out.rename = [rename.open, rename.dataset.id, el("dlg-project-rename-what").textContent, el("inp-project-rename").value, at()];
  rename.close();
  openProjectDialog("remove", "3");
  const remove = el("dlg-project-remove");
  out.remove = [remove.open, remove.dataset.id, el("dlg-project-remove-what").textContent];
  out.markup = [el("dlg-project-rename-what").html, el("dlg-project-remove-what").html];
  remove.close();
  const before = out.events.length;
  for (const id of ["9", "03", "", undefined]) { openProjectDialog("rename", id); openProjectDialog("remove", id); }
  out.unlisted = out.events.length - before;
  out.sent = sent.length;
""", hostile)
    assert out["atStart"] == ["loading", "1 rows"]
    who = f"Project #3: {HOSTILE} (alpha)"
    assert out["rename"] == [True, "3", who, HOSTILE, "dlg-project-rename-control"]
    assert out["remove"] == [True, "3", f"{who}. Only the registration is removed; the workspace directory stays: /w/{HOSTILE}"]
    assert out["markup"] == ["", ""]  # written as text, never as markup
    assert out["unlisted"] == 0 and out["sent"] == 0  # a number the list does not hold opens nothing


@needs_node
def test_a_rename_is_sent_once_and_its_success_is_said_then_the_list_read_again() -> None:
    out = _write_run("""
  document.activeElement = row;
  openProjectDialog("rename", "3");
  el("inp-project-rename").value = "Alpha, renamed";
  const first = submitProject("project-rename");
  out.sent = sent.map(s => [s.method, s.url, s.body]);
  out.whileOut = writing();
  out.atOnce = [el("dlg-project-rename").open, at(), result(), [...out.events]];
  // A second Enter, the add form's button and another row's button, while the answer is out.
  submitProject("project-rename"); submitProject("project-add");
  openProjectDialog("remove", "2");
  out.second = [sent.length, el("dlg-project-remove").open, result()];
  out.listBefore = [...out.listWrites];
  sent[0].resolve({id: "3", slug: "alpha", name: "Alpha, renamed", workspace_path: "/w/alpha"});
  await first;
  out.after = {result: result(), open: el("dlg-project-rename").open, at: at(), writing: writing(),
               asked: pending.map(p => p.url), listWrites: [...out.listWrites], kept: "projects" in viewData, events: out.events};
  pending.shift().resolve({...payload, projects: payload.projects.slice(0, 1)});
  await tick(); await tick();
  out.reloaded = [...out.listWrites];
""")
    assert out["sent"] == [["PATCH", "/api/registry/projects/3", {"name": "Alpha, renamed"}]]
    assert out["whileOut"] == ["1", ["true", "true", "true"]]
    # Sent, it cannot be taken back: the dialog, and its Cancel, are gone before the answer, the focus is
    # back on the row's button, and the wait is said where the answer will be.
    assert out["atOnce"] == [False, "row", ["waiting", "Sent: waiting for the registry's answer."], ["showModal dlg-project-rename", "close dlg-project-rename"]]
    assert out["second"] == [1, False, ["waiting", "Still waiting for the registry's answer to the last change."]]
    assert out["listBefore"] == ["loading", "2 rows"]
    after = out["after"]
    assert after["result"] == ["done", "Renamed project #3: Alpha, renamed (alpha)."]  # the waiting sentence is gone
    assert after["open"] is False and after["writing"] == ["", ["", "", ""]]
    # The row's button, which had the focus back, is about to be replaced by the reload: the sentence takes it.
    assert after["events"] == ["showModal dlg-project-rename", "close dlg-project-rename", "focus projects-result"]
    assert after["at"] == "projects-result"
    # No row was changed by the page: the list is asked for again and shows its loading card until then.
    assert after["asked"] == ["/api/registry/projects"] and after["kept"] is False
    assert after["listWrites"] == ["loading", "2 rows", "loading"]
    assert out["reloaded"] == ["loading", "2 rows", "loading", "1 rows"]


@needs_node
def test_a_refusal_is_said_and_nothing_is_read_again() -> None:
    out = _write_run("""
  document.activeElement = row;
  openProjectDialog("remove", "3");
  const first = submitProject("project-remove");
  out.sent = sent.map(s => [s.method, s.url, s.body === undefined]);
  sent[0].reject(refusal(404, "no project #3 in /r/projects.db"));
  await first;
  out.refused = {result: result(), open: el("dlg-project-remove").open, at: at(), writing: writing(), asked: pending.length, listWrites: [...out.listWrites], events: [...out.events]};
  // And an answer that is no answer.
  openProjectDialog("remove", "3");
  const second = submitProject("project-remove");
  sent[1].reject(new Error("Failed to fetch"));
  await second;
  out.unconfirmed = {result: result(), open: el("dlg-project-remove").open, at: at(), asked: pending.length, listWrites: [...out.listWrites]};
""")
    assert out["sent"] == [["DELETE", "/api/registry/projects/3", True]]
    refused = out["refused"]
    assert refused["result"] == ["refused", "Nothing was changed: no project #3 in /r/projects.db"]
    assert refused["open"] is False and refused["writing"] == ["", ["", "", ""]]
    # The row's button is still there, so the focus the browser gave back to it stays.
    assert refused["at"] == "row" and refused["events"] == ["showModal dlg-project-remove", "close dlg-project-remove"]
    assert refused["asked"] == 0 and refused["listWrites"] == ["loading", "2 rows"]
    unconfirmed = out["unconfirmed"]
    assert unconfirmed["result"][0] == "unconfirmed" and unconfirmed["result"][1].startswith("No usable answer from the server: the registry may or may not")
    assert unconfirmed["open"] is False and unconfirmed["at"] == "row" and unconfirmed["asked"] == 0 and unconfirmed["listWrites"] == ["loading", "2 rows"]


@needs_node
def test_the_focus_is_moved_only_when_what_holds_it_is_gone_or_about_to_go() -> None:
    out = _write_run("""
  const cases = {onTheRow: row, inTheForm: addButton, nowhere: body, nothing: null};
  for (const answer of ["done", "refused"]) {
    for (const [name, where] of Object.entries(cases)) {
      document.activeElement = row;
      openProjectDialog("rename", "3");
      const going = submitProject("project-rename");
      document.activeElement = where;             // where the viewer, or the browser, left the focus during the wait
      out.events.length = 0;
      const s = sent.pop();
      if (answer === "done") s.resolve({id: "3", slug: "alpha", name: "N", workspace_path: "/w/alpha"});
      else s.reject(refusal(409, "said"));
      await going;
      out[answer + " " + name] = [result()[0], at(), [...out.events]];
      if (pending.length) { pending.shift().resolve(payload); await tick(); await tick(); }
    }
  }
""")
    moved = ["focus projects-result"]
    # A success replaces the list, so a focus in it goes to the sentence; one in the form stays.
    assert out["done onTheRow"] == ["done", "projects-result", moved]
    assert out["done inTheForm"] == ["done", "add-button", []]
    # A refusal replaces nothing: the row's button keeps the focus.
    assert out["refused onTheRow"] == ["refused", "row", []]
    assert out["refused inTheForm"] == ["refused", "add-button", []]
    # Nothing focused (a browser that gave the focus back to nothing): the sentence takes it either way.
    for answer in ("done", "refused"):
        assert out[answer + " nowhere"] == [answer, "projects-result", moved]
        assert out[answer + " nothing"] == [answer, "projects-result", moved]


@needs_node
def test_an_add_empties_the_boxes_it_sent_and_keeps_what_was_typed_since() -> None:
    out = _write_run("""
  const added = {id: "4", slug: "gamma", name: "gamma", workspace_path: "/w/gamma"};
  const add = async (during, answer) => {
    document.activeElement = addButton;
    out.events.length = 0;
    const going = submitProject("project-add");
    const s = sent.pop();
    during();
    if (answer === "done") s.resolve(added); else s.reject(answer);
    await going;
    const got = {body: s.body, typed: typed(), result: result()[0], at: at(), events: [...out.events], asked: pending.length};
    while (pending.length) { pending.shift().resolve(payload); await tick(); await tick(); }
    return got;
  };
  type("gamma", "/w/gamma", "");
  out.plain = await add(() => {}, "done");
  type("gamma", "/w/gamma", "Gamma");
  out.named = await add(() => {}, "done");
  for (const i of [0, 1, 2]) {
    type("gamma", "/w/gamma", "Gamma");
    out["changed" + i] = (await add(() => { el(boxes[i]).value += "x"; }, "done")).typed;
  }
  type("gamma", "/w/gamma", "");
  out.nameTyped = (await add(() => { el(boxes[2]).value = "Next"; }, "done")).typed;
  type("alpha", "/w/alpha", "");
  out.refused = await add(() => {}, refusal(409, "project 'alpha' already exists in /r/projects.db"));
  type("gamma", "/w/gamma", "");
  out.intoTheList = await add(() => { document.activeElement = row; }, "done");
  type("gamma", "/w/gamma", "");
  out.intoTheListRefused = await add(() => { document.activeElement = row; }, refusal(409, "said"));
""")
    # The name box left empty is no `name` in the body, and still an unchanged box.
    assert out["plain"] == {"body": {"slug": "gamma", "workspace_path": "/w/gamma"}, "typed": ["", "", ""], "result": "done",
                            "at": "add-button", "events": [], "asked": 1}
    assert out["named"]["body"] == {"slug": "gamma", "workspace_path": "/w/gamma", "name": "Gamma"} and out["named"]["typed"] == ["", "", ""]
    # Whichever box was typed in during the wait, all three are kept.
    assert out["changed0"] == ["gammax", "/w/gamma", "Gamma"]
    assert out["changed1"] == ["gamma", "/w/gammax", "Gamma"]
    assert out["changed2"] == ["gamma", "/w/gamma", "Gammax"]
    assert out["nameTyped"] == ["gamma", "/w/gamma", "Next"]
    assert out["refused"]["typed"] == ["alpha", "/w/alpha", ""] and out["refused"]["asked"] == 0
    assert (out["refused"]["result"], out["refused"]["at"], out["refused"]["events"]) == ("refused", "add-button", [])
    # Tabbed into the list during the wait: a success replaces it, so the sentence takes the focus.
    assert (out["intoTheList"]["at"], out["intoTheList"]["events"]) == ("projects-result", ["focus projects-result"])
    assert (out["intoTheListRefused"]["at"], out["intoTheListRefused"]["events"]) == ("row", [])


@needs_node
def test_a_change_ends_with_its_answer_and_an_older_list_never_overwrites_a_newer() -> None:
    out = _write_run("""
  const added = n => ({id: String(n), slug: "p" + n, name: "p" + n, workspace_path: "/w/p" + n});
  const list = n => ({...payload, projects: [...payload.projects, ...[4, 5].slice(0, n).map(added)]});
  type("p4", "/w/p4", "");
  let going = submitProject("project-add");
  sent.shift().resolve(added(4));
  await going;
  // The first reload is still out, and the next change is sent all the same.
  out.afterFirst = {writing: writing()[0], asked: pending.length, kept: "projects" in viewData};
  type("p5", "/w/p5", "");
  going = submitProject("project-add");
  out.secondSent = sent.length;
  sent.shift().resolve(added(5));
  await going;
  out.asked = pending.length;
  const [older, newer] = pending.splice(0, 2);
  newer.resolve(list(2));
  await tick(); await tick();
  out.afterNewer = [...out.listWrites];
  older.resolve(list(1));
  await tick(); await tick();
  out.afterOlder = [...out.listWrites];
  out.kept = viewData.projects.projects.length;
""")
    assert out["afterFirst"] == {"writing": "", "asked": 1, "kept": False}
    assert out["secondSent"] == 1 and out["asked"] == 2
    assert out["afterNewer"] == ["loading", "2 rows", "loading", "loading", "4 rows"]
    assert out["afterOlder"] == out["afterNewer"] and out["kept"] == 4  # the older answer wrote nothing


@needs_node
def test_a_change_is_no_longer_out_whatever_its_answers_wording_did() -> None:
    out = _write_run("""
  type("gamma", "/w/gamma", "");
  const going = submitProject("project-add");
  sent[0].resolve(null);                         // an answer the wording cannot read
  await going;
  out.state = {writing: writing(), result: result()[0], typed: typed(), asked: pending.length};
  const again = submitProject("project-add");
  out.sentAgain = sent.length;
""")
    assert out["state"] == {"writing": ["", ["", "", ""]], "result": "unconfirmed", "typed": ["gamma", "/w/gamma", ""], "asked": 0}
    assert out["sentAgain"] == 2


@needs_node
@pytest.mark.parametrize(("name", "old", "new"), [
    ("a second change is sent while one is out", 'if (document.getElementById("tab-projects").dataset.writing) return;', ""),
    ("a dialog opens while a change is out", 'setProjectResult("waiting", "Still waiting for the registry\'s answer to the last change.");\n        return;', ""),
    ("the dialog stays open while its change is out", "if (dialog && dialog.open) dialog.close();", ""),
    ("the wait is not said", 'setProjectResult("waiting", "Sent: waiting for the registry\'s answer.");', ""),
    ("a focus on nothing is left there", "if (!at || at === document.body || (done && list.contains(at))) {", "if (done && list.contains(at)) {"),
    ("the boxes are emptied whatever they hold", "boxes.every((b, i) => b.value === typed[i])", "true"),
    ("the boxes are compared with the body", "boxes.every((b, i) => b.value === typed[i])", "boxes[0].value === typed[0] && boxes[1].value === typed[1]"),
    ("the focus is never moved", "(done && list.contains(at))) {\n        document.getElementById(\"projects-result\").focus();", "(done && list.contains(at))) {"),
    ("the focus is moved after a refusal too", "(done && list.contains(at))", "list.contains(at)"),
    ("the focus is always moved", "if (!at || at === document.body || (done && list.contains(at))) {", "if (true) {"),
    ("the list is read again after a refusal", "if (done) reloadProjects();", "reloadProjects();"),
    ("the list is not read again", "if (done) reloadProjects();", ""),
    ("the pending mark is cleared only on success", "} finally {\n        markProjectWriting(false);\n      }", "}\n      if (kind === \"done\") markProjectWriting(false);"),
    ("a refusal is worded as unconfirmed", '? "refused" : "unconfirmed"', '? "unconfirmed" : "unconfirmed"'),
    ("any answered failure is a refusal", "return code !== null && said && ((code >= 400 && code < 500) || code === 503)", "return code !== null"),
])
def test_each_rule_of_the_flow_is_held_by_a_check_above(monkeypatch, name, old, new) -> None:
    """Each of these wrong pages fails one of the flow checks: none of them is a check that cannot fail."""
    assert _all_functions(exclude=()).count(old) == 1, name
    monkeypatch.setitem(globals(), "_functions", lambda exclude=(): _all_functions(exclude=exclude).replace(old, new))
    checks = (
        test_a_rename_is_sent_once_and_its_success_is_said_then_the_list_read_again,
        test_a_refusal_is_said_and_nothing_is_read_again,
        test_the_focus_is_moved_only_when_what_holds_it_is_gone_or_about_to_go,
        test_an_add_empties_the_boxes_it_sent_and_keeps_what_was_typed_since,
        test_a_change_ends_with_its_answer_and_an_older_list_never_overwrites_a_newer,
        test_a_change_is_no_longer_out_whatever_its_answers_wording_did,
        test_the_outcome_is_worded_from_what_the_server_answered,
    )
    failed = []
    for check in checks:
        try:
            check()
        except AssertionError:
            failed.append(check.__name__)
    assert failed, name


def _listener(kind: str) -> str:
    script = PAGE[PAGE.index("<script>"):PAGE.index("</script>")]
    return re.search(rf'document\.addEventListener\("{kind}", ev => \{{.*?\n    \}}\);', script, re.S).group(0)


@needs_node
def test_the_forms_are_sent_by_the_page_and_no_other_form_is_touched() -> None:
    out = _node(f"""
const handlers = {{}};
const document = {{addEventListener: (type, fn) => {{ handlers[type] = fn; }}}};
const calls = [];
const submitProject = act => calls.push(act);
{_listener("submit")}
const prevented = [];
for (const [i, target] of [{{dataset: {{projectForm: "project-add"}}}}, {{dataset: {{projectForm: "project-rename"}}}}, {{dataset: {{projectForm: "project-remove"}}}},
                           {{dataset: {{projectForm: "project-open"}}}}, {{dataset: {{}}}}, {{}}, null].entries()) {{
  handlers.submit({{target, preventDefault: () => prevented.push(i)}});
}}
process.stdout.write(JSON.stringify({{calls, prevented}}));""")
    assert out == {"calls": ["project-add", "project-rename", "project-remove"], "prevented": [0, 1, 2]}


@needs_node
def test_a_rows_buttons_and_cancel_are_branches_of_the_one_click_listener() -> None:
    out = _node(f"""
const handlers = {{}};
const document = {{addEventListener: (type, fn) => {{ handlers[type] = fn; }}}};
const calls = [];
const openProjectDialog = (kind, id) => calls.push([kind, id]);
const projectUrl = () => null, viewNames = () => [];
{_listener("click")}
const press = dataset => handlers.click({{target: {{closest: () => ({{dataset, closest: sel => ({{close: () => calls.push(["close", sel])}})}})}}}});
press({{act: "project-rename-open", id: "3"}});
press({{act: "project-remove-open", id: "9007199254740993"}});
press({{act: "project-dialog-cancel"}});
process.stdout.write(JSON.stringify(calls));""")
    assert out == [["rename", "3"], ["remove", "9007199254740993"], ["close", "dialog"]]


def test_the_forms_and_dialogs_are_the_pages_own_markup() -> None:
    pane = PAGE[PAGE.index('<section id="tab-projects"'):PAGE.index("</main>")]
    # Three forms, each named for the change it sends; the submit listener reads that name.
    assert re.findall(r'data-project-form="([a-z-]+)"', pane) == ["project-add", "project-rename", "project-remove"]
    assert pane.count('<button type="submit" class="tool project-submit">') == 3
    # Each dialog is named by its heading and described by the sentence saying which project it is about.
    for kind in ("rename", "remove"):
        assert (f'<dialog id="dlg-project-{kind}" class="project-dialog" aria-labelledby="dlg-project-{kind}-title" '
                f'aria-describedby="dlg-project-{kind}-what">') in pane
        assert f'id="dlg-project-{kind}-title"' in pane and f'<p id="dlg-project-{kind}-what"></p>' in pane
    # Removing is two steps, and the second opens with the focus on Cancel.
    remove = pane[pane.index('<dialog id="dlg-project-remove"'):]
    assert remove.index('data-act="project-dialog-cancel" autofocus>Cancel<') < remove.index(">Remove the registration<")
    assert "autofocus" not in pane[:pane.index('<dialog id="dlg-project-remove"')]
    # The server's rule is the only rule: no box refuses or cuts what is typed, and none remembers it.
    boxes = re.findall(r"<input\b[^>]*>", pane)
    assert len(boxes) == 4
    for box in boxes:
        assert 'type="text"' in box and 'autocomplete="off"' in box
        assert not re.search(r"\b(maxlength|pattern|required|minlength|value)\b", box), box
    assert pane.count("novalidate") == 3
    # Nothing the page writes about a change is markup, and the page has no `confirm()`.
    script = PAGE[PAGE.index("<script>"):PAGE.index("</script>")]
    assert "confirm(" not in script and "alert(" not in script
    flow = script[script.index("function projectWriteRequest"):script.index("function reloadProjects")]
    assert "innerHTML" not in flow and "${" not in flow


def test_the_result_line_is_always_in_the_page_and_always_takes_the_same_room() -> None:
    # A status is announced only when the element was already there before its text changed, so the
    # line is never taken out (`display: none` would).
    css = PAGE[PAGE.index("<style>"):PAGE.index("</style>")]
    rules = re.findall(r"[^{}]*[.#]projects?-result[^{}]*\{[^}]*\}", css)
    assert len(rules) >= 6 and not any("display" in rule or "visibility" in rule for rule in rules)
    # It is in view wherever the list is scrolled to: a change made from a row far down is answered here.
    line = next(rule for rule in rules if re.search(r"\.project-result \{", rule))
    assert "position: sticky" in line and "bottom: 0" in line
    # One height whatever it says, a longer sentence scrolled inside it: an answer never puts the line
    # over what was in view. Of the rules that name the line by its class or id, no other (an empty
    # line, a kind of sentence) gives it another size; the heights a browser computes in each state
    # are the browser check's (project_writes_check.py, the scroll tag), not this test's.
    assert "height: var(--project-result-height);" in line and "overflow-y: auto;" in line
    assert "min-height" not in line and "max-height" not in line
    # The one exception is a window too short for a row between the bar above and the line: there the
    # line does not stick, and nothing is kept clear of it.
    short = ('    @media (max-height: 320px) {\n      :root[data-scope="registry"] { scroll-padding-bottom: 0; }\n'
             "      .project-result { position: static; }\n    }\n")
    assert css.count(short) == 1 and css.count("@media (max-height") == 1
    unstuck = next(rule for rule in rules if "position: static" in rule)
    for rule in rules:
        if rule is not line and rule is not unstuck:
            assert not re.search(r"\b(height|padding|border|margin|position|font-size)\b", rule.split("{", 1)[1]), rule
            assert ":empty" not in rule
    # A row the focus moves to is kept above the line, by that same length.
    scope = re.search(r':root\[data-scope="registry"\] \{([^}]*)\}', css).group(1)
    assert re.search(r"--project-result-height: \d+px;", scope)
    assert "scroll-padding-bottom: calc(var(--project-result-height) + 8px);" in scope
    # A sentence scrolled inside it has to be reached by the keyboard: the line is a Tab stop, with a
    # name and, before any change, a sentence of its own (never an empty stop).
    assert PAGE.count('<p id="projects-result" class="project-result" role="status" aria-label="Result of the last change" tabindex="0">'
                      "No change sent yet: the registry's answer to one is written here.</p>") == 1
