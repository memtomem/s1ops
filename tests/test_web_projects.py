"""`s1 web --db`: every registered project's workspace under `/p/<id>/`, each request answered by the
workbench app of that project's directory; the list of projects; the rule that all of it is this
computer's only; and the page's functions that tell which of the three pages an address is."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops import registry
from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from s1ops.web import projects as web_projects
from s1ops.web.app import STATIC_DIR, create_app
from s1ops.web.projects import NO_FRAMES, PREFIX_REFUSED, REGISTRY_REFUSED, create_projects_app
from tests.test_web_render import _all_functions, _node
from tests.test_web_sheet import LOOPBACK

PAGE = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
HOSTILE = "\"'><img src=x onerror=alert(1)>"
FRAMES = {"content-security-policy": "frame-ancestors 'none'", "x-frame-options": "DENY"}
BIG = ("9007199254740992", "9007199254740993", "9223372036854775807")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")
not_root = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads through any mode")


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("demo") / "w"
    generate_workspace(root)
    return root


@pytest.fixture
def made(tmp_path: Path, demo: Path) -> dict:
    """A registry of two projects: #1 alpha, a copy of the demo workspace, and #2 beta, an empty directory."""
    alpha, beta, db = tmp_path / "alpha", tmp_path / "beta", tmp_path / "projects.db"
    shutil.copytree(demo, alpha)
    beta.mkdir()
    registry.init_registry(db)
    registry.add_project(db, "alpha", str(alpha), "Alpha tickets")
    registry.add_project(db, "beta", str(beta))
    return {"db": db, "alpha": alpha, "beta": beta, "client": TestClient(create_projects_app(db), **LOOPBACK)}


@pytest.fixture
def spies(monkeypatch) -> list[str]:
    """The registry reads and the workspace apps made, by name, while still happening."""
    calls: list[str] = []

    def spy(owner, name: str) -> None:
        real = getattr(owner, name)
        monkeypatch.setattr(owner, name, lambda *a, **k: (calls.append(name), real(*a, **k))[1])

    spy(registry, "list_projects")
    spy(registry, "get_project")
    spy(web_projects, "create_app")
    return calls


def _get(client: TestClient, url: str, **kwargs):
    return client.get(url, follow_redirects=False, **kwargs)


def _asgi(app, path: str, *, method: str = "GET", raw: bytes | None | bool = True, root_path: str = "") -> tuple[int, dict, bytes]:
    """One request as a hand-made scope: the path is sent as written (a client folds `..` first), and
    `raw` is the scope's `raw_path` (`True`: the path's own bytes; `None`: the scope has none)."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
        "path": path, "query_string": b"", "root_path": root_path, "headers": [(b"host", b"127.0.0.1:8765")],
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8765),
    }
    if raw is not None:
        scope["raw_path"] = path.encode() if raw is True else raw
    got: dict = {"body": b""}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            got["status"] = message["status"]
            got["headers"] = {k.decode().lower(): v.decode() for k, v in message["headers"]}
        else:
            got["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return got["status"], got["headers"], got["body"]


def _raw(db: Path, *statements: str) -> None:
    with contextlib.closing(sqlite3.connect(db, isolation_level=None)) as conn:
        for statement in statements:
            conn.execute(statement)


# --- a project's workspace is the workbench app of its directory -----------------------------------


def _api_routes(workspace: Path) -> list[str]:
    return sorted(r.path for r in create_app(workspace_dir=workspace).routes if r.path.startswith("/api/"))


def test_every_workspace_route_answers_as_the_workspace_app_does(made) -> None:
    routes = _api_routes(made["alpha"])
    assert len(routes) >= 12 and "/api/overview" in routes and "/api/sheet" in routes
    alone = TestClient(create_app(workspace_dir=made["alpha"]), **LOOPBACK)
    for route in routes:
        params = {"n_boot": 20} if route == "/api/audit" else {}
        mine, theirs = _get(made["client"], "/p/1" + route, params=params), _get(alone, route, params=params)
        assert mine.status_code == theirs.status_code == 200, (route, mine.text[:200])
        assert mine.json() == theirs.json(), route


def test_a_query_and_the_inner_apps_own_refusals_pass_through(made) -> None:
    alone = TestClient(create_app(workspace_dir=made["alpha"]))
    for url in ("/api/audit?n_boot=5", "/api/slices?fdr_q=7", "/api/labelaudit?top_k=3"):
        mine, theirs = _get(made["client"], "/p/1" + url), _get(alone, url)
        assert (mine.status_code, mine.json()) == (theirs.status_code, theirs.json()), url
    assert _get(made["client"], "/p/1/api/audit?n_boot=5").status_code == 422
    # A broken workspace file is that file's 400, as in the workspace app.
    (made["alpha"] / "fits.json").write_text("{", encoding="utf-8")
    r = _get(made["client"], "/p/1/api/overview")
    assert r.status_code == 400 and r.json()["detail"].startswith(str(made["alpha"] / "fits.json"))
    # The inner app's own slash redirect stays inside the project.
    r = _get(made["client"], "/p/1/api/overview/")
    assert (r.status_code, r.headers["location"]) == (307, "http://127.0.0.1:8765/p/1/api/overview")


def test_each_project_reads_its_own_directory(made) -> None:
    assert _get(made["client"], "/p/1/api/overview").json()["n_records"] > 0
    assert _get(made["client"], "/p/2/api/overview").json()["n_records"] == 0


def test_an_app_is_made_for_every_request_so_a_new_file_shows(made, spies) -> None:
    before = _get(made["client"], "/p/2/api/overview").json()
    shutil.copy(made["alpha"] / "records.jsonl", made["beta"] / "records.jsonl")
    after = _get(made["client"], "/p/2/api/overview").json()
    assert before["n_records"] == 0 and after["n_records"] > 0
    # Not one app kept and asked twice, which would show the file too.
    assert spies == ["get_project", "create_app"] * 2


def test_a_project_added_renamed_or_removed_by_the_cli_shows_on_the_next_request(made, tmp_path) -> None:
    client = made["client"]
    listed = lambda: [(p["id"], p["slug"], p["name"]) for p in _get(client, "/api/registry/projects").json()["projects"]]
    assert listed() == [("1", "alpha", "Alpha tickets"), ("2", "beta", "beta")]
    registry.rename_project(made["db"], "Beta", slug="beta")
    registry.remove_project(made["db"], slug="alpha")
    again = registry.add_project(made["db"], "alpha", str(made["beta"]))
    assert again.id == 3 and listed() == [("2", "beta", "Beta"), ("3", "alpha", "alpha")]
    # The old number names no project again; the slug's new registration is another address.
    gone = _get(client, "/p/1/api/overview")
    assert gone.status_code == 404 and gone.json() == {"detail": f"no project #1 in {made['db']}"}
    assert _get(client, "/p/3/api/overview").json()["n_records"] == 0


# --- which addresses answer ------------------------------------------------------------------------


def test_the_pages_and_the_redirects_to_them(made, spies) -> None:
    client = made["client"]
    r = _get(client, "/")
    assert (r.status_code, r.headers["location"]) == (307, "/p/")
    for slashless in ("/p", "/p/1", "/p/9", "/p/01"):
        r = _get(client, slashless)
        assert (r.status_code, r.headers["location"]) == (307, "http://127.0.0.1:8765" + slashless + "/")
    for page in ("/p/", "/p/1/", "/p/2/", "/p/9/", "/p/1/?x=1", "/p/" + BIG[2] + "/"):
        r = _get(client, page)
        assert r.status_code == 200 and r.text == PAGE and r.headers["content-type"].startswith("text/html"), page
    # A page is the same file for every project and none: the registry is not read for it.
    assert spies == []


def test_a_page_opens_whatever_became_of_the_registry_or_the_directory(made) -> None:
    shutil.rmtree(made["alpha"])
    made["db"].unlink()
    assert _get(made["client"], "/p/1/").text == PAGE and _get(made["client"], "/p/").text == PAGE


@pytest.mark.parametrize("address", ["/p/%31/", "/%70/1/", "/%70/", "/p/1%2F", "/p/%31/?x=1"])
def test_a_page_is_answered_only_at_its_address_as_written(made, spies, address) -> None:
    # The browser keeps the escapes in the address the page reads its project from.
    r = _get(made["client"], address)
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"} and spies == []


def test_a_page_is_refused_to_a_scope_that_does_not_say_how_it_was_written(made) -> None:
    app = create_projects_app(made["db"])
    for path in ("/p/", "/p/1/"):
        assert _asgi(app, path)[0] == 200
        assert _asgi(app, path, raw=None)[0] == 404
        assert _asgi(app, path, raw=path.replace("p", "%70").encode())[0] == 404
        # A server that leaves the query in `raw_path` (an older test client does) is still read.
        assert _asgi(app, path, raw=path.encode() + b"?x=1")[0] == 200
    # A project's requests do not depend on it: only a page reads its address back.
    assert _asgi(app, "/p/1/api/groups", raw=None)[0] == 200


@pytest.mark.parametrize("key", ["0", "01", "1x", "-1", "+1", "1.0", "a", "1" * 20, "\u0661", "1\n"])
def test_a_key_that_is_not_a_registration_number_is_not_looked_up(made, spies, key) -> None:
    app = create_projects_app(made["db"])
    made["db"].unlink()
    for rest in ("/", "/api/overview"):
        status, _, body = _asgi(app, f"/p/{key}{rest}")
        assert status == 404 and json.loads(body) == {"detail": "Not Found"}, rest
    assert spies == []


def test_a_key_past_every_id_sqlite_holds_is_only_a_missing_project(made) -> None:
    r = _get(made["client"], "/p/" + "9" * 19 + "/api/overview")
    assert r.status_code == 404 and r.json() == {"detail": f"no project #{'9' * 19} in {made['db']}"}


@pytest.mark.parametrize("rest", ["/docs", "/redoc", "/openapi.json", "/api", "/static/index.html", "//api/overview", "/x/api/overview"])
def test_nothing_of_the_inner_app_but_its_api_is_reached(made, spies, rest) -> None:
    status, _, body = _asgi(create_projects_app(made["db"]), "/p/1" + rest)
    assert status == 404 and json.loads(body) == {"detail": "Not Found"} and spies == []


@pytest.mark.parametrize("path", ["/p/1/api/../docs", "/p/1/api/../openapi.json", "/p/1/../1/api/overview", "/p/1/api/./overview"])
def test_dot_segments_are_not_folded_into_another_route(made, path) -> None:
    # Sent as written: an HTTP client folds them before sending.
    assert _asgi(create_projects_app(made["db"]), path)[0] == 404


def test_the_registrys_app_documents_nothing(made) -> None:
    for url in ("/docs", "/redoc", "/openapi.json", "/api/overview", "/api/registry", "/favicon.ico"):
        assert _get(made["client"], url).status_code == 404, url


@pytest.mark.parametrize("method", ["POST", "HEAD", "OPTIONS", "PUT", "DELETE"])
def test_only_get_reaches_a_project_and_is_refused_before_the_registry_is_opened(made, spies, method) -> None:
    client = made["client"]
    for url in ("/p/1/", "/p/1/api/overview", "/p/9/", "/p/9/api/overview"):
        r = client.request(method, url, follow_redirects=False)
        assert (r.status_code, r.headers["allow"]) == (405, "GET"), url
    made["db"].unlink()
    assert client.request(method, "/p/1/api/overview").status_code == 405
    assert spies == []
    # Not a key, or not a page or an API path: nothing there for any method.
    for url in ("/p/01/", "/p/01/api/overview", "/p/1/docs"):
        assert client.request(method, url, follow_redirects=False).status_code == 404, url
    # Without the slash the request never reaches the project: the router redirects any key and method.
    assert client.request(method, "/p/01", follow_redirects=False).status_code == 307
    # The registry's own routes are GET only too.
    assert client.request(method, "/api/registry/projects").status_code == 405
    assert client.request(method, "/p/").status_code == 405


# --- a directory or a registry that cannot be used -----------------------------------------------------


def test_a_directory_that_is_gone_or_a_file_now_is_a_400_naming_the_project(made) -> None:
    client, alpha = made["client"], made["alpha"]
    shutil.rmtree(alpha)
    r = _get(client, "/p/1/api/overview")
    assert r.status_code == 400
    assert r.json() == {"detail": f"project #1 (alpha): workspace directory cannot be read: {alpha}: No such file or directory"}
    alpha.write_text("a file", encoding="utf-8")
    r = _get(client, "/p/1/api/overview")
    assert (r.status_code, r.json()) == (400, {"detail": f"project #1 (alpha): workspace is not a directory: {alpha}"})
    # The other project is untouched, and the page of the broken one still opens.
    assert _get(client, "/p/2/api/overview").status_code == 200 and _get(client, "/p/1/").status_code == 200


def test_a_symlink_loop_is_a_400(made) -> None:
    shutil.rmtree(made["alpha"])
    os.symlink(made["alpha"], made["alpha"])
    r = _get(made["client"], "/p/1/api/overview")
    assert r.status_code == 400 and r.json()["detail"].startswith(f"project #1 (alpha): workspace directory cannot be read: {made['alpha']}: ")


@not_root
def test_who_names_a_directory_that_cannot_be_searched(tmp_path, demo) -> None:
    # Under an unsearchable folder the directory cannot even be looked at: the dispatcher names the project.
    # Unsearchable itself, it is still a directory, and each file in it is the inner app's own 400.
    outer, db = tmp_path / "outer", tmp_path / "projects.db"
    inside = outer / "ws"
    shutil.copytree(demo, inside)
    registry.init_registry(db)
    registry.add_project(db, "inside", str(inside))
    client = TestClient(create_projects_app(db), **LOOPBACK)
    try:
        os.chmod(inside, 0o000)
        r = _get(client, "/p/1/api/overview")
        assert r.status_code == 400 and r.json()["detail"].startswith(str(inside / "records.jsonl") + ": cannot read: ")
        os.chmod(inside, 0o700)
        os.chmod(outer, 0o000)
        r = _get(client, "/p/1/api/overview")
        assert r.status_code == 400
        assert r.json() == {"detail": f"project #1 (inside): workspace directory cannot be read: {inside}: Permission denied"}
    finally:
        os.chmod(outer, 0o700)
        os.chmod(inside, 0o700)


def test_a_directory_removed_between_the_check_and_the_app_is_a_400(made, monkeypatch) -> None:
    def gone(workspace_dir):
        raise FileNotFoundError(f"workspace directory not found: {workspace_dir}")

    monkeypatch.setattr(web_projects, "create_app", gone)
    r = _get(made["client"], "/p/1/api/overview")
    assert r.status_code == 400
    assert r.json() == {"detail": f"project #1 (alpha): workspace directory cannot be read: {made['alpha']}: workspace directory not found: {made['alpha']}"}


def test_a_stored_path_the_filesystem_encoding_cannot_hold_is_a_400(made, monkeypatch) -> None:
    real = os.stat

    def stat(path, *args, **kwargs):
        # Only this directory: `os` is one module, and the test runner stats its own files through it.
        if path == str(made["alpha"]):
            raise UnicodeEncodeError("ascii", "\u00e9", 0, 1, "ordinal not in range(128)")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    r = _get(made["client"], "/p/1/api/overview")
    assert r.status_code == 400 and r.json()["detail"].startswith(f"project #1 (alpha): workspace directory cannot be read: {made['alpha']}: ")


def test_a_bug_while_making_the_inner_app_stays_a_500_and_is_not_framed_either(made, monkeypatch) -> None:
    def broken(workspace_dir):
        raise RuntimeError("a bug")

    monkeypatch.setattr(web_projects, "create_app", broken)
    r = TestClient(create_projects_app(made["db"]), raise_server_exceptions=False, **LOOPBACK).get("/p/1/api/overview")
    assert r.status_code == 500 and FRAMES.items() <= dict(r.headers).items()


def test_an_inner_app_that_fails_after_it_began_to_answer_sends_one_framed_start(made, monkeypatch) -> None:
    async def half(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        raise RuntimeError("a bug after the first bytes")

    monkeypatch.setattr(web_projects, "create_app", lambda workspace_dir: half)
    starts: list[dict] = []

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            starts.append(message)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": "/p/1/api/overview", "raw_path": b"/p/1/api/overview", "query_string": b"", "root_path": "",
             "headers": [(b"host", b"127.0.0.1:8765")], "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8765)}
    # The bug reaches the server, which logs it; no second response is begun over the first.
    with pytest.raises(RuntimeError, match="a bug after the first bytes"):
        asyncio.run(create_projects_app(made["db"])(scope, receive, send))
    assert [m["status"] for m in starts] == [200]
    assert [h for h in starts[0]["headers"] if h in NO_FRAMES] == list(NO_FRAMES)


def test_a_file_that_is_not_a_usable_registry_is_a_400_naming_it(made, tmp_path) -> None:
    client, db = made["client"], made["db"]
    _raw(db, "PRAGMA writable_schema = ON", "UPDATE projects SET name = x'00ff'")
    for url in ("/api/registry/projects", "/p/1/api/overview"):
        r = _get(client, url)
        assert r.status_code == 400 and str(db) in r.json()["detail"] and "project registry" in r.json()["detail"], url
    db.unlink()
    for url in ("/api/registry/projects", "/p/1/api/overview"):
        r = _get(client, url)
        assert (r.status_code, r.json()) == (400, {"detail": f"project registry not found: {db} (run `s1 project init` first)"}), url
    foreign = tmp_path / "other.db"
    _raw(foreign, "CREATE TABLE projects(id INTEGER)")
    r = _get(TestClient(create_projects_app(foreign), **LOOPBACK), "/api/registry/projects")
    assert (r.status_code, r.json()) == (400, {"detail": f"not a project registry: {foreign} (make one with `s1 project init`)"})


def test_a_registry_held_by_another_process_is_a_503_and_a_page_still_opens(made, monkeypatch) -> None:
    monkeypatch.setattr(registry, "BUSY_TIMEOUT_S", 0.05)
    client = made["client"]
    with contextlib.closing(sqlite3.connect(made["db"], isolation_level=None)) as holder:
        holder.execute("BEGIN EXCLUSIVE")
        for url in ("/api/registry/projects", "/p/1/api/overview"):
            r = _get(client, url)
            assert r.status_code == 503 and r.json()["detail"].startswith(f"project registry is busy: {made['db']}"), url
        assert _get(client, "/p/1/").status_code == 200 and _get(client, "/p/").status_code == 200
        holder.execute("ROLLBACK")
    assert _get(client, "/p/1/api/overview").status_code == 200


# --- the list ----------------------------------------------------------------------------------------


def test_the_list_is_the_clis_rows_with_each_id_as_text(made, capsys) -> None:
    capsys.readouterr()
    assert cli_main(["project", "list", "--db", str(made["db"])]) == 0
    rows = json.loads(capsys.readouterr().out)["projects"]
    body = _get(made["client"], "/api/registry/projects").json()
    assert body == {"registry": {"path": str(made["db"])}, "projects": [{**row, "id": str(row["id"])} for row in rows]}
    assert [row["id"] for row in rows] == [1, 2]


def test_ids_a_browser_would_round_are_served_and_opened_exactly(tmp_path) -> None:
    # Such ids are only in a file edited by hand, which the registry reads as long as its counter covers them.
    db = tmp_path / "projects.db"
    registry.init_registry(db)
    dirs = []
    for i, big in enumerate(BIG):
        d = tmp_path / f"ws{i}"
        d.mkdir()
        (d / "records.jsonl").write_text("", encoding="utf-8")
        registry.add_project(db, f"p{i}", str(d))
        dirs.append(d)
    _raw(db, *(f"UPDATE projects SET id = {big} WHERE slug = 'p{i}'" for i, big in enumerate(BIG)),
         f"UPDATE sqlite_sequence SET seq = {BIG[2]}")
    client = TestClient(create_projects_app(db), **LOOPBACK)
    listed = _get(client, "/api/registry/projects").json()["projects"]
    assert [p["id"] for p in listed] == list(BIG)
    assert int(float(BIG[0])) == int(float(BIG[1]))  # what a JSON number would have made of two of them
    for big, d in zip(BIG, dirs):
        files = _get(client, f"/p/{big}/api/overview").json()["files"]
        assert files["records"] == str(d / "records.jsonl"), big


def test_a_relative_registry_path_is_anchored_once(made, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(made["db"].parent)
    app = create_projects_app(made["db"].name)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    elsewhere.rmdir()  # no working directory left to ask for
    client = TestClient(app, **LOOPBACK)
    assert _get(client, "/api/registry/projects").json()["registry"] == {"path": str(Path(made["db"].parent) / made["db"].name)}
    assert _get(client, "/p/1/api/groups").status_code == 200
    made["db"].unlink()
    assert _get(client, "/api/registry/projects").status_code == 400
    monkeypatch.chdir(tmp_path)


def test_a_registry_path_that_is_not_utf8_is_refused_at_once(tmp_path) -> None:
    with pytest.raises(ValueError) as info:
        create_projects_app(str(tmp_path / "pro\udcffjects.db"))
    assert str(info.value) == "the project registry's path holds a lone surrogate (U+DCFF), which is not a character (a file name that is not UTF-8)"


# --- this computer only ------------------------------------------------------------------------------

ADDRESSES = ["/", "/p/", "/p/1/", "/p/1/api/overview", "/api/registry/projects", "/p/9/api/overview", "/docs"]
NOT_LOCAL = {
    "the test client's own address and Host": ({}, {}),
    "another address with a loopback Host": ({"client": ("10.0.0.5", 50000), "base_url": "http://127.0.0.1:8765"}, {}),
    "a loopback address with another Host": ({"client": ("127.0.0.1", 50000), "base_url": "http://evil.example:8765"}, {}),
    "an IPv6 authority": ({"client": ("::1", 50000), "base_url": "http://[::1]:8765"}, {}),
    "a forwarded request": (LOOPBACK, {"X-Forwarded-For": "127.0.0.1"}),
    "a request through a proxy": (LOOPBACK, {"Via": "1.1 proxy"}),
}


@pytest.mark.parametrize("who", sorted(NOT_LOCAL))
def test_a_request_that_is_not_this_computers_is_refused_before_anything_is_read(made, spies, who) -> None:
    client_args, headers = NOT_LOCAL[who]
    client = TestClient(create_projects_app(made["db"]), **client_args)
    for address in ADDRESSES:
        r = client.get(address, headers=headers, follow_redirects=False)
        assert (r.status_code, r.json()) == (403, {"detail": REGISTRY_REFUSED}), address
        assert FRAMES.items() <= dict(r.headers).items()
    # Not a refusal after a look: a guard behind the routes would have read the registry by now.
    assert spies == []


def test_two_hosts_are_refused(made) -> None:
    app = create_projects_app(made["db"])
    scope_headers = [(b"host", b"127.0.0.1:8765"), (b"host", b"evil.example")]
    got: dict = {}

    async def send(message: dict) -> None:
        got.setdefault("status", message.get("status"))

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {"type": "http", "method": "GET", "path": "/p/", "raw_path": b"/p/", "query_string": b"", "root_path": "",
             "headers": scope_headers, "client": ("127.0.0.1", 50000), "scheme": "http", "http_version": "1.1"}
    asyncio.run(app(scope, receive, send))
    assert got["status"] == 403


def test_every_response_refuses_to_be_framed(made) -> None:
    client = made["client"]
    seen = {}
    for method, url in [("GET", "/"), ("GET", "/p/"), ("GET", "/p/1/"), ("GET", "/p/1"), ("GET", "/p/1/api/overview"),
                        ("GET", "/api/registry/projects"), ("GET", "/p/9/api/overview"), ("GET", "/p/1/api/audit?n_boot=5"),
                        ("POST", "/p/1/"), ("GET", "/nothing"), ("GET", "/p/%31/")]:
        r = client.request(method, url, follow_redirects=False)
        seen[r.status_code] = seen.get(r.status_code, 0) + 1
        assert FRAMES.items() <= dict(r.headers).items(), (method, url)
        assert [v for k, v in r.headers.multi_items() if k == "x-frame-options"] == ["DENY"], (method, url)
    assert set(seen) == {200, 307, 404, 405, 422}
    assert NO_FRAMES == tuple((k.encode(), v.encode()) for k, v in FRAMES.items())


def test_a_server_prefix_is_refused_and_a_loopback_request_alone_is_told_so(made, spies) -> None:
    app = create_projects_app(made["db"])
    for path in ("/p/", "/p/1/", "/p/1/api/overview", "/api/registry/projects"):
        status, headers, body = _asgi(app, path, root_path="/tools")
        assert (status, json.loads(body)) == (404, {"detail": PREFIX_REFUSED}) and headers["x-frame-options"] == "DENY", path
    assert spies == []
    # The address rule comes first: a request that is not this computer's learns nothing of the setup.
    assert TestClient(app, root_path="/tools").get("/p/").json() == {"detail": REGISTRY_REFUSED}


def test_no_socket_but_http_is_opened(made) -> None:
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {"type": "websocket.connect"}

    scope = {"type": "websocket", "path": "/p/1/api/overview", "raw_path": b"/p/1/api/overview", "query_string": b"",
             "root_path": "", "headers": [(b"host", b"127.0.0.1:8765")], "client": ("127.0.0.1", 50000), "scheme": "ws"}
    asyncio.run(create_projects_app(made["db"])(scope, receive, send))
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_the_workspace_app_has_none_of_these_addresses(demo) -> None:
    client = TestClient(create_app(workspace_dir=demo), **LOOPBACK)
    for url in ("/p/", "/p/1/", "/p/1/api/overview", "/api/registry/projects"):
        assert client.get(url).status_code == 404, url
    assert "x-frame-options" not in client.get("/").headers


# --- s1 web --db -------------------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch) -> list:
    import uvicorn

    calls: list = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.append((app, host, port)))
    return calls


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_web_serves_the_registry_it_was_given(made, served, host) -> None:
    assert cli_main(["web", "--db", str(made["db"]), "--host", host, "--port", "8799"]) == 0
    app, bound, port = served[0]
    assert (bound, port) == (host, 8799)
    client = TestClient(app, **LOOPBACK)
    assert [p["slug"] for p in client.get("/api/registry/projects").json()["projects"]] == ["alpha", "beta"]
    assert client.get("/p/1/api/overview").json()["n_records"] > 0
    assert TestClient(app).get("/p/").status_code == 403


def _refused(capsys, argv: list[str]) -> str:
    capsys.readouterr()
    assert cli_main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("s1: ") and err.count("\n") == 1
    return err[len("s1: "):-1]


@pytest.mark.parametrize("there", [True, False])
def test_what_db_does_not_go_with_is_said_before_any_file_is_looked_at(made, served, capsys, tmp_path, there) -> None:
    db = str(made["db"] if there else tmp_path / "none.db")
    assert _refused(capsys, ["web", "--db", db, "--workspace", str(made["alpha"])]) == (
        "--db serves the projects of a registry, each from its own directory: it does not take --workspace")
    # A workspace named as the default is still one named.
    assert "does not take --workspace" in _refused(capsys, ["web", "--db", db, "--workspace", "."])
    assert _refused(capsys, ["web", "--db", db, "--sheet", str(tmp_path / "none.md")]) == (
        "--db does not take --sheet: a review sheet is served with one workspace (`s1 web --workspace DIR --sheet PATH`)")
    for host in ("0.0.0.0", "::1", "192.168.1.5"):
        assert _refused(capsys, ["web", "--db", db, "--host", host]) == (
            f"--db serves a registry's paths to this computer only: --host must be 127.0.0.1 or localhost, not {host!r}")
    assert served == []


def test_web_refuses_a_file_that_is_not_a_usable_registry_before_serving(made, served, capsys, tmp_path) -> None:
    missing = tmp_path / "none.db"
    assert _refused(capsys, ["web", "--db", str(missing)]) == f"project registry not found: {missing} (run `s1 project init` first)"
    foreign = tmp_path / "other.db"
    _raw(foreign, "CREATE TABLE t(x)")
    assert _refused(capsys, ["web", "--db", str(foreign)]) == f"not a project registry: {foreign} (make one with `s1 project init`)"
    assert _refused(capsys, ["web", "--db", str(tmp_path)]) == f"project registry is a directory: {tmp_path}"
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "registry.fifo"
        os.mkfifo(fifo)
        assert _refused(capsys, ["web", "--db", str(fifo)]) == f"project registry is not a regular file: {fifo}"
    assert served == []


def test_web_without_db_serves_a_workspace_as_before(demo, served, monkeypatch, capsys) -> None:
    monkeypatch.chdir(demo)
    for n, argv in enumerate((["web"], ["web", "--workspace", "."], ["web", "--workspace", str(demo)]), start=1):
        assert cli_main(argv) == 0 and len(served) == n, argv  # each reached the server
    for app, host, port in served:
        assert (host, port) == ("127.0.0.1", 8765)
        client = TestClient(app)
        assert client.get("/api/overview").json()["n_records"] > 0 and client.get("/p/").status_code == 404
    assert _refused(capsys, ["web", "--workspace", str(demo / "none")]) == f"workspace directory not found: {demo / 'none'}"


# --- the page: which of the three it is --------------------------------------------------------------


def _js(expr: str, prelude: str = "", functions: str | None = None) -> object:
    return _node(f"{prelude}\n{functions if functions is not None else _all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


@needs_node
def test_a_page_reads_its_scope_from_its_address() -> None:
    cases = {
        "/p/": {"kind": "registry"},
        "/p/1/": {"kind": "project", "id": "1"},
        "/p/" + BIG[1] + "/": {"kind": "project", "id": BIG[1]},
        "/p/" + BIG[2] + "/": {"kind": "project", "id": BIG[2]},
    }
    workspace = ["/", "", "/p", "/p/1", "/p/0/", "/p/01/", "/p/%31/", "/%70/", "/p/1/x", "/p/1//", "/x/p/1/", "/p/1/\n", "/p/" + "1" * 20 + "/", "/p/1e3/", "/p/-1/"]
    got = _js(f"{json.dumps(list(cases) + workspace)}.map(p => pageScope(p))")
    assert got == list(cases.values()) + [{"kind": "workspace"}] * len(workspace)
    assert _js("[pageScope(undefined), pageScope(null)]") == [{"kind": "workspace"}] * 2


@needs_node
def test_a_projects_page_asks_its_own_project_and_the_registry_as_it_is() -> None:
    urls = ["/api/overview", "/api/audit?n_boot=200", "/api/registry/projects", "/api/registryx"]
    at = lambda pathname: _js(f"{json.dumps(urls)}.map(u => requestUrl(u, {json.dumps(pathname)}))")
    assert at("/p/7/") == ["/p/7/api/overview", "/p/7/api/audit?n_boot=200", "/api/registry/projects", "/p/7/api/registryx"]
    assert at("/p/" + BIG[1] + "/") [0] == "/p/" + BIG[1] + "/api/overview"
    for pathname in ("/", "/p/", "/p/%37/", "/somewhere/"):
        assert at(pathname) == urls, pathname


@needs_node
def test_the_one_fetch_of_the_page_goes_where_the_scope_says() -> None:
    js = """
const asked = [];
const fetch = async url => { asked.push(url); return {ok: true, json: async () => ({})}; };
let location = {pathname: "/p/3/"};
%s
(async () => {
  await fetchJson("/api/groups");
  await fetchJson("/api/registry/projects");
  location = {pathname: "/"};
  await fetchJson("/api/groups");
  location = {pathname: "/p/"};
  await fetchJson("/api/registry/projects");
  process.stdout.write(JSON.stringify(asked));
})();
""" % _all_functions(exclude=())
    assert _node(js) == ["/p/3/api/groups", "/api/registry/projects", "/api/groups", "/api/registry/projects"]
    # Nothing else in the page requests anything.
    assert PAGE.count("fetch(") == 1


@needs_node
def test_a_registrys_page_has_one_view_whatever_the_hash_says() -> None:
    hashes = ["", "#/review", "#/calib?qid=q&model=m&prim=noul", "#/projects", "#/policy?lo=0.3"]
    views = _js(f"{json.dumps(hashes)}.map(h => stateFromLocation(h, '/p/'))")
    assert [v["view"] for v in views] == ["projects"] * len(hashes)
    assert all(v == {**_js("defaultState()"), "view": "projects"} for v in views)
    # Any other page reads its hash, where no link can name the projects view.
    assert _js(f"{json.dumps(hashes)}.map(h => stateFromLocation(h, '/p/1/').view)") == ["overview", "review", "calib", "overview", "policy"]
    assert _js("stateFromLocation('#/projects', '/')") == _js("defaultState()")
    assert "projects" not in _js("viewNames()") and "projects" in _js("Object.keys(viewDefs())")


@needs_node
def test_a_projects_address_is_built_only_from_a_registration_number() -> None:
    good = ["1", "42", *BIG]
    assert _js(f"{json.dumps(good)}.map(projectUrl)") == [f"/p/{i}/" for i in good]
    bad = ["", "0", "01", "1x", " 1", "1/", "../1", "1" * 20, HOSTILE, "1\n", "javascript:alert(1)"]
    assert _js(f"{json.dumps(bad)}.map(projectUrl)") == [None] * len(bad)
    assert _js("[projectUrl(1), projectUrl(null), projectUrl(undefined), projectUrl({})]") == [None] * 4
    # What `projectUrl` makes, `pageScope` reads back as that project.
    assert _js(f"{json.dumps(good)}.map(i => pageScope(projectUrl(i)).id)") == good


@needs_node
def test_the_header_names_the_project_as_listed_or_says_it_is_not() -> None:
    projects = [{"id": "1", "slug": "alpha", "name": "Alpha tickets"}, {"id": BIG[0], "slug": "a", "name": "A"}, {"id": BIG[1], "slug": "b", "name": HOSTILE}]
    label = lambda i, listed=projects: _js(f"projectLabel({json.dumps(i)}, {json.dumps(listed)})")
    assert label("1") == "Alpha tickets (alpha)" and label(BIG[0]) == "A (a)" and label(BIG[1]) == HOSTILE + " (b)"
    assert label("2") == "Not found: #2" and label("1", []) == "Not found: #1" and label("1", None) == "Not found: #1"
    # An id is compared as text: a number that would round to a listed one is another project.
    assert label("1", [{"id": 1, "slug": "x", "name": "X"}]) == "Not found: #1"


NAME_STEPS = """
let generation = 0;
const els = {};
const document = {getElementById: id => (els[id] ??= {textContent: "", innerHTML: ""})};
const pending = [];
const fetchJson = url => new Promise((resolve, reject) => pending.push({url, resolve, reject}));
const tick = () => new Promise(r => setTimeout(r, 0));
const out = {};
%s
(async () => {
  loadProjectName("2");
  out.atOnce = els["project-name"].textContent;
  out.urls = pending.map(p => p.url);
  pending.shift().resolve({projects: [{id: "2", slug: "beta", name: %s}]});
  await tick();
  out.named = els["project-name"].textContent;
  // A Refresh while the list is out: only its own answer is written, whichever arrives last.
  loadProjectName("2");
  generation += 1;
  loadProjectName("2");
  pending[1].resolve({projects: []});
  await tick();
  out.latest = els["project-name"].textContent;
  pending[0].resolve({projects: [{id: "2", slug: "old", name: "Old"}]});
  await tick();
  out.afterTheOlder = els["project-name"].textContent;
  generation += 1;
  loadProjectName("2");
  pending[2].reject(new Error(%s));
  await tick();
  out.failed = els["project-name"].textContent;
  out.markup = Object.values(els).map(e => e.innerHTML);
  process.stdout.write(JSON.stringify(out));
})();
"""


@needs_node
def test_the_header_name_is_text_and_only_the_latest_refreshs() -> None:
    out = _node(NAME_STEPS % (_all_functions(), json.dumps(HOSTILE), json.dumps(HOSTILE)))
    assert out["atOnce"] == "Project #2" and out["urls"] == ["/api/registry/projects"]
    assert out["named"] == HOSTILE + " (beta)"
    assert out["latest"] == "Not found: #2" and out["afterTheOlder"] == "Not found: #2"
    assert out["failed"] == f"Project #2 (project list unavailable: {HOSTILE})"
    assert out["markup"] == [""]  # written as text, never as markup


# --- the page: the projects list -----------------------------------------------------------------


def _shell_run(steps: str, payload: dict, functions: str | None = None) -> dict:
    js = f"""
let STATE = null;
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [], groupsState = "loading", groupsError = "";
const els = {{}};
const out = {{keptAtEachWrite: []}};
const el = id => (els[id] ??= {{id, html: "", textContent: "", value: "", hidden: false, disabled: false, attrs: {{}},
  set innerHTML(v) {{ this.html = v; if (id === "projects-container") out.keptAtEachWrite.push("projects" in viewData); }},
  get innerHTML() {{ return this.html; }},
  classList: {{toggle() {{}}}}, querySelector: () => null, querySelectorAll: () => [],
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const pending = [];
let fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const payload = {json.dumps(payload)};
const shown = () => el("projects-container").innerHTML;
{functions if functions is not None else _all_functions()}
STATE = stateFromLocation("#/review", "/p/");
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


def _listed(n: int = 12) -> dict:
    """A list as served: ids in order, names and paths that sort otherwise."""
    return {"registry": {"path": "/r/projects.db"},
            "projects": [{"id": str(i), "slug": f"s{n - i:02d}", "name": f"Name {(i * 7) % n:02d}", "workspace_path": f"/w/{chr(96 + i)}"} for i in range(1, n + 1)]}


TABLE_STEPS = r"""
  render();
  out.urls = pending.map(p => p.url);
  out.keptBeforeTheAnswer = "projects" in viewData;
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.kept = viewData.projects === payload;
  out.tables = Object.keys(currentTables());
  fetchJson = () => { throw new Error("no request"); };
  const box = {dataset: {table: "projects:all"}, body: {innerHTML: ""}, count: {textContent: ""},
    querySelector(sel) { return sel === ".tk-body" ? this.body : this.count; }, querySelectorAll: () => []};
  document.querySelectorAll = sel => sel === ".tk" ? [box] : [];
  const downloads = [];
  downloadText = (name, text) => downloads.push([name, text]);
  const ids = () => Array.from(box.body.innerHTML.matchAll(/data-id="(\d+)"/g)).map(m => m[1]);
  sortTable("projects:all", "id"); sortTable("projects:all", "id");
  out.byIdDown = ids();
  sortTable("projects:all", "slug");
  out.bySlug = ids();
  exportTable("projects:all", "json");
  out.json = [downloads[0][0], JSON.parse(downloads[0][1])];
  filterTable("projects:all", "/w/c");
  out.count = box.count.textContent;
  out.filtered = ids();
  exportTable("projects:all", "csv");
  out.csv = downloads[1][1];
"""


@needs_node
def test_the_list_goes_through_the_shared_table_handlers() -> None:
    d = _listed()
    out = _shell_run(TABLE_STEPS, d)
    assert out["urls"] == ["/api/registry/projects"]  # the one request of a registry's page
    assert out["keptBeforeTheAnswer"] is False and out["kept"] is True and out["tables"] == ["projects:all"]
    # Two writes, the skeleton and the list, and the payload was kept only after the second.
    assert out["keptAtEachWrite"] == [False, False]
    # Ids are text and sort in natural order, 10 after 9.
    assert out["byIdDown"] == [str(i) for i in range(12, 0, -1)]
    assert out["bySlug"] == [str(i) for i in range(12, 0, -1)]  # s00 is project 12
    assert out["json"] == ["projects.json", sorted(d["projects"], key=lambda p: p["slug"])]  # as served, ids as text
    assert out["count"] == "1 of 12 rows" and out["filtered"] == ["3"]
    assert out["csv"] == "\ufeffid,name,slug,workspace_path\r\n3,Name 09,s09,/w/c\r\n"


@needs_node
def test_without_its_tables_the_view_would_render_and_do_nothing() -> None:
    # The mutant the check above is for: the list still shows, sort, filter and export are silent.
    mutant = _all_functions().replace('if (STATE.view === "projects" && d) return projectsTables(d);', "")
    assert mutant != _all_functions()
    with pytest.raises(AssertionError):
        _shell_run(TABLE_STEPS, _listed(), functions=mutant)


@needs_node
def test_ids_past_what_a_number_holds_stay_apart_in_the_list() -> None:
    d = {"registry": {"path": "/r/p.db"}, "projects": [{"id": big, "slug": f"p{i}", "name": f"P{i}", "workspace_path": f"/w/{i}"} for i, big in enumerate(BIG)]}
    steps = r"""
  render();
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.opened = Array.from(shown().matchAll(/data-act="project-open" data-id="(\d+)"/g)).map(m => projectUrl(m[1]));
  const box = {dataset: {table: "projects:all"}, body: {innerHTML: ""}, count: {textContent: ""},
    querySelector(sel) { return sel === ".tk-body" ? this.body : this.count; }, querySelectorAll: () => []};
  document.querySelectorAll = sel => sel === ".tk" ? [box] : [];
  sortTable("projects:all", "id"); sortTable("projects:all", "id");
  out.down = Array.from(box.body.innerHTML.matchAll(/data-id="(\d+)"/g)).map(m => m[1]);
"""
    out = _shell_run(steps, d)
    assert out["opened"] == [f"/p/{big}/" for big in BIG]
    assert out["down"] == list(reversed(BIG))


@needs_node
def test_an_empty_registry_says_how_to_add_a_project_and_draws_no_table() -> None:
    steps = """
  render();
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.html = shown();
  out.tables = Object.keys(currentTables());
"""
    out = _shell_run(steps, {"registry": {"path": "/r/" + HOSTILE}, "projects": []})
    assert "No project is registered." in out["html"] and "s1 project add SLUG --workspace DIR --db FILE" in out["html"]
    assert "<table" not in out["html"] and out["tables"] == []
    assert "<img" not in out["html"] and "&lt;img src=x onerror=alert(1)&gt;" in out["html"]


@needs_node
def test_a_failed_list_is_an_error_card_and_nothing_is_kept() -> None:
    steps = """
  render();
  pending.shift().reject(Object.assign(new Error(%s), {status: %d}));
  await tick(); await tick();
  out.html = shown();
  out.kept = "projects" in viewData;
  out.tables = Object.keys(currentTables());
"""
    busy = _shell_run(steps % (json.dumps("project registry is busy: /r/p.db"), 503), {})
    assert "The project registry was busy" in busy["html"] and "traceback" not in busy["html"]
    assert busy["kept"] is False and busy["tables"] == []
    broken = _shell_run(steps % (json.dumps(HOSTILE), 400), {})
    # A registry that cannot be used is a file to fix, said without calling it a workspace's.
    assert "<img" not in broken["html"] and "A file or directory could not be used: fix the one named above, then Refresh." in broken["html"]


# --- the page: its markup and wiring -----------------------------------------------------------------


def _wiring() -> str:
    """The script's statements after its last function: listeners and startup."""
    script = PAGE[PAGE.index("<script>"):PAGE.index("</script>")]
    return script[script.rindex("\n    }\n") + len("\n    }\n"):]


def test_the_project_line_and_the_list_pane_are_in_the_page() -> None:
    assert '<div class="project-tools"><a href="/p/">All projects</a><span class="project-name" id="project-name"></span></div>' in PAGE
    assert '<section id="tab-projects" class="tab-pane" aria-label="Projects">\n      <div id="projects-container">' in PAGE
    # No tab opens the list, and it adds no control of its own to the page.
    assert PAGE.count('role="tab"') == 9 and 'data-view="projects"' not in PAGE
    pane = PAGE[PAGE.index('<section id="tab-projects"'):PAGE.index("</main>")]
    assert "<select" not in pane and "<input" not in pane and "<button" not in pane


def test_the_stylesheet_shows_each_scope_its_own() -> None:
    css = PAGE[PAGE.index("<style>"):PAGE.index("</style>")]
    for rule in (
        ".project-tools { display: none;",
        ':root[data-scope="project"] .project-tools { display: flex; }',
        ".only-project { display: none; }",
        ':root[data-scope="project"] .only-project { display: inline; }',
        ':root[data-scope="registry"] nav.views,',
        ':root[data-scope="registry"] #flt-group,',
        ':root[data-scope="registry"] #btn-clear-group { display: none; }',
    ):
        assert rule in css, rule
    # Refresh and the time of the last answer stay on a registry's page.
    assert '[data-scope="registry"] #btn-refresh' not in css and '[data-scope="registry"] .filter-bar {' not in css


def test_startup_and_history_read_the_state_through_the_scope() -> None:
    wiring = _wiring()
    assert "STATE = stateFromLocation(location.hash, location.pathname);" in wiring
    assert "parseHash(location.hash)" not in PAGE  # no path left that reads a hash without the scope
    assert "applyScope(pageScope(location.pathname));" in wiring and "loadShell(pageScope(location.pathname));" in wiring
    assert "loadMeta();" not in wiring and "loadGroups();" not in wiring
    # A document the browser restores on Back is left as it kept it, the focus included: nothing
    # reloads it but Refresh.
    assert "pageshow" not in PAGE and "persisted" not in PAGE


@needs_node
def test_a_registrys_page_loads_nothing_around_its_view_and_a_projects_its_name() -> None:
    js = """
let generation = 0;
const asked = [];
const els = {};
const document = {getElementById: id => (els[id] ??= {textContent: ""}), documentElement: {dataset: {}}};
let groupsSeq = 0, groupsState = "", metaSeq = 0;
const fetchJson = url => { asked.push(url); return new Promise(() => {}); };
%s
const out = {};
for (const pathname of ["/p/", "/p/4/", "/"]) {
  asked.length = 0;
  els["overview-meta"] = {textContent: "Loading workspace..."};
  applyScope(pageScope(pathname));
  loadShell(pageScope(pathname));
  out[pathname] = {asked: asked.slice(), scope: document.documentElement.dataset.scope, meta: els["overview-meta"].textContent,
                   name: (els["project-name"] || {}).textContent || ""};
  delete els["project-name"];
}
process.stdout.write(JSON.stringify(out));
""" % _all_functions()
    out = _node(js)
    assert out["/p/"] == {"asked": [], "scope": "registry", "meta": "Project registry", "name": ""}
    assert out["/p/4/"] == {"asked": ["/api/overview", "/api/groups", "/api/registry/projects"], "scope": "project",
                            "meta": "Loading workspace...", "name": "Project #4"}
    assert out["/"] == {"asked": ["/api/overview", "/api/groups"], "scope": "workspace", "meta": "Loading workspace...", "name": ""}


@needs_node
def test_the_empty_review_says_where_a_sheet_is_served_to_a_projects_page_only() -> None:
    html = _js("reviewEmptyHtml()")
    assert "s1 web --sheet PATH" in html
    assert '<span class="only-project">A workbench opened from a project registry (<code>s1 web --db</code>) serves no review sheet:' in html
    assert "<code>s1 web --workspace DIR --sheet PATH</code>.</span>" in html
