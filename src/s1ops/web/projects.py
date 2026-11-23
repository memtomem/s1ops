"""The workbench over a project registry (`s1 web --db`): every registered project's workspace under
`/p/<id>/`, each request answered by a `create_app` made for that project's directory, and the list of
projects at `/p/`, whose page can also add a project, rename one and remove one's registration (what
`s1 project` does, through `registry`): the only requests here that write anything.

Everything here is served to this computer only (`loopback_request_allowed`), since the list holds
absolute paths, and at the root of its origin only: the page reads which project it shows from its
address, so a page is answered at `/p/` and `/p/<id>/` exactly as written and nowhere else.

The key in the address is a project's registration number, which the registry never gives twice, so an
address cannot come to name another directory. A page is answered without reading the registry (an
address of no project opens a workbench whose every request says so); a project's `/api/` requests read
it each time, so a project added, renamed or removed by `s1 project` shows on the next request.

A write is a `POST`, `PATCH` or `DELETE` on the registry's own routes. Beyond being this computer's, it
must not name another site (`_own_request`: a page of another origin on this computer is still another
site), and a body is read only as `application/json`, counted as it arrives, and never quoted back.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect, HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from s1ops import __version__, registry
from s1ops.contracts.records import surrogate_problem
from s1ops.web.app import STATIC_DIR, create_app, loopback_request_allowed

REGISTRY_REFUSED = (
    "a project registry's workbench is served to this computer only: a loopback client address, a Host "
    "of 127.0.0.1 or localhost, and no proxy headers"
)
PREFIX_REFUSED = "a project registry's workbench is served at the root of its address, not under a prefix"
WRITE_REFUSED = (
    "a project registry is changed only from its own page or by a local client: this request names another "
    "site in Origin or Sec-Fetch-Site"
)
BODY_MAX = 65536
"""Bytes of a write's body that are kept: far above any slug, name and path, and a bound on what a client
that keeps sending can make the server hold."""
BODY_TYPE = "send the body as Content-Type: application/json"
BODY_NOT_OBJECT = "the body is not a JSON object"
ADD_SHAPE = "a project is added with slug, workspace_path and optionally name"
RENAME_SHAPE = "a project is renamed with name"
PATH_NOT_ABSOLUTE = (
    "the workbench takes an absolute workspace path (a relative one would be read from the server's "
    "working directory)"
)
NO_FRAMES = ((b"content-security-policy", b"frame-ancestors 'none'"), (b"x-frame-options", b"DENY"))
"""On every response: no other site may put these pages in a frame."""

# Every positive integer SQLite holds has at most 19 digits. A key is compared as a Python integer and
# never bound, so one past that range is only a project that does not exist.
_KEY = re.compile(r"[1-9][0-9]{0,18}")
_STATUS = {"missing": 404, "exists": 409, "busy": 503}


class _Refused(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _json(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


def _refusal(e: registry.RegistryError) -> _Refused:
    return _Refused(_STATUS.get(e.kind, 400), str(e))


class _Gone(Exception):
    """The client went away before its body had arrived: there is no one to answer."""


def _served(project: registry.Project) -> dict[str, Any]:
    """A project as a response holds it. The id is a decimal string: a browser reads a JSON number as a
    float, which cannot tell two ids past 2**53 apart."""
    return {**project.as_dict(), "id": str(project.id)}


async def _registry_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """One registry call, in the threadpool: a wait for the file's lock never holds the event loop."""
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except registry.RegistryError as e:
        raise _refusal(e) from None


def _own_request(scope: Scope) -> bool:
    """Whether a request that may write comes from this workbench's own page or from a client that is
    not a browser. A browser names the page a request was made by: `Origin` on every request that is
    not a GET or HEAD, and `Sec-Fetch-Site`. Either naming another origin, another port of this
    computer included, or given twice, is refused; neither is `curl` or a script, which the loopback
    rule has admitted."""
    headers = HTTPConnection(scope).headers
    origins, sites = headers.getlist("origin"), headers.getlist("sec-fetch-site")
    if len(origins) > 1 or len(sites) > 1:
        return False
    # The one Host the loopback rule accepted, as written: a browser writes both in lower case and
    # leaves a default port out of both.
    if origins and origins[0] != f"{scope.get('scheme', 'http')}://{headers.get('host', '')}":
        return False
    return not sites or sites[0] == "same-origin"


def _no_repeats(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    found = dict(pairs)
    if len(found) != len(pairs):
        raise ValueError("a key is given twice")
    return found


async def _json_object(request: Request, required: frozenset[str], optional: frozenset[str], shape: str) -> dict[str, Any]:
    """A write's body: a JSON object holding the `required` keys and any of the `optional` ones, read
    here and nowhere else. The type is checked by hand (the oldest FastAPI supported reads an untyped
    body as JSON, and what another site's page can send without asking is `text/plain`), and the bytes
    are counted before they are kept (`Request.body()` holds everything first). No message quotes the
    body: a key or a value may hold what a response cannot (an escaped lone surrogate)."""
    if request.headers.getlist("content-type") != ["application/json"]:
        raise _Refused(415, BODY_TYPE)
    kept = bytearray()
    try:
        async for chunk in request.stream():
            if len(kept) + len(chunk) > BODY_MAX:
                raise _Refused(413, f"the body is larger than {BODY_MAX} bytes")
            kept += chunk
    except ClientDisconnect:
        raise _Gone from None
    try:
        body = json.loads(bytes(kept).decode("utf-8"), object_pairs_hook=_no_repeats)
    except (ValueError, RecursionError):  # deep nesting is a RecursionError, which is not a ValueError
        raise _Refused(400, BODY_NOT_OBJECT) from None
    if not isinstance(body, dict):
        raise _Refused(400, BODY_NOT_OBJECT)
    if not required <= body.keys() <= required | optional:
        raise _Refused(400, shape)
    return body


def _page() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


def _canonical(scope: Scope, address: str) -> bool:
    """Whether the request was sent to `address` as written. The path a route matches is decoded, while
    the page reads its scope from the address the browser holds, escapes kept: answered at `/p/%31/`, the
    page of project 1 would take itself for a workspace's."""
    raw = scope.get("raw_path")
    # Cut at "?": the server gives the path alone, an older test client leaves the query in it.
    return isinstance(raw, bytes) and raw.split(b"?", 1)[0] == address.encode("ascii")


def _workspace_app(db: str, project_id: int) -> FastAPI:
    """The app for one project's directory, as `s1 web --workspace` makes it, or why there is none."""
    try:
        project = registry.get_project(db, project_id=project_id)
    except registry.RegistryError as e:
        raise _refusal(e) from None
    who, path = f"project #{project.id} ({project.slug})", project.workspace_path
    try:
        # A stored path this host's filesystem encoding cannot hold fails to encode before any lookup.
        is_dir = stat.S_ISDIR(os.stat(path).st_mode)
    except (OSError, UnicodeEncodeError) as e:
        raise _Refused(400, f"{who}: workspace directory cannot be read: {path}: {getattr(e, 'strerror', None) or e}") from None
    if not is_dir:
        raise _Refused(400, f"{who}: workspace is not a directory: {path}")
    try:
        return create_app(workspace_dir=path)
    except OSError as e:  # gone, or no longer a directory, since the check
        raise _Refused(400, f"{who}: workspace directory cannot be read: {path}: {e.strerror or e}") from None


class _LocalOnly:
    """The outermost callable: refuses every request that is not this computer's own, and every request
    but a GET that names another site, before anything behind it runs or a body is read, and marks
    every response, a 500 included, as not to be framed. It wraps the app rather than being one of its
    middlewares, which sit inside the handler that writes a 500."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return

        async def framed(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", ()), *NO_FRAMES]}
            await send(message)

        if not loopback_request_allowed(HTTPConnection(scope)):
            await _json(403, REGISTRY_REFUSED)(scope, receive, framed)
        elif scope.get("root_path"):
            # The page's requests and the redirects name addresses from the root.
            await _json(404, PREFIX_REFUSED)(scope, receive, framed)
        elif scope["method"] != "GET" and not _own_request(scope):
            await _json(403, WRITE_REFUSED)(scope, receive, framed)
        else:
            await self.app(scope, receive, framed)


def create_projects_app(db_path: str | os.PathLike[str]) -> ASGIApp:
    """The workbench of the registry at `db_path`, which `s1 project init` made. The file is opened on
    each request, not here. The address it is served on is the caller's: `s1 web --db` refuses any but
    127.0.0.1 and localhost, and every request not from this computer is a 403."""
    db = os.fspath(db_path)
    if not os.path.isabs(db):
        # Once, as the registry does on each call (nothing is folded), so no request depends on the
        # working directory still being there.
        db = os.path.join(os.getcwd(), db)
    # Every refusal names the file, and a response cannot hold half a character.
    problem = surrogate_problem("the project registry's path", db)
    if problem:
        raise ValueError(f"{problem} (a file name that is not UTF-8)")

    app = FastAPI(title="s1ops Workbench", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/p/", status_code=307)

    @app.get("/p/")
    def projects_page(request: Request) -> Response:
        if not _canonical(request.scope, "/p/"):
            raise HTTPException(status_code=404)
        return _page()

    @app.exception_handler(_Refused)
    async def refused(request: Request, e: _Refused) -> Response:
        return _json(e.status, e.detail)

    @app.exception_handler(_Gone)
    async def gone(request: Request, e: _Gone) -> Response:
        return Response(status_code=400)  # reaches no one

    # One handler for a path's methods, so a 405 names them all in `Allow`.
    @app.api_route("/api/registry/projects", methods=["GET", "POST"])
    async def api_projects(request: Request) -> Response:
        """GET: the registered projects in id order. POST: register a directory, as `s1 project add`
        does, except that the path must be absolute."""
        if request.method == "GET":
            projects = await _registry_call(registry.list_projects, db)
            return JSONResponse({"registry": {"path": db}, "projects": [_served(p) for p in projects]})
        body = await _json_object(request, frozenset({"slug", "workspace_path"}), frozenset({"name"}), ADD_SHAPE)
        path = body["workspace_path"]
        problem = registry.path_problem(path) or (None if os.path.isabs(path) else PATH_NOT_ABSOLUTE)
        if problem is None and "name" in body:
            # The registry reads None as no name given: only a body without the key is named by its slug.
            problem = registry.name_problem(body["name"])
        if problem:
            raise _Refused(400, problem)
        project = await _registry_call(registry.add_project, db, body["slug"], path, body.get("name"))
        return JSONResponse({"project": _served(project)}, status_code=201)

    @app.api_route("/api/registry/projects/{key}", methods=["PATCH", "DELETE"])
    async def api_project(key: str, request: Request) -> Response:
        """PATCH: change a project's display name. DELETE: remove its registration (no body is read);
        the directory stays. A removed number is never given again, so it answers 404 from then on."""
        if _KEY.fullmatch(key) is None:
            raise _Refused(404, "Not Found")
        if request.method == "PATCH":
            body = await _json_object(request, frozenset({"name"}), frozenset(), RENAME_SHAPE)
            project = await _registry_call(registry.rename_project, db, body["name"], project_id=int(key))
        else:
            project = await _registry_call(registry.remove_project, db, project_id=int(key))
        return JSONResponse({"project": _served(project)})

    async def dispatch(scope: Scope, receive: Receive, send: Send) -> None:
        """`/p/<key>/...`: the page at `/`, the project's workspace app under `/api/`, nothing else (the
        inner app's own page and its API documentation are never reached)."""
        key = scope.get("path_params", {}).get("project", "")
        root, path = scope.get("root_path", ""), scope["path"]
        # The mount leaves the whole path in the scope and names its own part as the root.
        rel = path[len(root):] if path.startswith(root) else None
        if _KEY.fullmatch(key) is None or rel is None or not (rel == "/" or rel.startswith("/api/")):
            response: Response = _json(404, "Not Found")
        elif scope["method"] != "GET":
            # Before the registry is opened: a workspace app has only GET routes.
            response = _json(405, "Method Not Allowed", {"Allow": "GET"})
        elif rel == "/":
            response = await run_in_threadpool(_page) if _canonical(scope, f"/p/{key}/") else _json(404, "Not Found")
        else:
            try:
                inner = await run_in_threadpool(_workspace_app, db, int(key))
            except _Refused as e:
                response = _json(e.status, e.detail)
            else:
                await inner(scope, receive, send)
                return
        await response(scope, receive, send)

    app.mount("/p/{project}", dispatch)
    return _LocalOnly(app)
