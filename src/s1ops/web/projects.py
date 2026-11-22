"""The workbench over a project registry (`s1 web --db`): every registered project's workspace under
`/p/<id>/`, each request answered by a `create_app` made for that project's directory, and the list of
projects at `/p/`. It reads the registry and writes nothing.

Everything here is served to this computer only (`loopback_request_allowed`), since the list holds
absolute paths, and at the root of its origin only: the page reads which project it shows from its
address, so a page is answered at `/p/` and `/p/<id>/` exactly as written and nowhere else.

The key in the address is a project's registration number, which the registry never gives twice, so an
address cannot come to name another directory. A page is answered without reading the registry (an
address of no project opens a workbench whose every request says so); a project's `/api/` requests read
it each time, so a project added, renamed or removed by `s1 project` shows on the next request.
"""

from __future__ import annotations

import os
import re
import stat
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from s1ops import __version__, registry
from s1ops.contracts.records import surrogate_problem
from s1ops.web.app import STATIC_DIR, create_app, loopback_request_allowed

REGISTRY_REFUSED = (
    "a project registry's workbench is served to this computer only: a loopback client address, a Host "
    "of 127.0.0.1 or localhost, and no proxy headers"
)
PREFIX_REFUSED = "a project registry's workbench is served at the root of its address, not under a prefix"
NO_FRAMES = ((b"content-security-policy", b"frame-ancestors 'none'"), (b"x-frame-options", b"DENY"))
"""On every response: no other site may put these pages in a frame."""

# Every positive integer SQLite holds has at most 19 digits. A key is compared as a Python integer and
# never bound, so one past that range is only a project that does not exist.
_KEY = re.compile(r"[1-9][0-9]{0,18}")
_STATUS = {"missing": 404, "busy": 503}


class _Refused(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _json(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


def _refusal(e: registry.RegistryError) -> _Refused:
    return _Refused(_STATUS.get(e.kind, 400), str(e))


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
    """The outermost callable: refuses every request that is not this computer's own before anything
    behind it runs, and marks every response, a 500 included, as not to be framed. It wraps the app
    rather than being one of its middlewares, which sit inside the handler that writes a 500."""

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

    @app.get("/api/registry/projects")
    def api_projects() -> dict[str, Any]:
        """The registered projects in id order. An id is a decimal string here: a browser reads a JSON
        number as a float, which cannot tell two ids past 2**53 apart."""
        try:
            projects = registry.list_projects(db)
        except registry.RegistryError as e:
            refused = _refusal(e)
            raise HTTPException(status_code=refused.status, detail=refused.detail) from None
        return {"registry": {"path": db}, "projects": [{**p.as_dict(), "id": str(p.id)} for p in projects]}

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
