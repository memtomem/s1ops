"""The workbench serves no generated API pages: FastAPI's `/docs` and `/redoc` load scripts from a CDN,
which would run on the workbench's origin, where `/api/sheet` serves state text to a loopback browser."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.web.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("ws")
    generate_workspace(ws)
    return TestClient(create_app(workspace_dir=ws))


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_no_generated_api_page_is_served(client, path):
    assert client.get(path).status_code == 404

