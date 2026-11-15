"""Web workbench robustness: malformed workspace files are HTTP 400 naming the file; optional files stay optional."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.web.app import STATIC_DIR, create_app

HOSTILE = "<img src=x onerror=alert(1)>"

# Server fields that carry identifiers or free text from workspace files.
_TEXT_FIELDS = (
    "question_id", "model_served", "primitive", "item_id", "dataset", "label", "predicted",
    "slice", "feature", "message", "fix", "rule", "question", "probe_question", "band",
    "primary_model", "shadow_model", "status", "method", "direction", "null_kind",
    "inverse_source", "direction_after", "metric", "series", "lane", "caption",
    "title", "evidence", "evidence_url", "criterion", "primary_pred", "shadow_pred",
)
_SAFE_INTERPOLATIONS = {
    "res.status",  # numeric HTTP status, only ever reaches the page via showError's esc()
}


def _statement_before(script: str, pos: int) -> str:
    """The statement an interpolation at `pos` belongs to, up to it."""
    start = max(script.rfind(";", 0, pos), script.rfind("{\n", 0, pos), script.rfind("}\n", 0, pos))
    return script[start + 1:pos]


def _leaks(script: str) -> list[str]:
    """Interpolations naming a server text field without esc(). One in a `.textContent =`
    assignment is text, never markup, so it is exempt there and nowhere else."""
    leaks = []
    for m in re.finditer(r"\$\{([^{}]*)\}", script):
        e = m.group(1)
        if "esc(" in e or e.strip() in _SAFE_INTERPOLATIONS or ".textContent =" in _statement_before(script, m.start()):
            continue
        if any(re.search(rf"\.{f}\b", e) for f in _TEXT_FIELDS):
            leaks.append(e)
    return leaks


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _client(ws: dict[str, str]) -> TestClient:
    return TestClient(create_app(workspace_dir=Path(ws["records"]).parent))


def _break_line(path: str, line_no: int, text: str = "{not json") -> None:
    lines = Path(path).read_text().splitlines()
    lines[line_no - 1] = text
    Path(path).write_text("\n".join(lines) + "\n")


def test_web_malformed_records_returns_400(ws) -> None:
    _break_line(ws["records"], 3)
    r = _client(ws).get("/api/audit?n_boot=20")
    assert r.status_code == 400
    assert "records.jsonl" in r.json()["detail"]
    assert "line 3" in r.json()["detail"]


def test_web_duplicate_record_returns_400(ws) -> None:
    lines = Path(ws["records"]).read_text().splitlines()
    Path(ws["records"]).write_text("\n".join(lines + [lines[0]]) + "\n")
    client = _client(ws)
    for url in ("/api/audit?n_boot=20", "/api/simulate-policy"):
        r = client.get(url)
        assert r.status_code == 400, url
        assert "appears twice" in r.json()["detail"]


def test_web_malformed_probe_and_canary_return_400(ws) -> None:
    _break_line(ws["probe_records"], 2)
    _break_line(ws["records_canary"], 5)
    client = _client(ws)
    r_lint = client.get("/api/lint")
    assert r_lint.status_code == 400
    assert "probe_records.jsonl" in r_lint.json()["detail"]
    r_drift = client.get("/api/drift")
    assert r_drift.status_code == 400
    assert "records_canary.jsonl" in r_drift.json()["detail"]


def test_web_null_feature_row_returns_400(ws) -> None:
    _break_line(ws["features"], 1, "null")
    r = _client(ws).get("/api/slices")
    assert r.status_code == 400
    assert "line 1: expected a JSON object" in r.json()["detail"]


def test_web_judge_missing_keys_returns_400(ws) -> None:
    Path(ws["judge_grades"]).write_text('{"criterion": "tone", "judge": 2}\n')
    r = _client(ws).get("/api/compare")
    assert r.status_code == 400
    assert "must be non-negative integers" in r.json()["detail"]


def test_web_compare_without_judge_file_returns_null_judge(ws) -> None:
    Path(ws["judge_grades"]).unlink()
    r = _client(ws).get("/api/compare")
    assert r.status_code == 200
    assert r.json()["judge"] is None
    assert len(r.json()["shadow"]["questions"]) >= 1


def test_web_overview_lists_missing_files(ws) -> None:
    Path(ws["records_canary"]).unlink()
    r = _client(ws).get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["missing_files"] == ["records_canary.jsonl"]
    assert body["files"]["records_canary"] is None


def test_create_app_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_app(workspace_dir=tmp_path / "missing")


def test_web_escapes_hostile_identifiers(ws) -> None:
    client = _client(ws)
    top = client.get("/api/labelaudit?top_k=15").json()["top_flags"][0]["item_id"]
    old, new = json.dumps({"item_id": top})[1:-1], json.dumps({"item_id": HOSTILE})[1:-1]
    for name in ("records", "features"):
        text = Path(ws[name]).read_text()
        assert old in text, name
        Path(ws[name]).write_text(text.replace(old, new))

    flags = client.get("/api/labelaudit?top_k=15").json()["top_flags"]
    assert HOSTILE in [f["item_id"] for f in flags]

    html = (STATIC_DIR / "index.html").read_text()
    script = html[html.index("<script>"):]
    assert "function esc(" in script
    assert _leaks(script) == []


def test_the_scan_exempts_an_error_message_only_where_it_is_text() -> None:
    html = (STATIC_DIR / "index.html").read_text()
    script = html[html.index("<script>"):]
    old = 'getElementById("overview-meta").textContent = `Workspace error: ${err.message}`'
    assert script.count(old) == 1
    as_markup = script.replace(old, 'getElementById("overview-meta").innerHTML = `Workspace error: ${err.message}`')
    assert _leaks(as_markup) == ["err.message"]


def test_index_html_has_no_external_refs() -> None:
    html = (STATIC_DIR / "index.html").read_text()
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src" not in html
    assert "<link" not in html


@pytest.mark.parametrize("row", ['{"judge": -1, "human": 2}', '{"judge": 0, "human": 0}'])
def test_web_unscorable_judge_grades_return_400(ws, row) -> None:
    Path(ws["judge_grades"]).write_text(row + "\n")
    r = _client(ws).get("/api/compare")
    assert r.status_code == 400
    assert "judge_grades.jsonl" in r.json()["detail"]


def test_web_errors_name_the_file_that_is_broken(ws) -> None:
    _break_line(ws["records"], 3)
    client = _client(ws)
    for url in ("/api/compare", "/api/drift"):
        detail = client.get(url).json()["detail"]
        assert detail.startswith(str(Path(ws["records"]))), (url, detail)
        assert "line 3" in detail


@pytest.mark.parametrize(
    "target, exc",
    [("compare_records", KeyError), ("compare_records", ValueError), ("judge_agreement", ValueError)],
)
def test_web_server_bug_stays_500(ws, monkeypatch, target, exc) -> None:
    import s1ops.web.app as web_app

    def broken(*args, **kwargs):
        raise exc(f"bug in {target}")

    monkeypatch.setattr(web_app, target, broken)
    client = TestClient(create_app(workspace_dir=Path(ws["records"]).parent), raise_server_exceptions=False)
    assert client.get("/api/compare").status_code == 500


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to execute the page's esc()")
def test_esc_neutralizes_markup() -> None:
    html = (STATIC_DIR / "index.html").read_text()
    script = html[html.index("<script>") + len("<script>"):html.index("</script>")]
    body = re.search(r"function esc\(v\) \{.*?\n    \}", script, re.S).group(0)
    js = f"{body}\nprocess.stdout.write(JSON.stringify([esc({json.dumps(HOSTILE)}), esc(null), esc(`a&'b\"`)]));"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == ["&lt;img src=x onerror=alert(1)&gt;", "", "a&amp;&#39;b&quot;"]


def test_web_unpairable_shadow_returns_400(ws) -> None:
    lines = Path(ws["records_kev"]).read_text().splitlines()
    Path(ws["records_kev"]).write_text("\n".join(lines + [lines[0]]) + "\n")
    r = _client(ws).get("/api/compare")
    assert r.status_code == 400
    assert r.json()["detail"].startswith("records.jsonl vs records_kev.jsonl: duplicate record key")


def test_web_constant_grades_report_undefined_kappa(ws) -> None:
    Path(ws["judge_grades"]).write_text(
        '{"criterion": "a", "judge": 1, "human": 1}\n{"criterion": "b", "judge": 2, "human": 2}\n'
        '{"criterion": "a", "judge": 1, "human": 1}\n{"criterion": "b", "judge": 2, "human": 2}\n'
    )
    r = _client(ws).get("/api/compare")
    assert r.status_code == 200
    judge = r.json()["judge"]
    assert {k: v["weighted_kappa"] for k, v in judge["by_criterion"].items()} == {"a": None, "b": None}
    assert judge["overall"]["weighted_kappa"] is not None  # defined across criteria
    html = (STATIC_DIR / "index.html").read_text()
    assert "weighted_kappa.toFixed" not in html
