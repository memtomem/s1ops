"""Policy simulator: it simulates the group asked for, and only noul groups get escalation bands."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.contracts.records import read_records, select
from s1ops.demo import generate_workspace
from s1ops.eval.metrics import top_label
from s1ops.improve.policy import select_threshold, threshold_curve
from s1ops.web.app import STATIC_DIR, create_app


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _mixed_workspace(ws: dict[str, str]) -> Path:
    """records.jsonl holding the noul group and the choice and score groups."""
    workspace = Path(ws["records"]).parent
    with open(ws["records"], "a") as f:
        f.write(Path(ws["records_categorical"]).read_text())
    return workspace


def _groups(path: str) -> dict:
    return {g.primitive: g for g in select(read_records(path, use_calib=True), allow_model_mismatch=True).groups}


def test_a_filter_that_matches_nothing_is_404_not_another_group(ws) -> None:
    client = TestClient(create_app(workspace_dir=Path(ws["records"]).parent))
    for query in ("question_id=q-nope", "model_served=nope", "primitive=choice"):
        r = client.get(f"/api/simulate-policy?{query}")
        assert r.status_code == 404, query
        assert r.json()["detail"] == "No matching labeled group found"


@pytest.mark.parametrize("primitive", ["choice", "score"])
def test_choice_and_score_groups_get_a_threshold_curve_and_no_band(ws, primitive: str) -> None:
    workspace = _mixed_workspace(ws)
    grp = _groups(ws["records"])[primitive]
    client = TestClient(create_app(workspace_dir=workspace))
    query = f"question_id={grp.question_id}&model_served={grp.model_served}&primitive={primitive}&error_budget=0.1"
    d = client.get(f"/api/simulate-policy?{query}").json()
    assert d["group"] == {"question_id": grp.question_id, "model_served": grp.model_served, "primitive": primitive, "n": grp.n}
    assert d["band_supported"] is False
    assert d["active_band"] is None and d["recommended_band_under_35pct_esc"] is None and d["pareto_frontier"] is None
    conf, correct = top_label(grp.P, grp.y)
    assert d["tuned_threshold"]["threshold"] == select_threshold(conf, correct, budget=0.1)
    assert d["threshold_curve"] == threshold_curve(conf, correct, c_s1=0.00004, c_delegate=0.004)


def test_without_a_filter_the_simulator_opens_on_a_noul_group(ws) -> None:
    workspace = _mixed_workspace(ws)
    d = TestClient(create_app(workspace_dir=workspace)).get("/api/simulate-policy").json()
    assert d["group"]["primitive"] == "noul"
    assert d["band_supported"] is True
    assert d["active_band"] is not None and d["pareto_frontier"]


def test_the_noul_simulation_is_unchanged(ws) -> None:
    # Same numbers as before the per-primitive split; only band_supported is new.
    d = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get(
        "/api/simulate-policy?tau_low=0.25&tau_high=0.75"
    ).json()
    assert set(d) == {
        "group", "band_supported", "active_band", "tuned_threshold",
        "recommended_band_under_35pct_esc", "pareto_frontier", "threshold_curve",
    }
    assert d["active_band"]["band"] == "[0.25, 0.75)"
    assert d["active_band"]["n"] == d["group"]["n"]


def test_index_html_wires_the_picker_and_drift_escalation() -> None:
    html = (STATIC_DIR / "index.html").read_text()
    assert 'id="inp-group"' in html
    assert 'fetchJson("/api/groups").then(d => fillGroupPicker(d.groups))' in html
    for param in ("question_id", "model_served", "primitive"):
        assert f'params.set("{param}", picked.{param})' in html
    assert "fmtRate(q.escalation_rate_ref)" in html and "fmtRate(q.escalation_rate_cur)" in html
    # Picker labels are set with textContent, never interpolated into HTML.
    assert re.search(r"option\.textContent = \[g\.question_id, g\.model_served, g\.primitive\]", html)
    # The band card dereferences active_band only after checking it exists.
    assert html.index("const b = d.band_supported ? d.active_band : null;") < html.index("b.escalation_rate * 100")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_index_html_script_parses() -> None:
    import subprocess

    html = (STATIC_DIR / "index.html").read_text()
    script = html[html.index("<script>") + len("<script>"):html.index("</script>")]
    r = subprocess.run(["node", "--check", "-"], input=script, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr


def test_groups_lists_every_labeled_group_for_the_picker(ws) -> None:
    workspace = _mixed_workspace(ws)
    groups = TestClient(create_app(workspace_dir=workspace)).get("/api/groups").json()["groups"]
    expected = _groups(ws["records"])
    assert {g["primitive"]: (g["question_id"], g["model_served"], g["n"]) for g in groups} == {
        p: (g.question_id, g.model_served, g.n) for p, g in expected.items()
    }
    json.dumps(groups)


def test_a_bad_fits_file_does_not_empty_the_picker(ws) -> None:
    workspace = _mixed_workspace(ws)
    Path(ws["fits"]).write_text("{}")
    client = TestClient(create_app(workspace_dir=workspace))
    r = client.get("/api/overview")
    assert r.status_code == 400 and "fits.json" in r.json()["detail"]
    assert len(client.get("/api/groups").json()["groups"]) == 3


def test_noul_band_metrics_match_the_policy_library(ws) -> None:
    from s1ops.improve.policy import Band, DevRecord, Prices, choose, pareto, sweep
    from s1ops.improve.policy import grid as band_grid
    from s1ops.improve.policy import evaluate as evaluate_band

    grp = _groups(ws["records"])["noul"]
    prices = Prices(jev_usd=0.00004, llm_usd=0.01)
    recs = [
        DevRecord(id=item_id, p_jev=float(grp.P[i, 1]), label=bool(grp.y[i] == 1), llm=bool(grp.y[i] == 1),
                  jev_cost_usd=0.00004, llm_cost_usd=0.01)
        for i, (_, item_id) in enumerate(grp.items)
    ]
    swept = sweep(recs, band_grid(0.10), prices)
    d = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get(
        "/api/simulate-policy?tau_low=0.1&tau_high=0.9&cost_sys2=0.01"
    ).json()
    assert d["active_band"] == evaluate_band(recs, Band(0.1, 0.9), prices).as_row()
    assert d["pareto_frontier"] == [r.as_row() for r in pareto(swept, metric="f1")]
    assert d["recommended_band_under_35pct_esc"] == choose(swept, metric="f1", max_escalation_rate=0.35).as_row()


def test_index_html_drops_stale_simulator_responses() -> None:
    # A response for an earlier pick must not overwrite the current one, success or error
    # (checked in headless Chrome by delaying one group's responses; here the guard's wiring).
    html = (STATIC_DIR / "index.html").read_text()
    body = html[html.index("async function loadPolicy()"):html.index("document.getElementById(\"inp-group\").addEventListener")]
    assert "const seq = ++policySeq;" in body
    assert body.count("if (seq !== policySeq) return;") == 2
    assert body.index("if (seq !== policySeq) return;") < body.index("const b = d.band_supported")
