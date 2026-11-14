"""The page's own renderers, run in node over what the API returns for a real workspace."""

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

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's renderers")

NO_FILES = "No shadow comparison records or judge grades found."
NO_PAIRS = "No items shared between the two record files"


def _functions(*names: str) -> str:
    html = (STATIC_DIR / "index.html").read_text()
    script = html[html.index("<script>") + len("<script>"):html.index("</script>")]
    bodies = (re.search(rf"(?:async )?function {n}\(.*?\n    \}}", script, re.S) for n in names)
    return "\n".join(m.group(0) for m in bodies)


def _node(js: str) -> object:
    r = subprocess.run(["node", "-"], input=js, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _load(loader: str, helpers: tuple[str, ...], payload: dict) -> str:
    """Run an async tab loader against a payload and return the HTML it wrote to its container."""
    js = f"""
const written = {{}};
const document = {{getElementById: id => (written[id] ??= {{}})}};
const fetchJson = async () => ({json.dumps(payload)});
{_functions(loader, *helpers)}
{loader}().then(() => process.stdout.write(JSON.stringify(Object.values(written).map(e => e.innerHTML))));
"""
    written = _node(js)
    assert len(written) == 1
    return written[0]


def _verdict(group: dict) -> str:
    return _node(f"{_functions('esc', 'fitVerdictHtml')}\nprocess.stdout.write(JSON.stringify(fitVerdictHtml({json.dumps(group)})));")


def _group(delta: list[float], after_ece: float = 0.05, p95: float = 0.10, direction: str = "no clear direction") -> dict:
    return {
        "after": {"ece_mass": after_ece},
        "delta_ci": {"ece_mass": delta},
        "ece_floor_after": {"mass": {"p95": p95}},
        "direction_after": direction,
    }


@pytest.mark.parametrize(
    "delta, cls, text",
    [
        ([0.024, 0.142], "badge-err", "Fit worsened ECE"),
        ([-0.142, -0.024], "badge-ok", "Fit improved ECE"),
        ([-0.04, 0.056], "badge-info", "No clear ECE change"),
        ([0.0, 0.056], "badge-info", "No clear ECE change"),  # a CI touching zero does not exclude it
        ([-0.056, 0.0], "badge-info", "No clear ECE change"),
    ],
)
def test_fit_verdict_reads_the_delta_ci(delta, cls, text) -> None:
    html = _verdict(_group(delta))
    assert re.search(rf'<span class="badge {cls}">{text} \(', html), html


def test_fit_verdict_compares_the_fitted_ece_with_its_own_floor() -> None:
    above = _verdict(_group([0.0, 0.1], after_ece=0.1514, p95=0.1239, direction="under-confident"))
    assert "Miscalibrated after fit (under-confident)" in above
    assert "floor 0.1239" in above
    within = _verdict(_group([0.0, 0.1], after_ece=0.1239, p95=0.1239))
    assert "Within Noise Floor after fit" in within
    assert "Miscalibrated" not in within


@pytest.mark.parametrize(
    "group",
    [
        {},
        {"after": None},
        {"after": {"ece_mass": 0.1}, "delta_ci": {}, "ece_floor_after": {"mass": {"p95": 0.1}}},
        {"after": {"ece_mass": 0.1}, "delta_ci": {"ece_mass": [0.0, 0.1]}, "ece_floor_after": {}},
    ],
)
def test_fit_verdict_is_absent_without_its_inputs(group) -> None:
    assert _verdict(group) == ""


def test_calibration_tab_shows_the_verdict_for_each_group(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    client = TestClient(
        create_app(
            workspace_dir=tmp_path / "ws",
            records_path=ws["records_categorical"],
            fits_path=ws["fits_categorical"],
        )
    )
    payload = client.get("/api/audit").json()
    html = _load("loadAudit", ("esc", "fitVerdictHtml", "renderReliabilitySvg"), payload)

    cards = {}
    for card in html.split('<div class="card">')[1:]:
        cards[re.search(r">(score|choice)</span>", card).group(1)] = card
    assert set(cards) == {"score", "choice"}
    # The demo's temperature fit, read through the real API: over-softens score, does little for choice.
    assert "Fit worsened ECE" in cards["score"]
    assert "Miscalibrated after fit (under-confident)" in cards["score"]
    assert "No clear ECE change" in cards["choice"]


@pytest.mark.parametrize("n_in_sample, shown", [(0, False), (1, True), (160, True)])
def test_fit_verdict_says_when_records_are_not_held_out(n_in_sample, shown) -> None:
    html = _verdict({**_group([-0.1, -0.01]), "n": 160, "fit": {"n_in_sample": n_in_sample}})
    expected = f"Not held out: {n_in_sample} of 160 records were in the fit's sample"
    assert (expected in html) is shown
    assert ("Not held out" in html) is shown


def _audit_noul(records: list[str], ws: dict[str, str], tmp_path: Path) -> tuple[dict, str]:
    path = tmp_path / "audited.jsonl"
    path.write_text("".join(line + "\n" for line in records))
    client = TestClient(create_app(workspace_dir=tmp_path / "ws", records_path=path, fits_path=ws["fits"]))
    payload = client.get("/api/audit").json()
    html = _load("loadAudit", ("esc", "fitVerdictHtml", "renderReliabilitySvg"), payload)
    assert len(payload["groups"]) == 1
    return payload["groups"][0], html


def test_calibration_tab_counts_the_records_in_the_fit_sample(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    train = Path(ws["records_train"]).read_text().splitlines()
    held_out = Path(ws["records"]).read_text().splitlines()

    group, html = _audit_noul(held_out, ws, tmp_path)
    assert group["fit"]["n_in_sample"] == 0
    assert "Not held out" not in html

    group, html = _audit_noul(train, ws, tmp_path)
    assert group["fit"]["n_in_sample"] == len(train) == group["n"]
    assert f"Not held out: {len(train)} of {len(train)} records" in html

    group, html = _audit_noul(train[:3] + held_out, ws, tmp_path)
    assert group["fit"]["n_in_sample"] == 3
    assert f"Not held out: 3 of {3 + len(held_out)} records" in html


def test_calibration_tab_has_no_verdict_without_a_fit(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    Path(ws["fits"]).unlink()
    payload = TestClient(create_app(workspace_dir=tmp_path / "ws")).get("/api/audit").json()
    html = _load("loadAudit", ("esc", "fitVerdictHtml", "renderReliabilitySvg"), payload)
    assert "No fit applied" in html
    assert "after fit" not in html
    assert "ECE (Δ" not in html


def _compare_tab(workspace: Path) -> str:
    payload = TestClient(create_app(workspace_dir=workspace)).get("/api/compare").json()
    return _load("loadCompare", ("esc", "fmtKappa", "fmtP"), payload)


def test_compare_tab_says_when_it_has_no_files(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    Path(ws["records_kev"]).unlink()
    Path(ws["judge_grades"]).unlink()
    html = _compare_tab(tmp_path / "ws")
    assert NO_FILES in html
    assert NO_PAIRS not in html


def test_compare_tab_tells_no_shared_items_from_no_files(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    Path(ws["judge_grades"]).unlink()
    rows = [json.loads(line) for line in Path(ws["records_kev"]).read_text().splitlines()]
    n = len(rows)
    for row in rows:
        row["item_id"] = "other-" + row["item_id"]
    Path(ws["records_kev"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
    html = _compare_tab(tmp_path / "ws")
    assert f"{NO_PAIRS} ({n} primary only, {n} shadow only)" in html
    assert NO_FILES not in html


def test_compare_tab_shows_judge_alone_without_a_notice(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    Path(ws["records_kev"]).unlink()
    html = _compare_tab(tmp_path / "ws")
    assert "Rubric Judge Audit" in html
    assert NO_FILES not in html
    assert NO_PAIRS not in html


def test_compare_tab_shows_both_sections_with_no_notice(tmp_path: Path) -> None:
    generate_workspace(tmp_path / "ws")
    html = _compare_tab(tmp_path / "ws")
    assert "Shadow Comparison:" in html
    assert "Rubric Judge Audit" in html
    assert NO_FILES not in html
    assert NO_PAIRS not in html
