"""Workbench wording, run as the page's own functions in node over what the API returns: the slices
tab's answerability label (#95), the band sliders that cannot cross (#96), LogLoss, small p-values
and the default group (#97), and the policy tab's budget and threshold wording (#125)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.web.app import STATIC_DIR, create_app
from tests.test_web_render import _all_functions, _load, _node

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")

HTML = (STATIC_DIR / "index.html").read_text()


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


def _call(fn: str, *args) -> object:
    js = f"{_all_functions()}\nprocess.stdout.write(JSON.stringify({fn}(...{json.dumps(list(args))})));"
    return _node(js)


# --- #95 answerability -----------------------------------------------------------------------


def _slice(n: int, ci: tuple[float, float], null_ci: tuple[float, float], answerable: bool) -> dict:
    return {"slice": "s", "n": n, "accuracy": sum(ci) / 2, "ci": list(ci), "null_accuracy": sum(null_ci) / 2,
            "null_ci": list(null_ci), "null_kind": "majority", "answerable": answerable}


@pytest.mark.parametrize(
    "m, cls, text",
    [
        pytest.param(_slice(20, (0.80, 0.95), (0.30, 0.60), True), "badge-ok", "ANSWERABLE", id="answerable"),
        pytest.param(_slice(9, (0.80, 0.95), (0.30, 0.60), False), "badge-info", "TOO FEW (n &lt; 10)", id="too-few"),
        pytest.param(_slice(10, (0.80, 0.95), (0.30, 0.60), True), "badge-ok", "ANSWERABLE", id="n-equals-min-n"),
        pytest.param(_slice(10, (0.40, 0.70), (0.30, 0.60), False), "badge-warn", "NOT DISTINGUISHABLE FROM NULL", id="n-equals-min-n-overlap"),
        pytest.param(_slice(20, (0.10, 0.25), (0.30, 0.60), False), "badge-err", "BELOW NULL", id="below"),
        # The demo's adversarial_trap: above the null on the point estimate, CIs overlapping.
        pytest.param(_slice(20, (0.6396, 0.9476), (0.3421, 0.7418), False), "badge-warn", "NOT DISTINGUISHABLE FROM NULL", id="overlap"),
        pytest.param(_slice(20, (0.10, 0.30), (0.30, 0.60), False), "badge-warn", "NOT DISTINGUISHABLE FROM NULL", id="touching-below"),
        pytest.param(_slice(20, (0.60, 0.90), (0.30, 0.60), False), "badge-warn", "NOT DISTINGUISHABLE FROM NULL", id="touching-above"),
    ],
)
def test_answerability_badge(m, cls, text) -> None:
    assert _call("answerabilityBadge", m, 10) == f'<span class="badge {cls}">{text}</span>'


def test_slices_tab_shows_the_null_ci_and_the_rule(ws) -> None:
    client = TestClient(create_app(Path(ws["records"]).parent))
    s, la = client.get("/api/slices").json(), client.get("/api/labelaudit").json()
    assert s["min_n"] == 10
    js = f"""
const written = {{}};
const document = {{getElementById: id => (written[id] ??= {{}})}};
const fetchJson = async url => url === "/api/slices" ? {json.dumps(s)} : {json.dumps(la)};
{_all_functions()}
loadSlicesAndLabels().then(() => process.stdout.write(JSON.stringify(written["slices-container"].innerHTML)));
"""
    html = _node(js)
    row = re.search(r"adversarial_trap</td>(.*?)</tr>", html, re.S).group(1)
    assert "55.0% [34.2%, 74.2%]" in row and "NOT DISTINGUISHABLE FROM NULL" in row
    assert "BELOW NULL CI" not in html
    assert "Answerable: n ≥ 10 and the accuracy CI's lower end above the null CI's upper end." in html


# --- #96 band sliders --------------------------------------------------------------------------


def _policy(payload: dict, *, lo: str = "0.25", hi: str = "0.75", moved: str | None = None, dist: str = "", budget: str = "0.10") -> dict:
    """Run `pushBand` (when a handle moved) and `loadPolicy` with a stub DOM; return the handles, their
    labels, the request's parameters and the policy container's HTML."""
    js = f"""
let policySeq = 0;
let GROUPS = [];
const els = {{}};
const el = id => (els[id] ??= {{value: "", textContent: "", innerHTML: "", style: {{}}}});
Object.assign(el("inp-tau-low"), {{value: {json.dumps(lo)}}});
Object.assign(el("inp-tau-high"), {{value: {json.dumps(hi)}}});
Object.assign(el("inp-cost-s2"), {{value: "0.004"}});
Object.assign(el("inp-err-budget"), {{value: {json.dumps(budget)}}});
Object.assign(el("inp-dist"), {{value: {json.dumps(dist)}}});
Object.assign(el("inp-group"), {{value: ""}});
const document = {{getElementById: el, querySelectorAll: () => [el("band-a"), el("band-b")]}};
const urls = [];
const fetchJson = async url => {{ urls.push(url); return {json.dumps(payload)}; }};
{_all_functions()}
{f"pushBand({json.dumps(moved)});" if moved else ""}
loadPolicy().then(() => process.stdout.write(JSON.stringify({{
  lo: el("inp-tau-low").value, hi: el("inp-tau-high").value,
  lblLo: el("lbl-tau-low").textContent, lblHi: el("lbl-tau-high").textContent,
  urls, html: el("policy-container").innerHTML,
}})));
"""
    out = _node(js)
    assert len(out["urls"]) == 1
    out["params"] = {k: v[0] for k, v in parse_qs(urlsplit(out["urls"][0]).query).items()}
    return out


@pytest.fixture(scope="module")
def noul(ws) -> dict:
    return TestClient(create_app(Path(ws["records"]).parent)).get("/api/simulate-policy").json()


@pytest.mark.parametrize(
    "moved, lo, hi, expected",
    [
        pytest.param("inp-tau-low", "0.80", "0.75", ("0.80", "0.80"), id="low-pushes-high"),
        pytest.param("inp-tau-high", "0.25", "0.20", ("0.20", "0.20"), id="high-pushes-low"),
        pytest.param("inp-tau-low", "0.30", "0.75", ("0.30", "0.75"), id="no-crossing"),
    ],
)
def test_a_handle_moved_past_the_other_pushes_it(noul, moved, lo, hi, expected) -> None:
    out = _policy(noul, lo=lo, hi=hi, moved=moved)
    assert (out["lo"], out["hi"]) == expected
    assert (out["lblLo"], out["lblHi"]) == tuple(f"{float(v):.2f}" for v in expected)
    assert (out["params"]["tau_low"], out["params"]["tau_high"]) == tuple(str(float(v)) for v in expected)


def test_each_slider_has_one_input_listener_that_pushes_before_loading() -> None:
    script = HTML[HTML.index("<script>"):HTML.index("</script>")]
    wiring = re.search(r'\["inp-tau-low", "inp-tau-high", "inp-cost-s2", "inp-err-budget"\]\.forEach\(id => \{.*?\n    \}\);', script, re.S).group(0)
    assert wiring.index("pushBand(id)") < wiring.index("run(loadPolicy")
    assert wiring.count('addEventListener("input"') == 1
    # The page's one other input listener is the table kit's filter box, which never loads.
    rest = script.replace(wiring, "")
    assert rest.count('addEventListener("input"') == 1
    tables = re.search(r'addEventListener\("input", ev => \{.*?\n    \}\);', rest, re.S).group(0)
    assert 'contains("tk-filter")' in tables and "run(" not in tables


# --- #125 policy wording ---------------------------------------------------------------------


def test_the_budget_label_says_what_the_budget_bounds() -> None:
    assert "<label>Error budget (error rate among accepted answers)</label>" in HTML
    assert "1 - Precision" not in HTML


def _note(html: str) -> str:
    return re.search(r"The cutoff <code>(.*?)</code> writes for this group and budget \(empirical error rate\)(.*?)\.</div>", html).groups()


def test_the_threshold_note_names_the_tune_command_for_the_distribution_used(ws, noul) -> None:
    client = TestClient(create_app(Path(ws["records"]).parent))
    raw = client.get("/api/simulate-policy", params={"use_calib": False}).json()
    assert (noul["distribution"], raw["distribution"]) == ("calibrated", "raw")  # the demo's auto is calibrated
    assert _note(_policy(noul)["html"]) == ("s1 policy tune --noul-routing threshold --use-calib", "; it does not bound the escalation band")
    # The note follows the response, not the picker: Raw was asked for here.
    assert _note(_policy(raw, dist="false")["html"]) == ("s1 policy tune --noul-routing threshold", "; it does not bound the escalation band")

    cat = TestClient(create_app(Path(ws["records"]).parent, records_path=ws["records_categorical"], fits_path=ws["fits_categorical"]))
    for primitive in ("choice", "score"):
        auto = cat.get("/api/simulate-policy", params={"primitive": primitive}).json()
        raw = cat.get("/api/simulate-policy", params={"primitive": primitive, "use_calib": False}).json()
        assert (auto["band_supported"], auto["distribution"], raw["distribution"]) == (False, "calibrated", "raw")
        assert _note(_policy(auto)["html"]) == ("s1 policy tune --use-calib", "")
        assert _note(_policy(raw)["html"]) == ("s1 policy tune", "")


# --- #97 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("p, shown", [(0.0, "&lt; 0.0001"), (0.00009, "&lt; 0.0001"), (0.0001, "0.0001"), (0.5, "0.5000"), (None, "N/A")])
def test_fmt_p(p, shown) -> None:
    assert _call("fmtP", p) == shown


def test_the_calibrated_card_shows_logloss_and_nothing_raw_without_a_fit(tmp_path: Path) -> None:
    w = generate_workspace(tmp_path / "ws")
    fitted = TestClient(create_app(tmp_path / "ws")).get("/api/audit").json()
    log_loss = fitted["groups"][0]["after"]["log_loss"]
    assert f"/ LogLoss {log_loss:.4f}" in _load("loadAudit", fitted)
    Path(w["fits"]).unlink()
    unfitted = TestClient(create_app(tmp_path / "ws")).get("/api/audit").json()
    html = _load("loadAudit", unfitted)
    assert re.search(r'Calibrated ECE / Brier / LogLoss</div>\s*<div class="kpi">N/A <span class="sub">/ Brier N/A / LogLoss N/A</span>', html)


def test_the_compare_tab_prints_a_tiny_p_as_below_its_resolution(ws) -> None:
    payload = TestClient(create_app(Path(ws["records"]).parent)).get("/api/compare").json()
    p_values = [q["mcnemar"]["p_holm"] for q in payload["shadow"]["questions"] if q.get("mcnemar")]
    assert p_values and min(p_values) < 0.0001  # the demo's case
    assert "p_holm = &lt; 0.0001" in _load("loadCompare", payload)


def test_the_default_group_entry_says_what_it_picks() -> None:
    assert '<option value="">Default group (first noul, else first)</option>' in HTML
    assert "First noul group" not in HTML
