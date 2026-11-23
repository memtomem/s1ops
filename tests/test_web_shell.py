"""The workbench shell: the view state kept in the URL hash, how a chosen group is resolved before
any group view requests, the per-view question pickers, the overview, and how failures are shown;
plus the group filters the shell sends to /api/audit, /api/slices and /api/labelaudit."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.web.app import STATIC_DIR, create_app
from tests.test_web_render import _all_functions, _node

HTML = (STATIC_DIR / "index.html").read_text()
HOSTILE = "\"'><img src=x onerror=alert(1)>"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")


def _js(expr: str, prelude: str = "") -> object:
    """Evaluate `expr` with every page function loaded; `prelude` declares what it needs."""
    return _node(f"{prelude}\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


# --- the hash ----------------------------------------------------------------------------------


@needs_node
def test_the_default_state_writes_no_hash() -> None:
    assert _js("formatHash(defaultState())") == ""
    assert _js('parseHash("")') == _js("defaultState()")


@needs_node
def test_a_state_round_trips_through_the_hash_with_only_what_differs() -> None:
    state = {"view": "policy", "qid": "q-1", "model": "m/1 b", "prim": "choice", "dq": "q-1", "cq": "", "ck": "",
             "dist": "true", "lo": 0.3, "hi": 0.75, "cs1": 0.0002, "cs2": 0.01, "budget": 0.05, "nboot": 1000, "cbins": "mass",
             "fdrq": 0.1, "topk": 15, "item": 3}
    hash_ = _js(f"formatHash({json.dumps(state)})")
    assert hash_.startswith("#/policy?")
    keys = [k for k, _ in (pair.split("=") for pair in hash_.split("?", 1)[1].split("&"))]
    assert keys == ["qid", "model", "prim", "dq", "dist", "lo", "cs1", "cs2", "budget", "nboot", "item"]  # cq, hi, cbins, fdrq and topk are defaults
    assert _js(f"parseHash({json.dumps(hash_)})") == state


@needs_node
@pytest.mark.parametrize(
    "hash_, field, expected",
    [
        ("#/nowhere", "view", "overview"),
        ("#/drift", "view", "drift"),
        ("drift", "view", "drift"),
        ("#/checkpoints?ck=tickets-r1", "view", "checkpoints"),
        ("#/checkpoints?ck=tickets-r1", "ck", "tickets-r1"),
        ("#/checkpoints", "ck", ""),
        ("#/policy?lo=0", "lo", 0.05),
        ("#/policy?hi=7", "hi", 0.95),
        ("#/policy?lo=NaN", "lo", 0.25),
        ("#/policy?lo=Infinity", "lo", 0.25),
        ("#/policy?lo=abc", "lo", 0.25),
        ("#/policy?budget=-1", "budget", 0),
        ("#/policy?cs2=1e400", "cs2", 0.004),
        ("#/policy?cs1=1e400", "cs1", 0.00004),
        ("#/policy?cs1=-3", "cs1", 0),
        ("#/policy?cs1=4e-7", "cs1", 4e-7),
        ("#/policy?cs1=", "cs1", 0.00004),
        ("#/policy?dist=maybe", "dist", ""),
        ("#/policy?dist=false", "dist", "false"),
    ],
)
def test_the_hash_is_checked_and_clamped(hash_, field, expected) -> None:
    assert _js(f"parseHash({json.dumps(hash_)})")[field] == expected


@needs_node
@pytest.mark.parametrize(
    "hash_, band",
    [
        ("#/policy?lo=0.9&hi=0.1", [0.1, 0.9]),  # crossed: the band the server would evaluate
        ("#/policy?lo=0.33&hi=0.71", [0.35, 0.7]),  # on the sliders' step
        ("#/policy?lo=0.5&hi=0.5", [0.5, 0.5]),
    ],
)
def test_the_hash_band_is_the_one_evaluated(hash_, band) -> None:
    st = _js(f"parseHash({json.dumps(hash_)})")
    assert [st["lo"], st["hi"]] == band


def _from_inputs(**values) -> dict | None:
    base = {"dist": "", "lo": "0.25", "hi": "0.75", "cs1": "0.00004", "cs2": "0.004", "budget": "0.10", **values}
    st = {"view": "policy", "qid": "q", "model": "m", "prim": "noul", "dq": "q", "cq": "q", "ck": "tickets-r1",
          "dist": "", "lo": 0.25, "hi": 0.75, "cs1": 0.00004, "cs2": 0.004, "budget": 0.1, "nboot": 200, "cbins": "mass",
          "fdrq": 0.1, "topk": 15, "item": 1}
    return _js(f"policyStateFromInputs({json.dumps(st)}, {json.dumps(base)})")


@needs_node
@pytest.mark.parametrize("values", [{"cs2": ""}, {"budget": " "}, {"cs2": "abc"}, {"budget": "1e400"}, {"cs1": ""}, {"cs1": "x"}])
def test_a_number_mid_edit_changes_nothing(values) -> None:
    assert _from_inputs(**values) is None


@needs_node
def test_the_inputs_go_through_the_hashs_rule() -> None:
    st = _from_inputs(budget="2", cs1="0.0003", cs2="-1", lo="0.8", hi="0.3", dist="true")
    assert (st["budget"], st["cs1"], st["cs2"], st["lo"], st["hi"], st["dist"]) == (1, 0.0003, 0, 0.3, 0.8, "true")
    # The group and the views' questions are kept, and the state is what its own hash reads back.
    assert (st["qid"], st["dq"], st["view"]) == ("q", "q", "policy")
    assert _js(f"parseHash(formatHash({json.dumps(st)}))") == st


# --- resolving the chosen group -----------------------------------------------------------------

GROUPS = [
    {"question_id": "q-a", "model_served": "m", "primitive": "noul", "n": 10},
    {"question_id": "q-b", "model_served": "m", "primitive": "choice", "n": 10},
]


def _resolve(qid: str, model: str, prim: str, groups_state: str = "ok") -> dict:
    st = {"qid": qid, "model": model, "prim": prim}
    return _js(f"resolveGroup({json.dumps(st)}, {json.dumps(GROUPS)}, {json.dumps(groups_state)})")


@needs_node
@pytest.mark.parametrize(
    "triple, groups_state, status, index",
    [
        (("", "", ""), "loading", "none", -1),
        (("q-b", "m", "choice"), "ok", "ok", 1),
        (("q-b", "m", ""), "ok", "invalid", -1),
        (("q-b", "m", "ordinal"), "ok", "invalid", -1),
        (("q-b", "m", "choice"), "loading", "loading", -1),
        (("q-b", "m", "choice"), "error", "error", -1),
        (("q-b", "m", "noul"), "ok", "unknown", -1),
    ],
)
def test_a_chosen_group_resolves_only_to_itself(triple, groups_state, status, index) -> None:
    assert _resolve(*triple, groups_state=groups_state) == {"status": status, "index": index}


@needs_node
def test_a_group_view_requests_only_with_no_group_or_its_own() -> None:
    ready = _js('["none", "ok", "loading", "error", "invalid", "unknown"].map(s => groupReady({status: s}))')
    assert ready == [True, True, False, False, False, False]
    st = {"qid": "q-b", "model": "m", "prim": "choice"}
    assert _js(f'groupParams({json.dumps(st)}, {{status: "ok"}})') == {"question_id": "q-b", "model_served": "m", "primitive": "choice"}
    assert _js(f'groupParams({json.dumps(st)}, {{status: "unknown"}})') == {}


@needs_node
def test_an_unresolved_group_shows_no_identifier_from_the_link() -> None:
    for status in ("loading", "error", "invalid", "unknown"):
        html = _js(f'groupBlockedHtml({{status: {json.dumps(status)}}})')
        assert "Nothing is requested until the group is resolved" in html
    # The note names the link's triple, and the bar writes it with textContent.
    note = _js(f'filterNote("calib", {{status: "unknown"}}, {json.dumps({"qid": HOSTILE, "model": "m", "prim": "noul"})})')
    assert HOSTILE in note
    assert 'getElementById("flt-note").textContent =' in HTML


@needs_node
def test_api_urls_carry_only_set_parameters() -> None:
    assert _js('apiUrl("/api/slices", {})') == "/api/slices"
    assert _js('apiUrl("/api/slices", undefined)') == "/api/slices"
    assert _js('apiUrl("/api/audit", {question_id: "q a&b", model_served: "", primitive: null})') == "/api/audit?question_id=q+a%26b"


@needs_node
def test_the_note_says_why_the_group_filter_does_not_apply() -> None:
    st = {"qid": "", "model": "", "prim": ""}
    for view in ("overview", "lint", "compare", "drift", "checkpoints"):
        assert _js(f'filterNote({json.dumps(view)}, {{status: "none"}}, {json.dumps(st)})').startswith("The group filter does not apply here:")
    assert "default group (first noul, else first)" in _js(f'filterNote("policy", {{status: "none"}}, {json.dumps(st)})')
    assert _js(f'filterNote("calib", {{status: "none"}}, {json.dumps(st)})') == "All labeled groups."


@needs_node
def test_arrow_keys_wrap_around_the_views() -> None:
    assert _js('["ArrowDown", "ArrowUp", "Home", "End", "x"].map(k => nextViewIndex(k, 0, 7))') == [1, 6, 0, 6, -1]
    assert _js('nextViewIndex("ArrowRight", 6, 7)') == 0


# The views' tablist, exactly. One flat tablist of the nine tabs in step order; each section's name is an
# `aria-hidden` sibling, and each tab but the overview names its section by `aria-describedby`, so the
# tablist owns only tabs and the arrow keys (`navKey`, which selects `[role=tab]`) walk all nine. What the
# accessibility tree makes of it is shown only in a browser (Chrome's: nine tabs, each section name as the
# description); this pins the markup that was checked there, so any change to it is a reviewed edit here.
NAV_MARKUP = "\n".join([
    '    <nav class="views" role="tablist" aria-label="Workbench views" aria-orientation="vertical">',
    '      <button type="button" role="tab" id="nav-overview" data-view="overview" aria-controls="tab-overview" aria-selected="true" tabindex="0"><span class="nav-step">0</span>Overview</button>',
    '      <span class="nav-section" id="navsec-route" aria-hidden="true">Calibrate &amp; route</span>',
    '      <button type="button" role="tab" id="nav-calib" data-view="calib" aria-controls="tab-calib" aria-selected="false" tabindex="-1" aria-describedby="navsec-route"><span class="nav-step">1</span>Calibration &amp; Noise Floor</button>',
    '      <button type="button" role="tab" id="nav-policy" data-view="policy" aria-controls="tab-policy" aria-selected="false" tabindex="-1" aria-describedby="navsec-route"><span class="nav-step">2</span>Threshold &amp; Escalation Simulator</button>',
    '      <span class="nav-section" id="navsec-questions" aria-hidden="true">Questions &amp; labels</span>',
    '      <button type="button" role="tab" id="nav-lint" data-view="lint" aria-controls="tab-lint" aria-selected="false" tabindex="-1" aria-describedby="navsec-questions"><span class="nav-step">3</span>Question Lint &amp; Probe Coherence</button>',
    '      <button type="button" role="tab" id="nav-slices" data-view="slices" aria-controls="tab-slices" aria-selected="false" tabindex="-1" aria-describedby="navsec-questions"><span class="nav-step">4</span>Slices &amp; Blind Label Audit</button>',
    '      <span class="nav-section" id="navsec-models" aria-hidden="true">Models over time</span>',
    '      <button type="button" role="tab" id="nav-compare" data-view="compare" aria-controls="tab-compare" aria-selected="false" tabindex="-1" aria-describedby="navsec-models"><span class="nav-step">5</span>Shadow Compare &amp; Judge Audit</button>',
    '      <button type="button" role="tab" id="nav-drift" data-view="drift" aria-controls="tab-drift" aria-selected="false" tabindex="-1" aria-describedby="navsec-models"><span class="nav-step">6</span>Drift &amp; Canary Monitor</button>',
    '      <button type="button" role="tab" id="nav-checkpoints" data-view="checkpoints" aria-controls="tab-checkpoints" aria-selected="false" tabindex="-1" aria-describedby="navsec-models"><span class="nav-step">7</span>Checkpoints &amp; Training Runs</button>',
    '      <span class="nav-section" id="navsec-review" aria-hidden="true">Human review</span>',
    '      <button type="button" role="tab" id="nav-review" data-view="review" aria-controls="tab-review" aria-selected="false" tabindex="-1" aria-describedby="navsec-review"><span class="nav-step">8</span>Blind Sheet Review</button>',
    '    </nav>',
])


def _live_nav(html: str) -> str:
    """The views' nav block, after comments are dropped (a commented-out block is not the live one); the page
    has one nav and one tablist."""
    live = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    navs = re.findall(r"<nav\b", live, re.I)
    tablists = re.findall(r"""(?<![\w-])role\s*=\s*["']?tablist\b""", live, re.I)
    assert len(navs) == 1 and len(tablists) == 1
    nav = re.search(r'    <nav class="views" role="tablist".*?</nav>', live, re.S)
    assert nav is not None
    return nav.group(0)


def test_the_views_are_grouped_under_hidden_section_labels() -> None:
    assert _live_nav(HTML) == NAV_MARKUP
    nav = re.search(r'    <nav class="views".*?</nav>', HTML, re.S).group(0)
    for mutant in (
        HTML.replace(nav, "<!--\n" + nav + "\n-->", 1),
        HTML.replace(nav, nav + '\n    <nav role="tablist" class="views"><button role="tab">x</button></nav>', 1),
        HTML.replace(nav, nav + "\n    <div role='tablist'><button role='tab'>x</button></div>", 1),
        HTML.replace(nav, nav + "\n    <div role=tablist><button role=tab>x</button></div>", 1),
        HTML.replace(nav, nav + '\n    <div ROLE="TABLIST"><button role="tab">x</button></div>', 1),
    ):
        with pytest.raises(AssertionError):
            _live_nav(mutant)
    # An attribute that only ends in "role" is not a second tablist.
    assert _live_nav(HTML.replace(nav, nav + '\n    <div data-role="tablist"></div>', 1)) == NAV_MARKUP


@needs_node
def test_the_theme_cycles_system_light_dark() -> None:
    assert _js('["system", "light", "dark", "bogus"].map(nextTheme)') == ["light", "dark", "system", "system"]


# --- per-view question pickers ------------------------------------------------------------------


def _drift(ws) -> dict:
    return TestClient(create_app(Path(ws["records"]).parent)).get("/api/drift").json()


@needs_node
def test_drift_shows_the_chosen_question_and_the_whole_reports_summary(ws) -> None:
    d = _drift(ws)
    qid = d["questions"][0]["question_id"]
    two = {**d, "questions": d["questions"] + [{**d["questions"][0], "question_id": "q-other"}]}
    shown = _js(f"renderDrift({json.dumps(two)}, {json.dumps(qid)})")
    assert shown.count("Drift Monitor:") == 1
    assert f'value="{qid}" selected' in shown
    sm = d["summary"]
    assert f"Every question in the report: {sm['ok']} OK · {sm['warn']} WARN · {sm['critical']} CRITICAL." in shown
    assert _js(f'renderDrift({json.dumps(two)}, "")').count("Drift Monitor:") == 2


@needs_node
def test_a_question_the_report_lacks_is_said_so_never_swapped(ws) -> None:
    html = _js(f"renderDrift({json.dumps(_drift(ws))}, {json.dumps(HOSTILE)})")
    assert "Drift Monitor:" not in html
    assert "No question &quot;&#39;&gt;&lt;img" in html
    assert "<img" not in html


@needs_node
def test_compare_filters_the_shadow_questions_and_says_the_judge_is_unfiltered(ws) -> None:
    d = TestClient(create_app(Path(ws["records"]).parent)).get("/api/compare").json()
    qid = d["shadow"]["questions"][0]["question_id"]
    html = _js(f"renderCompare({json.dumps(d)}, {json.dumps(qid)})")
    assert "Shadow Comparison:" in html and "Rubric Judge Audit" in html
    assert "the question filter does not apply to it" in html
    assert "the question filter does not apply" not in _js(f'renderCompare({json.dumps(d)}, "")')
    missing = _js(f'renderCompare({json.dumps(d)}, "q-none")')
    assert "No question q-none in the shadow comparison." in missing and "Shadow Comparison:" not in missing


# --- overview -----------------------------------------------------------------------------------


@needs_node
def test_the_overview_names_each_missing_file_and_what_writes_it(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    Path(ws["fits"]).unlink()
    Path(ws["records_kev"]).unlink()
    data = TestClient(create_app(tmp_path / "ws")).get("/api/overview").json()
    html = _js(f"renderOverview({json.dumps(data)})")
    assert "7 / 9" in html
    rows = dict(re.findall(r'<td class="mono">([\w.]+)</td>\s*<td>(.*?)</td>', html, re.S))
    assert "missing" in rows["fits.json"] and "missing" in rows["records_kev.jsonl"]
    assert "present" in rows["records.jsonl"]
    assert "<code>s1 fit</code>" in html
    assert '<span class="badge badge-none">no fit</span>' in html


@needs_node
def test_the_overview_shows_a_present_files_own_name(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    data = TestClient(create_app(tmp_path / "ws", records_path=ws["records_categorical"])).get("/api/overview").json()
    assert "records_categorical.jsonl" in _js(f"renderOverview({json.dumps(data)})")


# --- failures -----------------------------------------------------------------------------------


def _shown(status: int | None) -> str:
    err = f"Object.assign(new Error({json.dumps(HOSTILE)}), {{status: {json.dumps(status)}}})" if status else f"new Error({json.dumps(HOSTILE)})"
    js = f"""
const written = {{}};
const document = {{getElementById: id => (written[id] ??= {{}})}};
{_all_functions()}
showError("c", {err});
process.stdout.write(JSON.stringify(written.c.innerHTML));
"""
    return _node(js)


@needs_node
@pytest.mark.parametrize(
    "status, marker, hint",
    [
        (400, "ERROR", "A file or directory could not be used: fix the one named above, then Refresh."),
        (422, "ERROR", "fix the one named above, then Refresh"),
        (500, "ERROR", "has the traceback"),
        (503, "ERROR", "The project registry was busy"),
        (None, "ERROR", "is <code>s1 web</code> still running?"),
        (404, "Nothing to show.", None),
    ],
)
def test_each_kind_of_failure_says_what_to_do(status, marker, hint) -> None:
    html = _shown(status)
    assert marker in html
    assert "<img" not in html and "&lt;img" in html
    if hint:
        assert hint in html
    else:
        assert "ERROR" not in html


# --- the group filters the shell sends ------------------------------------------------------------


@pytest.fixture(scope="module")
def cat(ws) -> TestClient:
    return TestClient(create_app(Path(ws["records"]).parent, records_path=ws["records_categorical"], fits_path=ws["fits_categorical"]))


def test_audit_and_slices_take_the_primitive(cat) -> None:
    for url, base in (("/api/audit", {"n_boot": 20}), ("/api/slices", {})):
        every = cat.get(url, params=base).json()["groups"]
        assert {g["primitive"] for g in every} == {"choice", "score"}
        for primitive in ("choice", "score"):
            only = cat.get(url, params={**base, "primitive": primitive}).json()["groups"]
            assert [g["primitive"] for g in only] == [primitive]
            assert only == [g for g in every if g["primitive"] == primitive]


def test_label_audit_ranks_one_groups_records(cat) -> None:
    groups = cat.get("/api/groups").json()["groups"]
    every = cat.get("/api/labelaudit", params={"top_k": 100}).json()
    assert every["n_total_candidates"] == sum(g["n"] for g in groups)
    for g in groups:
        params = {"question_id": g["question_id"], "model_served": g["model_served"], "primitive": g["primitive"], "top_k": 100}
        one = cat.get("/api/labelaudit", params=params).json()
        assert one["n_total_candidates"] == g["n"]
        assert {f["question_id"] for f in one["top_flags"]} == {g["question_id"]}
        # The other primitive's group is left out by the primitive alone.
        alone = cat.get("/api/labelaudit", params={"primitive": g["primitive"], "top_k": 100}).json()
        assert alone["n_total_candidates"] == g["n"]


def test_label_audit_for_no_matching_group_is_empty_not_an_error(cat) -> None:
    r = cat.get("/api/labelaudit", params={"question_id": "q-none"})
    assert r.status_code == 200
    assert r.json() == {"n_total_candidates": 0, "skipped": {"no_response": 0, "unlabeled": 0}, "distribution": "calibrated", "top_flags": []}


def test_an_empty_filter_value_is_no_filter(ws) -> None:
    client = TestClient(create_app(Path(ws["records"]).parent))
    unfiltered = client.get("/api/labelaudit").json()
    assert unfiltered["n_total_candidates"] == 160
    assert client.get("/api/labelaudit?question_id=&model_served=&primitive=").json() == unfiltered
    audit = client.get("/api/audit?n_boot=20").json()
    assert client.get("/api/audit?n_boot=20&question_id=&model_served=&primitive=").json() == audit
    assert client.get("/api/slices?primitive=").json() == client.get("/api/slices").json()


# --- every control has a name ------------------------------------------------------------------


def _unnamed_controls(html: str) -> list[str]:
    """The `<input>` / `<select>` / `<textarea>` tags of the page, static or in a template, that a
    screen reader would announce by role alone: no `aria-label`, no `aria-labelledby`, and no
    `<label for>` naming their id."""
    labelled = set(re.findall(r'<label for="([^"]+)"', html))
    unnamed = []
    for tag in re.findall(r"<(?:input|select|textarea)\b[^>]*>", html):
        ident = re.search(r'\bid="([^"]+)"', tag)
        if "aria-label" in tag or (ident and ident.group(1) in labelled):
            continue
        unnamed.append(tag)
    return unnamed


def test_every_control_has_an_accessible_name() -> None:
    html = (STATIC_DIR / "index.html").read_text()
    assert len(re.findall(r"<(?:input|select|textarea)\b[^>]*>", html)) >= 16
    assert _unnamed_controls(html) == []
    # The check sees a control that lost its name, by either route.
    assert len(_unnamed_controls(html.replace('<label for="inp-cost-s2">', "<label>"))) == 1
    assert len(_unnamed_controls(html.replace(' aria-label="band_lo (Auto-FALSE below)"', ""))) == 1


def test_a_slider_is_named_without_its_live_value() -> None:
    # Its visible label holds the value's <span>; the slider announces its value itself.
    html = (STATIC_DIR / "index.html").read_text()
    for ident in ("inp-tau-low", "inp-tau-high"):
        assert f'<label for="{ident}"' not in html
        assert re.search(rf'<input type="range" id="{ident}" aria-label="band_(?:lo|hi) \([^"]+\)"', html)
