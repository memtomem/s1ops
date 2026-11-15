"""The overview's status table, run in node over real route payloads: each row counts every item of
its unit once, at the worst status badge its view shows for it, from the view's own helper; served
values are counted as served; No data, Not checked and the calibration gate are never counts; and a
check writes only when it is the route's latest of the current generation."""
from __future__ import annotations

import copy
import html as html_lib
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.web.app import STATIC_DIR, create_app
from tests.test_web_harness import HOSTILE, _hostile
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")

HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
ROUTES = {"audit": "/api/audit?n_boot=20", "lint": "/api/lint", "slices": "/api/slices", "compare": "/api/compare", "drift": "/api/drift", "checkpoints": "/api/checkpoints"}


def _get(root: Path, url: str) -> dict:
    r = TestClient(create_app(workspace_dir=root)).get(url)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("ws") / "demo"
    generate_workspace(root)
    return {k: _get(root, u) for k, u in ROUTES.items()}


def _js(expr: str, prelude: str = "") -> object:
    return _node(f"{prelude}\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


def _summary(fn: str, d: dict) -> dict:
    return _js(f"{fn}({json.dumps(d)})")


BUCKETS = ("attention", "warnings", "ok", "other")


def _buckets(s: dict) -> tuple:
    return tuple(s[k] for k in BUCKETS)


# --- the counting rule ----------------------------------------------------------------------------


@needs_node
def test_an_item_is_counted_at_its_worst_badge_and_ok_ranks_below_info() -> None:
    cases = [
        (["badge-ok"], "ok"), (["badge-ok", "badge-info"], "other"), (["badge-none", "badge-ok"], "other"),
        (["badge-ok", "badge-warn", "badge-info"], "warnings"), (["badge-warn", "badge-err"], "attention"), ([], "other"),
    ]
    got = _js(f"{json.dumps([c for c, _ in cases])}.map(badgeBucket)")
    assert got == [b for _, b in cases]


def _calib_variant(g: dict, *, raw_above: bool, fit: str | None, after_above: bool = False, in_sample: int = 0, warnings: tuple = ()) -> dict:
    g = copy.deepcopy(g)
    g["warnings"] = list(warnings)
    g["ece_floor"] = {"width": {"mean": 0.01, "p95": 0.05}, "mass": {"mean": 0.01, "p95": 0.05}}
    g["before"]["ece_mass"] = 0.2 if raw_above else 0.01
    if fit is None:
        g["after"] = None
        g["delta_ci"] = None
        g["fit"] = None
        return g
    g["after"] = {**(g.get("after") or g["before"]), "ece_mass": 0.2 if after_above else 0.01}
    g["ece_floor_after"] = {"width": {"mean": 0.01, "p95": 0.05}, "mass": {"mean": 0.01, "p95": 0.05}}
    g["delta_ci"] = {**(g.get("delta_ci") or {}), "ece_mass": {"worse": [0.01, 0.05], "better": [-0.05, -0.01], "unclear": [-0.01, 0.01]}[fit]}
    g["fit"] = {**(g.get("fit") or {}), "n_in_sample": in_sample}
    return g


@needs_node
def test_calibration_counts_each_group_once_and_matches_the_card(demo) -> None:
    g = demo["audit"]["groups"][0]
    groups = [
        _calib_variant(g, raw_above=True, fit="better"),  # a fit that fixed it: OK, not a warning forever
        _calib_variant(g, raw_above=True, fit="worse", in_sample=3),  # worst is the worsened fit
        _calib_variant(g, raw_above=False, fit="unclear"),  # no clear change, within the floor: other
        _calib_variant(g, raw_above=True, fit=None),  # no fit: the raw badge decides
        _calib_variant(g, raw_above=False, fit=None),
        _calib_variant(g, raw_above=False, fit="better", warnings=("few records",)),
        _calib_variant(g, raw_above=False, fit="better", after_above=True),
        _calib_variant(g, raw_above=False, fit="unclear", in_sample=2),  # warn + info: a warning
    ]
    s = _summary("calibSummary", {"groups": groups})
    assert s["state"] == "counted" and s["n_items"] == 8
    assert _buckets(s) == (1, 4, 2, 1)
    assert "2 of 8 groups: not held out" in s["detail"] and "1 of 8 groups: Fit worsened ECE" in s["detail"]
    # The card shows the badges the summary counted (the same helpers): class and words.
    cards = _js(f"{json.dumps(groups)}.map((g, i) => auditCardHtml(g, 'mass', {{}}, i))")
    status = [re.findall(r'<span class="badge (badge-(?:ok|warn|err|info))">\s*(Fit \w+ ECE|No clear ECE change|Not held out|Miscalibrated|Within|WARN)', c) for c in cards]
    assert ("badge-ok", "Fit improved ECE") in status[0] and ("badge-warn", "Miscalibrated") in status[0]
    assert ("badge-err", "Fit worsened ECE") in status[1] and ("badge-warn", "Not held out") in status[1]
    assert ("badge-info", "No clear ECE change") in status[2]
    assert ("badge-warn", "WARN") in status[5]
    badges = _js(f"{json.dumps(groups)}.map(g => calibStatusBadges(g).map(b => b.cls))")
    for card, cls in zip(cards, badges):
        for c in set(cls):
            assert f'class="badge {c}"' in card


@needs_node
def test_slices_count_the_worst_served_verdict_per_group(demo) -> None:
    d = copy.deepcopy(demo["slices"])
    base = d["groups"][0]
    m = base["answerability"][0]
    answerable_low = {**m, "n": 500, "answerable": True, "ci": [0.1, 0.2], "null_ci": [0.5, 0.6]}  # served true under the null
    not_answerable_high = {**m, "n": 500, "answerable": False, "ci": [0.8, 0.9], "null_ci": [0.1, 0.2]}  # served false above it
    sig_high_q = {**base["error_slices"][0], "q_value": 0.99, "significant": True}
    sig_low_q = {**base["error_slices"][0], "q_value": 0.001, "significant": False}
    d["groups"] = [
        {**base, "answerability": [answerable_low], "error_slices": [sig_low_q]},
        {**base, "answerability": [not_answerable_high], "error_slices": [sig_high_q]},
        {**base, "answerability": [not_answerable_high], "error_slices": [sig_high_q]},
    ]
    s = _summary("slicesSummary", d)
    assert _buckets(s) == (0, 2, 1, 0)
    assert "2 of 3 groups have a significant error slice" in s["detail"]
    # Several slices a group: its worst verdict, TOO FEW from the served min_n, any significant slice.
    ok = {**m, "n": 500, "answerable": True}
    few = {**m, "n": d["min_n"] - 1, "answerable": False, "ci": [0.1, 0.2], "null_ci": [0.5, 0.6]}
    indistinct = {**m, "n": 500, "answerable": False, "ci": [0.4, 0.6], "null_ci": [0.45, 0.55]}
    below = {**m, "n": 500, "answerable": False, "ci": [0.1, 0.2], "null_ci": [0.5, 0.6]}
    d["groups"] = [
        {**base, "answerability": [ok, few], "error_slices": [sig_low_q, sig_high_q]},
        {**base, "answerability": [ok, indistinct], "error_slices": [sig_low_q]},
        {**base, "answerability": [below, ok], "error_slices": []},
    ]
    s = _summary("slicesSummary", d)
    assert _buckets(s) == (1, 1, 0, 1) and s["n_items"] == 3
    assert f"1 of 3 groups: TOO FEW (n < {d['min_n']})" in s["detail"] and "1 of 3 groups have a significant error slice" in s["detail"]


@needs_node
def test_compare_and_judge_count_the_served_booleans(demo) -> None:
    d = copy.deepcopy(demo["compare"])
    q = d["shadow"]["questions"][0]
    # Served values against their numbers, in both directions, and counts no recomputation gives.
    d["shadow"]["questions"] = [{**q, "replaceable": True, "flip_rate": 0.9}, {**q, "replaceable": True, "flip_rate": 0.5}, {**q, "replaceable": False, "flip_rate": 0.0}]
    d["judge"]["overall"] = {**d["judge"]["overall"], "passes_bar": False, "weighted_kappa": 1.0}
    s = _summary("compareSummaries", d)
    assert _buckets(s["compare"]) == (0, 1, 2, 0) and s["compare"]["n_items"] == 3
    assert _buckets(s["judge"]) == (0, 1, 0, 0) and "BELOW BAR" in s["judge"]["detail"]
    d["judge"]["overall"] = {**d["judge"]["overall"], "passes_bar": True, "weighted_kappa": None}
    assert _buckets(_summary("compareSummaries", d)["judge"]) == (0, 0, 1, 0)


@needs_node
def test_lint_and_drift_count_their_served_counts_as_served(demo) -> None:
    lint = {**demo["lint"], "counts": {"error": 7, "warning": 0, "info": 1, "suppressed": 2}}
    s = _summary("lintSummary", lint)
    assert _buckets(s) == (7, 0, None, 1)
    assert s["detail"].startswith(f"{demo['lint']['n_findings']} unsuppressed findings; 2 suppressed.")
    probes_only = _summary("lintSummary", {"questions_file": None, "probe_audit": [{}], "counts": {}})
    assert probes_only == {"state": "nodata", "detail": "No questions file. 1 probe group(s) are in the lint view."}
    clean = {**demo["lint"], "n_findings": 0, "findings": [], "counts": {"error": 0, "warning": 0, "info": 0, "suppressed": 0}}
    assert _summary("lintSummary", clean)["detail"].startswith("0 unsuppressed findings")
    assert [q["status"] for q in demo["drift"]["questions"]].count("CRITICAL") != 4
    drift = {**demo["drift"], "summary": {"n_questions": 9, "ok": 3, "warn": 2, "critical": 4}}
    s = _summary("driftSummary", drift)
    assert _buckets(s) == (4, 2, 3, None) and s["n_items"] == 9


@needs_node
def test_checkpoints_count_status_and_notes_together(demo) -> None:
    d = copy.deepcopy(demo["checkpoints"])
    s = _summary("checkpointsSummary", d)  # the demo has no manifest: every model's training set is unknown
    assert s["state"] == "counted" and _buckets(s) == (0, 0, 0, len(d["checkpoints"]))
    c = d["checkpoints"][0]
    held = {**c, "n_trained": 5, "n_trained_hits": 0, "n_records": 10}
    other_run = {**held, "policies": [{"question_id": "q-1", "trainrun_id": "tr-x", "run_matches": False}]}
    run = {"trainrun_id": "tr-1", "manifest_loaded": True, "parent_loaded": True, "record": {"manifest": {"sha256": "a" * 64}, "parent": {"checkpoint": "base", "trainrun_id": "tr-0"}}}
    differs = {**held, "run": {**run, "manifest_loaded": False}}  # served false, whatever the hashes say
    orphan = {**held, "run": {**run, "parent_loaded": False}}
    d["checkpoints"] = [held, other_run, differs, orphan]
    s = _summary("checkpointsSummary", d)
    assert _buckets(s) == (1, 1, 1, 1)
    for words in ("1 of 4 checkpoints: POLICY NAMES ANOTHER RUN", "1 of 4 checkpoints: MANIFEST DIFFERS", "1 of 4 checkpoints: PARENT NOT HERE", "4 of 4 checkpoints: HELD OUT"):
        assert words in s["detail"]
    assert "<" not in s["detail"]  # names, never the notes' markup


# --- no data, not checked, the gate ------------------------------------------------------------------


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@needs_node
def test_each_no_data_case_says_what_it_is(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    paths = generate_workspace(ws)
    rows = [json.loads(x) for x in Path(paths["records_kev"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    _write(Path(paths["records_kev"]), [{**r, "question_id": "q-ffffffffffffffff"} for r in rows])  # same items, another question
    Path(paths["judge_grades"]).write_text("", encoding="utf-8")
    s = _summary("compareSummaries", _get(ws, "/api/compare"))
    assert s["compare"] == {"state": "nodata", "detail": "The two record files share no answered (item, question) pair."}
    assert s["judge"] == {"state": "nodata", "detail": "No judge grades (no file, or an empty one)."}
    canary = [json.loads(x) for x in Path(paths["records_canary"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    _write(Path(paths["records_canary"]), [{**r, "question_id": "q-eeeeeeeeeeeeeeee"} for r in canary])  # no question in common
    d = _summary("driftSummary", _get(ws, "/api/drift"))
    assert d == {"state": "nodata", "detail": "The reference or the canary records file is missing, or they share no question."}
    Path(paths["records"]).unlink()  # the shadow file is there, the primary is not
    assert _summary("compareSummaries", _get(ws, "/api/compare"))["compare"]["detail"] == "The primary or the shadow records file is missing."
    bare = tmp_path / "bare"
    bare.mkdir()
    for fn, url in (("calibSummary", "/api/audit?n_boot=20"), ("slicesSummary", "/api/slices"), ("lintSummary", "/api/lint"), ("driftSummary", "/api/drift"), ("checkpointsSummary", "/api/checkpoints")):
        assert _summary(fn, _get(bare, url))["state"] == "nodata", fn


def _rows(entries: dict, gate: str = "", groups: int = 1, err: str = "") -> list[dict]:
    return _js(f"overviewStatusRows({json.dumps(entries)}, {json.dumps(gate)}, {groups}, {json.dumps(err)})")


@needs_node
def test_a_row_without_a_count_holds_null_and_shows_na(demo) -> None:
    counted = {"calib": {"state": "counted", "attention": 0, "warnings": 0, "ok": 1, "other": 0, "n_items": 1, "detail": "", "at": "12:00", "setting": 20}}
    for st in ("checking", "nodata", "error"):
        counted["lint"] = {"state": st, "detail": "x", "attention": 9, "warnings": 9, "ok": 9, "other": 9}
        rows = {r["row"]: r for r in _rows(counted)}
        assert [rows["lint"][k] for k in BUCKETS] == [None] * 4, st
        assert [rows["calib"][k] for k in BUCKETS] == [0, 0, 1, 0]
        assert rows["calib"]["source"] == "checked 12:00, n_boot 20"
    rows = _rows({})
    lacking = {r["row"]: r for r in rows}
    assert lacking["policy"]["unit"] is None and lacking["lint"]["source"] is None  # null sorts last, never ""
    assert [r["state"] for r in rows] == ["none", "nostatus", "none", "none", "none", "none", "none", "none"]
    assert _js(f"overviewStateLine({json.dumps(rows)})").startswith("8 rows: 7 not checked, 1 no status.")
    mixed = _rows({"lint": {"state": "counted", "attention": 0, "warnings": 0, "ok": None, "other": 0}, "drift": {"state": "nodata", "detail": ""}})
    assert _js(f"overviewStateLine({json.dumps(mixed)})").startswith("8 rows: 5 not checked, 1 no status, 1 counted, 1 no data.")
    t = _js(f"tableHtml(overviewStatusTable({json.dumps(rows)}).spec, {json.dumps(rows)}, {{}})")
    lint_row = [tr for tr in re.findall(r"<tr>(.*?)</tr>", t, re.S) if ">Lint<" in tr][0]
    cells = re.findall(r"<td[^>]*>(.*?)</td>", lint_row, re.S)
    assert cells[3:7] == ["N/A"] * 4
    head = re.findall(r"<th[ >]", t)
    assert all(len(re.findall(r"<td", tr)) == len(head) for tr in re.findall(r"<tr>(.*?)</tr>", t, re.S)[1:])
    assert "Check: Simulator" not in t and "Open: Simulator" in t


@needs_node
def test_the_calibration_gate_replaces_its_check() -> None:
    for gate, words in (("gated", "Not audited"), ("waiting", "Waiting for groups"), ("groups_error", "Groups failed")):
        rows = {r["row"]: r for r in _rows({}, gate, 9, HOSTILE)}
        assert rows["calib"]["state_text"] == words and rows["calib"]["can_check"] is False
        t = _js(f"overviewStatusHtml(overviewStatusTable({json.dumps(list(rows.values()))}), {{}})")
        assert "<img" not in t and "Check: Calibration" not in t
    assert _js("[calibGate([], 'ok'), calibGate(new Array(6).fill({}), 'ok'), calibGate(new Array(7).fill({}), 'ok'), calibGate([], 'loading'), calibGate([], 'error')]") == ["", "", "gated", "waiting", "groups_error"]


@needs_node
def test_hostile_route_payloads_render_as_text(demo) -> None:
    fns = {"audit": "calibSummary", "lint": "lintSummary", "slices": "slicesSummary", "drift": "driftSummary", "checkpoints": "checkpointsSummary"}
    expr = "(() => { const e = {}; " + "; ".join(f"e.{row} = {{...{fn}({json.dumps(_hostile(demo[k]))}), at: 'x'}}" for k, fn, row in
        [(k, fn, {"audit": "calib"}.get(k, k)) for k, fn in fns.items()]) + \
        f"; const c = compareSummaries({json.dumps(_hostile(demo['compare']))}); e.compare = c.compare; e.judge = c.judge; " + \
        "return overviewStatusHtml(overviewStatusTable(overviewStatusRows(e, '', 1, '')), {}); })()"
    t = _js(expr)
    assert "<img" not in t and "Status of each view" in t


@needs_node
def test_hostile_store_text_is_escaped() -> None:
    entries = {r: {"state": "error", "detail": HOSTILE, "at": HOSTILE} for r in ("calib", "lint", "slices", "compare", "judge", "drift", "checkpoints")}
    t = _js(f"overviewStatusHtml(overviewStatusTable(overviewStatusRows({json.dumps(entries)}, '', 1, '')), {{}})")
    assert "<img" not in t
    assert html_lib.escape(HOSTILE, quote=True).replace("&#x27;", "&#39;") in t or html_lib.escape(HOSTILE) in t  # escaped once
    assert "&amp;lt;" not in t


# --- checks ----------------------------------------------------------------------------------------


def _check_run(steps: str) -> dict:
    js = f"""
let STATE = {{}};
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [{{}}], groupsState = "ok", groupsError = "";
const els = {{}};
const el = id => (els[id] ??= {{id, textContent: "", html: "", contains: () => false, querySelectorAll: () => [], attrs: {{}},
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }},
  set innerHTML(v) {{ this.html = v; }}, get innerHTML() {{ return this.html; }}}});
const document = {{getElementById: el, querySelectorAll: () => [], activeElement: null}};
const pending = [];
let fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const entry = r => overviewStore().entries[r];
const out = {{}};
{_all_functions()}
STATE = {{...defaultState(), view: "overview"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


@needs_node
def test_a_check_settles_and_only_the_latest_writes(demo) -> None:
    steps = f"""
  const lint = {json.dumps(demo["lint"])};
  checkRoute("lint");
  out.checking = entry("lint").state;
  out.url = pending[0].url;
  pending.shift().resolve(lint); await tick(); await tick();
  out.counted = entry("lint").state;
  out.shown = el("overview-status").html.includes("Counted");
  // Two checks: the first answers last, a failure after a success, both orders.
  checkRoute("lint"); checkRoute("lint");
  const [a, b] = pending.splice(0, 2);
  b.resolve({{...lint, counts: {{error: 3, warning: 0, info: 0, suppressed: 0}}}}); await tick(); await tick();
  a.reject(Object.assign(new Error("old"), {{status: 400}})); await tick(); await tick();
  out.latest = [entry("lint").state, entry("lint").attention];
  checkRoute("lint"); checkRoute("lint");
  const [c, d] = pending.splice(0, 2);
  d.reject(Object.assign(new Error("newer"), {{status: 400}})); await tick(); await tick();
  c.resolve(lint); await tick(); await tick();
  out.failure = [entry("lint").state, entry("lint").detail];
  // A Refresh in between, then a check of the same route after it (the same seq, 1, in the new
  // store): the old answer writes nothing, success or failure.
  checkRoute("drift");
  generation += 1; viewData = {{}};
  checkRoute("drift");
  const [old, fresh] = pending.splice(0, 2);
  old.resolve({{...{json.dumps(demo["drift"])}, summary: {{n_questions: 9, ok: 9, warn: 0, critical: 0}}}}); await tick(); await tick();
  out.afterRefresh = entry("drift").state;
  fresh.reject(Object.assign(new Error("fresh"), {{status: 400}})); await tick(); await tick();
  checkRoute("drift"); generation += 1; viewData = {{}}; checkRoute("drift");
  const [old2, fresh2] = pending.splice(0, 2);
  fresh2.resolve({json.dumps(demo["drift"])}); await tick(); await tick();
  old2.reject(Object.assign(new Error("old"), {{status: 400}})); await tick(); await tick();
  out.afterRefresh2 = entry("drift").state;
  // A 200 whose summary throws is an error, not a row left checking; compare and judge alike.
  checkRoute("compare"); pending.shift().resolve(null); await tick(); await tick();
  out.thrown = [entry("compare").state, entry("judge").state];
  // The judge row's check settles both rows from one request.
  checkRoute("compare");
  out.compareUrls = pending.map(p => p.url);
  pending.shift().resolve({json.dumps(demo["compare"])}); await tick(); await tick();
  out.both = [entry("compare").state, entry("judge").state];
  // Check all: six routes, five past the calibration gate.
  pending.length = 0; checkAllRoutes(); out.all = pending.map(p => p.url.split("?")[0]).sort();
  pending.length = 0; GROUPS = new Array(7).fill({{}}); checkAllRoutes(); out.gated = pending.length;
  pending.length = 0; GROUPS = []; groupsState = "loading"; checkAllRoutes(); out.loading = pending.map(p => p.url.split("?")[0]).includes("/api/audit");
  pending.length = 0; groupsState = "error"; checkAllRoutes(); out.failed = pending.map(p => p.url.split("?")[0]).includes("/api/audit");
"""
    out = _check_run(steps)
    assert out["checking"] == "checking" and out["url"] == "/api/lint"
    assert out["counted"] == "counted" and out["shown"] is True
    assert out["latest"] == ["counted", 3]
    assert out["failure"] == ["error", "newer"]
    assert out["afterRefresh"] == "checking" and out["afterRefresh2"] == "counted"
    assert out["thrown"] == ["error", "error"]
    assert out["compareUrls"] == ["/api/compare"] and out["both"] == ["counted", "counted"]
    assert out["all"] == ["/api/audit", "/api/checkpoints", "/api/compare", "/api/drift", "/api/lint", "/api/slices"]
    assert out["gated"] == 5
    assert out["loading"] is False and out["failed"] is False


@needs_node
def test_a_check_keeps_the_setting_it_was_requested_under(demo) -> None:
    steps = f"""
  STATE = {{...STATE, fdrq: 0.05, nboot: 200}};
  checkRoute("slices"); checkRoute("audit");
  out.urls = pending.map(p => p.url);
  STATE = {{...STATE, fdrq: 0.2, nboot: 1000}};
  pending.shift().resolve({json.dumps(demo["slices"])});
  pending.shift().resolve({json.dumps(demo["audit"])}); await tick(); await tick();
  const rows = overviewStatusRows(overviewStore().entries, "", 1, "");
  const row = id => rows.find(r => r.row === id);
  out.source = [row("slices").source, row("calib").source];
  out.exported = JSON.parse(toJsonRows(overviewStatusTable(rows).spec, rows)).filter(r => r.setting !== null).map(r => [r.row, r.setting]);
  out.shown = el("overview-status").html.includes("FDR q ≤ 0.05") && el("overview-status").html.includes("n_boot 200");
"""
    out = _check_run(steps)
    assert out["urls"] == ["/api/slices?fdr_q=0.05", "/api/audit?n_boot=200"]
    assert out["source"][0].endswith("FDR q ≤ 0.05") and out["source"][1].endswith("n_boot 200")
    assert out["exported"] == [["calib", 200], ["slices", 0.05]]
    assert out["shown"] is True


@needs_node
def test_every_visit_to_the_overview_draws_the_status_block() -> None:
    steps = """
  STATE = {...STATE, view: "drift"};
  runView("drift");
  STATE = {...STATE, view: "overview"};
  runView("overview");
  out.cold = el("overview-status").html.includes("Status of each view");
  el("overview-status").html = "";
  runView("overview");  // the same signature: a cache hit
  out.hit = el("overview-status").html.includes("Status of each view");
  out.tables = Object.keys(currentTables());
"""
    out = _check_run(steps)
    assert out == {"cold": True, "hit": True, "tables": ["overview-views"]}


@needs_node
def test_a_check_settling_while_another_view_is_shown_draws_nothing(demo) -> None:
    steps = f"""
  STATE = {{...STATE, view: "drift"}};
  checkRoute("lint");
  pending.shift().resolve({json.dumps(demo["lint"])}); await tick(); await tick();
  out.written = el("overview-status").html;
  out.state = entry("lint").state;
"""
    assert _check_run(steps) == {"written": "", "state": "counted"}


def _click(dataset: dict) -> list:
    """Run the page's own delegated `click` listener for an element and return what it called."""
    script = HTML[HTML.index("<script>"):HTML.index("</script>")]
    listener = re.search(r'document\.addEventListener\("click", ev => \{.*?\n    \}\);', script, re.S).group(0)
    js = f"""
let STATE;
const handlers = {{}};
const calls = [];
const document = {{addEventListener: (type, fn) => {{ handlers[type] = fn; }}, getElementById: id => ({{focus: () => calls.push(["focus", id])}})}};
{_all_functions()}
STATE = {{...defaultState(), view: "overview"}};
checkRoute = route => calls.push(["check", route]);
checkAllRoutes = () => calls.push(["all"]);
commit = (next, mode) => calls.push(["commit", next.view, mode]);
{listener}
const el = {{dataset: {json.dumps(dataset)}}};
handlers.click({{target: {{closest: () => el}}}});
process.stdout.write(JSON.stringify(calls));
"""
    return _node(js)


def _buttons(t: str) -> dict[str, dict]:
    """The rendered buttons by accessible name (or text), each with its data-* attributes as a dataset."""
    out = {}
    for attrs, text in re.findall(r"<button([^>]*)>([^<]*)</button>", t):
        data = {k.replace("-", "_"): html_lib.unescape(v) for k, v in re.findall(r'data-([a-z-]+)="([^"]*)"', attrs)}
        label = re.search(r'aria-label="([^"]*)"', attrs)
        out[html_lib.unescape(label.group(1)) if label else text.strip()] = data
    return out


@needs_node
def test_the_rendered_row_buttons_act_on_their_route_and_view() -> None:
    rows = _rows({})
    b = _buttons(_js(f"overviewStatusHtml(overviewStatusTable({json.dumps(rows)}), {{}})"))
    assert _click(b["Check: Judge audit"]) == [["check", "compare"]]
    assert _click(b["Check: Calibration"]) == [["check", "audit"]]
    assert _click(b["Check all"]) == [["all"]]
    assert _click(b["Open: Judge audit"]) == [["commit", "compare", "push"], ["focus", "nav-compare"]]
    assert _click(b["Open: Simulator"]) == [["commit", "policy", "push"], ["focus", "nav-policy"]]
    assert _click({"act": "ov-open", "row": "x", "view": "nowhere"}) == []


def _live_run(active: dict) -> dict:
    """renderOverviewStatus with its table already mounted and `active` (detached) holding the focus."""
    js = f"""
let STATE = {{}};
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [{{}}], groupsState = "ok", groupsError = "";
const focused = [];
const button = d => ({{dataset: d, focus: () => focused.push(d)}});
const parts = {{body: {{innerHTML: ""}}, count: {{textContent: ""}}}};
const box = {{dataset: {{table: "overview-views"}}, querySelector: s => s === ".tk-body" ? parts.body : parts.count}};
const status = {{html: "SENTINEL", set innerHTML(v) {{ this.html = v; }}, get innerHTML() {{ return this.html; }},
  contains: x => x === box,
  querySelectorAll: () => [button({{act: "ov-check", row: "compare"}}), button({{act: "ov-check", row: "judge"}}),
    button({{act: "tk-sort", col: "ok"}}), button({{act: "tk-sort", col: "attention"}})]}};
const line = {{textContent: "old"}};
const document = {{getElementById: id => id === "overview-status" ? status : line, querySelectorAll: () => [box],
  activeElement: {{dataset: {json.dumps(active)}}}}};
{_all_functions()}
STATE = {{...defaultState(), view: "overview"}};
overviewStore().entries.lint = {{state: "counted", attention: 2, warnings: 0, ok: null, other: 0, n_items: 2, detail: "d", at: "12:00", setting: null}};
renderOverviewStatus();
process.stdout.write(JSON.stringify({{outer: status.html, line: line.textContent, body: parts.body.innerHTML.includes(">Counted<"),
  count: parts.count.textContent, focused}}));
"""
    return _node(js)


@needs_node
def test_a_settle_updates_the_mounted_table_in_place_and_keeps_the_focus() -> None:
    out = _live_run({"act": "ov-check", "row": "judge"})
    assert out["outer"] == "SENTINEL"  # the block (and its filter box) is not rewritten
    assert out["line"].startswith("8 rows: 6 not checked, 1 no status, 1 counted.") and out["body"] and out["count"] == "8 rows"
    assert out["focused"] == [{"act": "ov-check", "row": "judge"}]
    assert _live_run({"act": "tk-sort", "col": "attention"})["focused"] == [{"act": "tk-sort", "col": "attention"}]
    assert _live_run({"act": "tk-filter"})["focused"] == []
