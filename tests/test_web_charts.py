"""The calibration view: the per-bin Wilson intervals /api/audit adds for it, the chart kit, and the
view rendered in node over real API payloads (the reliability diagram, its bins and the group list)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main
from s1ops.demo import generate_workspace
from s1ops.eval.metrics import reliability, reliability_counts
from s1ops.improve.slices import wilson
from s1ops.web.app import create_app
from tests.test_web_harness import ESCAPED, HOSTILE, _hostile, _mutated
from tests.test_web_render import _all_functions, _declarations, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


@pytest.fixture(scope="module")
def payloads(ws) -> dict[str, dict]:
    root = Path(ws["records"]).parent
    cat = TestClient(create_app(root, records_path=ws["records_categorical"], fits_path=ws["fits_categorical"]))
    unfitted = TestClient(create_app(root, fits_path=root / "no-fits.json"))
    return {
        "noul": TestClient(create_app(root)).get("/api/audit", params={"n_boot": 20}).json(),
        "categorical": cat.get("/api/audit", params={"n_boot": 20}).json(),
        "unfitted": unfitted.get("/api/audit", params={"n_boot": 20}).json(),
    }


def _groups(payloads: dict[str, dict]) -> list[dict]:
    return [g for p in payloads.values() for g in p["groups"]]


# --- display.reliability_ci ----------------------------------------------------------------------


def test_each_bins_interval_is_counted_from_its_labels(payloads) -> None:
    groups = _groups(payloads)
    assert {g["primitive"] for g in groups} == {"noul", "choice", "score"}
    sides = 0
    for g in groups:
        ci = g["display"]["reliability_ci"]
        expected = {"before": "reliability", "after": "reliability_after"}
        assert set(ci) == {side for side, key in expected.items() if key in g}
        for side, key in expected.items():
            if key not in g:
                continue
            sides += 1
            for strategy in ("width", "mass"):
                rows, bins = ci[side][strategy], g[key][strategy]
                assert [r["n"] for r in rows] == [b["n"] for b in bins]
                for r, b in zip(rows, bins):
                    # Bit for bit: `observed` is the mean of the same 0/1 targets.
                    assert r["successes"] / r["n"] == b["observed"]
                    assert r["ci"] == list(wilson(r["successes"], r["n"]))
    assert sides == 7  # before and after for the fitted noul, choice and score groups; before for the unfitted one


def test_a_count_is_never_recovered_from_a_rounded_rate() -> None:
    # 29 of 50: the rate times n is 28.999999999999996, which truncation would read as 28.
    score, target = np.full(50, 0.55), np.array([1.0] * 29 + [0.0] * 21)
    (row,) = reliability(score, target, 10, "width")
    assert row.observed * row.n != 29
    assert reliability_counts(score, target, 10, "width") == [(50, 29)]


@pytest.mark.parametrize("strategy", ["width", "mass"])
def test_the_counts_follow_the_reliability_bins_in_hard_cases(strategy) -> None:
    rng = np.random.default_rng(3)
    cases = [
        (np.full(100, 0.7), rng.integers(0, 2, 100).astype(float)),  # every score tied
        (np.array([0.0, 1.0, 1.0, 0.95, 0.05]), np.array([0.0, 1.0, 0.0, 1.0, 0.0])),  # fewer records than bins, score 1.0
        (np.r_[np.full(40, 0.15), np.full(3, 0.85)], rng.integers(0, 2, 43).astype(float)),  # empty width bins
    ]
    for score, target in cases:
        bins = reliability(score, target, 10, strategy)
        counts = reliability_counts(score, target, 10, strategy)
        assert [n for n, _ in counts] == [b.n for b in bins]
        assert [k / n for n, k in counts] == [b.observed for b in bins]


def test_the_page_only_field_is_not_in_the_cli_report(ws, tmp_path, capsys) -> None:
    root = Path(ws["records"]).parent
    web = TestClient(create_app(root)).get("/api/audit", params={"n_boot": 20}).json()["groups"]
    assert main(["audit", ws["records"], "--fit", ws["fits"], "--boot", "20"]) == 0
    cli = json.loads(capsys.readouterr().out)["groups"]
    assert [{k: v for k, v in g.items() if k not in ("display", "fit")} for g in web] == cli
    assert all("display" not in g for g in cli)


def test_n_boot_reaches_the_bootstrap(ws) -> None:
    client = TestClient(create_app(Path(ws["records"]).parent))
    assert client.get("/api/audit", params={"n_boot": 100}).json()["groups"][0]["bootstrap"]["reps"] == 100


# --- chart kit -------------------------------------------------------------------------------------


def _js(expr: str, prelude: str = "") -> object:
    return _node(f"{prelude}\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


@needs_node
@pytest.mark.parametrize(
    "args, ticks",
    [
        ((0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]),
        ((0, 37, 4), [0, 10, 20, 30]),
        ((0.13, 0.91, 4), [0.2, 0.4, 0.6, 0.8]),
        ((-1, 1, 4), [-1, -0.5, 0, 0.5, 1]),
        ((0.5, 0.5, 5), [0.5]),
        # Ranges too narrow for distinct float64 ticks: one tick, never an endless loop.
        ((2000, 2000 + 1e-12, 10), [2000]),
        ((0.6, 0.6000000000000001, 5), [0.6]),
        # A narrow range near zero keeps distinct ticks.
        ((0, 1e-13, 5), [0, 2e-14, 4e-14, 6e-14, 8e-14, 1e-13]),
        ((0.3, 0.1 + 0.2, 5), [0.3]),
        ((1, 1 + 2**-52, 5), [1]),
        ((-1e308, 1e308, 5), [-1e308, -5e307, 0, 5e307, 1e308]),  # hi - lo overflows
        ((0, 1, float("inf")), []),
        ((0, 1, 0.5), []),
    ],
)
def test_nice_ticks(args, ticks) -> None:
    # A timeout, since the failure this guards against is a loop that never ends.
    js = f"{_all_functions()}\nprocess.stdout.write(JSON.stringify(niceTicks(...{json.dumps(list(args))})));"
    r = subprocess.run(["node", "-"], input=js, capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ticks


@needs_node
def test_scale_linear_maps_both_ends() -> None:
    assert _js("[0, 0.5, 1].map(scaleLinear(0, 1, 280, 10))") == [280, 145, 10]


@needs_node
def test_the_frame_escapes_its_title_and_description() -> None:
    svg = _js(f"svgFrame({{w: 10, h: 10, title: {json.dumps(HOSTILE)}, desc: {json.dumps(HOSTILE)}}}, '')")
    assert "<img" not in svg and svg.count(ESCAPED) == 2
    assert svg.startswith('<svg class="chart" viewBox="0 0 10 10" role="img"><title>')
    assert "xmlns" not in svg


@needs_node
def test_a_mark_at_a_missing_value_is_left_out() -> None:
    out = _js('[svgDots([{x: NaN, y: 1, tip: "a"}, {x: 1, y: 1, tip: "b"}], "ser-1", "circle"), svgLine([{x: 1, y: NaN}, {x: 2, y: 2}], "ser-1")]')
    assert out[0].count('class="hit"') == 1 and 'data-tip="b"' in out[0]
    assert out[1] == ""


@needs_node
def test_bins_over_one_tied_score_are_one_bar_of_their_summed_count() -> None:
    bins = [{"lo": 0.7, "hi": 0.7, "n": 10, "mean_score": 0.7, "observed": 0.5}] * 3 + [
        {"lo": 0.8, "hi": 0.9, "n": 4, "mean_score": 0.85, "observed": 0.75}]
    strip = _js(f'countStrip({{name: "before fit", bins: {json.dumps(bins)}}}, scaleLinear(0, 1, 0, 100))')
    assert [(b["n"], b["bins"]) for b in strip] == [(30, 3), (4, 1)]
    assert strip[0]["tip"].startswith("3 bins over one score range, 30 records:")
    assert strip[0]["tip"].count("n = 10,") == 3


# --- the calibration view --------------------------------------------------------------------------


def _calib(payload: dict, cbins: str = "mass") -> str:
    return _js(f"renderCalib({json.dumps(payload)}, {{cbins: {json.dumps(cbins)}, tstate: {{}}}})")


def _cards(html: str) -> list[str]:
    return html.split('<div class="card">')[1:]


@needs_node
@pytest.mark.parametrize("case", ["noul", "categorical", "unfitted"])
@pytest.mark.parametrize("strategy", ["width", "mass"])
def test_the_diagram_draws_every_bin_of_each_side(payloads, case, strategy) -> None:
    html = _calib(payloads[case], strategy)
    cards = _cards(html)
    assert len(cards) == len(payloads[case]["groups"])
    for g, card in zip(payloads[case]["groups"], cards):
        before = g["reliability"][strategy]
        after = g.get("reliability_after", {}).get(strategy, [])
        svg = re.search(r'<svg class="chart".*?</svg>', card, re.S).group(0)
        assert len(re.findall(r'<circle class="dot ser-1"', svg)) == len(before)
        assert len(re.findall(r'<rect class="dot ser-2"', svg)) == len(after)
        assert svg.count('class="whisker') == len(before) + len(after)
        assert svg.count('class="hit"') == len(before) + len(after)
        # Series colors are the chart tokens, never a status color or a literal.
        assert "var(--" not in svg and not re.search(r"#[0-9a-fA-F]{3,6}\b", svg)
        assert ("after fit" in card) is bool(after)
        # The bin table holds every bin of both sides.
        bins_table = re.search(r'data-table="calib:bins:.*?</table>', card, re.S).group(0)
        assert bins_table.count("<tr>") == 1 + len(before) + len(after)


@needs_node
def test_the_bins_tooltip_gives_the_count_and_the_interval(payloads) -> None:
    g = payloads["noul"]["groups"][0]
    b, ci = g["reliability"]["mass"][0], g["display"]["reliability_ci"]["before"]["mass"][0]
    html = _calib(payloads["noul"])
    tip = f"n = {b['n']}, successes = {ci['successes']}"
    assert tip in html
    assert f"95% Wilson CI [{ci['ci'][0]:.3f}, {ci['ci'][1]:.3f}]" in html


@needs_node
def test_the_warnings_and_metric_intervals_are_shown(payloads) -> None:
    g = json.loads(json.dumps(payloads["noul"]["groups"][0]))
    g["warnings"] = ["n=12 is below 15x10 bins"]
    html = _calib({"groups": [g]})
    assert '<span class="badge badge-warn">WARN</span> n=12 is below 15x10 bins' in html
    lo, hi = g["before_ci"]["brier"]
    assert f"[{lo:.4f}, {hi:.4f}]" in html
    lo, hi = g["delta_ci"]["ece_mass"]
    assert f"[{lo:.4f}, {hi:.4f}]" in html
    assert "Bootstrap: 20 resamples" in html


@needs_node
def test_the_primitive_badge_is_not_a_status_color(payloads) -> None:
    html = _calib(payloads["categorical"])
    assert re.findall(r'<span class="badge (badge-\w+)">(?:score|choice)</span>', html) == ["badge-none", "badge-none"]


# --- more than six groups ----------------------------------------------------------------------------

GROUPS7 = [{"question_id": f"q-{i}", "model_served": "m", "primitive": "noul", "n": 10 + i} for i in range(7)]


@needs_node
@pytest.mark.parametrize(
    "resolved, groups, state, mode",
    [
        ("none", GROUPS7, "ok", "list"),
        ("none", GROUPS7[:6], "ok", "audit"),
        ("none", [], "loading", "wait"),
        ("none", [], "error", "error"),
        ("ok", GROUPS7, "ok", "audit"),
    ],
)
def test_audit_mode(resolved, groups, state, mode) -> None:
    assert _js(f"auditMode({{status: {json.dumps(resolved)}}}, {json.dumps(groups)}, {json.dumps(state)})") == mode


def _list(groups: list[dict], functions: str | None = None) -> list[str]:
    js = f"""
const els = {{}};
const document = {{getElementById: id => (els[id] ??= {{innerHTML: ""}})}};
const fetchJson = async () => {{ throw new Error("the list requests nothing"); }};
{functions if functions is not None else _all_functions()}
loadAudit({{mode: "list", groups: {json.dumps(groups)}, opts: () => ({{tstate: {{}}}})}}).then(() =>
  process.stdout.write(JSON.stringify(Object.values(els).map(e => e.innerHTML))));
"""
    return _node(js)


@needs_node
def test_more_than_six_groups_are_listed_without_a_request() -> None:
    (html,) = _list(GROUPS7)
    assert "7 labeled groups" in html
    assert html.count('data-act="open-group"') == 7
    assert 'data-qid="q-6" data-model="m" data-prim="noul"' in html


@needs_node
def test_the_group_list_renders_hostile_identifiers_as_text() -> None:
    hostile = _hostile(GROUPS7)
    (html,) = _list(hostile)
    assert "<img" not in html and ESCAPED in html
    # The check sees a dropped escape in a data attribute and in a cell.
    for old, new in (('data-${esc(attr)}="${esc(r[key])}"', 'data-${esc(attr)}="${r[key]}"'),
                     ("${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>")):
        assert "<img" in _list(hostile, functions=_mutated(old, new))[0]


@needs_node
def test_the_list_signature_follows_the_groups() -> None:
    prelude = "let GROUPS = []; let groupsState = 'ok'; let generation = 0;"
    sig = lambda groups: _js(f"(GROUPS = {json.dumps(groups)}, viewSig('calib', defaultState()))", prelude)
    other = [dict(g) for g in GROUPS7]
    other[3]["model_served"] = "m2"
    assert sig(GROUPS7) != sig(other)
    assert sig(GROUPS7) == sig(json.loads(json.dumps(GROUPS7)))
    assert json.loads(sig(GROUPS7[:6]))[1][1] == "audit"


@needs_node
def test_n_boot_is_in_the_signature_and_the_bins_are_not() -> None:
    prelude = "let GROUPS = []; let groupsState = 'ok'; let generation = 0;"
    base = "defaultState()"
    assert _js(f"viewSig('calib', {{...{base}, nboot: 1000}}) !== viewSig('calib', {base})", prelude) is True
    assert _js(f"viewSig('calib', {{...{base}, cbins: 'width'}}) === viewSig('calib', {base})", prelude) is True


@needs_node
@pytest.mark.parametrize(
    "hash_, nboot, cbins",
    [
        ("#/calib?nboot=1000&cbins=width", 1000, "width"),
        ("#/calib?nboot=300", 200, "mass"),  # not a size the picker shows
        ("#/calib?nboot=2000.0", 2000, "mass"),
        ("#/calib?nboot=abc&cbins=quantile", 200, "mass"),
        ("#/calib?nboot=", 200, "mass"),
    ],
)
def test_the_hash_reads_only_the_pickers_values(hash_, nboot, cbins) -> None:
    st = _js(f"parseHash({json.dumps(hash_)})")
    assert (st["nboot"], st["cbins"]) == (nboot, cbins)


@needs_node
def test_the_audit_request_sends_the_bootstrap_size() -> None:
    js = f"""
const urls = [];
const document = {{getElementById: () => ({{}})}};
const fetchJson = async url => {{ urls.push(url); return {{groups: []}}; }};
{_all_functions()}
loadAudit({{params: {{question_id: "q a", model_served: "m", primitive: "noul"}}, nboot: 500}}).then(() => process.stdout.write(JSON.stringify(urls)));
"""
    assert _node(js) == ["/api/audit?question_id=q+a&model_served=m&primitive=noul&n_boot=500"]


# --- the shell keeps no payload of an older request -----------------------------------------------


def _shell_run(steps: str, payload: dict, functions: str | None = None) -> dict:
    """Run `runView("calib")` with the shell's state declared and `fetchJson` answering only when a
    step settles it; return what the steps record."""
    group = {k: payload["groups"][0][k] for k in ("question_id", "model_served", "primitive")}
    js = f"""
let STATE = {{view: "calib"}};
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [{{...{json.dumps(group)}, n: 160}}], groupsState = "ok", groupsError = "";
const els = {{}};
const el = id => (els[id] ??= {{id, innerHTML: "", textContent: "", value: "", attrs: {{}},
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const pending = [];
const fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const payload = {json.dumps(payload)};
const out = {{}};
{functions if functions is not None else _all_functions()}
STATE = {{...defaultState(), view: "calib"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


CACHE_STEPS = """
  runView("calib");
  pending.shift().resolve(payload);
  await tick();
  out.first = el("calib-container").innerHTML.includes('<svg class="chart"');
  // A bigger bootstrap is pending; changing the bins meanwhile must not bring back the old payload.
  STATE = {...STATE, nboot: 1000};
  runView("calib");
  out.url = pending[0].url;
  STATE = {...STATE, cbins: "width"};
  runView("calib");
  out.pendingChart = el("calib-container").innerHTML.includes('<svg class="chart"');
  out.pendingBusy = el("calib-container").attrs["aria-busy"] === "true";
  out.pendingTables = Object.keys(currentTables()).length;
  // It fails; a bins change after the failure keeps the error and offers nothing to export.
  pending.shift().reject(Object.assign(new Error("boom"), {status: 500}));
  await tick(); await tick();
  STATE = {...STATE, cbins: "mass"};
  runView("calib");
  out.failedError = el("calib-container").innerHTML.includes("ERROR</span> boom");
  out.failedChart = el("calib-container").innerHTML.includes('<svg class="chart"');
  out.failedTables = Object.keys(currentTables()).length;
"""


@needs_node
def test_a_bins_change_never_shows_an_older_requests_payload(payloads) -> None:
    out = _shell_run(CACHE_STEPS, payloads["noul"])
    assert out["first"] is True
    assert out["url"].endswith("n_boot=1000")
    assert (out["pendingChart"], out["pendingBusy"], out["pendingTables"]) == (False, True, 0)
    assert (out["failedError"], out["failedChart"], out["failedTables"]) == (True, False, 0)


@needs_node
def test_the_payload_check_catches_a_kept_payload(payloads) -> None:
    functions = _mutated("      delete viewData[view];\n", "")
    out = _shell_run(CACHE_STEPS, payloads["noul"], functions=functions)
    assert out["pendingChart"] is True and out["failedChart"] is True


@needs_node
def test_each_chart_names_the_table_holding_its_values(payloads) -> None:
    html = _calib(payloads["categorical"])
    ids = []
    for card in _cards(html):
        target = re.search(r'<svg class="chart"[^>]* aria-details="([^"]+)"', card).group(1)
        # The element with that id is the card's bin table, and nothing else carries it.
        assert re.search(rf'<div class="tk" data-table="calib:bins:[^"]*" id="{target}">', card)
        assert html.count(f'"{target}"') == 2
        ids.append(target)
    assert len(ids) == len(set(ids)) == 2
    # No mark takes focus: the table is the keyboard path.
    assert "tabindex" not in html


UNRENDERABLE_STEPS = """
  // A response the renderer throws on (a group without its metrics): the error card, nothing kept.
  runView("calib");
  pending.shift().resolve({groups: [{...payload.groups[0], before: null}]});
  await tick(); await tick();
  out.error = el("calib-container").innerHTML.includes("ERROR</span>");
  out.kept = "calib" in viewData;
  out.tables = Object.keys(currentTables()).length;
"""


@needs_node
def test_a_response_the_view_cannot_render_is_not_kept(payloads) -> None:
    assert _shell_run(UNRENDERABLE_STEPS, payloads["noul"]) == {"error": True, "kept": False, "tables": 0}
    # Kept before rendering, it would stay behind the error card.
    functions = _mutated(
        "      el.innerHTML = renderCalib(data, ctx.opts ? ctx.opts() : {});\n      if (ctx.keep) ctx.keep(data);\n",
        "      if (ctx.keep) ctx.keep(data);\n      el.innerHTML = renderCalib(data, ctx.opts ? ctx.opts() : {});\n",
    )
    assert _shell_run(UNRENDERABLE_STEPS, payloads["noul"], functions=functions)["kept"] is True


@needs_node
def test_the_group_list_is_kept_once_it_is_on_screen() -> None:
    script = "\n".join(src for name, src in _declarations() if name == "loadAudit")
    assert script.index("el.innerHTML = renderCalib(kept") < script.index("ctx.keep(kept)")


REFRESH_STEPS = """
  runView("calib");
  // A Refresh while the request is out, as refresh() does it (a hidden view is not run again).
  generation += 1;
  loadedSig = {};
  viewData = {};
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.kept = "calib" in viewData;
  out.rendered = el("calib-container").innerHTML.includes('<svg class="chart"');
  out.loadedAt = el("loaded-at").textContent;
"""


@needs_node
def test_a_request_a_refresh_outdated_keeps_and_renders_nothing(payloads) -> None:
    assert _shell_run(REFRESH_STEPS, payloads["noul"]) == {"kept": False, "rendered": False, "loadedAt": ""}


@needs_node
def test_the_refresh_check_catches_a_missing_generation_guard(payloads) -> None:
    functions = _mutated("isCurrent: () => viewSeq[view] === seq && generation === gen,", "isCurrent: () => viewSeq[view] === seq,")
    functions = functions.replace("if (viewSeq[view] !== seq || generation !== gen) return;", "if (viewSeq[view] !== seq) return;")
    out = _shell_run(REFRESH_STEPS, payloads["noul"], functions=functions)
    assert out["kept"] is True and out["rendered"] is True and out["loadedAt"].startswith("Loaded ")
