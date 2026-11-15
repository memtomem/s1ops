"""The policy simulator view: what /api/simulate-policy adds for it (`display`), the chart kit's
additions, and the view rendered in node over real API payloads (the threshold curve of every
primitive, the cost breakdown, the confusion table, the recommended band and the Pareto chart)."""

from __future__ import annotations

import html as html_lib
import inspect
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.contracts.records import read_records, select
from s1ops.demo import generate_workspace
from s1ops.eval.metrics import routed_confidence
from s1ops.improve.policy import Band, DevRecord, Prices, choose, pareto, select_threshold, sweep, threshold_curve
from s1ops.improve.policy import evaluate as evaluate_band
from s1ops.improve.policy import grid as band_grid
from s1ops.web.app import create_app
from tests.test_categorical_ties import _rows, _write
from tests.test_web_harness import _mutated
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")

CURVE_DEFAULTS = {name: inspect.signature(threshold_curve).parameters[name].default for name in ("c_error", "delegate_accuracy")}


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


@pytest.fixture(scope="module")
def clients(ws) -> dict[str, TestClient]:
    root = Path(ws["records"]).parent
    return {
        "noul": TestClient(create_app(root)),
        "categorical": TestClient(create_app(root, records_path=ws["records_categorical"], fits_path=ws["fits_categorical"])),
    }


def _get(client: TestClient, **params) -> dict:
    r = client.get("/api/simulate-policy", params=params)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def payloads(clients) -> dict[str, dict]:
    return {
        "noul": _get(clients["noul"]),
        "noul-raw": _get(clients["noul"], use_calib=False),
        "choice": _get(clients["categorical"], primitive="choice"),
        "score": _get(clients["categorical"], primitive="score"),
    }


def _noul_records(ws: dict, cost_s1: float, cost_sys2: float, use_calib: bool = True) -> list[DevRecord]:
    (grp,) = select(read_records(ws["records"], use_calib=use_calib), allow_model_mismatch=True).groups
    return [
        DevRecord(id=item_id, p_jev=float(grp.P[i, 1]), label=bool(grp.y[i] == 1), llm=bool(grp.y[i] == 1),
                  jev_cost_usd=cost_s1, llm_cost_usd=cost_sys2)
        for i, (_, item_id) in enumerate(grp.items)
    ]


# --- display ---------------------------------------------------------------------------------------


def test_display_holds_the_costs_the_response_was_computed_with(clients) -> None:
    assert CURVE_DEFAULTS == {"c_error": 0.10, "delegate_accuracy": 0.98}  # what the page's caption states
    assert _get(clients["noul"])["display"]["costs"] == {"cost_s1": 0.00004, "cost_sys2": 0.004, **CURVE_DEFAULTS}
    for name, params in (("noul", {}), ("categorical", {"primitive": "choice"})):
        d = _get(clients[name], cost_s1=0.002, cost_sys2=0.05, **params)
        assert d["display"]["costs"] == {"cost_s1": 0.002, "cost_sys2": 0.05, **CURVE_DEFAULTS}


def _served_curve_is_the_librarys(records: str, use_calib: bool, **costs: float) -> int:
    """The served curve equals `threshold_curve` over what the group routes, at the served costs;
    returns how many records the group escalates at every threshold."""
    d = _get(TestClient(create_app(records_path=records)), use_calib=use_calib, **costs)
    (grp,) = select(read_records(records, use_calib=use_calib), allow_model_mismatch=True).groups
    conf, correct, escalated = routed_confidence(grp.primitive, grp.P, grp.y)
    served = d["display"]["costs"]
    expected = threshold_curve(conf, correct, c_s1=served["cost_s1"], c_delegate=served["cost_sys2"], c_error=served["c_error"],
                               delegate_accuracy=served["delegate_accuracy"], always_escalated=escalated)
    assert d["threshold_curve"] == expected
    if escalated and expected:
        # The comparison has teeth: without the ties the curve is another one.
        assert d["threshold_curve"] != threshold_curve(conf, correct, c_s1=served["cost_s1"], c_delegate=served["cost_sys2"])
    return escalated


def test_the_served_curve_is_the_librarys_at_the_served_costs(ws, tmp_path) -> None:
    assert _served_curve_is_the_librarys(ws["records"], True) == 0
    assert _served_curve_is_the_librarys(ws["records"], False, cost_s1=0.001, cost_sys2=0.02) == 0
    for primitive in ("choice", "score"):
        only = tmp_path / f"{primitive}.jsonl"
        only.write_text("".join(line + "\n" for line in Path(ws["records_categorical"]).read_text(encoding="utf-8").splitlines()
                                if json.loads(line)["primitive"] == primitive), encoding="utf-8")
        _served_curve_is_the_librarys(str(only), True)
    # Tied and untied tops in one group: the two ties count in every point's denominator.
    mixed = _write(tmp_path / "mixed.jsonl", _rows(
        (6, {"a": 0.9, "b": 0.1}, "a"), (2, {"a": 0.7, "b": 0.3}, "b"), (2, {"a": 0.5, "b": 0.5}, "a")))
    assert _served_curve_is_the_librarys(mixed, False, cost_s1=0.001, cost_sys2=0.02) == 2


def test_swept_bands_is_the_sweep_the_frontier_and_the_recommendation_come_from(ws, clients) -> None:
    for use_calib in (True, False):
        d = _get(clients["noul"], use_calib=use_calib, cost_s1=0.0001, cost_sys2=0.01)
        swept = sweep(_noul_records(ws, 0.0001, 0.01, use_calib), band_grid(0.10), Prices(jev_usd=0.0001, llm_usd=0.01))
        rows = d["display"]["swept_bands"]
        assert rows == [r.as_row() for r in swept] and len(rows) == 66
        assert d["pareto_frontier"] == [r.as_row() for r in pareto(swept, metric="f1")]
        assert all(row in rows for row in d["pareto_frontier"])
        assert d["recommended_band_under_35pct_esc"] == choose(swept, metric="f1", max_escalation_rate=0.35).as_row()
        assert d["recommended_band_under_35pct_esc"] in rows
    for primitive in ("choice", "score"):
        d = _get(clients["categorical"], primitive=primitive)
        assert d["display"]["swept_bands"] is None and d["pareto_frontier"] is None


def test_the_system_1_cost_reaches_the_curve_and_the_band(ws, clients) -> None:
    d = _get(clients["noul"], cost_s1=0.001, cost_sys2=0.02, tau_low=0.3, tau_high=0.7)
    assert {p["s1_per_1k_usd"] for p in d["threshold_curve"]} == {1.0}
    assert d["active_band"] == evaluate_band(_noul_records(ws, 0.001, 0.02), Band(0.3, 0.7), Prices(jev_usd=0.001, llm_usd=0.02)).as_row()
    cheap = _get(clients["noul"], cost_s1=0.0, cost_sys2=0.02, tau_low=0.3, tau_high=0.7)
    assert d["active_band"]["cost_usd"] - cheap["active_band"]["cost_usd"] == pytest.approx(0.001 * d["group"]["n"], abs=1e-9)


def test_the_bands_cost_parts_are_its_served_cost(payloads) -> None:
    # What the page prints as the two parts: counts times the served prices.
    for d in (payloads["noul"], payloads["noul-raw"]):
        b, costs = d["active_band"], d["display"]["costs"]
        assert b["n"] * costs["cost_s1"] + b["escalated"] * costs["cost_sys2"] == pytest.approx(b["cost_usd"], abs=1e-9)


# --- chart kit additions -----------------------------------------------------------------------------


def _js(expr: str, prelude: str = "") -> object:
    return _node(f"{prelude}\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


@needs_node
def test_a_band_runs_along_one_bound_and_back_along_the_other() -> None:
    assert _js('svgBand([{x: 1, y0: 9, y1: 5}], "ser-1")') == ""
    assert _js('svgBand([{x: 1, y0: 9, y1: 5}, {x: NaN, y0: 1, y1: 1}], "ser-1")') == ""
    path = _js('svgBand([{x: 1, y0: 9, y1: 5}, {x: 2, y0: 8, y1: NaN}, {x: 3, y0: 7, y1: 4}], "ser-1")')
    assert path == '<path class="area ser-1" d="M1.0 5.0L3.0 4.0L3.0 7.0L1.0 9.0Z" />'


def _hover(svg: str) -> tuple[dict, list]:
    """A chart's hover layer: the rectangle's attributes as numbers, and its [x, tip] entries."""
    rects = re.findall(r'<rect class="hit plot" x="([\d.-]+)" y="([\d.-]+)" width="([\d.-]+)" height="([\d.-]+)" data-points="([^"]*)" />', svg)
    assert len(rects) == 1, svg[:200]
    x, y, w, h, points = rects[0]
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}, json.loads(html_lib.unescape(points))


BOX = "{left: 70, right: 440, top: 5, bottom: 105}"


@needs_node
def test_a_line_charts_hover_layer_is_one_rectangle_over_the_plot() -> None:
    layer = _js(f'svgHoverPlot([{{x: 400, tip: "c"}}, {{x: 100, tip: "a"}}, {{x: 200.04, tip: "b"}}, {{x: NaN, tip: "z"}}], {BOX})')
    rect, entries = _hover(layer)
    assert rect == {"x": 70.0, "y": 5.0, "w": 370.0, "h": 100.0}
    assert entries == [[100, "a"], [200, "b"], [400, "c"]]
    # The crosshair comes first (moveTip finds it as the rectangle's previous sibling), hidden.
    assert layer.startswith('<line class="cross" x1="0" y1="5.0" x2="0" y2="105.0" visibility="hidden" /><rect class="hit plot"')
    assert _js(f"svgHoverPlot([], {BOX})") == "" and _js(f"svgHoverPlot([{{x: NaN, tip: 'z'}}], {BOX})") == ""
    # Points written at one x are one entry naming each; a tooltip is escaped in the attribute.
    _, merged = _hover(_js(f'svgHoverPlot([{{x: 10, tip: "a"}}, {{x: 10.01, tip: "b"}}, {{x: 30, tip: "c"}}], {BOX})'))
    assert merged == [[10, "a\n\nb"], [30, "c"]]
    hostile = _js(f'svgHoverPlot([{{x: 1, tip: "\\"><img src=x>"}}], {BOX})')
    assert "<img" not in hostile and _hover(hostile)[1] == [[1, '"><img src=x>']]


@needs_node
def test_the_pointer_reads_the_point_nearest_in_x() -> None:
    entries = '[[100, "a"], [200, "b"], [400, "c"]]'
    picks = _js(f"[99, 149.9, 150, 150.1, 300.1, 1e9, -5].map(x => plotPick({entries}, x)[1])")
    assert picks == ["a", "a", "a", "b", "c", "c", "a"]  # halfway goes to the first
    assert _js("plotPick([], 5)") is None and _js(f"plotPick({entries}, NaN)") is None
    # A chart shown at another size than it is drawn: the pointer's x in the chart's own units.
    assert _js("[chartX(150, 100, 920, 460), chartX(100, 100, 920, 460), chartX(1020, 100, 920, 460), chartX(5, 0, 230, 460)]") == [25, 0, 460, 10]
    assert _js("[chartX(5, 0, 0, 460)].map(Number.isNaN)") == [True]


def _curve_layer(coverages: list[float]) -> tuple[dict, list]:
    points = ", ".join(f'{{x: f.x({c!r}), tip: "p{i}"}}' for i, c in enumerate(coverages))
    return _hover(_js(f"(f => svgHoverPlot([{points}], f))(curveFrame())"))


@needs_node
@pytest.mark.parametrize(
    "coverages, entries",
    [
        # Each case broke the per-point columns this layer replaced (the review rounds of PR 162):
        # the outermost columns of a sparse curve ran past the plot's sides;
        pytest.param([0.5, 1.0], [[255, "p0"], [440, "p1"]], id="sparse"),
        # a one-pixel minimum width made two columns a third of a pixel apart overlap and cross the side;
        pytest.param([0.999, 1.0], [[439.6, "p0"], [440, "p1"]], id="a-third-of-a-pixel-apart"),
        # edges rounded separately left neighbours overlapping by a tenth of a pixel;
        pytest.param([0.022, 0.708, 1.0], [[78.1, "p0"], [332, "p1"], [440, "p2"]], id="edges-rounded-apart"),
        # and a share of 0.037px rounded to nothing, so the end point had no tooltip at all.
        pytest.param([4999 / 5000, 1.0], [[439.9, "p0"], [440, "p1"]], id="a-share-that-rounded-away"),
    ],
)
def test_no_point_of_a_curve_loses_its_tooltip_or_leaves_the_plot(coverages, entries) -> None:
    rect, got = _curve_layer(coverages)
    frame = _js("(f => [f.left, f.right, f.top, f.bottom])(curveFrame())")
    # One rectangle, exactly the plot: nothing to overlap and nothing outside the sides.
    assert [rect["x"], rect["x"] + rect["w"], rect["y"], rect["y"] + rect["h"]] == frame
    assert got == entries
    # Every point is the answer somewhere on the plot: at its own x.
    for x, tip in entries:
        assert _js(f"plotPick({json.dumps(got)}, {x})") == [x, tip]


@needs_node
def test_a_point_outside_the_plot_still_answers_for_the_part_nearest_it() -> None:
    # Deliberately unlike the columns, which drew nothing for a lone point outside the sides: the
    # layer is the plot, and the nearest point answers wherever the pointer is on it.
    rect, entries = _hover(_js(f'svgHoverPlot([{{x: 500, tip: "out"}}], {BOX})'))
    assert (rect["x"], rect["w"]) == (70.0, 370.0) and entries == [[500, "out"]]
    _, both = _hover(_js(f'svgHoverPlot([{{x: 500, tip: "out"}}, {{x: 100, tip: "in"}}], {BOX})'))
    assert _js(f"[299, 301].map(x => plotPick({json.dumps(both)}, x)[1])") == ["in", "out"]


@needs_node
def test_the_curves_hover_layer_is_the_plot(tmp_path) -> None:
    rows = _rows((50, {"a": 0.9, "b": 0.1}, "a"), (29, {"a": 0.8, "b": 0.2}, "b"), (21, {"a": 0.8, "b": 0.2}, "a"))
    d = _get(TestClient(create_app(records_path=_write(tmp_path / "r.jsonl", rows))), use_calib=False)
    assert [p["coverage"] for p in d["threshold_curve"]] == [0.5, 1.0]
    frame = _js("(f => [f.left, f.right])(curveFrame())")
    for svg in _charts(_view(d)["html"]):
        rect, entries = _hover(svg)
        assert [rect["x"], rect["x"] + rect["w"]] == frame and [x for x, _ in entries] == [255, 440]
        assert "threshold 0.9000" in entries[0][1] and "threshold 0.8000" in entries[1][1]


@needs_node
def test_a_long_curve_offers_the_pointer_spaced_points_and_the_kept_one() -> None:
    assert _js("thinned([1, 2, 3], 5, -1)") == [1, 2, 3]
    picked = _js("thinned(Array.from({length: 1000}, (_, i) => i), 400, 502)")
    assert len(picked) == 401 and picked[0] == 0 and picked[-1] == 999 and 502 in picked
    # One the spacing already holds is not counted twice.
    assert len(_js("thinned(Array.from({length: 1000}, (_, i) => i), 400, 501)")) == 400
    assert picked == sorted(set(picked))
    assert len(_js("thinned(Array.from({length: 1000}, (_, i) => i), 400, -1)")) == 400


@needs_node
@pytest.mark.parametrize("value, top", [(0.95, 1), (0.31, 0.4), (28.79, 30), (0.6464, 0.8), (1, 1), (0, 0)])
def test_an_axis_ends_on_a_tick(value, top) -> None:
    assert _js(f"niceMax({value}, 5)") == top


@needs_node
@pytest.mark.parametrize("ticks, digits", [([0, 0.2, 0.4], 1), ([0, 10, 20], 0), ([0, 0.05, 0.1], 2), ([0, 0.002], 3), ([0, 1], 0), ([5], 2), ([], 2),
                                           ([0, 2e-10], 10), ([0, 1e-300], 20)])
def test_tick_labels_take_the_places_their_step_needs(ticks, digits) -> None:
    assert _js(f"tickDigits({json.dumps(ticks)})") == digits


@needs_node
@pytest.mark.parametrize(
    "value, digits, label",
    [(0.2, 1, "0.2"), (20, 0, "20"), (0.05, 2, "0.05"), (0, 1, "0.0"), (999999, 0, "999999"), (0.0005, 4, "0.0005"),
     (0.00005, 5, "5e-5"), (0.000003, 6, "3e-6"), (2e-10, 10, "2e-10"), (1.5e-10, 11, "1.5e-10"), (1e-9, 10, "1e-9"),
     (0, 10, "0"), (2e9, 0, "2e+9"), (-3e6, 0, "-3e+6"), (1000000, 0, "1e+6")],
)
def test_a_tick_too_small_or_too_large_for_fixed_places_is_scientific(value, digits, label) -> None:
    assert _js(f"tickLabel({value}, {digits})") == label
    assert _js("[tickLabel(NaN, 1), tickLabel(null, 1), tickLabel('7', 1)]") == ["N/A", "N/A", "N/A"]


def _y_labels(svg: str) -> list[str]:
    return re.findall(r'text-anchor="end">([^<]*)</text>', svg)


@needs_node
def test_an_axis_of_tiny_or_huge_amounts_keeps_its_ticks_apart(tmp_path) -> None:
    # One all-correct point at a System 1 cost of 1e-12: the total is 1e-9 per 1,000, where six
    # fixed places would label every tick 0.000000.
    records = _write(tmp_path / "r.jsonl", _rows((4, {"a": 0.9, "b": 0.1}, "a")))
    client = TestClient(create_app(records_path=records))
    tiny = _get(client, use_calib=False, cost_s1=1e-12, cost_sys2=0)
    assert [p["cost_per_1k_usd"] for p in tiny["threshold_curve"]] == [1e-9]
    labels = _y_labels(_charts(_view(tiny)["html"])[1])
    assert labels == ["0", "2e-10", "4e-10", "6e-10", "8e-10", "1e-9"]
    # Six fixed places ("0.000003") would be 48px wide and run under the axis name.
    small = _get(client, use_calib=False, cost_s1=5e-9, cost_sys2=0)
    assert _y_labels(_charts(_view(small)["html"])[1]) == ["0", "1e-6", "2e-6", "3e-6", "4e-6", "5e-6"]
    huge = _get(client, use_calib=False, cost_s1=1e6, cost_sys2=0)
    labels = _y_labels(_charts(_view(huge)["html"])[1])
    assert len(labels) == len(set(labels)) > 2 and max(len(text) for text in labels) <= 6


def _name_x(axes: str) -> float:
    return float(re.search(r'<text class="axis-name" transform="translate\(([\d.-]+) ', axes).group(1))


@needs_node
def test_the_axis_name_clears_the_widest_tick_label() -> None:
    axes = "svgAxes({{x: scaleLinear(0, 1, 64, 440), y: scaleLinear(0, {top}, 234, 14), xTicks: [0, 1], yTicks: {ticks}, left: 64, right: 440, top: 14, bottom: 234, yDigits: {digits}, nameGap: 50, xName: 'x', yName: 'y'}})"
    # Short labels: the name stays where the chart put it.
    assert _name_x(_js(axes.format(top=20, ticks="[0, 10, 20]", digits=0))) == 14.0
    # "1.5e-10" is seven characters (about 43px from 6px left of the axis): the name moves out to clear it.
    wide = _js(axes.format(top=2.5e-10, ticks="[0, 5e-11, 1e-10, 1.5e-10, 2e-10, 2.5e-10]", digits=11))
    assert "1.5e-10" in wide
    label_left = 64 - 6 - 7 * 6.1
    assert _name_x(wide) + 5.5 <= label_left and _name_x(wide) >= 5.5  # clear of the label, and on the canvas


@needs_node
def test_the_pointer_reads_the_nearest_target() -> None:
    points = "[{x: 0, y: 0}, {x: 10, y: 0}, {x: 10, y: 10}]"
    assert _js(f"[nearestIndex({points}, 9, 1), nearestIndex({points}, 4, 0), nearestIndex({points}, 9, 9), nearestIndex([], 1, 1)]") == [1, 0, 2, -1]
    # Two at one distance: the first, so the answer does not depend on which was drawn last.
    assert _js(f"nearestIndex({points}, 5, 0)") == 0


@needs_node
def test_markers_and_legend_keys_come_in_the_new_shapes() -> None:
    dots = _js('[svgDots([{x: 5, y: 5}], "mark", "diamond"), svgDots([{x: 5, y: 5}], "ctx", "small"), svgDots([{x: 5, y: 5, tip: "t"}], "ser-1", "circle")]')
    assert dots[0] == '<path class="dot mark" d="M5.0 0.0L10.0 5.0L5.0 10.0L0.0 5.0Z" />'  # no tip: no hover target
    assert dots[1] == '<circle class="dot ctx" cx="5.0" cy="5.0" r="2.5" />'
    assert dots[2].endswith('<circle class="hit" cx="5.0" cy="5.0" r="10" data-tip="t" />')
    legend = _js('svgLegend([{name: "a", cls: "ser-3", dotted: true}, {name: "b", cls: "mark", shape: "diamond", noLine: true}, {name: "c", cls: "ser-1", shape: "area", noLine: true}])')
    assert '<line class="key ser-3 dotted"' in legend and legend.count("<line") == 1
    assert '<path class="mark fill"' in legend and '<rect class="ser-1 area"' in legend


# --- the view ----------------------------------------------------------------------------------------

_INPUTS = {"inp-tau-low": "0.25", "inp-tau-high": "0.75", "inp-cost-s1": "0.00004", "inp-cost-s2": "0.004",
           "inp-err-budget": "0.10", "inp-dist": "", "inp-group": ""}


def _view(payload: dict | None, *, functions: str | None = None, inputs: dict | None = None, steps: str | None = None) -> dict:
    """Run the simulator's loader as the page does (`run(loadPolicy, ...)`) over `payload` (None: the
    request fails); return the container's HTML, whether the payload was kept, the tables on offer
    and the URLs requested. `steps` replaces the single run (it has `pending`, the held requests)."""
    answer = "throw Object.assign(new Error('boom'), {status: 500});" if payload is None else f"return {json.dumps(payload)};"
    js = f"""
let policySeq = 0, GROUPS = [], viewData = {{}}, TSTATE = {{}};
let STATE = {{view: "policy"}};
const els = {{}};
const inputs = {json.dumps({**_INPUTS, **(inputs or {})})};
const el = id => (els[id] ??= {{value: inputs[id] ?? "", textContent: "", innerHTML: "", style: {{}}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const urls = [];
const pending = [];
const tick = () => new Promise(r => setTimeout(r, 0));
const held = {json.dumps(steps is not None)};
const fetchJson = async url => {{
  urls.push(url);
  if (held) return new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
  {answer}
}};
const out = {{}};
const snapshot = () => ({{html: el("policy-container").innerHTML, kept: "policy" in viewData, tables: Object.keys(currentTables()).sort(), urls}});
{functions if functions is not None else _all_functions()}
(async () => {{
{steps if steps is not None else 'await run(loadPolicy, "policy-container"); Object.assign(out, snapshot());'}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


def _charts(html: str) -> list[str]:
    return re.findall(r'<svg class="chart".*?</svg>', html, re.S)


@needs_node
@pytest.mark.parametrize("case, charts, tables", [("noul", 3, 2), ("noul-raw", 3, 2), ("choice", 2, 1), ("score", 2, 1)])
def test_every_primitive_gets_the_curve_and_noul_the_bands(payloads, case, charts, tables) -> None:
    out = _view(payloads[case])
    html = out["html"]
    assert len(_charts(html)) == charts and out["kept"] is True and len(out["tables"]) == tables
    assert "NaN" not in html and "undefined" not in html
    # Every chart names a table that is in the view, and no mark takes focus.
    for svg in _charts(html):
        target = re.search(r'aria-details="([^"]+)"', svg).group(1)
        assert html.count(f' id="{target}"') == 1
    assert "tabindex" not in html
    g = payloads[case]["group"]
    key = json.dumps([g["question_id"], g["model_served"], g["primitive"]], separators=(",", ":"))
    assert out["tables"] == sorted(f"policy:{name}:{key}" for name in (("bands", "curve") if tables == 2 else ("curve",)))
    # The curve's line is drawn over every point; a 160-point curve draws no marker at each.
    points = len(payloads[case]["threshold_curve"])
    assert points > 40 and all(len(_hover(svg)[1]) == points for svg in _charts(html)[-2:])
    assert 'class="dot ser-1" cx' not in _charts(html)[-2]


@needs_node
def test_the_request_carries_the_system_1_cost() -> None:
    out = _view(None, inputs={"inp-cost-s1": "0.0002"})
    assert "cost_s1=0.0002" in out["urls"][0] and "cost_sys2=0.004" in out["urls"][0]
    prelude = "let GROUPS = []; let groupsState = 'ok'; let generation = 0;"
    assert _js("viewSig('policy', {...defaultState(), cs1: 0.001}) !== viewSig('policy', defaultState())", prelude) is True


def _tuned_marks(html: str) -> list[str]:
    return re.findall(r'<path class="dot mark" d="M([\d.]+) ', html)


@needs_node
def test_the_marked_point_is_the_one_the_server_tuned(clients, payloads) -> None:
    d = payloads["choice"]
    tau = d["tuned_threshold"]["threshold"]
    curve = _js(f"policyCurve({json.dumps(d)})")
    (tuned,) = [r for r in curve["rows"] if r["tuned"]]
    assert tuned["threshold"] == tau and curve["rows"][curve["tuned"]] == tuned
    html = _view(d)["html"]
    # One mark on each of the two curve charts, at the tuned point's coverage.
    x = _js(f"px(curveFrame().x({tuned['coverage']}))")
    assert _tuned_marks(html) == [x, x]
    assert f"the threshold the server tuned ({tau:.4f})" in html
    # No threshold within budget: nothing is marked, and the caption says so.
    none = _get(clients["categorical"], primitive="score", error_budget=0)
    assert none["tuned_threshold"]["threshold"] is None
    html = _view(none, inputs={"inp-err-budget": "0"})["html"]
    assert _tuned_marks(html) == [] and "No point is marked: no threshold meets the error budget." in html
    assert "tuned threshold</span>" not in html  # no legend entry for a mark that is not drawn


@needs_node
def test_the_page_marks_the_servers_point_even_where_the_curve_seems_to_allow_more(tmp_path) -> None:
    # 29 errors of 100 at budget 0.29: `0.29 * 100` is 28.999999999999996, so the server tunes 0.9
    # although the 0.8 point's error rate equals the budget. The page marks what the server tuned.
    rows = _rows((50, {"a": 0.9, "b": 0.1}, "a"), (29, {"a": 0.8, "b": 0.2}, "b"), (21, {"a": 0.8, "b": 0.2}, "a"))
    records = _write(tmp_path / "r.jsonl", rows)
    d = _get(TestClient(create_app(records_path=records)), error_budget=0.29, use_calib=False)
    assert [(p["threshold"], p["error_rate"]) for p in d["threshold_curve"]] == [(0.9, 0.0), (0.8, 0.29)]
    assert d["tuned_threshold"]["threshold"] == 0.9 == select_threshold([0.9] * 50 + [0.8] * 50, [True] * 50 + [False] * 29 + [True] * 21, budget=0.29)
    curve = _js(f"policyCurve({json.dumps(d)})")
    assert curve["tuned"] == 0 and [r["tuned"] for r in curve["rows"]] == ["tuned", ""]
    html = _view(d, inputs={"inp-err-budget": "0.29"})["html"]
    x = _js("px(curveFrame().x(0.5))")
    assert _tuned_marks(html) == [x, x]
    # A two-point curve draws a marker at each point, on each chart's main line.
    assert html.count('<circle class="dot ser-1"') == 4


@needs_node
def test_a_group_with_nothing_to_accept_has_no_curve(tmp_path) -> None:
    records = _write(tmp_path / "r.jsonl", _rows((3, {"a": 0.5, "b": 0.5}, "a"), (2, {"a": 0.5, "b": 0.5}, "b")))
    d = _get(TestClient(create_app(records_path=records)), use_calib=False)
    assert d["threshold_curve"] == []
    out = _view(d)
    assert _charts(out["html"]) == [] and "No curve: no record of this group can be accepted at any threshold" in out["html"]
    assert out["kept"] is True and "NaN" not in out["html"]


@needs_node
def test_a_one_point_curve_and_zero_costs_still_draw(tmp_path, clients) -> None:
    records = _write(tmp_path / "r.jsonl", _rows((4, {"a": 0.9, "b": 0.1}, "a"), (1, {"a": 0.9, "b": 0.1}, "b")))
    d = _get(TestClient(create_app(records_path=records)), use_calib=False)
    assert len(d["threshold_curve"]) == 1
    html = _view(d)["html"]
    assert len(_charts(html)) == 2 and "NaN" not in html and [len(_hover(svg)[1]) for svg in _charts(html)] == [1, 1]
    free = _get(clients["noul"], cost_s1=0, cost_sys2=0)
    assert {b["cost_usd"] for b in free["display"]["swept_bands"]} == {0.0}
    html = _view(free, inputs={"inp-cost-s1": "0", "inp-cost-s2": "0"})["html"]
    assert len(_charts(html)) == 3 and "NaN" not in html
    # Every band is drawn at cost 0: one hover target per distinct F1, naming its bands.
    spots = len({b["f1"] for b in _js(f"policyBands({json.dumps(free)})")})
    assert _charts(html)[0].count('class="hit"') <= spots


@needs_node
def test_a_long_curve_thins_only_its_hover_targets(payloads) -> None:
    d = json.loads(json.dumps(payloads["choice"]))
    point = d["threshold_curve"][0]
    d["threshold_curve"] = [{**point, "threshold": 1 - i / 2000, "accepted": i + 1, "coverage": (i + 1) / 1000} for i in range(1000)]
    d["tuned_threshold"]["threshold"] = d["threshold_curve"][502]["threshold"]
    html = _view(d)["html"]
    first, second = _charts(html)
    assert len(_hover(first)[1]) == len(_hover(second)[1]) == 401
    assert len(re.search(r'<polyline class="line ser-1" points="([^"]+)"', first).group(1).split(" ")) == 1000
    assert "Hover reads 400 evenly spaced points and the tuned one; the table below holds all 1000." in html
    assert len(_tuned_marks(html)) == 2


# --- typing a number -----------------------------------------------------------------------------------


@needs_node
@pytest.mark.parametrize(
    "held, value, after",
    [("0.0", 0, "0.0"), ("0.10", 0.1, "0.10"), ("1e-5", 0.00001, "1e-5"), ("0.00005", 0.00005, "0.00005"),
     ("2", 1, "1"), ("", 0.5, "0.5"), (" ", 0, "0"), ("abc", 0.25, "0.25"), ("0.3", 0.25, "0.25")],
)
def test_an_input_that_names_the_number_is_left_as_typed(held, value, after) -> None:
    js = f"const box = {{value: {json.dumps(held)}}}; const document = {{getElementById: () => box}};"
    assert _js(f"(setNumberInput('x', {value}), box.value)", js) == after


TYPING = """
let STATE, GROUPS = [], groupsState = "ok", groupsError = "", generation = 0, loadedSig = {}, viewData = {}, TSTATE = {}, policySeq = 0;
const els = {};
const defaults = {"inp-dist": "", "inp-tau-low": "0.25", "inp-tau-high": "0.75", "inp-cost-s1": "0.00004", "inp-cost-s2": "0.004", "inp-err-budget": "0.1", "inp-group": ""};
const el = id => (els[id] ??= {value: defaults[id] ?? "", textContent: "", innerHTML: "", style: {}});
const document = {getElementById: el, querySelectorAll: () => []};
const location = {pathname: "/", search: ""};
const replaced = [];
const history = {replaceState: (a, b, url) => replaced.push(url)};
"""


def _typed(field: str, text: str, functions: str | None = None) -> dict:
    """Type `text` into a simulator input a key at a time, running what the page's `input` listener
    runs after each key; return the field after each key, the final state and hash."""
    key = {"inp-cost-s1": "cs1", "inp-cost-s2": "cs2", "inp-err-budget": "budget"}[field]
    js = f"""
{TYPING}
{functions if functions is not None else _all_functions()}
STATE = {{...defaultState(), view: "policy"}};
const steps = [];
el({json.dumps(field)}).value = "";
for (const ch of {json.dumps(text)}) {{
  el({json.dumps(field)}).value += ch;
  keepPolicyInputs();
  steps.push(el({json.dumps(field)}).value);
}}
process.stdout.write(JSON.stringify({{steps, value: STATE[{json.dumps(key)}], hash: replaced[replaced.length - 1]}}));
"""
    return _node(js)


@needs_node
@pytest.mark.parametrize(
    "field, text, value, key",
    [("inp-cost-s1", "0.00005", 0.00005, "cs1"), ("inp-cost-s2", "0.01", 0.01, "cs2"),
     ("inp-err-budget", "0.05", 0.05, "budget"), ("inp-err-budget", "0.105", 0.105, "budget")],
)
def test_a_decimal_typed_key_by_key_is_the_number_requested(field, text, value, key) -> None:
    # The state follows every key ("0.0" is 0 on the way to 0.05), and the field keeps what was typed.
    out = _typed(field, text)
    assert out["steps"] == [text[: i + 1] for i in range(len(text))]
    assert out["value"] == value and out["hash"] == f"#/policy?{key}={text}"


@needs_node
def test_a_number_out_of_range_is_still_rewritten_as_clamped() -> None:
    out = _typed("inp-err-budget", "25")
    assert out["steps"] == ["1", "1"] and out["value"] == 1  # "2" is clamped to 1, then "15" to 1


@needs_node
def test_the_typing_check_catches_a_field_rewritten_after_every_key() -> None:
    # As the page was: every valid key rewrote the field from the state, so 0.05 ended as 5.
    functions = _mutated('      if (String(el.value).trim() !== "" && Number(el.value) === v) return;\n', "")
    out = _typed("inp-cost-s1", "0.00005", functions=functions)
    assert out["steps"][-1] == "5" and out["value"] == 5


# --- bands -------------------------------------------------------------------------------------------


def _bands(d: dict) -> list[dict]:
    return _js(f"policyBands({json.dumps(d)})")


@needs_node
def test_the_bands_table_holds_every_band_the_chart_draws(clients, payloads) -> None:
    d = payloads["noul"]
    rows = _bands(d)
    # The default band is not on the 0.10 grid: it is a row of its own, first.
    assert len(rows) == 67 and rows[0]["band"] == "[0.25, 0.75)" and rows[0]["role"] == "active"
    assert [r["band"] for r in rows[1:]] == [b["band"] for b in d["display"]["swept_bands"]]
    assert sorted(r["band"] for r in rows if "frontier" in r["role"]) == sorted(b["band"] for b in d["pareto_frontier"])
    assert [r["band"] for r in rows if "recommended" in r["role"]] == [d["recommended_band_under_35pct_esc"]["band"]]
    chart = _charts(_view(d)["html"])[0]
    # Each row is drawn once, in the shape of its role.
    drawn = chart.count('class="dot ctx"') + chart.count('<circle class="dot ser-1"') + chart.count('class="dot ser-3"') + chart.count('class="dot ser-2"')
    assert drawn == len(rows)
    assert chart.count('<rect class="dot ser-2"') == 1 and chart.count('<path class="dot ser-3"') == 1
    assert chart.count('class="hit"') == len(_js(f"bandSpots(policyBands({json.dumps(d)}), b => ({{x: b.cost_usd * 500, y: b.f1 * 240}})).map(s => s.bands.length)"))


@needs_node
def test_a_band_on_the_grid_is_one_row_with_every_role_it_has(clients, payloads) -> None:
    on_grid = _get(clients["noul"], tau_low=0.3, tau_high=0.7)
    rows = _bands(on_grid)
    assert len(rows) == 66
    (active,) = [r for r in rows if "active" in r["role"]]
    assert (active["lo"], active["hi"]) == (0.3, 0.7)
    rec = payloads["noul"]["recommended_band_under_35pct_esc"]
    both = _get(clients["noul"], tau_low=rec["lo"], tau_high=rec["hi"])
    (row,) = [r for r in _bands(both) if "active" in r["role"]]
    assert row["role"].startswith("active, recommended") and len(_bands(both)) == 66
    chart = _charts(_view(both, inputs={"inp-tau-low": str(rec["lo"]), "inp-tau-high": str(rec["hi"])})["html"])[0]
    # A band that is both is drawn as the active one.
    assert chart.count('<rect class="dot ser-2"') == 1 and chart.count('<path class="dot ser-3"') == 0
    empty = _get(clients["noul"], tau_low=0.5, tau_high=0.5)
    (row,) = [r for r in _bands(empty) if "active" in r["role"]]
    assert (row["lo"], row["hi"], row["escalated"]) == (0.5, 0.5, 0) and len(_bands(empty)) == 66


@needs_node
def test_the_confusion_table_and_the_cost_parts_are_the_served_bands(payloads) -> None:
    d = payloads["noul"]
    b, costs = d["active_band"], d["display"]["costs"]
    html = _view(d)["html"]
    cells = re.findall(r'<tr><th scope="row">Label (true|false)</th><td class="mono num">(\d+)</td><td class="mono num">(\d+)</td></tr>', html)
    assert cells == [("true", str(b["tp"]), str(b["fn"])), ("false", str(b["fp"]), str(b["tn"]))]
    assert f"Precision {b['precision']:.3f} · recall {b['recall']:.3f} · unresolved 0." in html
    assert f"{b['n']} × $0.00004</td><td class=\"mono num\">${b['n'] * costs['cost_s1']:.4f}</td>" in html
    assert f"{b['escalated']} × $0.004</td><td class=\"mono num\">${b['escalated'] * costs['cost_sys2']:.4f}</td>" in html
    assert f"Total (as served)</th><td></td><td class=\"mono num\">${b['cost_usd']:.4f}</td>" in html
    rec = d["recommended_band_under_35pct_esc"]
    assert f'<div class="kpi">{rec["band"]}</div>' in html and f"F1 {rec['f1']:.3f} · accuracy {rec['accuracy']:.3f}" in html
    none = {**d, "recommended_band_under_35pct_esc": None}
    html = _view(none)["html"]
    assert "No band of the 0.10 grid escalates 35% or less." in html and "recommended band</span>" not in html


@needs_node
def test_the_cost_caption_states_the_served_assumptions(clients) -> None:
    html = _view(_get(clients["categorical"], primitive="choice", cost_s1=0.0002, cost_sys2=0.05), inputs={"inp-cost-s1": "0.0002", "inp-cost-s2": "0.05"})["html"]
    assert "System 1 on every record ($0.0002 a call" in html and "not accepted ($0.05 a call)" in html
    assert "$0.1 per wrong answer" in html and "at an assumed accuracy of 0.98." in html


# --- the payload kept is the one on screen -------------------------------------------------------------


def _broken(payloads: dict) -> list[dict]:
    """A response each branch's template throws on, after the tables were built."""
    band = json.loads(json.dumps(payloads["noul"]))
    band["active_band"]["f1"] = None
    no_band = json.loads(json.dumps(payloads["choice"]))
    no_band["group"] = None
    return [band, no_band]


@needs_node
def test_a_failed_or_unrenderable_response_keeps_nothing(payloads) -> None:
    out = _view(None)
    assert (out["kept"], out["tables"]) == (False, []) and "ERROR</span> boom" in out["html"]
    for bad in _broken(payloads):
        out = _view(bad)
        assert (out["kept"], out["tables"]) == (False, []), out["html"][:200]
        assert "ERROR</span>" in out["html"] and "<svg" not in out["html"]
    assert _view({**payloads["noul"], "tuned_threshold": None})["kept"] is False


@needs_node
def test_the_kept_payload_check_catches_keeping_before_rendering(payloads) -> None:
    functions = _mutated("      const tables = policyTables(d);\n", "      viewData.policy = d;\n      const tables = policyTables(d);\n")
    for bad in _broken(payloads):
        out = _view(bad, functions=functions)
        assert out["kept"] is True and "ERROR</span>" in out["html"]


HELD_STEPS = """
  const first = __FIRST__, second = __SECOND__;
  run(loadPolicy, "policy-container");
  pending.shift().resolve(first);
  await tick();
  out.first = snapshot().kept && viewData.policy.group.primitive;
  // A new request is out: nothing is kept until it answers.
  run(loadPolicy, "policy-container");
  out.pendingKept = "policy" in viewData;
  out.pendingTables = Object.keys(currentTables()).length;
  // Another starts before the second answers; the second's late answer changes nothing.
  run(loadPolicy, "policy-container");
  const [older, newer] = [pending.shift(), pending.shift()];
  newer.resolve(second);
  await tick();
  older.resolve(first);
  await tick();
  out.last = viewData.policy.group.primitive;
  out.shown = el("policy-container").innerHTML.includes("Escalation bands are noul-only");
"""


@needs_node
def test_a_new_request_drops_the_kept_payload_and_a_late_answer_does_not_bring_it_back(payloads) -> None:
    steps = HELD_STEPS.replace("__FIRST__", json.dumps(payloads["noul"])).replace("__SECOND__", json.dumps(payloads["choice"]))
    out = _view({}, steps=steps)
    assert out == {"first": "noul", "pendingKept": False, "pendingTables": 0, "last": "choice", "shown": True}
    mutant = _mutated("      delete viewData.policy;\n", "")
    assert _view({}, steps=steps, functions=mutant)["pendingKept"] is True


@needs_node
def test_a_sort_or_filter_survives_a_slider_move(payloads) -> None:
    d = payloads["noul"]
    g = d["group"]
    table = "policy:bands:" + json.dumps([g["question_id"], g["model_served"], g["primitive"]], separators=(",", ":"))
    steps = f"""
  TSTATE[{json.dumps(table)}] = {{sort: "f1", dir: "desc", q: "0.4"}};
  run(loadPolicy, "policy-container");
  pending.shift().resolve({json.dumps(d)});
  await tick();
  const html = el("policy-container").innerHTML;
  out.sorted = html.includes('aria-sort="descending"');
  out.filter = html.includes('value="0.4"');
  const entry = currentTables()[{json.dumps(table)}];
  const shown = viewRows(entry.spec, entry.rows, tableState(TSTATE, {json.dumps(table)}));
  out.count = shown.length;
  out.total = entry.rows.length;
  out.best = shown[0].f1 >= shown[shown.length - 1].f1;
  out.csvRows = toCsv(entry.spec, shown).trim().split("\\r\\n").length - 1;
  out.header = toCsv(entry.spec, shown).split("\\r\\n")[0];
"""
    out = _view({}, steps=steps)
    assert out["sorted"] and out["filter"] and out["best"]
    assert 0 < out["count"] < out["total"] == 67 and out["csvRows"] == out["count"]
    assert out["header"].startswith("question_id,model_served,primitive,distribution,band,lo,hi,role,n,escalated,")
