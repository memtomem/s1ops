"""The drift view rendered in node over real /api/drift/histograms payloads: the questions table,
each question's distribution overlay, quantiles and tables, empty sides, and the shell keeping a
response only once it is on screen."""

from __future__ import annotations

import html as html_lib
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.contracts.records import read_records
from s1ops.demo import generate_workspace
from s1ops.web.app import create_app
from tests.test_drift_histograms import _row, _write
from tests.test_web_harness import _mutated
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")


def _get(root: Path) -> dict:
    r = TestClient(create_app(workspace_dir=root)).get("/api/drift/histograms")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def payloads(tmp_path_factory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("ws")
    ws = generate_workspace(root / "noul")
    # Choice and score: the demo's categorical records split in two (the demo has no categorical canary).
    cat = root / "categorical"
    cat.mkdir()
    lines = Path(ws["records_categorical"]).read_text(encoding="utf-8").splitlines(keepends=True)
    (cat / "records.jsonl").write_text("".join(lines[::2]), encoding="utf-8")
    (cat / "records_canary.jsonl").write_text("".join(lines[1::2]), encoding="utf-8")
    # One side without an answered record, and neither side.
    empty = root / "empty-cur"
    empty.mkdir()
    _write(empty / "records.jsonl", [_row("noul", i, 0.1 + i / 50) for i in range(30)] + [_row("noul", 0, 0.5, qid="q2", served=None)])
    _write(empty / "records_canary.jsonl", [_row("noul", i, 0.0, served=None) for i in range(5)] + [_row("noul", 1, 0.5, qid="q2", served=None)])
    return {"noul": _get(root / "noul"), "categorical": _get(cat), "empty": _get(empty)}


def _render(d: dict, dq: str = "", hist: bool = True) -> str:
    opts = "driftOpts(d, {tstate: {}})" if hist else "undefined"
    return _node(f"const d = {json.dumps(d)};\n{_all_functions()}\nprocess.stdout.write(JSON.stringify(renderDrift(d.report, {json.dumps(dq)}, {opts})));")


def _js(expr: str, prelude: str = "") -> object:
    return _node(f"{prelude}\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


def _cards(html: str) -> list[str]:
    return re.split(r'<div class="card">\n            <h3[^>]*>Drift Monitor:', html)[1:]


def _svg(card: str, title: str) -> str:
    m = re.search(rf'<svg class="chart"[^>]*>(?:(?!</svg>).)*?<title>{title}.*?</svg>', card, re.S)
    assert m, title
    return m.group(0)


# --- the overlay ---------------------------------------------------------------------------------


@needs_node
@pytest.mark.parametrize("case", ["noul", "categorical"])
def test_each_side_draws_a_bar_per_bin_it_has_records_in(payloads, case) -> None:
    d = payloads[case]
    html = _render(d)
    cards = _cards(html)
    assert len(cards) == len(d["report"]["questions"])
    for q, card in zip(d["report"]["questions"], cards):
        h = next(h for h in d["histograms"]["questions"] if h["question_id"] == q["question_id"])
        svg = _svg(card, "Score distribution of")
        drawn = []
        for cls, side in (("ser-1", "ref"), ("ser-2", "cur")):
            bars = re.findall(rf'<rect class="bar {cls}" x="[^"]+" y="[^"]+" width="[^"]+" height="([^"]+)"', svg)
            counts = [c for c in h[side]["counts"] if c > 0]
            assert len(bars) == len(counts)
            drawn += [(float(b), c / h[side]["n_answered"]) for b, c in zip(bars, counts)]
        # Heights are the shares, on one scale for both sides (pixels are written to 0.1).
        tallest = max(drawn)
        ratio = tallest[0] / tallest[1]
        assert all(abs(hh - s * ratio) <= 0.2 for hh, s in drawn)
        # The chart names its card's bins table, which holds all ten bins of both sides.
        target = re.search(r'aria-details="([^"]+)"', svg).group(1)
        assert f'data-table="drift:bins:{q["question_id"]}" id="{target}"' in card
        table = card[card.index(f'id="{target}"'):]
        table = table[:table.index("</table>")]
        assert table.count("<tr>") == 1 + len(h["ref"]["counts"]) == 11
    if case == "categorical":
        assert "at least 1/K" in html and {q["primitive"] for q in d["report"]["questions"]} == {"choice", "score"}
    else:
        assert "at least 1/K" not in html


@needs_node
def test_the_bins_and_metrics_tables_hold_every_mark(payloads) -> None:
    d = payloads["noul"]
    q = d["report"]["questions"][0]
    tables = _js(f"Object.fromEntries(Object.entries(driftTables({json.dumps(d)}, {{}})).map(([k, t]) => [k, t.rows]))")
    bins = tables["drift:bins:" + q["question_id"]]
    h = d["histograms"]["questions"][0]
    assert [r["ref_count"] for r in bins] == h["ref"]["counts"] and [r["cur_count"] for r in bins] == h["cur"]["counts"]
    assert [r["psi_term"] for r in bins] == h["psi_terms"]
    assert [r["closed"] for r in bins] == [False] * 9 + [True]
    metrics = {r["metric"]: r for r in tables["drift:metrics:" + q["question_id"]]}
    for k in ("p10", "p50", "p90", "mean"):
        assert (metrics["score " + k]["ref"], metrics["score " + k]["cur"]) == (q["quantiles_ref"][k], q["quantiles_cur"][k])
    assert metrics["mean entropy (bits)"]["change"] == q["entropy_delta"]
    assert metrics["latency p95 (ms)"]["ref"] == q["latency_ref"]["p95_ms"]
    # The quantile chart names the metrics table.
    card = _cards(_render(d))[0]
    target = re.search(r'aria-details="([^"]+)"', _svg(card, "Score quantiles of")).group(1)
    assert f'data-table="drift:metrics:{q["question_id"]}" id="{target}"' in card


@needs_node
def test_a_bar_tooltip_is_its_own_bins_numbers(payloads) -> None:
    d = payloads["noul"]
    h = d["histograms"]["questions"][0]
    svg = _svg(_cards(_render(d))[0], "Score distribution of")
    for cls, side, name in (("ser-1", "ref", "reference"), ("ser-2", "cur", "current")):
        tips = [html_lib.unescape(t) for t in re.findall(rf'<rect class="bar {cls}"[^>]* data-tip="([^"]*)"', svg)]
        bins = [i for i, c in enumerate(h[side]["counts"]) if c > 0]
        assert len(tips) == len(bins)
        for tip, i in zip(tips, bins):
            lo, hi = h["edges"][i], h["edges"][i + 1]
            # The edges as the PSI uses them, so a score that prints as the upper edge is not shown as excluded.
            assert tip.startswith(f"{name}, bin [{lo!r}, {hi!r}{']' if i == 9 else ')'}".replace("0.0,", "0,").replace("1.0]", "1]"))
            assert f"\n{h[side]['counts'][i]} of {h[side]['n_answered']} answered" in tip
            assert f"PSI term {h['psi_terms'][i]:.4f}" in tip


# --- empty sides -----------------------------------------------------------------------------------


@needs_node
def test_a_side_without_answers_shows_no_measurement(payloads) -> None:
    d = payloads["empty"]
    by_q = {q["question_id"]: q for q in d["report"]["questions"]}
    assert by_q["q"]["n_cur_answered"] == 0 and by_q["q"]["quantiles_cur"]["p50"] == 0.0  # the report's zeros
    cards = dict(zip([q["question_id"] for q in d["report"]["questions"]], _cards(_render(d))))
    one = cards["q"]
    assert "No answered records" in _svg(one, "Score quantiles of")
    assert '<rect class="bar ser-2"' not in one
    assert re.search(r'<div class="kpi">[0-9.]+ → N/A</div>', one)
    tables = _js(f"Object.fromEntries(Object.entries(driftTables({json.dumps(d)}, {{}})).map(([k, t]) => [k, t.rows]))")
    metrics = {r["metric"]: r for r in tables["drift:metrics:q"]}
    assert metrics["score p50"]["cur"] is None and metrics["score p50"]["change"] is None
    assert metrics["mean entropy (bits)"]["cur"] is None and metrics["mean entropy (bits)"]["change"] is None
    assert {r["cur_share"] for r in tables["drift:bins:q"]} == {None}
    rows = {r["question_id"]: r for r in _js(f"driftQuestionsTable({json.dumps(d['report'])}).rows")}
    assert rows["q"]["p50_cur"] is None and rows["q"]["p50_ref"] is not None
    assert "Neither side has an answered record." in cards["q2"] and '<svg class="chart"' not in cards["q2"]


@needs_node
def test_latency_prints_zero_and_none() -> None:
    assert _js('[fmtMs(0), fmtMs(39.71), fmtMs(null), fmtMs(undefined)]') == ["0 ms", "39.71 ms", "N/A", "N/A"]


@needs_node
def test_a_measured_zero_latency_is_printed(payloads) -> None:
    d = json.loads(json.dumps(payloads["noul"]))
    d["report"]["questions"][0]["latency_ref"]["p95_ms"] = 0
    assert "p95 Latency: 0 ms → " in _render(d)


# --- the questions table -----------------------------------------------------------------------------


def _three(d: dict) -> dict:
    q = d["report"]["questions"][0]
    qs = [{**q, "question_id": f"q-{s.lower()}", "status": s} for s in ("OK", "CRITICAL", "WARN")]
    return {**d["report"], "questions": qs}


@needs_node
def test_questions_are_listed_most_severe_first_and_sort_by_severity(payloads) -> None:
    report = _three(payloads["noul"])
    out = _js(f"""(() => {{
      const t = driftQuestionsTable({json.dumps(report)});
      return {{given: t.rows.map(r => r.status), asc: sortRows(t.spec, t.rows, "status", "asc").map(r => r.status),
               desc: sortRows(t.spec, t.rows, "status", "desc").map(r => r.status), csv: toCsv(t.spec, t.rows).split("\\r\\n")[0]}};
    }})()""")
    assert out["given"] == ["CRITICAL", "WARN", "OK"]
    assert out["asc"] == ["OK", "WARN", "CRITICAL"] and out["desc"] == ["CRITICAL", "WARN", "OK"]
    assert out["csv"].startswith("status,severity,question_id,")


@needs_node
def test_show_chooses_the_question_and_names_it(payloads) -> None:
    report = _three(payloads["noul"])
    html = _js(f"renderDrift({json.dumps(report)}, '', undefined)")
    for qid in ("q-ok", "q-critical", "q-warn"):
        assert f'data-act="drift-show" data-qid="{qid}" aria-label="Show: {qid}">Show</button>' in html
    # The whole report's table stays above one chosen question's card.
    one = _js(f"renderDrift({json.dumps(report)}, 'q-warn', undefined)")
    assert one.count('data-act="drift-show"') == 3 and len(_cards(one)) == 1
    assert "Distribution not loaded." in one


# --- the shell keeps a response only once it is on screen ---------------------------------------


def _shell_run(steps: str, payload: dict, functions: str | None = None) -> dict:
    js = f"""
let STATE = {{view: "drift"}};
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [], groupsState = "ok", groupsError = "";
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
STATE = {{...defaultState(), view: "drift"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


RERENDER_STEPS = """
  runView("drift");
  out.url = pending[0].url;
  pending.shift().resolve(payload);
  await tick(); await tick();
  const qid = payload.report.questions[0].question_id;
  out.first = el("drift-container").innerHTML.includes("<title>Score distribution of");
  // Choosing the question re-renders from what is kept (no request), and so does coming back.
  STATE = {...STATE, dq: qid};
  runView("drift");
  out.requests = pending.length;
  out.chosen = el("drift-container").innerHTML.includes("<title>Score distribution of");
  STATE = {...STATE, view: "lint"};
  STATE = {...STATE, view: "drift"};
  runView("drift");
  out.back = el("drift-container").innerHTML.includes("<title>Score distribution of");
  const t = currentTables()["drift:bins:" + qid];
  out.binRows = t ? t.rows.length : 0;
  out.binCounts = t ? t.rows.map(r => r.ref_count) : null;
"""


@needs_node
def test_a_question_chosen_after_loading_keeps_its_overlay_and_tables(payloads) -> None:
    d = payloads["noul"]
    out = _shell_run(RERENDER_STEPS, d)
    assert out["url"] == "/api/drift/histograms"
    assert (out["first"], out["requests"], out["chosen"], out["back"]) == (True, 0, True, True)
    assert out["binRows"] == 10 and out["binCounts"] == d["histograms"]["questions"][0]["ref"]["counts"]


@needs_node
def test_the_rerender_check_catches_a_misread_kept_response(payloads) -> None:
    # The round 2 finding: a re-render reading another key than the loader's loses the overlay.
    functions = _mutated("hist: d && d.histograms ? d.histograms : null", "hist: d && d.hist ? d.hist : null")
    out = _shell_run(RERENDER_STEPS, payloads["noul"], functions=functions)
    assert out["first"] is False and out["chosen"] is False


FAIL_STEPS = """
  runView("drift");
  pending.shift().reject(Object.assign(new Error("boom"), {status: 400}));
  await tick(); await tick();
  out.failedError = el("drift-container").innerHTML.includes("ERROR</span> boom");
  out.failedKept = "drift" in viewData;
  // A response the renderer throws on (a question without its quantiles): the error card, nothing kept.
  generation += 1;
  loadedSig = {};
  runView("drift");
  pending.shift().resolve({...payload, report: {...payload.report, questions: [{...payload.report.questions[0], quantiles_ref: null}]}});
  await tick(); await tick();
  out.error = el("drift-container").innerHTML.includes("ERROR</span>");
  out.kept = "drift" in viewData;
  out.tables = Object.keys(currentTables()).length;
"""


@needs_node
def test_a_failed_or_unrenderable_response_keeps_nothing(payloads) -> None:
    assert _shell_run(FAIL_STEPS, payloads["noul"]) == {"failedError": True, "failedKept": False, "error": True, "kept": False, "tables": 0}
    # Kept before rendering, it would stay behind the error card.
    functions = _mutated(
        '      document.getElementById("drift-container").innerHTML = renderDrift(d.report, ctx.dq ? ctx.dq() : "", driftOpts(d, ctx.opts ? ctx.opts() : {}));\n      if (ctx.keep) ctx.keep(d);\n',
        '      if (ctx.keep) ctx.keep(d);\n      document.getElementById("drift-container").innerHTML = renderDrift(d.report, ctx.dq ? ctx.dq() : "", driftOpts(d, ctx.opts ? ctx.opts() : {}));\n',
    )
    assert _shell_run(FAIL_STEPS, payloads["noul"], functions=functions)["kept"] is True


REFRESH_STEPS = """
  runView("drift");
  generation += 1;
  loadedSig = {};
  viewData = {};
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.kept = "drift" in viewData;
  out.rendered = el("drift-container").innerHTML.includes("<title>Score distribution of");
"""


@needs_node
def test_a_request_a_refresh_outdated_keeps_and_renders_nothing(payloads) -> None:
    assert _shell_run(REFRESH_STEPS, payloads["noul"]) == {"kept": False, "rendered": False}


@needs_node
def test_the_quantile_marks_sit_on_the_served_quantiles(payloads) -> None:
    d = payloads["noul"]
    q = d["report"]["questions"][0]
    svg = _svg(_cards(_render(d))[0], "Score quantiles of")
    x = lambda v: 76 + v * (460 - 14 - 76)  # the chart's scale: [0, 1] onto [ml, W - mr]
    bars = re.findall(r'<rect class="bar (ser-[12])" x="([^"]+)" y="[^"]+" width="([^"]+)"', svg)
    ticks = re.findall(r'<polyline class="line (ser-[12])" points="([^ ]+),[^ ]+ ([^,]+),', svg)
    dots = re.findall(r'<path class="dot (ser-[12])" d="M([^ ]+) ', svg)
    for cls, side in (("ser-1", "quantiles_ref"), ("ser-2", "quantiles_cur")):
        qs = q[side]
        bx, bw = next((float(b[1]), float(b[2])) for b in bars if b[0] == cls)
        assert abs(bx - (x(qs["p10"]) + 1)) <= 0.06 and abs(bw - (x(qs["p90"]) - x(qs["p10"]) - 2)) <= 0.11
        tx = next(float(t[1]) for t in ticks if t[0] == cls)
        assert abs(tx - x(qs["p50"])) <= 0.06
        mx = next(float(m[1]) for m in dots if m[0] == cls)
        assert abs(mx - x(qs["mean"])) <= 0.06


@needs_node
def test_only_a_chosen_cards_heading_can_take_the_focus(payloads) -> None:
    d = payloads["categorical"]
    qid = d["report"]["questions"][0]["question_id"]
    assert "tabindex" not in _render(d)
    one = _render(d, qid)
    assert one.count('tabindex="-1"') == 1 and '<h3 tabindex="-1">Drift Monitor:' in one
