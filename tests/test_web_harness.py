"""The shape the page's tests depend on, and every tab rendering hostile workspace strings as text.

The escape scan in tests/test_web_errors.py reads the template literals; it cannot see a string
that reaches `innerHTML` through a helper or as a mapping key. Here each loader runs in node over
real API responses whose strings, and an extra key in every object, are all hostile."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import shutil

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import save_policies
from s1ops.demo import generate_workspace
from s1ops.eval.compare import compare_records
from s1ops.improve.trainrun import load_trainrun
from s1ops.web.app import STATIC_DIR, create_app
from tests.test_checkpoints import build_workspace, policy_entry
from tests.test_web_render import DECLARATION, _all_functions, _declarations, _node, _script

HTML = (STATIC_DIR / "index.html").read_text()
HOSTILE = "\"'><img src=x onerror=alert(1)>"
ESCAPED = "&lt;img src=x onerror=alert(1)&gt;"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")


# --- structure ---------------------------------------------------------------------------------


def test_the_page_has_one_script_element() -> None:
    # Every extraction slices from the first `<script>` to the first `</script>`.
    assert HTML.count("<script") == 1
    assert HTML.count("</script>") == 1


def test_every_top_level_function_is_extracted_once() -> None:
    # Found by a looser pattern than the extractor's, so a declaration shape it does not know (an
    # extra space, a generator) shows here instead of being left out.
    starts = re.findall(r"^    (?:async\s+)?function\b\s*\*?\s*(\w+)", _script(), re.M)
    names = [name for name, _ in _declarations()]
    assert names == starts
    assert len(names) == len(set(names))


def _parse_failures(declarations: dict[str, str], top_level: str) -> list[str]:
    """Names of the declarations that do not parse as one function expression, and "(top level)"
    when the remaining code does not parse as a script."""
    js = f"""
const vm = require("vm");
const parses = src => {{ try {{ new vm.Script(src); return true; }} catch (_) {{ return false; }} }};
const declarations = {json.dumps(declarations)};
const broken = Object.keys(declarations).filter(name => !parses("(" + declarations[name] + "\\n)"));
if (!parses({json.dumps(top_level)})) broken.push("(top level)");
process.stdout.write(JSON.stringify(broken));
"""
    return _node(js)


def _whole_problems(script: str) -> list[str]:
    """Why the page's declarations would not extract exactly: an extraction that is not one function
    expression, or the code left once every declaration is cut out not parsing.

    A body line that is exactly four spaces and a closing brace ends an extraction early, leaving
    the cut-off tail at the top level even when the head happens to parse. A closing line that is
    not exactly that (`    };`) runs the extraction on into the next top-level block, which would
    then run in every test; wrapped in parentheses, the over-long source no longer parses."""
    declarations = _declarations(script)
    rest = script
    for _, src in declarations:
        rest = rest.replace(src, "", 1)
    return _parse_failures(dict(declarations), rest)


@needs_node
def test_every_extracted_function_is_whole() -> None:
    assert _whole_problems(_script()) == []


@needs_node
def test_a_closing_line_inside_a_body_is_caught() -> None:
    script = _script()
    old = "        return;\n      }\n      const data = await fetchJson(apiUrl(\"/api/audit\""
    assert script.count(old) == 1
    assert _whole_problems(script.replace(old, "        return;\n      }\n    }\n      const data = await fetchJson(apiUrl(\"/api/audit\"")) == ["(top level)"]


@needs_node
def test_an_extraction_running_into_top_level_code_is_caught() -> None:
    script = _script()
    old = "\"'\": \"&#39;\"}[c]));\n    }\n"
    assert script.count(old) == 1
    overrun = "\"'\": \"&#39;\"}[c]));\n    };\n    if (true) {\n      run(loadAudit, \"calib-container\");\n    }\n"
    mutated = script.replace(old, overrun)
    assert "run(loadAudit" in dict(_declarations(mutated))["esc"]  # the overrun this guards against
    assert _whole_problems(mutated) == ["esc"]


def test_the_declaration_pattern_needs_the_closing_line_alone() -> None:
    assert DECLARATION.search("    function f() {\n      return 1;\n    }\n")
    assert not DECLARATION.search("    function f() {\n      return 1;\n    });\n")


@pytest.mark.parametrize("head", ["function f(", "async function f(", "function* f(", "function *f(", "async function* f("])
def test_the_declaration_pattern_extracts_every_declaration_shape(head) -> None:
    script = f"    {head}) {{\n      return 1;\n    }}\n"
    assert _declarations(script) == [("f", script.rstrip("\n"))]
    assert not DECLARATION.search("    functionf() {\n      return 1;\n    }\n")


# --- hostile strings ---------------------------------------------------------------------------


def _hostile(value: object) -> object:
    """Every string replaced, and every non-empty object given one more key, hostile, holding a copy
    of its first value: a renderer that reads fields by name ignores the key, while one that
    iterates an object (criterion names, for one) renders it."""
    if isinstance(value, str):
        return HOSTILE
    if isinstance(value, list):
        return [_hostile(v) for v in value]
    if isinstance(value, dict):
        out = {k: _hostile(v) for k, v in value.items()}
        if value:
            out[HOSTILE] = _hostile(next(iter(value.values())))
        return out
    return value


# Inputs the policy loader reads, at the page's defaults.
_INPUTS = {
    "inp-tau-low": "0.25", "inp-tau-high": "0.75", "inp-cost-s1": "0.00004", "inp-cost-s2": "0.004",
    "inp-err-budget": "0.10", "inp-dist": "", "inp-group": "",
}


def _render(loader: str, routes: dict[str, dict], functions: str | None = None) -> list[str]:
    """Run `loader` with `fetchJson` answering from `routes` (by path, query ignored) and return the
    HTML of every element it wrote."""
    js = f"""
let policySeq = 0;
let GROUPS = [];
let viewData = {{}}, TSTATE = {{}};
const els = {{}};
const inputs = {json.dumps(_INPUTS)};
const el = id => (els[id] ??= {{value: inputs[id] ?? "", textContent: "", innerHTML: "", style: {{}}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const routes = {json.dumps(routes)};
const fetchJson = async url => {{
  const path = url.split("?")[0];
  if (!(path in routes)) throw new Error("no route for " + path);
  return routes[path];
}};
{functions if functions is not None else _all_functions()}
{loader}().then(() => process.stdout.write(JSON.stringify(Object.values(els).map(e => e.innerHTML).filter(Boolean))));
"""
    return _node(js)


@pytest.fixture(scope="module")
def payloads(tmp_path_factory) -> dict[str, dict[str, dict]]:
    """Each loader's routes, read from the demo workspace through the real API."""
    root = tmp_path_factory.mktemp("ws")
    ws = generate_workspace(root / "w")
    client = TestClient(create_app(workspace_dir=root / "w"))
    cat = TestClient(create_app(workspace_dir=root / "w", records_path=ws["records_categorical"], fits_path=ws["fits_categorical"]))

    def get(c: TestClient, url: str, **params) -> dict:
        r = c.get(url, params=params)
        assert r.status_code == 200, r.text
        return r.json()

    lint, drift = get(client, "/api/lint"), get(client, "/api/drift/histograms")
    # Strings the demo leaves empty or null, filled so that their rendering is exercised too.
    lint_filled, drift_filled = copy.deepcopy(lint), copy.deepcopy(drift)
    for p in lint_filled["probe_audit"]:
        p["inverse_source"] = p["inverse_source"] or "generated"
    for q in drift_filled["report"]["questions"]:
        q["notes"] = q["notes"] or ["a note"]
        q["alerts"] = q["alerts"] or ["an alert"]
    # A file-level finding: no question, no evidence.
    lint_file_level = copy.deepcopy(lint)
    lint_file_level["findings"].append({
        "rule": "JL000", "severity": "error", "question": None, "question_id": None,
        "message": "m", "fix": "f", "evidence": None, "evidence_url": None,
    })
    # Two checkpoints with manifests, run records, step metrics and trained items; the second's
    # manifest is exported again and its policy names the first's run, so every note is rendered.
    (root / "checkpoints").mkdir()
    ck = build_workspace(root / "checkpoints")
    save_policies([policy_entry("tickets-r2", load_trainrun(ck / "trainrun.json").trainrun_id)], ck / "policy.json")
    r2_manifest = ck / "training_manifest-r2.json"
    r2_manifest.write_text(r2_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    (ck / "trainrun-r2.json").rename(root / "trainrun-r2.json")
    checkpoints = get(TestClient(create_app(workspace_dir=ck, training_run_paths=[root / "trainrun-r2.json"])), "/api/checkpoints")
    # The review sheet README's walkthrough writes, read by a loopback client (the route refuses the
    # test client's own address and Host).
    sheet, key = root / "blind_sheet.md", root / "blind_key.json"
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli_main(["labelaudit", "sheet", ws["records"], "--states", ws["states_local"], "--questions", ws["questions"],
                         "-o", str(sheet), "-k", str(key), "--top", "3", "--controls", "3"]) == 0
    loopback = TestClient(create_app(workspace_dir=root / "w", sheet_path=sheet), client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8765")
    review = get(loopback, "/api/sheet")
    # Item 1 answered in the file, and no question named (a 0.6.0 sheet): both lines are rendered.
    review_answered = copy.deepcopy(review)
    review_answered["sheet"]["items"][0]["answer"] = review_answered["sheet"]["options"][0]
    review_answered["sheet"]["question_id"] = None
    audit_filled = get(client, "/api/audit")
    for g in audit_filled["groups"]:
        g["warnings"] = g["warnings"] or ["a warning"]
    return {
        "overview": {"/api/overview": get(client, "/api/overview")},
        "audit-noul": {"/api/audit": get(client, "/api/audit")},
        "audit-categorical": {"/api/audit": get(cat, "/api/audit")},
        "audit-filled": {"/api/audit": audit_filled},
        "policy-noul": {"/api/simulate-policy": get(client, "/api/simulate-policy")},
        "policy-choice": {"/api/simulate-policy": get(cat, "/api/simulate-policy", primitive="choice")},
        "policy-score": {"/api/simulate-policy": get(cat, "/api/simulate-policy", primitive="score")},
        "lint": {"/api/lint": lint},
        "lint-filled": {"/api/lint": lint_filled},
        "lint-file-level": {"/api/lint": lint_file_level},
        "slices": {"/api/slices": get(client, "/api/slices"), "/api/labelaudit": get(client, "/api/labelaudit")},
        "slices-categorical": {"/api/slices": get(cat, "/api/slices"), "/api/labelaudit": get(cat, "/api/labelaudit")},
        "compare": {"/api/compare": get(client, "/api/compare")},
        "compare-categorical": {"/api/compare": {"shadow": _categorical_compare(), "judge": None}},
        "drift": {"/api/drift/histograms": drift},
        "drift-filled": {"/api/drift/histograms": drift_filled},
        "checkpoints": {"/api/checkpoints": checkpoints},
        "checkpoints-demo": {"/api/checkpoints": get(client, "/api/checkpoints")},
        "review": {"/api/sheet": review},
        "review-answered": {"/api/sheet": review_answered},
    }


def _categorical_compare() -> dict:
    """A comparison report whose flipped pairs hold option names (choice) and levels (score), which
    the demo's noul question never does."""
    def row(kind: str, qid: str, probs: dict[str, float], model: str, **extra: object) -> dict:
        return {"schema": "s1-record/1", "primitive": kind, "question_id": qid, "dataset": "d", "item_id": "i0",
                "model_served": model, "probabilities": probs, **extra}

    primary = [row("choice", "q-choice", {"refund": 0.7, "other": 0.3}, "old", choice="refund", label="refund"),
               row("score", "q-score", {"0": 0.1, "1": 0.9}, "old")]
    shadow = [row("choice", "q-choice", {"refund": 0.2, "other": 0.8}, "new", choice="other", label="refund"),
              row("score", "q-score", {"0": 0.8, "1": 0.2}, "new")]
    return compare_records(primary, shadow)


LOADERS = {
    "overview": "loadOverview",
    "audit-noul": "loadAudit",
    "audit-categorical": "loadAudit",
    "audit-filled": "loadAudit",
    "policy-noul": "loadPolicy",
    "policy-choice": "loadPolicy",
    "policy-score": "loadPolicy",
    "lint": "loadLint",
    "lint-filled": "loadLint",
    "lint-file-level": "loadLint",
    "slices": "loadSlicesAndLabels",
    "slices-categorical": "loadSlicesAndLabels",
    "compare": "loadCompare",
    "compare-categorical": "loadCompare",
    "drift": "loadDrift",
    "drift-filled": "loadDrift",
    "checkpoints": "loadCheckpoints",
    "checkpoints-demo": "loadCheckpoints",
    "review": "loadReview",
    "review-answered": "loadReview",
}


def _hostile_routes(routes: dict[str, dict]) -> dict[str, dict]:
    return {path: _hostile(body) for path, body in routes.items()}


@needs_node
@pytest.mark.parametrize("case", sorted(LOADERS))
def test_every_tab_renders_hostile_strings_as_text(payloads, case) -> None:
    html = "".join(_render(LOADERS[case], _hostile_routes(payloads[case])))
    assert "<img" not in html
    # The hostile strings did reach the page, escaped: the case exercised its rendering.
    assert ESCAPED in html


@needs_node
@pytest.mark.parametrize("case", sorted(LOADERS))
def test_every_tab_renders_the_unchanged_payload(payloads, case) -> None:
    # The same loaders over the payloads as served, so a hostile-only failure is the escaping's.
    assert _render(LOADERS[case], payloads[case])


def _mutated(old: str, new: str) -> str:
    script = _script()
    assert script.count(old) == 1, old
    return _all_functions(script=script.replace(old, new))


@needs_node
@pytest.mark.parametrize(
    "case, old, new",
    [
        pytest.param("drift", "Drift Monitor: <span class=\"badge badge-info\">${esc(q.question_id)}</span>",
                     "Drift Monitor: <span class=\"badge badge-info\">${q.question_id}</span>", id="string-value"),
        pytest.param("policy-noul", "On ${esc(d.distribution || 'unknown')} probabilities",
                     "On ${d.distribution || 'unknown'} probabilities", id="policy-value"),
        # Empty or null in the demo: caught only by the filled cases.
        pytest.param("drift-filled", "margin:4px 0;\">${esc(n)}</div>", "margin:4px 0;\">${n}</div>", id="drift-note"),
        # The calibration view's kits: a table cell (metric names are mapping keys), the chart's
        # title, and the warnings (empty in the demo).
        pytest.param("audit-noul", "${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>", id="table-cell"),
        pytest.param("audit-noul", "<title>${esc(spec.title)}</title>", "<title>${spec.title}</title>", id="chart-title"),
        pytest.param("audit-filled", "WARN</span> ${esc(w)}</div>", "WARN</span> ${w}</div>", id="audit-warning"),
        # The simulator's sections: the recommended band (a served string), a band in a tooltip, a
        # chart's title and a band cell of the table kit.
        pytest.param("policy-noul", '<div class="kpi">${esc(rec.band)}</div>', '<div class="kpi">${rec.band}</div>', id="policy-recommended"),
        pytest.param("policy-noul", 'r="10" data-tip="${esc(s.tip)}" />', 'r="10" data-tip="${s.tip}" />', id="policy-band-tip"),
        pytest.param("policy-choice", "<title>${esc(spec.title)}</title>", "<title>${spec.title}</title>", id="policy-chart-title"),
        pytest.param("policy-noul", "${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>", id="policy-table-cell"),
        # The drift view's kits: a questions-table cell, a row button's name (it holds a question
        # id), and the overlay's title.
        pytest.param("drift", "${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>", id="drift-table-cell"),
        pytest.param("drift", 'aria-label="${esc(c.text + ": " + r[c.labelKey])}"', 'aria-label="${c.text + ": " + r[c.labelKey]}"', id="drift-show-name"),
        pytest.param("drift", "<title>${esc(spec.title)}</title>", "<title>${spec.title}</title>", id="drift-chart-title"),
        # The slices view: a lane's name, a mark's tooltip (both hold a slice or feature name), the
        # chart's title, a table cell (the ranking's ids), and the two served distributions.
        pytest.param("slices", 'text-anchor="end">${esc(text)}</text>', 'text-anchor="end">${text}</text>', id="slices-lane-name"),
        pytest.param("slices", 'r="10" data-tip="${esc(p.tip)}" />', 'r="10" data-tip="${p.tip}" />', id="slices-mark-tip"),
        pytest.param("slices", "<title>${esc(spec.title)}</title>", "<title>${spec.title}</title>", id="slices-chart-title"),
        pytest.param("slices-categorical", "${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>", id="slices-table-cell"),
        pytest.param("slices", "Margins from ${esc(l.distribution)} probabilities.", "Margins from ${l.distribution} probabilities.", id="slices-ranking-distribution"),
        pytest.param("slices-categorical", "Predictions from ${esc(o.dist || 'unknown')} probabilities.", "Predictions from ${o.dist || 'unknown'} probabilities.", id="slices-distribution"),
        # The compare view: a lane's name and a mark's tooltip (both hold a question id), a flipped
        # pair's cell (an option name), and a model's name in a card.
        pytest.param("compare", 'text-anchor="end">${esc(text)}</text>', 'text-anchor="end">${text}</text>', id="compare-lane-name"),
        pytest.param("compare", 'r="10" data-tip="${esc(p.tip)}" />', 'r="10" data-tip="${p.tip}" />', id="compare-mark-tip"),
        pytest.param("compare-categorical", "${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>", id="compare-table-cell"),
        pytest.param("compare", '<div class="sub">${esc(q.primary_model)} vs', '<div class="sub">${q.primary_model} vs', id="compare-model"),
        # The lint view: a probe item's tooltip, a table cell, the questions file's path and a probe
        # group's name in its block heading.
        pytest.param("lint", 'r="10" data-tip="${esc(s.tip)}" />', 'r="10" data-tip="${s.tip}" />', id="lint-point-tip"),
        pytest.param("lint-file-level", "${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>", id="lint-table-cell"),
        pytest.param("lint", 'File: <span class="mono">${esc(d.questions_file)}</span>', 'File: <span class="mono">${d.questions_file}</span>', id="lint-file-path"),
        pytest.param("lint", "Group ${Number(index) + 1}: ${esc(p.probe_question)}", "Group ${Number(index) + 1}: ${p.probe_question}", id="lint-probe-name"),
        # The checkpoint view: a name in a card's heading, a note's item, a picker's option, a step
        # metric's name (a mapping key) as an axis name, and a table cell.
        pytest.param("checkpoints", 'Checkpoint: <span class="badge badge-info">${esc(c.checkpoint)}</span>',
                     'Checkpoint: <span class="badge badge-info">${c.checkpoint}</span>', id="checkpoint-name"),
        pytest.param("checkpoints", "first <span class=\"mono\">${esc(itemText(hit.first_trained_hit))}</span>",
                     "first <span class=\"mono\">${itemText(hit.first_trained_hit)}</span>", id="checkpoint-trained-item"),
        pytest.param("checkpoints", "${selected}>${esc(name)}</option>", "${selected}>${name}</option>", id="checkpoint-picker"),
        pytest.param("checkpoints", 'text-anchor="middle">${esc(ax.yName)}</text>', 'text-anchor="middle">${ax.yName}</text>', id="checkpoint-metric-name"),
        pytest.param("checkpoints", "names ${esc(p.trainrun_id)}, and", "names ${p.trainrun_id}, and", id="checkpoint-policy-run"),
        pytest.param("checkpoints-demo", "${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>", id="checkpoint-table-cell"),
        # The review view: an item's state text, the sheet's header and file name, an option's label
        # (a radio names its option by position), and the shown answer (only in a sheet that has one).
        pytest.param("review", '<pre class="review-state">\\n${esc(it.state)}</pre>', '<pre class="review-state">\\n${it.state}</pre>', id="review-state"),
        pytest.param("review", '<pre class="review-state">\\n${esc(sheet.header)}</pre>', '<pre class="review-state">\\n${sheet.header}</pre>', id="review-header"),
        pytest.param("review", "Blind label review: ${esc(sheet.file_name)}</h3>", "Blind label review: ${sheet.file_name}</h3>", id="review-file-name"),
        pytest.param("review", '` : ""}${esc(opt)}</label>`', '` : ""}${opt}</label>`', id="review-option-label"),
        pytest.param("review-answered", "Your answer: <strong>${esc(answer)}</strong>", "Your answer: <strong>${answer}</strong>", id="review-answer"),
    ],
)
def test_the_hostile_check_catches_a_dropped_esc(payloads, case, old, new) -> None:
    html = "".join(_render(LOADERS[case], _hostile_routes(payloads[case]), functions=_mutated(old, new)))
    assert "<img" in html


def _column(html: str, caption: str, index: int) -> list[str]:
    """The cells of one column of the table named `caption`, as written (still escaped)."""
    table = re.search(rf'<table aria-label="{re.escape(caption)}">.*?</table>', html, re.S).group(0)
    rows = re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>")[1], re.S)
    return [re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)[index] for row in rows]


def _escaped_once(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


@needs_node
@pytest.mark.parametrize("case", ["criterion", "inverse-wording"])
def test_one_hostile_value_reaches_its_own_cell_escaped_once(payloads, case) -> None:
    # The two values the page once wrote by hand (a criterion name, which is a mapping key, and a
    # probe group's inverse wording) now go through the table kit. Only that value is hostile here,
    # so the check cannot pass on another cell's: it must be in its own column, escaped once, and
    # the kit's cell escape must be what keeps it text.
    if case == "criterion":
        loader, route, caption, index = "loadCompare", "/api/compare", "Judge agreement by criterion", 0
        body = copy.deepcopy(payloads["compare"][route])
        name = next(iter(body["judge"]["by_criterion"]))
        body["judge"]["by_criterion"] = {HOSTILE: body["judge"]["by_criterion"].pop(name), **body["judge"]["by_criterion"]}
    else:
        loader, route, caption, index = "loadLint", "/api/lint", "Probe groups", 9
        body = copy.deepcopy(payloads["lint"][route])
        body["probe_audit"][0]["inverse_source"] = HOSTILE
    html = "".join(_render(loader, {route: body}))
    assert "<img" not in html
    assert _column(html, caption, index).count(_escaped_once(HOSTILE)) == 1
    mutant = "".join(_render(loader, {route: body}, functions=_mutated("${esc(cellText(c, r))}</td>", "${cellText(c, r)}</td>")))
    assert _column(mutant, caption, index).count(HOSTILE) == 1


# --- hostile error messages --------------------------------------------------------------------

# Each view's loader and the container its errors replace. A 400's detail names workspace files and
# identifiers, which reach innerHTML through showError's interpolation of `err.message`.
CONTAINERS = {
    "loadOverview": "overview-container",
    "loadAudit": "calib-container",
    "loadPolicy": "policy-container",
    "loadLint": "lint-container",
    "loadSlicesAndLabels": "slices-container",
    "loadCompare": "compare-container",
    "loadDrift": "drift-container",
    "loadCheckpoints": "checkpoints-container",
    "loadReview": "review-container",
}


def test_the_error_containers_are_the_pages_own() -> None:
    # The views' table in the page (`viewDefs`), which the shell runs each loader from.
    defs = re.findall(r'^        \w+: \{loader: (\w+), container: "([^"]+)"', _script(), re.M)
    assert len(defs) == 9
    assert CONTAINERS == dict(defs)
    assert set(CONTAINERS) == set(LOADERS.values())


def _render_error(loader: str, functions: str | None = None) -> list[list[str]]:
    """Run `run(loader, container)` with `fetchJson` failing with a hostile message, as it does for a
    400 whose detail holds a workspace identifier; return `[id, html]` for every element written."""
    js = f"""
let policySeq = 0;
let GROUPS = [];
let viewData = {{}}, TSTATE = {{}};
const els = {{}};
const inputs = {json.dumps(_INPUTS)};
const el = id => (els[id] ??= {{value: inputs[id] ?? "", textContent: "", innerHTML: "", style: {{}}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const fetchJson = async () => {{ throw new Error({json.dumps(HOSTILE)}); }};
{functions if functions is not None else _all_functions()}
run({loader}, {json.dumps(CONTAINERS[loader])}).then(() => process.stdout.write(JSON.stringify(
  Object.entries(els).filter(([, e]) => e.innerHTML).map(([id, e]) => [id, e.innerHTML]))));
"""
    return _node(js)


@needs_node
@pytest.mark.parametrize("loader", sorted(CONTAINERS))
def test_every_tab_shows_a_hostile_error_message_as_text(loader) -> None:
    written = _render_error(loader)
    assert [id_ for id_, _ in written] == [CONTAINERS[loader]]
    html = written[0][1]
    assert "<img" not in html
    assert ESCAPED in html


@needs_node
def test_the_error_check_catches_a_dropped_esc() -> None:
    functions = _mutated("ERROR</span> ${esc(err.message)}</div>", "ERROR</span> ${err.message}</div>")
    assert "<img" in _render_error("loadAudit", functions=functions)[0][1]
