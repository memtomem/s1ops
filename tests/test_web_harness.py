"""The shape the page's tests depend on, and every tab rendering hostile workspace strings as text.

The escape scan in tests/test_web_errors.py reads the template literals; it cannot see a string
that reaches `innerHTML` through a helper or as a mapping key. Here each loader runs in node over
real API responses whose strings, and an extra key in every object, are all hostile."""

from __future__ import annotations

import copy
import json
import re
import shutil

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.web.app import STATIC_DIR, create_app
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
    old = "        return;\n      }\n      el.innerHTML = data.groups.map"
    assert script.count(old) == 1
    assert _whole_problems(script.replace(old, "        return;\n      }\n    }\n      el.innerHTML = data.groups.map")) == ["(top level)"]


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
    "inp-tau-low": "0.25", "inp-tau-high": "0.75", "inp-cost-s2": "0.004",
    "inp-err-budget": "0.10", "inp-dist": "", "inp-group": "",
}


def _render(loader: str, routes: dict[str, dict], functions: str | None = None) -> list[str]:
    """Run `loader` with `fetchJson` answering from `routes` (by path, query ignored) and return the
    HTML of every element it wrote."""
    js = f"""
let policySeq = 0;
let GROUPS = [];
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

    lint, drift = get(client, "/api/lint"), get(client, "/api/drift")
    # Strings the demo leaves empty or null, filled so that their rendering is exercised too.
    lint_filled, drift_filled = copy.deepcopy(lint), copy.deepcopy(drift)
    for p in lint_filled["probe_audit"]:
        p["inverse_source"] = p["inverse_source"] or "generated"
    for q in drift_filled["questions"]:
        q["notes"] = q["notes"] or ["a note"]
        q["alerts"] = q["alerts"] or ["an alert"]
    return {
        "audit-noul": {"/api/audit": get(client, "/api/audit")},
        "audit-categorical": {"/api/audit": get(cat, "/api/audit")},
        "policy-noul": {"/api/simulate-policy": get(client, "/api/simulate-policy")},
        "policy-choice": {"/api/simulate-policy": get(cat, "/api/simulate-policy", primitive="choice")},
        "policy-score": {"/api/simulate-policy": get(cat, "/api/simulate-policy", primitive="score")},
        "lint": {"/api/lint": lint},
        "lint-filled": {"/api/lint": lint_filled},
        "slices": {"/api/slices": get(client, "/api/slices"), "/api/labelaudit": get(client, "/api/labelaudit")},
        "compare": {"/api/compare": get(client, "/api/compare")},
        "drift": {"/api/drift": drift},
        "drift-filled": {"/api/drift": drift_filled},
    }


LOADERS = {
    "audit-noul": "loadAudit",
    "audit-categorical": "loadAudit",
    "policy-noul": "loadPolicy",
    "policy-choice": "loadPolicy",
    "policy-score": "loadPolicy",
    "lint": "loadLint",
    "lint-filled": "loadLint",
    "slices": "loadSlicesAndLabels",
    "compare": "loadCompare",
    "drift": "loadDrift",
    "drift-filled": "loadDrift",
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
        # A criterion name is a mapping key: the extra hostile key is what catches it.
        pytest.param("compare", "<td class=\"mono\">${esc(c)}</td>", "<td class=\"mono\">${c}</td>", id="mapping-key"),
        pytest.param("drift", "Drift Monitor: <span class=\"badge badge-info\">${esc(q.question_id)}</span>",
                     "Drift Monitor: <span class=\"badge badge-info\">${q.question_id}</span>", id="string-value"),
        pytest.param("policy-noul", "On ${esc(d.distribution || 'unknown')} probabilities",
                     "On ${d.distribution || 'unknown'} probabilities", id="policy-value"),
        # Empty or null in the demo: caught only by the filled cases.
        pytest.param("drift-filled", "margin:4px 0;\">${esc(n)}</div>", "margin:4px 0;\">${n}</div>", id="drift-note"),
        pytest.param("lint-filled", "esc(p.inverse_source)", "p.inverse_source", id="probe-inverse-source"),
    ],
)
def test_the_hostile_check_catches_a_dropped_esc(payloads, case, old, new) -> None:
    html = "".join(_render(LOADERS[case], _hostile_routes(payloads[case]), functions=_mutated(old, new)))
    assert "<img" in html
