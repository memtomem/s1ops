"""The question lint and probe coherence view, rendered in node over real /api/lint payloads: the
findings with the served counts, the rule catalog, the probe groups, each group's two scatters and
its items (one list for both), and what the shell keeps."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.lint.rules import RULES
from s1ops.web.app import STATIC_DIR, create_app
from tests.test_web_harness import _mutated
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")

HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
FINDING_KEYS = ["rule", "severity", "question", "question_id", "message", "fix", "evidence", "evidence_url"]
POINT_KEYS = ["dataset", "item_id", "p_noul", "p_twin_yes", "twin_gap", "p_inverse", "inverse_sum"]


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _lint(root: Path, *, probes: list[dict] | None = None, questions: object = None, drop: tuple[str, ...] = ()) -> dict:
    """/api/lint over a fresh demo workspace with its probe records or questions file replaced."""
    ws = generate_workspace(root)
    if probes is not None:
        Path(ws["probe_records"]).write_text("".join(json.dumps(r) + "\n" for r in probes), encoding="utf-8")
    if questions is not None:
        Path(ws["questions"]).write_text(questions if isinstance(questions, str) else json.dumps(questions), encoding="utf-8")
    for key in drop:
        Path(ws[key]).unlink()
    r = TestClient(create_app(root)).get("/api/lint")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


@pytest.fixture(scope="module")
def demo(ws) -> dict:
    r = TestClient(create_app(Path(ws["records"]).parent)).get("/api/lint")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def probe_rows(ws) -> list[dict]:
    return _rows(ws["probe_records"])


def _group(rows: list[dict], question: str, model: str) -> list[dict]:
    return [{**r, "probe_question": question, "model_served": model} for r in rows]


def _colliding(rows: list[dict]) -> list[dict]:
    """Two probe groups of three items each whose question and model, joined by " · ", read the same."""
    one = [r for r in rows if r["item_id"] in {"eval-0000", "eval-0001", "eval-0002"}]
    other = [r for r in rows if r["item_id"] in {"eval-0003", "eval-0004", "eval-0005"}]
    return [*_group(one, "a · b", "c"), *_group(other, "a", "b · c")]


def _js(expr: str, prelude: str = "", functions: str | None = None) -> object:
    return _node(f"{prelude}\n{functions if functions is not None else _all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


def _render(d: dict, tstate: dict | None = None, functions: str | None = None) -> str:
    return _js(f"renderLint(d, {{tstate: {json.dumps(tstate or {})}}})", prelude=f"const d = {json.dumps(d)};", functions=functions)


def _sections(html: str) -> dict[str, str]:
    """The view's three cards, by what they hold."""
    findings, rest = html.split("<h3>Probe Coherence Audit")
    probes, rules = rest.split("<h3>Lint rules</h3>")
    return {"findings": findings, "probes": probes, "rules": rules}


def _blocks(probes: str) -> list[str]:
    """Each probe group's block of the probe card, in order."""
    return probes.split('<h4 class="block-head">')[1:]


def _table(html: str, caption: str) -> str:
    m = re.search(rf'<table aria-label="{re.escape(caption)}">.*?</table>', html, re.S)
    assert m, caption
    return m.group(0)


def _head(table: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", th).strip() for th in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S)]


def _cells(table: str) -> list[list[str]]:
    return [re.findall(r"<td[^>]*>(.*?)</td>", row, re.S) for row in re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>")[1], re.S)]


def _svgs(block: str) -> dict[str, str]:
    """A block's charts by kind ("Twin" / "Inverse")."""
    return {m.group(1): m.group(0) for m in re.finditer(r'<svg class="chart"(?:(?!</svg>).)*?<title>(Twin|Inverse) coherence of .*?</svg>', block, re.S)}


def _dots(svg: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in re.findall(r'<circle class="dot ser-1" cx="([^"]+)" cy="([^"]+)" r="2.5" />', svg)]


def _tips(svg: str) -> list[str]:
    return re.findall(r'<circle class="hit"[^>]*data-tip="([^"]*)"', svg)


def _x(v: float) -> float:
    return 52 + v * 394  # the scatter's plot: 394 units a side from (52, 12)


def _y(v: float) -> float:
    return 12 + 394 - v * 394


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= 0.06  # pixels are written to one place


# --- the route ---------------------------------------------------------------------------------------


def test_the_route_answers_the_same_keys(demo) -> None:
    assert list(demo) == ["questions_file", "n_findings", "counts", "findings", "probe_audit", "rules"]
    assert all(list(f) == FINDING_KEYS for f in demo["findings"]) and demo["findings"]
    assert all(list(r) == ["id", "title", "severity", "fix"] for r in demo["rules"])
    assert [r["id"] for r in demo["rules"]] == list(RULES)
    assert {f["rule"] for f in demo["findings"]} <= set(RULES)  # every finding's title can be looked up
    (group,) = demo["probe_audit"]
    assert all(list(p) == POINT_KEYS for p in group["points"]) and len(group["points"]) == group["n_states"] == 50


# --- the findings ------------------------------------------------------------------------------------


@needs_node
def test_a_workspace_without_a_questions_file_is_not_a_clean_one(tmp_path) -> None:
    d = _lint(tmp_path / "w", drop=("questions", "probe_records"))
    assert (d["questions_file"], d["n_findings"], d["counts"]) == (None, 0, {})
    s = _sections(_render(d))
    assert "No questions file found in this workspace (questions.json): nothing was linted." in s["findings"]
    assert "findings)" not in s["findings"] and "<table" not in s["findings"] and "badge" not in s["findings"]
    # The catalog is still the code's, without a count of findings nobody looked for.
    assert _head(_table(s["rules"], "Rule catalog")) == ["Rule", "Severity", "Title", "Fix"]
    assert "No questions file was read, so no rule has a count." in s["rules"]
    assert "No probe coherence group: the probe records file is missing or holds no answered record" in s["probes"]
    assert "<table" not in s["probes"] and "<svg" not in s["probes"]


@needs_node
def test_the_counters_are_the_served_counts_not_a_count_of_the_rows(demo) -> None:
    assert demo["counts"] == {"error": 0, "warning": 4, "info": 2, "suppressed": 0}
    card = _sections(_render(demo))["findings"]
    assert "Static Question Lint Diagnostics (6 findings)" in card
    assert re.findall(r'<span class="badge (badge-\w+)">(\w+ \d+)</span>', card) == [
        ("badge-err", "error 0"), ("badge-warn", "warning 4"), ("badge-info", "info 2"), ("badge-none", "suppressed 0"),
    ]
    assert f'File: <span class="mono">{demo["questions_file"]}</span>' in card
    # Counts that disagree with the rows listed (the suppressed ones are not): the served ones are shown.
    odd = {**demo, "counts": {"error": 7, "warning": 0, "info": 0, "suppressed": 3}, "n_findings": 1}
    card = _sections(_render(odd))["findings"]
    assert "(1 finding)" in card and [t for _, t in re.findall(r'<span class="badge (badge-\w+)">(\w+ \d+)</span>', card)] == ["error 7", "warning 0", "info 0", "suppressed 3"]
    # A count the response lacks is N/A, never a 0.
    assert "error N/A" in _sections(_render({**demo, "counts": {}}))["findings"]


@needs_node
def test_a_finding_shows_its_rules_title_its_evidence_and_where_it_is(demo, tmp_path) -> None:
    card = _sections(_render(demo))["findings"]
    table = _table(card, "Findings")
    assert _head(table) == ["Rule", "Rule title", "Severity", "Question", "Question ID", "Message", "Fix", "Evidence", "Evidence reference"]
    first = demo["findings"][0]
    row = _cells(table)[0]
    assert row == [first["rule"], RULES[first["rule"]].title, first["severity"], first["question"], first["question_id"],
                   first["message"].replace('"', "&quot;"), first["fix"].replace('"', "&quot;"), first["evidence"], first["evidence_url"]]
    assert "<a" not in card and "which is not shipped with s1ops" in card  # a reference, never a link
    # A file-level finding (a file that does not parse) has no question; a rule the catalog lacks has no title.
    broken = _lint(tmp_path / "broken", questions="not json")
    (finding,) = broken["findings"]
    assert (finding["rule"], finding["question"], finding["evidence"]) == ("JL000", None, None)
    odd = copy.deepcopy(broken)
    odd["findings"].append({**finding, "rule": "JL999", "question": "q"})
    rows = _cells(_table(_sections(_render(odd))["findings"], "Findings"))
    assert (rows[0][1], rows[0][3], rows[0][7:]) == (RULES["JL000"].title, "(file)", ["", ""])
    assert (rows[1][0], rows[1][1], rows[1][3]) == ("JL999", "", "q")
    # The export is each finding as served: its eight keys, the null question, no title and no rank.
    out = _js('[JSON.parse(toJsonRows(lintTables(d)["lint:findings"].spec, lintTables(d)["lint:findings"].rows)), toCsv(lintTables(d)["lint:findings"].spec, lintTables(d)["lint:findings"].rows).split("\\r\\n")[0]]',
              prelude=f"const d = {json.dumps(odd)};")
    assert out[0] == odd["findings"] and [list(r) for r in out[0]] == [FINDING_KEYS] * 2 and out[1] == ",".join(FINDING_KEYS)


@needs_node
def test_a_file_with_only_suppressed_findings_does_not_read_as_clean(tmp_path) -> None:
    question = {"type": "noul", "instructions": "Is the user message usually appropriate for general audiences without major issues?"}
    plain = _lint(tmp_path / "plain", questions={"hedge": question})
    assert [f["rule"] for f in plain["findings"]] == ["JL010"]
    d = _lint(tmp_path / "quiet", questions={"hedge": {**question, "x-s1-lint": {"ignore": ["JL010"]}}})
    assert (d["findings"], d["counts"]["suppressed"]) == ([], 1)
    s = _sections(_render(d))
    assert "(0 findings)" in s["findings"] and "suppressed 1" in s["findings"]
    assert "No unsuppressed finding in this file." in s["findings"] and "<table" not in s["findings"]
    assert "x-s1-lint</code> / <code>x-jev-lint" in s["findings"]
    # The catalog's count says what it counts, and that a 0 is not the absence of suppressed ones.
    rules = _table(s["rules"], "Rule catalog")
    assert _head(rules)[-1] == "Unsuppressed findings" and {r[-1] for r in _cells(rules)} == {"0"}
    assert "a rule with 0 may still have suppressed ones" in s["rules"]


@needs_node
def test_severity_sorts_by_rank_and_no_rank_is_exported(demo) -> None:
    prelude = f"const d = {json.dumps(demo)};"
    order = _js('["asc", "desc"].map(dir => viewRows(lintTables(d)["lint:rules"].spec, lintTables(d)["lint:rules"].rows, {sort: "severity", dir, q: ""}).map(r => r.severity))', prelude=prelude)
    ranks = {"error": 0, "warning": 1, "info": 2}
    assert order[0] == sorted(order[0], key=ranks.get) and order[1] == sorted(order[1], key=ranks.get, reverse=True)
    assert set(order[0]) == set(ranks)  # alphabetical order would put info before warning
    findings = _js('viewRows(lintTables(d)["lint:findings"].spec, lintTables(d)["lint:findings"].rows, {sort: "severity", dir: "asc", q: ""}).map(r => r.severity)', prelude=prelude)
    assert findings == ["warning"] * 4 + ["info"] * 2
    out = _js('[Object.keys(JSON.parse(toJsonRows(lintTables(d)["lint:rules"].spec, lintTables(d)["lint:rules"].rows))[0]), lintTables(d)["lint:rules"].rows.length]', prelude=prelude)
    assert out == [["id", "severity", "title", "fix"], len(RULES)]
    # The catalog's count column is the listed findings per rule.
    counts = {r[0]: r[-1] for r in _cells(_table(_sections(_render(demo))["rules"], "Rule catalog"))}
    assert (counts["JL010"], counts["JL023"], counts["JL014"], counts["JL000"]) == ("2", "2", "1", "0")


# --- the probe groups --------------------------------------------------------------------------------


@needs_node
def test_the_probe_table_shows_every_served_statistic(demo) -> None:
    (p,) = demo["probe_audit"]
    s = _sections(_render(demo))["probes"]
    table = _table(s, "Probe groups")
    assert _head(table) == ["Probe question", "Model served", "Items", "Mean twin gap", "Max twin gap", "Twin flip rate",
                            "Mean inverse error", "Max inverse error", "Inverse flip rate", "Inverse wording"]
    assert _cells(table) == [[p["probe_question"], p["model_served"], "50", f"{p['mean_twin_gap']:.4f}", f"{p['max_twin_gap']:.4f}", f"{p['twin_flip_rate']:.3f}",
                              f"{p['mean_inverse_error']:.4f}", f"{p['max_inverse_error']:.4f}", f"{p['inverse_flip_rate']:.3f}", "N/A"]]
    assert "A probe asks one question three ways" in s and "0.5 itself counts as the upper side" in s
    prelude = f"const d = {json.dumps(demo)};"
    exported = _js('JSON.parse(toJsonRows(lintTables(d)["lint:probes"].spec, lintTables(d)["lint:probes"].rows))[0]', prelude=prelude)
    assert exported["inverse_source"] is None and "inverse_text" not in exported
    tagged = copy.deepcopy(demo)
    tagged["probe_audit"][0]["inverse_source"] = "generated"
    assert _cells(_table(_sections(_render(tagged))["probes"], "Probe groups"))[0][-1] == "generated"


@needs_node
def test_each_scatter_draws_its_own_answer_against_the_question_as_asked(demo) -> None:
    d = copy.deepcopy(demo)
    point = {"dataset": "ds", "item_id": "it", "p_noul": 0.2, "p_twin_yes": 0.35, "twin_gap": 0.15, "p_inverse": 0.7, "inverse_sum": 0.9}
    d["probe_audit"][0].update(points=[point], n_states=1)
    (block,) = _blocks(_sections(_render(d))["probes"])
    svgs = _svgs(block)
    ((tx, ty),), ((ix, iy),) = _dots(svgs["Twin"]), _dots(svgs["Inverse"])
    assert _near(tx, _x(0.2)) and _near(ty, _y(0.35)) and _near(ix, _x(0.2)) and _near(iy, _y(0.7))
    assert _tips(svgs["Twin"]) == ["ds · it: p 0.2000, twin p(yes) 0.3500, gap 0.1500"]
    assert _tips(svgs["Inverse"]) == ["ds · it: p 0.2000, inverse p 0.7000, sum 0.9000"]
    # The reference lines: y = x and y = 1 - x, corner to corner of a plot as tall as it is wide.
    ref = {kind: [float(v) for v in re.search(r'<line class="ref" x1="([^"]+)" y1="([^"]+)" x2="([^"]+)" y2="([^"]+)"', svg).groups()] for kind, svg in svgs.items()}
    assert ref["Twin"] == [52.0, 406.0, 446.0, 12.0] and ref["Inverse"] == [52.0, 12.0, 446.0, 406.0]
    assert abs(ref["Twin"][2] - ref["Twin"][0]) == abs(ref["Twin"][3] - ref["Twin"][1]) == 394.0
    for svg in svgs.values():
        assert re.findall(r'text-anchor="middle">([\d.]+)</text>', svg) == ["0.00", "0.25", "0.50", "0.75", "1.00"]  # the x ticks; 0.5 is a gridline
    # A null dataset is left out of the tooltip, not printed.
    d["probe_audit"][0]["points"][0]["dataset"] = None
    assert _tips(_svgs(_blocks(_sections(_render(d))["probes"])[0])["Twin"])[0].startswith("it: p ")


@needs_node
def test_each_chart_counts_the_items_that_have_its_answer(probe_rows, tmp_path) -> None:
    # The twin answer of the first ten items is missing; every item has an inverse answer.
    first_ten = {f"eval-{i:04d}" for i in range(10)}
    d = _lint(tmp_path / "w", probes=[r for r in probe_rows if not (r["probe_role"] == "twin" and r["item_id"] in first_ten)])
    (p,) = d["probe_audit"]
    assert p["n_states"] == 50 and sum("p_twin_yes" in pt for pt in p["points"]) == 40
    (block,) = _blocks(_sections(_render(d))["probes"])
    svgs = _svgs(block)
    assert (len(_dots(svgs["Twin"])), len(_dots(svgs["Inverse"]))) == (40, 50)
    assert "40 of the 50 items shown have a twin answer." in block and "50 of the 50 items shown have an inverse answer." in block
    rows = _cells(_table(block, f"Items of group 1: {p['probe_question']} · {p['model_served']}"))
    assert len(rows) == 50 and rows[0][3:5] == ["N/A", "N/A"] and rows[10][3] != "N/A"
    exported = _js('JSON.parse(toJsonRows(Object.values(lintTables(d))[3].spec, Object.values(lintTables(d))[3].rows))', prelude=f"const d = {json.dumps(d)};")
    assert [list(r) for r in exported[:1]] == [POINT_KEYS] and exported[0]["p_twin_yes"] is None and exported[10] == p["points"][10]


def _big(probe_rows: list[dict], n: int, no_twin: int) -> list[dict]:
    """`n` items in one probe group (the demo's 50, repeated), the first `no_twin` without a twin answer."""
    by_item: dict[str, list[dict]] = {}
    for r in probe_rows:
        by_item.setdefault(r["item_id"], []).append(r)
    base = [by_item[k] for k in sorted(by_item)]
    return [{**r, "item_id": f"big-{i:05d}"} for i in range(n) for r in base[i % len(base)] if not (r["probe_role"] == "twin" and i < no_twin)]


@needs_node
def test_a_large_group_shows_its_first_items_in_the_charts_and_the_table_alike(probe_rows, tmp_path) -> None:
    d = _lint(tmp_path / "w", probes=_big(probe_rows, 2500, 600))
    (p,) = d["probe_audit"]
    assert p["n_states"] == 2500 and _js("scatterLimit()") == 2000
    s = _sections(_render(d))["probes"]
    (block,) = _blocks(s)
    assert "This group has 2500 items; the page shows the first 2000 in the report's order." in block
    rows = _cells(_table(block, f"Items of group 1: {p['probe_question']} · {p['model_served']}"))
    assert [r[1] for r in rows] == [pt["item_id"] for pt in p["points"][:2000]]
    shown = {r[1] for r in rows}
    svgs = _svgs(block)
    # The twin chart draws the shown items that have a twin answer, not the first 2000 that have one.
    assert "1400 of the 2000 items shown have a twin answer." in block and "2000 of the 2000 items shown have an inverse answer." in block
    for kind, expected in (("Twin", 1400), ("Inverse", 2000)):
        assert len(_dots(svgs[kind])) == expected
        named = set(re.findall(r"big-\d{5}", " ".join(_tips(svgs[kind]))))
        assert named and named <= shown, kind  # every mark a tooltip names has its row
    # Nothing else is cut to the shown items: the group's row is the served one, over every item.
    assert _cells(_table(s, "Probe groups"))[0][2:4] == ["2500", f"{p['mean_twin_gap']:.4f}"]
    # A group at the limit says nothing about a cut.
    small = copy.deepcopy(d)
    small["probe_audit"][0]["points"] = p["points"][:2000]
    assert "the page shows the first" not in _render(small)


@needs_node
def test_points_drawn_at_one_place_are_one_target_naming_each(demo) -> None:
    d = copy.deepcopy(demo)
    crowd = [{"dataset": "ds", "item_id": f"same-{i:02d}", "p_noul": 0.4, "p_twin_yes": 0.6, "twin_gap": 0.2} for i in range(12)]
    d["probe_audit"][0].update(points=[*crowd, {"dataset": "ds", "item_id": "apart", "p_noul": 0.9, "p_twin_yes": 0.1, "twin_gap": 0.8}], n_states=13)
    (block,) = _blocks(_sections(_render(d))["probes"])
    twin = _svgs(block)["Twin"]
    assert len(_dots(twin)) == 13 and len(_tips(twin)) == 2
    tip = _tips(twin)[0]
    assert [f"same-{i:02d}" in tip for i in (0, 7, 8)] == [True, True, False] and tip.endswith("+4 more at this point (all are in the table)")
    assert "0 of the 13 items shown have an inverse answer." in block and "Inverse" not in _svgs(block)  # no inverse answer: no chart
    assert _js("pointSpots([{v: 1}, {v: 1}, {v: 2}, {v: NaN}], o => ({x: o.v, y: 0})).map(s => [s.x, s.items.length])") == [[1, 2], [2, 1]]
    assert _js("bandSpots([{v: 1}, {v: 1}], o => ({x: o.v, y: 0}))") == [{"x": 1, "y": 0, "bands": [{"v": 1}, {"v": 1}]}]


@needs_node
def test_a_group_without_a_noul_answer_says_so(probe_rows, tmp_path) -> None:
    twin_only = [r for r in _group(probe_rows, "twin_only", "m") if r["probe_role"] == "twin"]
    d = _lint(tmp_path / "w", probes=[*probe_rows, *twin_only])
    groups = {p["probe_question"]: p for p in d["probe_audit"]}
    assert (groups["twin_only"]["n_states"], groups["twin_only"]["points"]) == (0, [])
    s = _sections(_render(d))["probes"]
    lone, full = sorted(_blocks(s), key=lambda b: "twin_only" not in b)
    assert lone.startswith("Group 1: twin_only · m (0 items with a noul answer)</h4>")  # groups come by question, then model
    assert "No item of this group has an answered noul probe, so there is nothing to pair." in lone
    assert "<svg" not in lone and "<table" not in lone
    assert len(_svgs(full)) == 2 and "<table" in full
    assert len(_cells(_table(s, "Probe groups"))) == 2


@needs_node
def test_two_groups_whose_joined_names_are_equal_keep_separate_tables(probe_rows, tmp_path) -> None:
    d = _lint(tmp_path / "w", probes=_colliding(probe_rows))
    ids = _js('Object.keys(lintTables(d)).filter(k => k.startsWith("lint:points:"))', prelude=f"const d = {json.dumps(d)};")
    assert sorted(ids) == sorted(['lint:points:["a · b","c"]', 'lint:points:["a","b · c"]'])
    # Sorting one by item, descending, leaves the other in the served order.
    html = _render(d, tstate={ids[0]: {"sort": "item_id", "dir": "desc", "q": ""}})
    # Groups come by question, then model: ("a", "b · c") is group 1 and the one sorted here.
    assert ids[0] == 'lint:points:["a","b · c"]'
    assert [r[1] for r in _cells(_table(html, "Items of group 1: a · b · c"))] == ["eval-0005", "eval-0004", "eval-0003"]
    assert [r[1] for r in _cells(_table(html, "Items of group 2: a · b · c"))] == ["eval-0000", "eval-0001", "eval-0002"]
    # Their names for a reader differ too: each holds its place in the report.
    names = re.findall(r'<table aria-label="(Items of [^"]*)">', html)
    titles = re.findall(r"<title>([^<]*)</title>", html)
    assert len(set(names)) == 2 and len(set(titles)) == 4 and "Twin coherence of group 2: a · b · c" in titles
    # The id survives the attribute it is written into, which is how the shell finds the table.
    written = re.findall(r'<div class="tk" data-table="(lint:points:[^"]*)"', html)
    assert [w.replace("&quot;", '"') for w in written] == ids


@needs_node
def test_every_chart_and_table_of_the_view_is_named_and_linked(probe_rows, tmp_path) -> None:
    d = _lint(tmp_path / "w", probes=[*probe_rows, *_group(probe_rows, "second", "other-model")])
    html = _render(d)
    blocks = _blocks(_sections(html)["probes"])
    assert len(blocks) == 2
    for i, block in enumerate(blocks):
        assert re.findall(r'aria-details="([^"]+)"', block) == [f"lint-points-table-{i}"] * 2
        assert block.count(f'id="lint-points-table-{i}"') == 1 and html.count(f'id="lint-points-table-{i}"') == 1
    titles = re.findall(r"<title>([^<]*)</title>", html)
    assert len(titles) == len(set(titles)) == 4 and "Twin coherence of group 1: second · other-model" in titles
    names = re.findall(r'<table aria-label="([^"]*)">', html)
    assert len(names) == len(set(names)) == 5 and "Items of group 1: second · other-model" in names
    assert "the reference line y = 1 − x" in html
    # A long unbroken name wraps in its block's heading.
    assert ".card h4.block-head { margin: 20px 0 6px; font-size: 13px; font-weight: 600; overflow-wrap: anywhere; }" in HTML


@needs_node
def test_a_probe_groups_items_are_one_list() -> None:
    points = [{"item_id": str(i), "p_noul": 0.5} for i in range(2100)]
    out = _js("[probeShown(p).length, probeShown(p)[1999].item_id, probeShown({}).length, probeShown({points: null}).length, lintPointsTable(p).rows.length]",
              prelude=f"const p = {json.dumps({'probe_question': 'q', 'model_served': 'm', 'points': points})};")
    assert out == [2000, "1999", 0, 0, 2000]


@needs_node
def test_a_prose_column_keeps_a_readable_width(demo) -> None:
    spec = {"id": "t", "caption": "c", "file": "f", "columns": [
        {"key": "a", "name": "A", "kind": "text"}, {"key": "b", "name": "B", "kind": "text", "wide": True}, {"key": "n", "name": "N", "kind": "num", "digits": 0},
    ]}
    html = _js(f"tableHtml({json.dumps(spec)}, [{{a: 'x', b: 'some prose', n: 1}}], {{}})")
    assert '<td class="mono">x</td><td class="mono wide">some prose</td><td class="mono num">1</td>' in html
    assert ".tk td.wide { min-width: 26ch; }" in HTML
    # The findings' and the catalog's sentences are such columns; identifiers and numbers are not.
    table = _table(_sections(_render(demo))["findings"], "Findings")
    first = re.findall(r'<td class="([^"]*)">', table.split("<tbody>")[1].split("</tr>")[0])
    assert first == ["mono", "mono wide", "mono", "mono", "mono", "mono wide", "mono wide", "mono", "mono"]


# --- what the shell keeps -----------------------------------------------------------------------------


def _shell_run(steps: str, payload: dict, functions: str | None = None) -> dict:
    js = f"""
let STATE = {{view: "lint"}};
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [], groupsState = "ok", groupsError = "";
const els = {{}};
const el = id => (els[id] ??= {{id, innerHTML: "", textContent: "", value: "", attrs: {{}},
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const pending = [];
let fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const payload = {json.dumps(payload)};
const shown = () => el("lint-container").innerHTML;
const out = {{}};
{functions if functions is not None else _all_functions()}
STATE = {{...defaultState(), view: "lint"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


KEEP_STEPS = """
  runView("lint");
  out.urls = pending.map(p => p.url);
  pending.splice(0, 1)[0].resolve(payload);
  await tick(); await tick();
  out.kept = viewData.lint === payload;
  out.tables = Object.keys(currentTables()).map(k => k.split(":").slice(0, 2).join(":"));
  const first = shown();
  // Coming back to the view re-renders from what was kept, with the tables' state; nothing is fetched.
  fetchJson = () => { throw new Error("no request"); };
  runView("lint");
  out.same = shown() === first;
  TSTATE = {"lint:rules": {sort: "id", dir: "desc", q: ""}};
  runView("lint");
  out.sorted = shown() !== first && / aria-sort="descending"><button[^>]*data-col="id"/.test(shown());
  out.requests = pending.length;
"""


@needs_node
def test_the_view_is_kept_once_shown_and_rerendered_from_what_was_kept(demo) -> None:
    assert _shell_run(KEEP_STEPS, demo) == {
        "urls": ["/api/lint"], "kept": True, "tables": ["lint:findings", "lint:rules", "lint:probes", "lint:points"],
        "same": True, "sorted": True, "requests": 0,
    }


FAIL_STEPS = """
  runView("lint");
  pending.splice(0, 1)[0].reject(Object.assign(new Error("boom"), {status: 400}));
  await tick(); await tick();
  out.failedError = shown().includes("ERROR</span> boom");
  out.failedKept = "lint" in viewData;
  // A response the renderer throws on (findings that are not a list): the error card, nothing kept.
  generation += 1;
  loadedSig = {};
  runView("lint");
  pending.splice(0, 1)[0].resolve({...payload, findings: 5});
  await tick(); await tick();
  out.error = shown().includes("ERROR</span>");
  out.kept = "lint" in viewData;
  try { out.tables = Object.keys(currentTables()).length; } catch (e) { out.tables = "threw"; }
"""


@needs_node
def test_a_failed_or_unrenderable_response_keeps_nothing(demo) -> None:
    assert _shell_run(FAIL_STEPS, demo) == {"failedError": True, "failedKept": False, "error": True, "kept": False, "tables": 0}
    render = '      document.getElementById("lint-container").innerHTML = renderLint(d, ctx.opts ? ctx.opts() : {});\n'
    keep = "      if (ctx.keep) ctx.keep(d);\n"
    mutant = _shell_run(FAIL_STEPS, demo, functions=_mutated(render + keep, keep + render))
    assert (mutant["error"], mutant["kept"], mutant["tables"]) == (True, True, "threw")


TABLE_STEPS = r"""
  runView("lint");
  pending.splice(0, 1)[0].resolve(payload);
  await tick(); await tick();
  const ids = Object.keys(currentTables());
  const boxes = ids.map(id => ({dataset: {table: id}, body: {innerHTML: ""}, count: {textContent: ""},
    querySelector(sel) { return sel === ".tk-body" ? this.body : this.count; }, querySelectorAll: () => []}));
  document.querySelectorAll = sel => sel === ".tk" ? boxes : [];
  const downloads = [];
  downloadText = (name, text) => downloads.push([name, text]);
  const [a, b] = ids.filter(id => id.startsWith("lint:points:"));
  out.ids = [a, b];
  sortTable(a, "item_id"); sortTable(a, "item_id");
  out.touched = boxes.filter(box => box.body.innerHTML).map(box => box.dataset.table);
  exportTable(a, "json"); exportTable(b, "json");
  out.exported = downloads.map(([name, text]) => [name, JSON.parse(text)]);
  filterTable(b, payload.probe_audit[1].points[1].item_id);
  out.count = boxes.find(box => box.dataset.table === b).count.textContent;
  exportTable(b, "csv");
  out.csv = downloads[2][1].split("\r\n").slice(1).filter(Boolean).map(line => line.split(",")[1]);
"""


@needs_node
def test_a_table_action_reaches_its_own_groups_rows(probe_rows, tmp_path) -> None:
    # Through the shell's own sort, filter and export, on two groups whose joined names are equal.
    d = _lint(tmp_path / "w", probes=_colliding(probe_rows))
    first, second = d["probe_audit"]
    out = _shell_run(TABLE_STEPS, d)
    assert out["ids"] == [f'lint:points:{json.dumps([g["probe_question"], g["model_served"]], ensure_ascii=False, separators=(",", ":"))}' for g in (first, second)]
    assert out["touched"] == [out["ids"][0]]  # a sort re-renders that table only
    (name_a, rows_a), (name_b, rows_b) = out["exported"]
    assert rows_a == first["points"][::-1] and rows_b == second["points"]  # sorted descending; untouched
    assert {r["item_id"] for r in rows_a}.isdisjoint(r["item_id"] for r in rows_b)
    assert name_a != name_b and name_a.startswith("probe-items-") and name_a.endswith(".json")
    assert out["count"] == "1 of 3 rows" and out["csv"] == [second["points"][1]["item_id"]]


STALE_STEPS = """
  runView("lint");
  const older = pending.splice(0, 1)[0];
  generation += 1;
  loadedSig = {};
  viewData = {};
  runView("lint");
  pending.splice(0, 1)[0].resolve(payload);
  await tick(); await tick();
  out.shown = shown().includes("(6 findings)");
  older.resolve({...payload, questions_file: null});
  await tick(); await tick();
  out.still = shown().includes("(6 findings)") && !shown().includes("nothing was linted");
  out.kept = viewData.lint === payload;
"""


@needs_node
def test_an_older_answer_never_replaces_a_newer_one(demo) -> None:
    assert _shell_run(STALE_STEPS, demo) == {"shown": True, "still": True, "kept": True}
    functions = _mutated('      const d = await fetchJson("/api/lint");\n      if (isStale(ctx)) return;\n', '      const d = await fetchJson("/api/lint");\n')
    assert _shell_run(STALE_STEPS, demo, functions=functions)["still"] is False
