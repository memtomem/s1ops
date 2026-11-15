"""The slices and blind label audit view: what /api/labelaudit says about its ranking, the two
settings the page sends, and the view rendered in node over real /api/slices and /api/labelaudit
payloads (the forest, the dumbbell, their tables, the ranking's card, and what the shell keeps)."""

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
from s1ops.improve.calibrate import load_fits
from s1ops.improve.labelaudit import rank_with_skips
from s1ops.web.app import SLICE_MIN_SUPPORT, STATIC_DIR, create_app
from tests.test_web_harness import _mutated
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")

HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
COMBINING_ACUTE = chr(0x301)  # U+0301, written by number so the source shows it
E_ACUTE = "e" + COMBINING_ACUTE  # one grapheme cluster of two code points


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


@pytest.fixture(scope="module")
def client(ws) -> TestClient:
    return TestClient(create_app(Path(ws["records"]).parent))


@pytest.fixture(scope="module")
def cat(ws) -> TestClient:
    return TestClient(create_app(Path(ws["records"]).parent, records_path=ws["records_categorical"], fits_path=ws["fits_categorical"]))


def _get(c: TestClient, url: str, **params) -> dict:
    r = c.get(url, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _view(c: TestClient, *, fdrq: float | None = 0.1, topk: int | None = 15, grouped: bool = False, **group) -> dict:
    """What `loadSlicesAndLabels` keeps: both responses and the settings they were asked with."""
    slices = _get(c, "/api/slices", **group, **({} if fdrq is None else {"fdr_q": fdrq}))
    labels = _get(c, "/api/labelaudit", **group, **({} if topk is None else {"top_k": topk}))
    return {"slices": slices, "labels": labels, "fdrq": fdrq, "topk": topk, "grouped": grouped}


def _js(expr: str, prelude: str = "", functions: str | None = None) -> object:
    return _node(f"{prelude}\n{functions if functions is not None else _all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


def _render(d: dict, functions: str | None = None, prelude: str = "") -> str:
    return _js("renderSlices(d, {tstate: {}})", prelude=f"const d = {json.dumps(d)};\n{prelude}", functions=functions)


def _cards(html: str) -> list[str]:
    """The group cards, in order (the ranking's card is cut off the last one)."""
    parts = re.split(r'<div class="card">\n          <h3>Group:', html)[1:]
    return [p.split("<h3>Blind Label Audit:")[0] for p in parts]


def _svg(card: str, title: str) -> str:
    m = re.search(rf'<svg class="chart"[^>]*>(?:(?!</svg>).)*?<title>{title}.*?</svg>', card, re.S)
    assert m, title
    return m.group(0)


def _x(v: float) -> float:
    return 126 + v * (460 - 14 - 126)  # a lane chart's scale: [0, 1] onto [ml, W - mr]


def _mid(i: int) -> float:
    return 12 + 24 * i + 12


def _circles(svg: str, cls: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in re.findall(rf'<circle class="dot {cls}" cx="([^"]+)" cy="([^"]+)" r="4" />', svg)]


def _squares(svg: str, cls: str) -> list[tuple[float, float]]:
    """Centres of the square markers."""
    return [(float(x) + 4, float(y) + 4) for x, y in re.findall(rf'<rect class="dot {cls}" x="([^"]+)" y="([^"]+)" width="8" height="8" />', svg)]


def _whiskers(svg: str, cls: str) -> list[tuple[float, float, float]]:
    """(x0, x1, y) of the horizontal intervals."""
    return [(float(x0), float(x1), float(y)) for x0, y, x1 in re.findall(rf'<path class="whisker {cls}" d="M([^ ]+) ([^H]+)H([^M]+)M', svg)]


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= 0.06  # pixels are written to one place


# --- the routes ------------------------------------------------------------------------------------


def test_the_ranking_says_which_probabilities_it_used(ws, client, tmp_path) -> None:
    with_fit = _get(client, "/api/labelaudit")
    # No fit file for this app, though the workspace's fits.json exists: `raw` says only that none was read.
    no_fit = _get(TestClient(create_app(Path(ws["records"]).parent, fits_path=tmp_path / "absent.json")), "/api/labelaudit")
    assert (with_fit["distribution"], no_fit["distribution"]) == ("calibrated", "raw")
    assert [f["item_id"] for f in with_fit["top_flags"]] == [f["item_id"] for f in no_fit["top_flags"]]  # one temperature keeps the order
    assert with_fit["top_flags"][0]["score"] != no_fit["top_flags"][0]["score"]


def test_the_web_ranking_is_the_balanced_one(ws, client) -> None:
    rows = [r.raw for r in read_records(ws["records"])]
    fits = load_fits(ws["fits"])
    balanced, _ = rank_with_skips(rows, temperature=fits)
    by_margin, _ = rank_with_skips(rows, temperature=fits, balance=False)
    web = [f["item_id"] for f in _get(client, "/api/labelaudit")["top_flags"]]
    assert web == [f.item_id for f in balanced[:15]] != [f.item_id for f in by_margin[:15]]
    # The page's caption: the given labels take turns, so rank is not margin order.
    flags = _get(client, "/api/labelaudit")["top_flags"]
    assert [f["label"] for f in flags[:4]] == ["False", "True", "False", "True"]
    assert flags[0]["score"] < flags[1]["score"]


def test_a_groups_ranking_counts_no_unanswered_record(ws, tmp_path) -> None:
    """Why the page words the skip line differently with a group chosen: the route filters by
    `model_served` before counting, and a failed probe has none."""
    rows = _rows(ws["records"])
    for r in rows[:5]:
        for k in ("p", "calib", "latency_ms"):
            r.pop(k, None)
        r.update(model_served=None, error_code="timeout")
    rows[5].pop("label")
    c = TestClient(create_app(Path(ws["records"]).parent, records_path=_write(tmp_path / "r.jsonl", rows)))
    g = _get(c, "/api/groups")["groups"][0]
    group = {"question_id": g["question_id"], "model_served": g["model_served"], "primitive": g["primitive"]}
    assert _get(c, "/api/labelaudit")["skipped"] == {"no_response": 5, "unlabeled": 1}
    assert _get(c, "/api/labelaudit", **group)["skipped"] == {"no_response": 0, "unlabeled": 1}


def test_the_slices_response_has_no_new_key(client) -> None:
    d = _get(client, "/api/slices", fdr_q=0.05)
    assert set(d) == {"distribution", "min_n", "groups"}
    assert set(d["groups"][0]) == {"question_id", "model_served", "primitive", "n", "answerability", "error_slices"}


@needs_node
def test_every_setting_the_pickers_offer_is_one_its_route_takes(client) -> None:
    fdr, topk = _js("[fdrChoices(), topkChoices()]")
    assert (fdr, topk) == ([0.01, 0.05, 0.1, 0.2], [15, 30, 50, 100])
    # The pickers' options are those values, as JavaScript prints them (what the hash and the request carry).
    for id_, values in (("inp-fdrq", fdr), ("inp-topk", topk)):
        select = re.search(rf'<select id="{id_}"[^>]*>(.*?)</select>', HTML, re.S).group(1)
        assert re.findall(r'<option value="([^"]+)"', select) == [_js(f"String({v})") for v in values]
    n = _get(client, "/api/labelaudit")["n_total_candidates"]
    for q in fdr:
        rows = _get(client, "/api/slices", fdr_q=q)["groups"][0]["error_slices"]
        assert rows and all(e["significant"] == (e["q_value"] <= q) for e in rows)
    assert len({sum(e["significant"] for e in _get(client, "/api/slices", fdr_q=q)["groups"][0]["error_slices"]) for q in fdr}) > 1
    for k in topk:
        assert len(_get(client, "/api/labelaudit", top_k=k)["top_flags"]) == min(k, n)
    assert _js("[defaultState().fdrq, defaultState().topk]") == [0.1, 15]


def test_the_page_names_the_routes_min_support() -> None:
    assert SLICE_MIN_SUPPORT == 5
    assert HTML.count(f"A feature is tested when at least {SLICE_MIN_SUPPORT} of the group's items carry it and at least one does not.") == 1


# --- the settings in the link --------------------------------------------------------------------


@needs_node
@pytest.mark.parametrize(
    "hash_, expected",
    [
        ("#/slices?fdrq=0.05&topk=50", [0.05, 50]),
        ("#/slices?fdrq=0.2", [0.2, 15]),
        ("#/slices?fdrq=0.07&topk=17", [0.1, 15]),  # not a choice: the default, never a value the picker cannot show
        ("#/slices?fdrq=abc&topk=", [0.1, 15]),
        ("#/slices?fdrq=0.5&topk=1000", [0.1, 15]),
    ],
)
def test_a_link_sets_only_a_setting_the_picker_offers(hash_, expected) -> None:
    st = _js(f"parseHash({json.dumps(hash_)})")
    assert [st["fdrq"], st["topk"]] == expected
    assert st["view"] == "slices"


@needs_node
def test_the_settings_round_trip_and_defaults_are_not_written() -> None:
    assert _js('formatHash({...defaultState(), view: "slices", fdrq: 0.05, topk: 100})') == "#/slices?fdrq=0.05&topk=100"
    assert _js('formatHash({...defaultState(), view: "slices"})') == "#/slices"
    assert _js('parseHash(formatHash({...defaultState(), view: "slices", fdrq: 0.01, topk: 30}))')["fdrq"] == 0.01


# --- the forest ------------------------------------------------------------------------------------


@needs_node
def test_each_forest_lane_draws_the_slices_accuracy_and_its_null(client) -> None:
    d = _view(client)
    (g,) = d["slices"]["groups"]
    (card,) = _cards(_render(d))
    svg = _svg(card, "Answerability of")
    slices = g["answerability"]
    circles, squares = _circles(svg, "ser-1"), _squares(svg, "ser-2")
    acc, null = _whiskers(svg, "ser-1"), _whiskers(svg, "ser-2")
    assert len(circles) == len(squares) == len(acc) == len(null) == len(slices) == 3
    assert not _circles(svg, "ser-2") and not _squares(svg, "ser-1")
    for i, m in enumerate(slices):
        # The four values differ, so a mark drawn at another's value fails here.
        assert len({m["accuracy"], m["null_accuracy"], *m["ci"], *m["null_ci"]}) == 6
        assert _near(circles[i][0], _x(m["accuracy"])) and _near(circles[i][1], _mid(i) - 4)
        assert _near(squares[i][0], _x(m["null_accuracy"])) and _near(squares[i][1], _mid(i) + 4)
        assert _near(acc[i][0], _x(m["ci"][0])) and _near(acc[i][1], _x(m["ci"][1])) and _near(acc[i][2], _mid(i) - 4)
        assert _near(null[i][0], _x(m["null_ci"][0])) and _near(null[i][1], _x(m["null_ci"][1])) and _near(null[i][2], _mid(i) + 4)
    assert re.findall(r'text-anchor="end">([^<]*)</text>', svg) == [m["slice"] for m in slices]
    # 3 lanes on a 24px pitch under a 12px top margin, plus the 40px axis band.
    assert 'viewBox="0 0 460 124"' in svg


@needs_node
def test_both_marks_of_a_lane_carry_the_slices_numbers(client) -> None:
    d = _view(client)
    (card,) = _cards(_render(d))
    svg = _svg(card, "Answerability of")
    tips = [html_lib.unescape(t) for t in re.findall(r'<circle class="hit"[^>]* data-tip="([^"]*)"', svg)]
    slices = d["slices"]["groups"][0]["answerability"]
    assert len(tips) == 2 * len(slices) and tips[0::2] == tips[1::2]
    trap = next(m for m in slices if m["slice"] == "adversarial_trap")
    assert tips[2 * slices.index(trap)] == (
        "adversarial_trap\nn = 20\naccuracy 85.0% [64.0%, 94.8%]\nnull (majority) 55.0% [34.2%, 74.2%]\nNOT DISTINGUISHABLE FROM NULL"
    )


@needs_node
def test_the_summary_counts_each_verdict_with_the_badge(client) -> None:
    (card,) = _cards(_render(_view(client)))
    assert ('3 slices: 2 × <span class="badge badge-warn">NOT DISTINGUISHABLE FROM NULL</span> · '
            '1 × <span class="badge badge-ok">ANSWERABLE</span>') in card


# --- the dumbbell ----------------------------------------------------------------------------------


@needs_node
def test_each_dumbbell_lane_draws_inside_as_the_circle_and_the_rest_as_the_square(client) -> None:
    d = _view(client, fdrq=0.05)
    (g,) = d["slices"]["groups"]
    (card,) = _cards(_render(d))
    svg = _svg(card, "Error slices of")
    rows = g["error_slices"]
    circles, squares = _circles(svg, "ser-1"), _squares(svg, "ser-2")
    links = [(float(a), float(b)) for a, b in re.findall(r'<line class="link" x1="([^"]+)" y1="[^"]+" x2="([^"]+)"', svg)]
    assert len(circles) == len(squares) == len(links) == len(rows) == 7
    for i, e in enumerate(rows):
        assert e["slice_error_rate"] != e["rest_error_rate"]
        assert _near(circles[i][0], _x(e["slice_error_rate"])) and _near(circles[i][1], _mid(i))
        assert _near(squares[i][0], _x(e["rest_error_rate"])) and _near(squares[i][1], _mid(i))
        assert _near(links[i][0], _x(e["slice_error_rate"])) and _near(links[i][1], _x(e["rest_error_rate"]))
    # " *" is the server's verdict at the q asked for, in the served order.
    names = [html_lib.unescape(t) for t in re.findall(r'text-anchor="end">([^<]*)</text>', svg)]
    assert [n.endswith(" *") for n in names] == [e["significant"] for e in rows]
    assert [e["significant"] for e in rows] == [True, True, True, False, False, False, False]
    assert "* marks a feature significant at q ≤ 0.05." in card


@needs_node
def test_the_significance_mark_is_the_servers_verdict_never_the_pages(client) -> None:
    d = _view(client, fdrq=0.01)
    rows = d["slices"]["groups"][0]["error_slices"]
    # At q 0.01 the third feature (q 0.0249) is no longer significant; a page deciding at 0.10 would mark it.
    assert [e["significant"] for e in rows[:3]] == [True, True, False] and rows[2]["q_value"] < 0.1
    # And a verdict the page could not derive from q at all: the first row served as not significant.
    rows[0]["significant"] = False
    (card,) = _cards(_render(d))
    names = [html_lib.unescape(t) for t in re.findall(r'text-anchor="end">([^<]*)</text>', _svg(card, "Error slices of"))]
    assert [n.endswith(" *") for n in names[:3]] == [False, True, False]
    table = card[card.index('id="slices-errors-table-0"'):]
    assert [re.findall(r"<td[^>]*>(.*?)</td>", r)[-1] for r in re.findall(r"<tr><td class=\"mono\">.*?</tr>", table, re.S)[:3]] == ["no", "yes", "no"]


@needs_node
def test_a_dumbbell_tooltip_gives_the_counts_the_test_and_the_q_asked_for(client) -> None:
    d = _view(client, fdrq=0.05)
    (card,) = _cards(_render(d))
    tips = [html_lib.unescape(t) for t in re.findall(r'<circle class="hit"[^>]* data-tip="([^"]*)"', _svg(card, "Error slices of"))]
    assert tips[0::2] == tips[1::2]
    assert tips[0] == ("lang=ko_slang\ninside: 27 of 27 wrong (100.0%)\noutside: 19 of 133 wrong (14.3%)\nodds ratio N/A\n"
                       "p = < 0.0001, q = < 0.0001\nsignificant at q ≤ 0.05")
    code = next(t for t in tips if t.startswith("has_code\n"))
    assert "odds ratio 3.68\np = 0.0107, q = 0.0249\nsignificant at q ≤ 0.05" in code
    assert tips[-1].endswith("\nnot significant at q ≤ 0.05")


@needs_node
def test_a_group_without_a_tested_feature_draws_no_dumbbell(cat) -> None:
    d = _view(cat)
    cards = _cards(_render(d))
    assert len(cards) == 2
    for g, card in zip(d["slices"]["groups"], cards):
        assert g["error_slices"] == [] and [m["slice"] for m in g["answerability"]] == ["default"]
        assert "No feature was tested in this group. A feature is tested when at least 5 of the group&#39;s items" in card
        assert "<title>Error slices of" not in card and card.count('<svg class="chart"') == 1
        assert '1 slice: 1 × <span class="badge badge-ok">ANSWERABLE</span>' in card
        # One lane is still a comparison of two intervals.
        assert len(_circles(_svg(card, "Answerability of"), "ser-1")) == 1


@needs_node
def test_a_workspace_without_a_features_file_has_one_default_slice(ws, tmp_path) -> None:
    c = TestClient(create_app(Path(ws["records"]).parent, features_path=tmp_path / "absent.jsonl"))
    d = _view(c)
    (g,) = d["slices"]["groups"]
    assert [m["slice"] for m in g["answerability"]] == ["default"] and g["error_slices"] == []
    assert "No feature was tested in this group." in _render(d)


# --- more rows than lanes, and names ---------------------------------------------------------------


def _many(d: dict, n: int) -> dict:
    """The demo's group with `n` slices and `n` tested features, built from its own first rows."""
    g = d["slices"]["groups"][0]
    slices = [{**g["answerability"][k % 3], "slice": f"slice-{k:02d}"} for k in range(n)]
    features = [{**g["error_slices"][k % 7], "feature": f"feature-{k:02d}"} for k in range(n)]
    return {**d, "slices": {**d["slices"], "groups": [{**g, "answerability": slices, "error_slices": features}]}}


@needs_node
def test_a_lane_chart_draws_thirty_lanes_and_says_how_many_it_left_out(client) -> None:
    d = _many(_view(client), 60)
    (card,) = _cards(_render(d))
    forest, dumbbell = _svg(card, "Answerability of"), _svg(card, "Error slices of")
    assert len(_circles(forest, "ser-1")) == len(_circles(dumbbell, "ser-1")) == 30 == _js("laneLimit()")
    assert 'viewBox="0 0 460 772"' in forest and 'viewBox="0 0 460 772"' in dumbbell
    assert "The first 30 of 60 slices are drawn; every one is in the table." in card
    assert "The first 30 of 60 features are drawn; every one is in the table." in card
    tables = _js("Object.fromEntries(Object.entries(slicesTables(d)).map(([k, t]) => [k.split(':')[1], t.rows.length]))", prelude=f"const d = {json.dumps(d)};")
    assert tables == {"answerability": 60, "errors": 60, "flags": 15}
    # At the limit nothing is said.
    assert "are drawn; every one" not in _render(_many(_view(client), 30))


@needs_node
@pytest.mark.parametrize(
    "name, cells, expected",
    [
        ("short", 18, "short"),
        ("x" * 18, 18, "x" * 18),
        ("x" * 200, 18, "x" * 17 + "…"),
        (E_ACUTE * 30, 18, E_ACUTE * 17 + "…"),  # a combining mark stays on its base
        ("한국어_은어_슬라이스_이름", 18, "한국어_은어_슬라…"),  # a wide character takes two cells: 7 × 2 + 2 underscores + the ellipsis = 17
        ("\U0001f600" * 12, 18, "\U0001f600" * 8 + "…"),
        ("שלום_עולם_בדיקה_ארוכה_מאוד", 18, "שלום_עולם_בדיקה_א…"),
        ("", 18, ""),
    ],
)
def test_a_lane_name_is_cut_by_cells_and_never_inside_a_character(name, cells, expected) -> None:
    assert _js(f"laneName({json.dumps(name)}, {cells})") == expected
    # Without Intl.Segmenter the cut is by code point: a surrogate pair is still never split.
    plain = _js(f"laneName({json.dumps(name)}, {cells})", prelude="Intl.Segmenter = undefined;")
    assert not re.search(r"[\ud800-\udfff]", json.dumps(plain, ensure_ascii=False))
    if COMBINING_ACUTE not in name:
        assert plain == expected


@needs_node
def test_a_long_name_is_cut_in_the_lane_and_whole_in_the_tooltip_and_the_table(client) -> None:
    d = _view(client)
    g = d["slices"]["groups"][0]
    long = "긴_이름_" + "y" * 200
    g["answerability"][0]["slice"] = long
    g["error_slices"][0]["feature"] = long  # a significant one
    (card,) = _cards(_render(d))
    forest, dumbbell = _svg(card, "Answerability of"), _svg(card, "Error slices of")
    cut = _js(f"laneName({json.dumps(long)}, 18)")
    assert f'text-anchor="end">{cut}</text>' in forest and long not in re.sub(r'data-tip="[^"]*"', "", forest)
    # The significance mark comes after the cut, so it is never the part cut off.
    assert f'text-anchor="end">{_js(f"laneName({json.dumps(long)}, 16)")} *</text>' in dumbbell
    assert f'data-tip="{long}\n' in forest and f'data-tip="{long}\n' in dumbbell
    assert card.count(f'<td class="mono">{long}</td>') == 2


# --- the tables ------------------------------------------------------------------------------------


@needs_node
def test_the_answerability_table_shows_the_text_and_exports_the_served_values(client) -> None:
    d = _view(client)
    g = d["slices"]["groups"][0]
    key = json.dumps([g["question_id"], g["model_served"], g["primitive"]], separators=(",", ":"))
    out = _js(
        "(() => { const t = slicesTables(d)[id]; return {json: JSON.parse(toJsonRows(t.spec, t.rows)), csv: toCsv(t.spec, t.rows).split('\\r\\n')[0],"
        " shown: shownColumns(t.spec).map(c => c.name), filtered: filterRows(t.spec, t.rows, 'distinguishable').map(r => r.slice)}; })()",
        prelude=f"const d = {json.dumps(d)}; const id = {json.dumps('slices:answerability:' + key)};",
    )
    assert out["shown"] == ["Slice", "n", "Accuracy (95% CI)", "Null baseline (95% CI)", "Baseline kind", "Verdict"]
    served = g["answerability"]
    # The page's text columns are not exported; every served value is, with its type.
    assert out["json"] == [{"question_id": g["question_id"], "model_served": g["model_served"], "primitive": g["primitive"],
                            "slice": m["slice"], "n": m["n"], "null_kind": m["null_kind"], "accuracy": m["accuracy"], "ci": m["ci"],
                            "null_accuracy": m["null_accuracy"], "null_ci": m["null_ci"], "answerable": m["answerable"]} for m in served]
    assert {type(r["answerable"]) for r in out["json"]} == {bool}
    assert out["csv"] == "question_id,model_served,primitive,slice,n,null_kind,accuracy,ci_lo,ci_hi,null_accuracy,null_ci_lo,null_ci_hi,answerable"
    # The filter still matches the shown text.
    assert out["filtered"] == [m["slice"] for m in served if not m["answerable"]] == ["adversarial_trap", "unanswerable"]


@needs_node
def test_the_percent_columns_sort_by_their_numbers(client) -> None:
    d = _view(client)
    g = d["slices"]["groups"][0]
    # As text "9.0%…" sorts after "85.0%…"; by the number it comes first.
    g["answerability"][0].update(accuracy=0.09, ci=[0.05, 0.2])
    order = _js(
        "(() => { const t = Object.values(slicesTables(d))[0]; return sortRows(t.spec, t.rows, 'accuracy_text', 'asc').map(r => r.accuracy); })()",
        prelude=f"const d = {json.dumps(d)};",
    )
    assert order == sorted(m["accuracy"] for m in g["answerability"]) and order[0] == 0.09


@needs_node
def test_the_error_slice_table_prints_tiny_p_values_and_a_null_odds_ratio(client) -> None:
    d = _view(client, fdrq=0.2)
    g = d["slices"]["groups"][0]
    (card,) = _cards(_render(d))
    table = card[card.index('id="slices-errors-table-0"'):]
    row = re.search(r"<tr><td class=\"mono\">lang=ko_slang</td>(.*?)</tr>", table, re.S).group(1)
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row)
    # slice n, errors, rate, rest n, errors, rate, odds ratio, p, q, significant
    assert cells == ["27", "27", "1.000", "133", "19", "0.143", "N/A", "&lt; 0.0001", "&lt; 0.0001", "yes"]
    assert "&amp;lt;" not in card
    assert '<span class="tk-title">Error slices (Benjamini-Hochberg, q ≤ 0.20)</span>' in card
    out = _js("(() => { const t = Object.values(slicesTables(d))[1]; return {json: JSON.parse(toJsonRows(t.spec, t.rows)), sorted: sortRows(t.spec, t.rows, 'p_value', 'asc').map(r => r.p_value)}; })()",
              prelude=f"const d = {json.dumps(d)};")
    assert [list(r)[3:] for r in out["json"]] == [list(e) for e in g["error_slices"]]  # the served keys, in order, after the group's three
    assert [{k: r[k] for k in e} for r, e in zip(out["json"], g["error_slices"])] == g["error_slices"]
    assert out["json"][0]["odds_ratio"] is None and out["json"][0]["p_value"] < 1e-15 and out["json"][0]["significant"] is True
    assert out["sorted"] == sorted(e["p_value"] for e in g["error_slices"])


@needs_node
@pytest.mark.parametrize("p, shown", [(0.0, "< 0.0001"), (1.4e-18, "< 0.0001"), (0.0001, "0.0001"), (0.5, "0.5000"), (None, "N/A"), ("x", "N/A")])
def test_a_p_cell_never_prints_an_exact_zero(p, shown) -> None:
    assert _js(f'cellText({{key: "p", kind: "p"}}, {{p: {json.dumps(p)}}})') == shown
    assert _js(f"pText({json.dumps(p)})") == shown


@needs_node
def test_each_chart_names_its_own_cards_table(client, cat) -> None:
    d = _view(client)
    d["slices"]["groups"] += [{**g, "error_slices": d["slices"]["groups"][0]["error_slices"]} for g in _view(cat)["slices"]["groups"]]
    html = _render(d)
    cards = _cards(html)
    assert len(cards) == 3
    for i, (g, card) in enumerate(zip(d["slices"]["groups"], cards)):
        key = html_lib.escape(json.dumps([g["question_id"], g["model_served"], g["primitive"]], separators=(",", ":")), quote=True).replace("&#x27;", "&#39;")
        for title, table in (("Answerability of", "answerability"), ("Error slices of", "errors")):
            target = re.search(r'aria-details="([^"]+)"', _svg(card, title)).group(1)
            assert target == f"slices-{table}-table-{i}"
            assert html.count(f'id="{target}"') == 1
            assert f'data-table="slices:{table}:{key}" id="{target}"' in card
    # No two charts of the view share a name, and each long table is in a scroll box of its own.
    names = re.findall(r'<svg class="chart"[^>]*><title>([^<]*)</title>', html)
    assert len(names) == len(set(names)) == 6
    assert all(len(re.findall(r'<div class="panel-block scroll-table">\s*<div class="tk"', card)) == 2 for card in cards)
    assert _audit(html).count('<div class="panel-block scroll-table">') == 1
    # Each chart's own values are the rows of the table it names.
    for i, (g, card) in enumerate(zip(d["slices"]["groups"], cards)):
        table = card[card.index(f'id="slices-answerability-table-{i}"'):card.index(f'id="slices-errors-table-{i}"')]
        assert [m["slice"] for m in g["answerability"]] == re.findall(r'<tr><td class="mono">([^<]*)</td>', table)


# --- the ranking's card ----------------------------------------------------------------------------


def _audit(html: str) -> str:
    return html[html.index("<h3>Blind Label Audit:"):]


@needs_node
def test_the_ranking_card_says_what_was_ranked_and_what_was_left_out(ws, client, tmp_path) -> None:
    d = _view(client, topk=30)
    card = _audit(_render(d))
    assert "<h3>Blind Label Audit: disagreement candidates (safe-v1: ids only, no state text)</h3>" in card and "Calibrated" not in card
    assert "Margins from calibrated probabilities (the fit file&#39;s fits)." not in card  # written by the page, not escaped twice
    assert "Margins from calibrated probabilities (the fit file's fits). Showing 30 of 160 labeled, answered records (top_k 30)." in card
    assert "Left out: 0 without a response, 0 unlabeled. Every group's records are ranked together" in card
    assert card.count("<tr>") == 1 + 30

    raw = _view(TestClient(create_app(Path(ws["records"]).parent, fits_path=tmp_path / "absent.json")))
    assert "Margins from raw probabilities (no fit file was read). Showing 15 of 160" in _audit(_render(raw))


@needs_node
def test_the_ranking_card_prints_the_top_k_asked_for_not_the_rows_returned(client) -> None:
    d = _view(client, topk=100)
    d["labels"]["top_flags"] = d["labels"]["top_flags"][:3]
    d["labels"]["n_total_candidates"] = 3
    assert "Showing 3 of 3 labeled, answered records (top_k 100)." in _audit(_render(d))


@needs_node
def test_the_skip_line_has_a_wording_for_a_chosen_group(client) -> None:
    d = _view(client)
    d["labels"]["skipped"] = {"no_response": 5, "unlabeled": 2}
    every = _audit(_render(d))
    assert "Left out: 5 without a response, 2 unlabeled." in every and "ranked together" in every and "belong to no group" not in every
    one = _audit(_render({**d, "grouped": True}))
    assert "Left out: 2 unlabeled. Records without a response have no served model, so they belong to no group and are not counted here." in one
    assert "without a response," not in one and "ranked together" not in one


@needs_node
def test_the_ranking_card_without_settings_or_counts_claims_none(client) -> None:
    d = _view(client, topk=None, fdrq=None)
    d["labels"]["skipped"] = {}
    del d["labels"]["n_total_candidates"]
    html = _render(d)
    assert "Showing 15 of N/A labeled, answered records (the server's default top_k). Left out: N/A without a response, N/A unlabeled." in _audit(html)
    assert "Error slices (Benjamini-Hochberg, q ≤ the server&#39;s default)" in html
    assert "significant at q ≤ the server&#39;s default." in html


@needs_node
def test_empty_records_rank_nothing_and_list_no_group(tmp_path) -> None:
    (tmp_path / "records.jsonl").write_text("")
    c = TestClient(create_app(tmp_path))
    d = _view(c)
    assert d["labels"] == {"top_flags": [], "skipped": {}} and d["slices"]["groups"] == []
    html = _render(d)
    assert '<div class="card">No labeled groups found in workspace.</div>' in html
    assert "The workspace has no records to rank." in _audit(html) and "<table" not in html
    assert _js("Object.values(slicesTables(d)).map(t => t.rows.length)", prelude=f"const d = {json.dumps(d)};") == [0]


@needs_node
def test_a_flag_without_a_dataset_is_an_empty_cell_and_a_null_in_the_export(client) -> None:
    d = _view(client)
    d["labels"]["top_flags"][0]["dataset"] = None
    card = _audit(_render(d))
    first = re.search(r"<tbody><tr>(.*?)</tr>", card, re.S).group(1)
    f = d["labels"]["top_flags"][0]
    assert re.findall(r"<td[^>]*>(.*?)</td>", first) == ["1", "", f["item_id"], f["question_id"], f["label"], f["predicted"], f"{f['score']:.4f}"]
    exported = _js("JSON.parse(toJsonRows(flagsTable(d.labels).spec, flagsTable(d.labels).rows))", prelude=f"const d = {json.dumps(d)};")
    assert exported[0] == {k: f[k] for k in ("rank", "dataset", "item_id", "question_id", "label", "predicted", "score")}


# --- what the shell sends and keeps ---------------------------------------------------------------


def _shell_run(steps: str, payload: dict, functions: str | None = None, groups: list[dict] | None = None) -> dict:
    js = f"""
let STATE = {{view: "slices"}};
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = {json.dumps(groups or [])}, groupsState = "ok", groupsError = "";
const els = {{}};
const el = id => (els[id] ??= {{id, innerHTML: "", textContent: "", value: "", attrs: {{}},
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const pending = [];
const fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const payload = {json.dumps(payload)};
const answer = () => {{ pending.splice(0, 2).forEach(p => p.resolve(p.url.startsWith("/api/slices") ? payload.slices : payload.labels)); }};
const shown = () => el("slices-container").innerHTML;
const out = {{}};
{functions if functions is not None else _all_functions()}
STATE = {{...defaultState(), view: "slices"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


@needs_node
def test_the_loader_without_settings_asks_for_the_bare_routes(client) -> None:
    out = _shell_run("""
  loadSlicesAndLabels();
  out.urls = pending.map(p => p.url);
""", _view(client))
    assert out["urls"] == ["/api/slices", "/api/labelaudit"]


SETTINGS_STEPS = """
  STATE = {...STATE, qid: GROUPS[0].question_id, model: GROUPS[0].model_served, prim: GROUPS[0].primitive, fdrq: 0.05, topk: 30};
  runView("slices");
  out.urls = pending.map(p => p.url);
  out.inputs = [el("inp-fdrq").value, el("inp-topk").value];
  answer();
  await tick(); await tick();
  out.kept = [viewData.slices.fdrq, viewData.slices.topk, viewData.slices.grouped];
  out.title = shown().includes("Error slices (Benjamini-Hochberg, q ≤ 0.05)");
  out.tables = Object.keys(currentTables()).map(k => k.split(":")[1]);
  // The state moves on without a request (a pending change): a re-render still shows what the
  // kept responses were asked with.
  STATE = {...STATE, fdrq: 0.2, topk: 100};
  rerenderView("slices");
  out.rerendered = [shown().includes("q ≤ 0.05)"), shown().includes("q ≤ 0.20"), shown().includes("(top_k 30)")];
  out.exportTitle = Object.values(currentTables())[1].spec.caption;
  // Asking for the view with the new settings requests again and drops what was kept.
  runView("slices");
  out.again = pending.map(p => p.url.split("?")[0] + "?" + p.url.split("&").slice(-1)[0]);
  out.droppedBeforeTheAnswer = !("slices" in viewData);
"""


@needs_node
def test_the_settings_are_sent_kept_with_the_answer_and_shown_from_it(client) -> None:
    groups = _get(client, "/api/groups")["groups"]
    g = groups[0]
    out = _shell_run(SETTINGS_STEPS, _view(client, fdrq=0.05, topk=30), groups=groups)
    query = f"question_id={g['question_id']}&model_served={g['model_served']}&primitive={g['primitive']}"
    assert out["urls"] == [f"/api/slices?{query}&fdr_q=0.05", f"/api/labelaudit?{query}&top_k=30"]
    assert out["inputs"] == ["0.05", "30"]
    assert out["kept"] == [0.05, 30, True] and out["title"] is True
    assert out["tables"] == ["answerability", "errors", "flags"]
    assert out["rerendered"] == [True, False, True] and out["exportTitle"] == "Error slices (Benjamini-Hochberg, q ≤ 0.05)"
    assert out["again"] == ["/api/slices?fdr_q=0.2", "/api/labelaudit?top_k=100"] and out["droppedBeforeTheAnswer"] is True


@needs_node
def test_the_title_check_catches_a_title_read_from_the_state(client) -> None:
    groups = _get(client, "/api/groups")["groups"]
    functions = _mutated("fdrq: d.fdrq, tstate, index}", "fdrq: STATE.fdrq, tstate, index}")
    out = _shell_run(SETTINGS_STEPS, _view(client, fdrq=0.05, topk=30), functions=functions, groups=groups)
    assert out["rerendered"][:2] == [False, True]


FAIL_STEPS = """
  runView("slices");
  pending[1].reject(Object.assign(new Error("boom"), {status: 400}));
  pending[0].resolve(payload.slices);
  pending.splice(0, 2);
  await tick(); await tick();
  out.failedError = shown().includes("ERROR</span> boom");
  out.failedKept = "slices" in viewData;
  // A response the renderer throws on (a slice without its interval): the error card, nothing kept.
  generation += 1;
  loadedSig = {};
  runView("slices");
  const broken = JSON.parse(JSON.stringify(payload.slices));
  broken.groups[0].answerability[0].ci = null;
  broken.groups[0].answerability[0].answerable = false;
  pending.splice(0, 2).forEach(p => p.resolve(p.url.startsWith("/api/slices") ? broken : payload.labels));
  await tick(); await tick();
  out.error = shown().includes("ERROR</span>");
  out.kept = "slices" in viewData;
  try { out.tables = Object.keys(currentTables()).length; } catch (e) { out.tables = "threw"; }
"""


@needs_node
def test_a_failed_or_unrenderable_response_keeps_nothing(client) -> None:
    d = _view(client)
    assert _shell_run(FAIL_STEPS, d) == {"failedError": True, "failedKept": False, "error": True, "kept": False, "tables": 0}
    # Kept before rendering, it would stay behind the error card, for a table action to fail on.
    functions = _mutated(
        '      document.getElementById("slices-container").innerHTML = renderSlices(d, ctx.opts ? ctx.opts() : {});\n      if (ctx.keep) ctx.keep(d);\n',
        '      if (ctx.keep) ctx.keep(d);\n      document.getElementById("slices-container").innerHTML = renderSlices(d, ctx.opts ? ctx.opts() : {});\n',
    )
    mutant = _shell_run(FAIL_STEPS, d, functions=functions)
    assert (mutant["kept"], mutant["tables"]) == (True, "threw")


STALE_STEPS = """
  runView("slices");
  const older = pending.splice(0, 2);
  // A setting changes while the first pair is out; the older pair answers last.
  STATE = {...STATE, fdrq: 0.2};
  runView("slices");
  out.second = pending.map(p => p.url);
  answer();
  await tick(); await tick();
  out.shown = shown().includes("q ≤ 0.20)");
  older.forEach(p => p.resolve(p.url.startsWith("/api/slices") ? payload.slices : {...payload.labels, top_flags: []}));
  await tick(); await tick();
  out.still = shown().includes("q ≤ 0.20)") && !shown().includes("q ≤ 0.10)");
  out.kept = viewData.slices.fdrq;
  // A Refresh outdates a request too.
  generation += 1;
  loadedSig = {};
  viewData = {};
  runView("slices");
  const late = pending.splice(0, 2);
  generation += 1;
  late.forEach(p => p.resolve(p.url.startsWith("/api/slices") ? payload.slices : payload.labels));
  await tick(); await tick();
  out.afterRefresh = "slices" in viewData;
"""


@needs_node
def test_an_older_answer_never_replaces_a_newer_one(client) -> None:
    out = _shell_run(STALE_STEPS, _view(client))
    assert out == {"second": ["/api/slices?fdr_q=0.2", "/api/labelaudit?top_k=15"], "shown": True, "still": True, "kept": 0.2, "afterRefresh": False}
    # Without the staleness check the older pair is rendered and kept.
    functions = _mutated("      if (isStale(ctx)) return;\n      const d = {slices, labels,", "      const d = {slices, labels,")
    assert _shell_run(STALE_STEPS, _view(client), functions=functions)["kept"] == 0.1


# --- the pickers' wiring ---------------------------------------------------------------------------


def _change(act: str, value: str) -> list:
    """Run the page's own delegated `change` listener for a control and return what it commits."""
    script = HTML[HTML.index("<script>"):HTML.index("</script>")]
    listener = re.search(r'document\.addEventListener\("change", ev => \{.*?\n    \}\);', script, re.S).group(0)
    js = f"""
let STATE;
const handlers = {{}};
const document = {{addEventListener: (type, fn) => {{ handlers[type] = fn; }}}};
const calls = [];
{_all_functions()}
STATE = {{...defaultState(), view: "slices", qid: "q", model: "m", prim: "noul"}};
commit = (next, mode) => calls.push([next.fdrq, next.topk, next.qid, next.view, mode]);
{listener}
const el = {{dataset: {{act: {json.dumps(act)}}}, value: {json.dumps(value)}}};
handlers.change({{target: {{closest: () => el}}}});
process.stdout.write(JSON.stringify(calls));
"""
    return _node(js)


@needs_node
@pytest.mark.parametrize(
    "act, value, committed",
    [
        ("slices-fdrq", "0.05", [0.05, 15]),
        ("slices-fdrq", "0.2", [0.2, 15]),
        ("slices-topk", "100", [0.1, 100]),
        ("slices-fdrq", "0.07", [0.1, 15]),  # through the hash's rule: never a value the picker cannot show
    ],
)
def test_a_picker_change_rewrites_the_current_history_entry(act, value, committed) -> None:
    # One commit, keeping the group and the view, with "replace": a setting is not a place to go Back to.
    assert _change(act, value) == [[*committed, "q", "slices", "replace"]]


def test_a_long_identifier_wraps_in_a_text_cell_only() -> None:
    # Numeric cells stay on one line (`td.num` is nowrap); only text cells get the width cap.
    assert ".tk td.mono:not(.num) { max-width: 44ch; overflow-wrap: break-word; }" in HTML
    assert "td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }" in HTML


# --- the kit additions -----------------------------------------------------------------------------


@needs_node
def test_a_horizontal_whisker_spans_its_interval_with_caps() -> None:
    assert _js('svgWhiskersH([{y: 20, x0: 10, x1: 50}, {y: NaN, x0: 1, x1: 2}], "ser-1")') == (
        '<path class="whisker ser-1" d="M10.0 20.0H50.0M10.0 17.0V23.0M50.0 17.0V23.0" />'
    )


@needs_node
def test_a_show_only_column_is_shown_and_filtered_and_never_exported() -> None:
    spec = {"id": "t", "caption": "c", "file": "f", "columns": [
        {"key": "a", "name": "A", "kind": "text"},
        {"key": "shown", "name": "Shown", "kind": "text", "showOnly": True},
        {"key": "raw", "name": "raw", "kind": "num", "exportOnly": True},
    ]}
    rows = [{"a": "x", "shown": "12.0%", "raw": 0.12}, {"a": "y", "shown": "50.0%", "raw": 0.5}]
    out = _js("({shown: shownColumns(spec).map(c => c.key), data: dataColumns(spec).map(c => c.key), csv: toCsv(spec, rows), hit: filterRows(spec, rows, '12.0').length})",
              prelude=f"const spec = {json.dumps(spec)}, rows = {json.dumps(rows)};")
    assert out == {"shown": ["a", "shown"], "data": ["a", "raw"], "csv": "a,raw\r\nx,0.12\r\ny,0.5\r\n", "hit": 1}
