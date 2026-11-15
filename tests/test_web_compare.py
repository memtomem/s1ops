"""The shadow compare and judge view: the numbers its captions state against the code's, and the
view rendered in node over real /api/compare payloads and reports built by `compare_records` (the
paired-accuracy dumbbell, the questions table, each question's flipped pairs, the judge's overall
figures and criteria, and what the shell keeps)."""

from __future__ import annotations

import copy
import inspect
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.eval import judge
from s1ops.eval.compare import compare_records
from s1ops.web import app as web_app
from s1ops.web.app import JUDGE_PASS_LEVEL, STATIC_DIR, create_app
from tests.test_web_harness import _mutated
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")

HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
QUESTION_KEYS = {
    "question_id", "primitive", "primary_model", "shadow_model", "n", "flip_rate", "mean_distance", "max_distance",
    "n_labeled", "primary_accuracy", "shadow_accuracy", "accuracy_delta", "mcnemar", "replaceable", "flipped_items",
}
FLIPPED_KEYS = ["dataset", "item_id", "question_id", "primitive", "distance", "flipped", "primary_pred", "shadow_pred", "label"]
AGREEMENT_KEYS = {"n", "exact", "within_one", "weighted_kappa", "bias", "pass_agreement", "passes_bar"}


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws") / "w")


@pytest.fixture(scope="module")
def demo(ws) -> dict:
    r = TestClient(create_app(Path(ws["records"]).parent)).get("/api/compare")
    assert r.status_code == 200, r.text
    return r.json()


def _noul(item: str, p: float, model: str, qid: str = "q-a", label: object = None, dataset: str | None = "d") -> dict:
    row = {"schema": "s1-record/1", "primitive": "noul", "question_id": qid, "dataset": dataset, "item_id": item, "model_served": model, "p": p}
    if label is not None:
        row["label"] = label
    return row


def _dist(kind: str, item: str, probs: dict[str, float], model: str, qid: str, label: object = None) -> dict:
    row = {"schema": "s1-record/1", "primitive": kind, "question_id": qid, "dataset": "d", "item_id": item, "model_served": model, "probabilities": probs}
    if kind == "choice":
        row["choice"] = max(probs, key=probs.get)
    if label is not None:
        row["label"] = label
    return row


def _report(primary: list[dict], shadow: list[dict], judge_part: dict | None = None) -> dict:
    """What /api/compare answers for these rows: `compare_records`' own report."""
    return {"shadow": compare_records(primary, shadow), "judge": judge_part}


def _questions(n: int) -> dict:
    """`n` noul questions of four labeled pairs each: the primary right on three, the shadow on all."""
    primary, shadow = [], []
    for k in range(n):
        qid = f"q-{k:016x}"
        for i, label in enumerate([True, False, True, False]):
            primary.append(_noul(f"i{i}", 0.9 if (label or i == 3) else 0.1, "old", qid, label))
            shadow.append(_noul(f"i{i}", 0.8 if label else 0.2, "new", qid, label))
    return _report(primary, shadow)


def _js(expr: str, prelude: str = "", functions: str | None = None) -> object:
    return _node(f"{prelude}\n{functions if functions is not None else _all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


def _render(d: dict, cq: str = "", functions: str | None = None) -> str:
    return _js(f"renderCompare(d, {json.dumps(cq)}, {{tstate: {{}}}})", prelude=f"const d = {json.dumps(d)};", functions=functions)


def _cards(html: str) -> list[str]:
    """The question cards, in order (the judge's card is cut off the last one)."""
    parts = re.split(r'<div class="card">\n          <h3[^>]*>Shadow Comparison:', html)[1:]
    return [p.split("<h3>Rubric Judge Audit")[0] for p in parts]


def _overview(html: str) -> str:
    return html.split("<h3>Every compared question</h3>")[1].split('<h3')[0]


def _svg(html: str) -> str:
    m = re.search(r'<svg class="chart"[^>]*>(?:(?!</svg>).)*?<title>Paired accuracy by question.*?</svg>', html, re.S)
    assert m
    return m.group(0)


def _x(v: float) -> float:
    return 138 + v * (460 - 14 - 138)  # the lane chart's scale with a 20-cell name margin


def _mid(i: int) -> float:
    return 12 + 24 * i + 12


def _circle(svg: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in re.findall(r'<circle class="dot ser-1" cx="([^"]+)" cy="([^"]+)" r="4" />', svg)]


def _square(svg: str) -> list[tuple[float, float]]:
    return [(float(x) + 4, float(y) + 4) for x, y in re.findall(r'<rect class="dot ser-2" x="([^"]+)" y="([^"]+)" width="8" height="8" />', svg)]


def _lanes(svg: str) -> list[str]:
    return re.findall(r'text-anchor="end">([^<]*)</text>', svg)


def _table(html: str, caption: str) -> str:
    """The rendered table whose accessible name is `caption`."""
    m = re.search(rf'<table aria-label="{re.escape(caption)}">.*?</table>', html, re.S)
    assert m, caption
    return m.group(0)


def _cells(table: str) -> list[list[str]]:
    return [re.findall(r"<td[^>]*>(.*?)</td>", row, re.S) for row in re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>")[1], re.S)]


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= 0.06  # pixels are written to one place


# --- the route and the numbers the captions state ---------------------------------------------------


def test_the_route_answers_the_same_keys(demo) -> None:
    assert set(demo) == {"shadow", "judge"}
    assert set(demo["shadow"]) == {"schema", "shared_pairs", "primary_only", "shadow_only", "questions"}
    assert all(set(q) == QUESTION_KEYS for q in demo["shadow"]["questions"])
    assert set(demo["judge"]) == {"overall", "by_criterion"}
    assert set(demo["judge"]["overall"]) == AGREEMENT_KEYS
    assert all(list(item) == FLIPPED_KEYS for q in demo["shadow"]["questions"] for item in q["flipped_items"])


@needs_node
def test_the_captions_state_the_codes_own_numbers() -> None:
    guard = inspect.signature(compare_records).parameters
    rule = _js("replaceableRuleText()")
    assert f"flip rate at most {guard['max_flip_rate'].default * 100:g}%" in rule
    assert f"at most {guard['max_accuracy_drop'].default * 100:g} points under" in rule
    bar = inspect.signature(judge.agreement).parameters
    # The criteria are held to the same bar as the overall figures: `by_criterion` passes no other.
    assert "min_pass_agreement" not in inspect.signature(judge.by_criterion).parameters
    assert inspect.signature(judge.by_criterion).parameters["min_kappa"].default == bar["min_kappa"].default
    text = _js("judgeRuleText()")
    assert f"Pass is a grade of {JUDGE_PASS_LEVEL} or more" in text and f"every item under {JUDGE_PASS_LEVEL}," in text
    assert f"weighted κ at least {bar['min_kappa'].default:.2f}" in text
    assert f"pass agreement at least {bar['min_pass_agreement'].default * 100:g}%" in text
    # The route's pass level is the constant, both times it is passed.
    source = inspect.getsource(web_app.create_app)
    assert source.count("pass_level=JUDGE_PASS_LEVEL") == 2 and "pass_level=3" not in source


@needs_node
def test_the_flip_count_is_read_back_from_the_served_rate(demo) -> None:
    (q,) = demo["shadow"]["questions"]
    assert _js(f"compareFlips({json.dumps(q)})") == len(q["flipped_items"]) == 44
    # More flips than the report lists: 120 of 120 pairs flip, 50 are listed.
    many = _report([_noul(f"i{i:03d}", 0.9, "old") for i in range(120)], [_noul(f"i{i:03d}", 0.1, "new") for i in range(120)])
    (big,) = many["shadow"]["questions"]
    assert len(big["flipped_items"]) == 50 and _js(f"compareFlips({json.dumps(big)})") == 120
    # Every (flips, n) a report of up to 400 pairs can hold.
    assert _js("(() => { for (let n = 1; n <= 400; n++) for (let k = 0; k <= n; k++) if (compareFlips({flip_rate: k / n, n}) !== k) return [k, n]; return null; })()") is None
    assert _js("[compareFlips({n: 3}), compareFlips({flip_rate: 0.5})].map(v => Number.isNaN(v))") == [True, True]


# --- the dumbbell -----------------------------------------------------------------------------------


@needs_node
def test_the_circle_is_the_primarys_accuracy_and_the_square_the_shadows(demo) -> None:
    (q,) = demo["shadow"]["questions"]
    assert q["primary_accuracy"] != q["shadow_accuracy"]
    svg = _svg(_render(demo))
    ((cx, cy),), ((sx, sy),) = _circle(svg), _square(svg)
    assert _near(cx, _x(q["primary_accuracy"])) and _near(sx, _x(q["shadow_accuracy"]))
    assert _near(cy, _mid(0)) and _near(sy, _mid(0))
    link = re.search(r'<line class="link" x1="([^"]+)" y1="[^"]+" x2="([^"]+)"', svg)
    assert _near(float(link.group(1)), cx) and _near(float(link.group(2)), sx)
    assert 'aria-details="compare-questions-table"' in svg
    # Both marks of the lane carry the lane's one tooltip.
    tips = set(re.findall(r'data-tip="([^"]*)"', svg))
    assert len(tips) == 1
    tip = tips.pop()
    assert f"primary {q['primary_model']}: 71.3%" in tip and f"shadow {q['shadow_model']}: 93.8% (+22.5 points)" in tip
    assert "over 160 labeled pairs of 160" in tip and "only primary correct 4, only shadow correct 40" in tip
    assert "Holm-adjusted p = &lt; 0.0001, significant at 0.05" in tip and tip.endswith("not replaceable")


@needs_node
def test_the_significance_mark_and_the_verdicts_are_the_servers_never_the_pages(demo) -> None:
    qid = demo["shadow"]["questions"][0]["question_id"]
    # A report whose booleans contradict its numbers: the page shows the booleans.
    odd = copy.deepcopy(demo)
    q = odd["shadow"]["questions"][0]
    q["mcnemar"].update(p_holm=1e-9, significant_05=False)
    q.update(flip_rate=0.9, replaceable=True)
    html = _render(odd)
    assert _lanes(_svg(html)) == [qid]
    row = _cells(_table(html, "Compared questions"))[0]
    assert row[-2:] == ["no", "yes"] and "1 replaceable · 0 not replaceable" in _overview(html)
    assert "not significant at 0.05 after Holm" in _cards(html)[0] and "not significant at 0.05" in _svg(html)
    q["mcnemar"].update(p_holm=0.9, significant_05=True)
    q.update(flip_rate=0.0, replaceable=False)
    html = _render(odd)
    assert _lanes(_svg(html)) == [qid + " *"]
    assert _cells(_table(html, "Compared questions"))[0][-2:] == ["yes", "no"] and "0 replaceable · 1 not replaceable" in _overview(html)
    assert "· significant at 0.05 after Holm" in _cards(html)[0] and "Exceeds Flip/Drop Guard" in _cards(html)[0]
    # The exports carry the served booleans, not the page's words.
    exported = _js('JSON.parse(toJsonRows(compareTables(d)["compare:questions"].spec, compareTables(d)["compare:questions"].rows))', prelude=f"const d = {json.dumps(odd)};")
    assert (exported[0]["significant_05"], exported[0]["replaceable"]) == (True, False)


@needs_node
def test_a_question_without_labels_has_a_row_and_no_lane() -> None:
    primary = [_noul("i0", 0.9, "old", "q-labeled", True), _noul("i1", 0.2, "old", "q-labeled", False), _noul("i0", 0.9, "old", "q-plain")]
    shadow = [_noul("i0", 0.8, "new", "q-labeled", True), _noul("i1", 0.7, "new", "q-labeled", False), _noul("i0", 0.1, "new", "q-plain")]
    d = _report(primary, shadow)
    html = _render(d)
    assert _lanes(_svg(html)) == ["q-labeled"]
    assert "1 of 2 questions have no labeled pair, so no accuracy and no lane; every question is in the table." in _overview(html)
    rows = _cells(_table(html, "Compared questions"))
    assert [r[0] for r in rows] == ["q-labeled", "q-plain"]
    plain = rows[1]
    assert plain[6] == "0" and plain[10:17] == ["N/A"] * 7 and plain[17:] == ["", "no"]
    assert "No labeled pair: accuracy and McNemar's test need labels" in _cards(html)[1]
    # No question with labels at all: the sentence, and no chart.
    none = _render(_report([_noul("i0", 0.9, "old")], [_noul("i0", 0.1, "new")]))
    assert "<svg" not in _overview(none) and "1 of 1 questions have no labeled pair" in _overview(none)


@needs_node
def test_accuracy_is_over_the_labeled_pairs_not_every_pair() -> None:
    # Four pairs, two of them labeled; the primary is right on one, the shadow on both.
    primary = [_noul("i0", 0.9, "old", label=True), _noul("i1", 0.9, "old", label=False), _noul("i2", 0.9, "old"), _noul("i3", 0.1, "old")]
    shadow = [_noul("i0", 0.8, "new", label=True), _noul("i1", 0.2, "new", label=False), _noul("i2", 0.9, "new"), _noul("i3", 0.2, "new")]
    d = _report(primary, shadow)
    (q,) = d["shadow"]["questions"]
    assert (q["n"], q["n_labeled"], q["primary_accuracy"], q["shadow_accuracy"]) == (4, 2, 0.5, 1.0)
    html = _render(d)
    assert "Accuracy over 2 labeled pairs of 4 (+50.0 points)" in _cards(html)[0]
    assert "over 2 labeled pairs of 4" in _svg(html)
    assert _cells(_table(html, "Compared questions"))[0][5:7] == ["4", "2"]
    ((cx, _),), ((sx, _),) = _circle(_svg(html)), _square(_svg(html))
    assert _near(cx, _x(0.5)) and _near(sx, _x(1.0))


@needs_node
def test_many_questions_are_listed_not_opened(demo) -> None:
    forty = {**_questions(40), "judge": demo["judge"]}
    html = _render(forty)
    svg = _svg(html)
    assert len(_lanes(svg)) == len(_circle(svg)) == len(_square(svg)) == 30
    assert "The first 30 of 40 questions with labels are drawn; every one is in the table." in _overview(html)
    assert len(_cells(_table(html, "Compared questions"))) == 40
    assert _cards(html) == [] and "40 questions: choose one (Show, or the picker) to open its details." in _overview(html)
    assert "Rubric Judge Audit" in html  # the judge's card does not depend on the cards
    # One more than the limit lists; the limit itself opens every card; a chosen question opens one.
    assert _js("compareCardLimit()") == 6
    assert len(_cards(_render(_questions(6)))) == 6 and "choose one" not in _render(_questions(6))
    seven = _questions(7)
    assert _cards(_render(seven)) == [] and "7 questions: choose one" in _render(seven)
    chosen = _render(seven, cq="q-0000000000000003")
    assert len(_cards(chosen)) == 1 and "q-0000000000000003" in _cards(chosen)[0] and "choose one" not in chosen
    assert len(_cells(_table(chosen, "Compared questions"))) == 7  # the overview is always every question


@needs_node
def test_a_lane_name_keeps_its_mark_and_the_other_lane_charts_keep_their_margin() -> None:
    assert _js("[laneFrame(3).ml, laneFrame(3, 20).ml, laneFrame(3).H, laneFrame(3).x(0), laneFrame(3).x(1), laneFrame(3, 20).x(1)]") == [126, 138, 124, 126, 446, 446]
    d = _questions(2)
    long_id = "q-" + "long" * 20
    d["shadow"]["questions"][1]["question_id"] = long_id
    for q in d["shadow"]["questions"]:
        q["mcnemar"]["significant_05"] = True
    lanes = _lanes(_svg(_render(d)))
    assert lanes[0] == "q-0000000000000000 *"  # an 18-character id and its mark, whole
    assert lanes[1] == long_id[:17] + "… *" and long_id in _svg(_render(d))  # cut in the lane, whole in the tooltip


# --- the tables -------------------------------------------------------------------------------------


@needs_node
def test_the_flipped_pairs_say_how_many_the_report_lists(demo) -> None:
    card = _cards(_render(demo))[0]
    assert "The report lists 44 of this question's 44 flipped pairs." in card
    assert "its first" not in card
    qid = demo["shadow"]["questions"][0]["question_id"]
    rows = _cells(_table(card, "Flipped pairs of " + qid))
    first = demo["shadow"]["questions"][0]["flipped_items"][0]
    assert len(rows) == 44 and rows[0] == [first["dataset"], first["item_id"], f"{first['distance']:.4f}", "false", "true", "true"]
    many = _report([_noul(f"i{i:03d}", 0.9, "old") for i in range(120)], [_noul(f"i{i:03d}", 0.1, "new") for i in range(120)])
    card = _cards(_render(many))[0]
    assert "The report lists 50 of this question's 120 flipped pairs: its first 50 in dataset, item order." in card
    assert len(_cells(_table(card, "Flipped pairs of q-a"))) == 50
    none = _cards(_render(_report([_noul("i0", 0.9, "old")], [_noul("i0", 0.8, "new")])))[0]
    assert "No flipped pair." in none and "<table" not in none


@needs_node
def test_choice_and_score_answers_are_text_cells_and_export_as_served() -> None:
    primary = [
        _dist("choice", "c0", {"refund": 0.7, "other": 0.3}, "old", "q-choice", "refund"),
        _dist("score", "s0", {"0": 0.1, "1": 0.2, "2": 0.7}, "old", "q-score"),
    ]
    shadow = [
        _dist("choice", "c0", {"refund": 0.2, "other": 0.8}, "new", "q-choice", "refund"),
        _dist("score", "s0", {"0": 0.8, "1": 0.1, "2": 0.1}, "new", "q-score"),
    ]
    d = _report(primary, shadow)
    html = _render(d)
    choice, score = _cards(html)
    assert _cells(_table(choice, "Flipped pairs of q-choice"))[0][3:] == ["refund", "other", "refund"]
    assert _cells(_table(score, "Flipped pairs of q-score"))[0][3:] == ["2", "0", ""]  # levels, and no label
    out = _js(
        "Object.fromEntries(['compare:flipped:q-choice', 'compare:flipped:q-score'].map(id => "
        "[id, [JSON.parse(toJsonRows(compareTables(d)[id].spec, compareTables(d)[id].rows)), toCsv(compareTables(d)[id].spec, compareTables(d)[id].rows).split('\\r\\n')[0]]]))",
        prelude=f"const d = {json.dumps(d)};",
    )
    (choice_row,), header = out["compare:flipped:q-choice"]
    (score_row,), _ = out["compare:flipped:q-score"]
    # The export is each flipped item as served: its keys in order, its values with their types.
    assert list(choice_row) == FLIPPED_KEYS and header == ",".join(FLIPPED_KEYS)
    assert choice_row == d["shadow"]["questions"][0]["flipped_items"][0]
    assert score_row == d["shadow"]["questions"][1]["flipped_items"][0]
    assert (score_row["primary_pred"], score_row["shadow_pred"], score_row["label"], score_row["flipped"]) == (2, 0, None, True)


@needs_node
def test_the_exports_hold_served_values_under_fixed_keys(demo) -> None:
    out = _js(
        "Object.fromEntries(Object.entries(compareTables(d)).map(([id, t]) => [id, [Object.keys(JSON.parse(toJsonRows(t.spec, t.rows))[0]), toCsv(t.spec, t.rows).split('\\r\\n')[0], t.rows.length]]))",
        prelude=f"const d = {json.dumps(demo)};",
    )
    qid = demo["shadow"]["questions"][0]["question_id"]
    assert sorted(out) == sorted(["compare:questions", f"compare:flipped:{qid}", "compare:judge"])
    question_keys = [
        "question_id", "primitive", "primary_model", "shadow_model", "n", "n_labeled", "flip_rate", "mean_distance", "max_distance",
        "primary_accuracy", "shadow_accuracy", "accuracy_delta", "a_only", "b_only", "p", "p_holm", "significant_05", "replaceable",
    ]
    assert out["compare:questions"] == [question_keys, ",".join(question_keys), 1]
    assert out[f"compare:flipped:{qid}"] == [FLIPPED_KEYS, ",".join(FLIPPED_KEYS), 44]
    judge_keys = ["criterion", "n", "exact", "within_one", "weighted_kappa", "bias", "pass_agreement", "passes_bar"]
    assert out["compare:judge"] == [judge_keys, ",".join(judge_keys), 3]
    # A table of the view by id is that question's rows, value for value.
    rows = _js(f'compareTables(d)[{json.dumps("compare:flipped:" + qid)}].rows', prelude=f"const d = {json.dumps(demo)};")
    assert rows == demo["shadow"]["questions"][0]["flipped_items"]


@needs_node
def test_the_summary_counts_answered_records(demo) -> None:
    text = re.sub(r"\s+", " ", _overview(_render(demo)))
    assert "1 question compared: 0 replaceable · 1 not replaceable." in text
    assert "160 pairs answered in both files; 0 answered only in the primary file, 0 only in the shadow file." in text
    assert "A record without a response is not compared" in text
    # A failed probe on one side is not compared, so its item counts as answered only on the other.
    failed = {**_noul("i1", 0.5, "old"), "model_served": None}
    d = _report([_noul("i0", 0.9, "old"), failed], [_noul("i0", 0.8, "new"), _noul("i1", 0.8, "new")])
    assert (d["shadow"]["shared_pairs"], d["shadow"]["primary_only"], d["shadow"]["shadow_only"]) == (1, 0, 1)
    assert "1 pairs answered in both files; 0 answered only in the primary file, 1 only in the shadow file." in re.sub(r"\s+", " ", _overview(_render(d)))
    # Nothing shared: the pinned sentence, then the same explanation.
    apart = _render(_report([_noul("i0", 0.9, "old")], [_noul("i1", 0.8, "new")]))
    assert "No items shared between the two record files (1 primary only, 1 shadow only). A record without a response is not compared" in apart


# --- the judge --------------------------------------------------------------------------------------


@needs_node
def test_the_judge_card_leads_with_the_overall_figures(demo) -> None:
    j = demo["judge"]
    assert all(st["n"] != j["overall"]["n"] for st in j["by_criterion"].values())  # the overall is not a criterion's
    html = _render(demo)
    card = html.split("<h3>Rubric Judge Audit")[1]
    o = j["overall"]
    assert f'<div class="kpi">{o["n"]}</div>' in card
    assert f'<div class="kpi">{o["exact"] * 100:.1f}% · {o["within_one"] * 100:.1f}%</div>' in card
    assert f'<div class="kpi">{o["bias"]:+.3f}</div>' in card
    assert '<span class="badge badge-ok">PASS</span>' in card
    rows = _cells(_table(card, "Judge agreement by criterion"))
    assert [r[0] for r in rows] == sorted(j["by_criterion"])
    first = j["by_criterion"][rows[0][0]]
    assert rows[0][1:] == [str(first["n"]), f"{first['exact']:.3f}", f"{first['within_one']:.3f}", f"{first['weighted_kappa']:.3f}", f"{first['bias']:.3f}", f"{first['pass_agreement']:.3f}", "PASS"]
    assert "Pass agreement is the share of items on which the judge and the human agree about reaching it" in card
    # The verdicts are the served booleans: a report whose numbers clear the bar and whose booleans
    # say otherwise shows BELOW BAR, overall and for the criterion.
    name = sorted(j["by_criterion"])[0]
    odd = copy.deepcopy(demo)
    odd["judge"]["overall"]["passes_bar"] = False
    odd["judge"]["by_criterion"][name]["passes_bar"] = False
    assert odd["judge"]["overall"]["weighted_kappa"] > 0.9 and odd["judge"]["by_criterion"][name]["weighted_kappa"] > 0.9
    card = _render(odd).split("<h3>Rubric Judge Audit")[1]
    assert '<span class="badge badge-warn">BELOW BAR</span>' in card and "badge-ok" not in card
    assert [r[-1] for r in _cells(_table(card, "Judge agreement by criterion"))] == ["BELOW BAR", "PASS", "PASS"]
    # An undefined kappa is N/A in the heading and the cell, and decides no verdict on the page.
    odd = copy.deepcopy(demo)
    odd["judge"]["overall"]["weighted_kappa"] = None
    odd["judge"]["by_criterion"][name]["weighted_kappa"] = None
    card = _render(odd).split("<h3>Rubric Judge Audit")[1]
    assert card.startswith(" (Quadratic-Weighted κ = N/A,") and '<span class="badge badge-ok">PASS</span>' in card
    row = _cells(_table(card, "Judge agreement by criterion"))[0]
    assert (row[4], row[-1]) == ("N/A", "PASS")


# --- names for assistive technology -----------------------------------------------------------------


@needs_node
def test_every_chart_and_table_of_the_view_is_named_and_linked(demo) -> None:
    six = {**_questions(6), "judge": demo["judge"]}
    html = _render(six)
    targets = re.findall(r'aria-details="([^"]+)"', html)
    assert targets == ["compare-questions-table"] and html.count('id="compare-questions-table"') == 1
    ids = re.findall(r'<div class="tk" data-table="[^"]*" id="([^"]+)"', html)
    assert ids == ["compare-questions-table", *[f"compare-flipped-table-{i}" for i in range(6)], "compare-judge-table"]
    names = re.findall(r'<table aria-label="([^"]*)">', html)
    assert len(names) == len(set(names)) == 8 and "Flipped pairs of q-0000000000000002" in names
    assert re.findall(r"<title>([^<]*)</title>", html) == ["Paired accuracy by question, primary and shadow"]
    # Each flipped table sits in its own question's card.
    for i, card in enumerate(_cards(html)):
        assert f'id="compare-flipped-table-{i}"' in card and f"Flipped pairs of q-{i:016x}" in card


@needs_node
def test_every_table_of_the_page_is_named_by_its_caption() -> None:
    spec = {"id": "t", "caption": 'A "quoted" <caption>', "file": "f", "columns": [{"key": "a", "name": "A", "kind": "text"}]}
    html = _js(f"tableHtml({json.dumps(spec)}, [{{a: 1}}], {{}})")
    assert '<table aria-label="A &quot;quoted&quot; &lt;caption&gt;"><thead>' in html


# --- the question picker and Show --------------------------------------------------------------------


@needs_node
def test_a_question_the_report_lacks_gets_its_own_picker_entry(ws) -> None:
    qs = [{"question_id": "q-a"}, {"question_id": "q-b"}]
    known = _js(f'questionPickerHtml("compare-question", {json.dumps(qs)}, "q-b", "All")')
    assert '<option value="q-b" selected>q-b</option>' in known and "Unresolved" not in known
    assert "Unresolved" not in _js(f'questionPickerHtml("compare-question", {json.dumps(qs)}, "", "All")')
    unknown = _js(f'questionPickerHtml("compare-question", {json.dumps(qs)}, {json.dumps("q-<none>")}, "All")')
    assert unknown.count(" selected") == 1 and '<option value="q-&lt;none&gt;" selected>Unresolved: q-&lt;none&gt;</option>' in unknown
    # Both views that have a question of their own use it: an edited link with many questions.
    seven = _render(_questions(7), cq="q-none")
    assert '<option value="q-none" selected>Unresolved: q-none</option>' in seven and "No question q-none in the shadow comparison." in seven
    assert _cards(seven) == [] and len(_cells(_table(seven, "Compared questions"))) == 7
    drift = TestClient(create_app(Path(ws["records"]).parent)).get("/api/drift/histograms").json()
    assert '<option value="q-none" selected>Unresolved: q-none</option>' in _js(f'renderDrift({json.dumps(drift["report"])}, "q-none", {{}})')


def _event(kind: str, dataset: dict, value: str = "") -> list:
    """Run the page's own delegated listener of `kind` for an element and return what it does."""
    script = HTML[HTML.index("<script>"):HTML.index("</script>")]
    listener = re.search(rf'document\.addEventListener\("{kind}", ev => \{{.*?\n    \}}\);', script, re.S).group(0)
    js = f"""
let STATE;
const handlers = {{}};
const document = {{addEventListener: (type, fn) => {{ handlers[type] = fn; }}}};
const calls = [];
{_all_functions()}
STATE = {{...defaultState(), view: "compare", cq: "q-old", dq: "d-old"}};
commit = (next, mode) => calls.push(["commit", next.view, next.cq, next.dq, mode]);
focusCard = id => calls.push(["focus", id]);
{listener}
const el = {{dataset: {json.dumps(dataset)}, value: {json.dumps(value)}}};
handlers[{json.dumps(kind)}]({{target: {{closest: () => el}}}});
process.stdout.write(JSON.stringify(calls));
"""
    return _node(js)


@needs_node
def test_show_and_the_picker_open_a_question_with_a_history_entry() -> None:
    # Show: the question, pushed, then the focus to this view's card; drift's Show keeps its own.
    assert _event("click", {"act": "compare-show", "qid": "q-new"}) == [["commit", "compare", "q-new", "d-old", "push"], ["focus", "compare-container"]]
    assert _event("click", {"act": "drift-show", "qid": "q-new"}) == [["commit", "compare", "q-old", "q-new", "push"], ["focus", "drift-container"]]
    assert _event("change", {"act": "compare-question"}, "q-new") == [["commit", "compare", "q-new", "d-old", "push"]]


@needs_node
def test_the_shown_cards_heading_takes_the_focus_below_the_filter_bar(demo) -> None:
    qid = demo["shadow"]["questions"][0]["question_id"]
    assert '<h3 tabindex="-1">Shadow Comparison:' in _render(demo, cq=qid) and "tabindex" not in "".join(_cards(_render(demo)))
    out = _js(
        "(() => { focusCard('compare-container'); focusCard('absent'); return calls; })()",
        prelude="""
const calls = [];
const heading = {focus: o => calls.push(["focus", o.preventScroll]), getBoundingClientRect: () => ({top: 300})};
const boxes = {"compare-container": {querySelector: sel => (calls.push(["query", sel]), heading)}, "filter-bar": {offsetHeight: 40}};
const document = {getElementById: id => boxes[id] || null};
const window = {scrollY: 100, scrollTo: o => calls.push(["scroll", o.top])};
""",
    )
    assert out == [["query", ".card h3[tabindex]"], ["focus", True], ["scroll", 300 + 100 - 40 - 12]]


# --- what the shell keeps -----------------------------------------------------------------------------


def _shell_run(steps: str, payload: dict, functions: str | None = None) -> dict:
    js = f"""
let STATE = {{view: "compare"}};
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [], groupsState = "ok", groupsError = "";
const els = {{}};
const log = [];
const el = id => (els[id] ??= {{id, textContent: "", value: "", attrs: {{}}, html: "",
  set innerHTML(v) {{ this.html = v; log.push("write " + id); }}, get innerHTML() {{ return this.html; }},
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const pending = [];
let fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const payload = {json.dumps(payload)};
const shown = () => el("compare-container").innerHTML;
const out = {{}};
{functions if functions is not None else _all_functions()}
STATE = {{...defaultState(), view: "compare"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


KEEP_STEPS = """
  runView("compare");
  out.urls = pending.map(p => p.url);
  out.keptBeforeTheAnswer = "compare" in viewData;
  pending.splice(0, 1)[0].resolve(payload);
  await tick(); await tick();
  out.kept = viewData.compare === payload;
  out.cards = shown().split("Shadow Comparison:").length - 1;
  out.tables = Object.keys(currentTables()).map(k => k.split(":").slice(0, 2).join(":"));
  // A question chosen later, and a table's state, are read with the kept payload: nothing is fetched.
  fetchJson = () => { throw new Error("no request"); };
  const qid = payload.shadow.questions[2].question_id;
  STATE = {...STATE, cq: qid};
  TSTATE = {["compare:flipped:" + qid]: {sort: "item_id", dir: "desc", q: ""}};
  runView("compare");
  out.one = [shown().split("Shadow Comparison:").length - 1, shown().includes('<h3 tabindex="-1">Shadow Comparison: <span class="badge badge-info">' + qid)];
  out.sorted = / aria-sort="descending"><button[^>]*data-col="item_id"/.test(shown());
  out.requests = pending.length;
"""


@needs_node
def test_the_view_is_kept_once_shown_and_rerendered_from_what_was_kept() -> None:
    d = _questions(4)
    for q in d["shadow"]["questions"]:
        assert q["flipped_items"]
    out = _shell_run(KEEP_STEPS, d)
    assert out == {
        "urls": ["/api/compare"], "keptBeforeTheAnswer": False, "kept": True, "cards": 4,
        "tables": ["compare:questions", *["compare:flipped"] * 4], "one": [1, True], "sorted": True, "requests": 0,
    }


TABLE_STEPS = r"""
  runView("compare");
  pending.splice(0, 1)[0].resolve(payload);
  await tick(); await tick();
  const ids = Object.keys(currentTables());
  const boxes = ids.map(id => ({dataset: {table: id}, body: {innerHTML: ""}, count: {textContent: ""},
    querySelector(sel) { return sel === ".tk-body" ? this.body : this.count; }, querySelectorAll: () => []}));
  document.querySelectorAll = sel => sel === ".tk" ? boxes : [];
  const downloads = [];
  downloadText = (name, text) => downloads.push([name, text]);
  const second = payload.shadow.questions[1].question_id;
  const id = "compare:flipped:" + second;
  sortTable(id, "item_id"); sortTable(id, "item_id");
  out.touched = boxes.filter(box => box.body.innerHTML).map(box => box.dataset.table);
  out.order = Array.from(boxes.find(box => box.dataset.table === id).body.innerHTML.matchAll(/<td class="mono">(i\d)<\/td>/g)).map(m => m[1]);
  exportTable(id, "json");
  out.exported = [downloads[0][0], JSON.parse(downloads[0][1])];
  filterTable("compare:questions", second);
  out.count = boxes.find(box => box.dataset.table === "compare:questions").count.textContent;
  exportTable("compare:questions", "csv");
  out.csv = downloads[1][1].split("\r\n").slice(1).filter(Boolean).map(line => line.split(",")[0]);
  exportTable("compare:judge", "json");
  out.judge = JSON.parse(downloads[2][1]).map(r => [r.criterion, r.n]);
"""


@needs_node
def test_a_table_action_reaches_its_own_questions_rows(demo) -> None:
    # Through the shell's own sort, filter and export: three questions, every pair of each flipped.
    primary = [_noul(f"i{i}", 0.9, "old", qid) for qid in ("q-a", "q-b", "q-c") for i in range(3)]
    shadow = [_noul(f"i{i}", 0.1 + i / 100, "new", qid) for qid in ("q-a", "q-b", "q-c") for i in range(3)]
    d = _report(primary, shadow, demo["judge"])
    listed = d["shadow"]["questions"][1]["flipped_items"]
    assert [f["question_id"] for f in listed] == ["q-b"] * 3
    out = _shell_run(TABLE_STEPS, d)
    assert out["touched"] == ["compare:flipped:q-b"] and out["order"] == ["i2", "i1", "i0"]  # that table only
    assert out["exported"] == ["compare-flipped-q-b.json", listed[::-1]]  # that question's rows, as sorted, as served
    assert out["count"] == "1 of 3 rows" and out["csv"] == ["q-b"]
    assert out["judge"] == [[name, st["n"]] for name, st in d["judge"]["by_criterion"].items()]


FAIL_STEPS = """
  runView("compare");
  pending.splice(0, 1)[0].reject(Object.assign(new Error("boom"), {status: 400}));
  await tick(); await tick();
  out.failedError = shown().includes("ERROR</span> boom");
  out.failedKept = "compare" in viewData;
  // A response the renderer throws on (a question without its distance): the error card, nothing kept.
  generation += 1;
  loadedSig = {};
  log.length = 0;
  runView("compare");
  const broken = JSON.parse(JSON.stringify(payload));
  broken.shadow.questions[0].mean_distance = null;
  pending.splice(0, 1)[0].resolve(broken);
  await tick(); await tick();
  out.error = shown().includes("ERROR</span>");
  out.kept = "compare" in viewData;
  try { out.tables = Object.keys(currentTables()).length; } catch (e) { out.tables = "threw"; }
"""


@needs_node
def test_a_failed_or_unrenderable_response_keeps_nothing(demo) -> None:
    assert _shell_run(FAIL_STEPS, demo) == {"failedError": True, "failedKept": False, "error": True, "kept": False, "tables": 0}
    # Kept before rendering (as this loader did before), it stays behind the error card.
    render = '      document.getElementById("compare-container").innerHTML = renderCompare(d, ctx.cq ? ctx.cq() : "", ctx.opts ? ctx.opts() : {});\n'
    keep = "      if (ctx.keep) ctx.keep(d);\n"
    mutant = _shell_run(FAIL_STEPS, demo, functions=_mutated(render + keep, keep + render))
    assert (mutant["error"], mutant["kept"]) == (True, True)


STALE_STEPS = """
  runView("compare");
  const older = pending.splice(0, 1)[0];
  // A Refresh while the first request is out; the older one answers last.
  generation += 1;
  loadedSig = {};
  viewData = {};
  runView("compare");
  pending.splice(0, 1)[0].resolve(payload);
  await tick(); await tick();
  out.shown = shown().includes("1 question compared");
  older.resolve({shadow: {questions: []}, judge: null});
  await tick(); await tick();
  out.still = shown().includes("1 question compared") && !shown().includes("No shadow comparison records");
  out.kept = viewData.compare === payload;
  // A Refresh outdates a request that has not been replaced, too.
  loadedSig = {};
  viewData = {};
  runView("compare");
  const late = pending.splice(0, 1)[0];
  generation += 1;
  late.resolve(payload);
  await tick(); await tick();
  out.afterRefresh = "compare" in viewData;
"""


@needs_node
def test_an_older_answer_never_replaces_a_newer_one(demo) -> None:
    assert _shell_run(STALE_STEPS, demo) == {"shown": True, "still": True, "kept": True, "afterRefresh": False}
    functions = _mutated('      const d = await fetchJson("/api/compare");\n      if (isStale(ctx)) return;\n', '      const d = await fetchJson("/api/compare");\n')
    mutant = _shell_run(STALE_STEPS, demo, functions=functions)
    assert (mutant["still"], mutant["afterRefresh"]) == (False, True)
