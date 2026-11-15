"""The table kit: cells, sorting, filtering, and the CSV and JSON exports of the rows a table shows."""

from __future__ import annotations

import csv
import io
import json
import shutil

import pytest

from tests.test_web_harness import ESCAPED, HOSTILE
from tests.test_web_render import _all_functions, _node

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")

SPEC = {
    "id": "t:1",
    "caption": "Things",
    "file": "things",
    "columns": [
        {"key": "group", "name": "group", "kind": "text", "exportOnly": True},
        {"key": "name", "name": "Name", "kind": "text"},
        {"key": "v", "name": "Value", "kind": "num", "digits": 2},
        {"key": "ci", "name": "CI", "kind": "range", "digits": 2},
        {"key": "open", "name": "Open", "kind": "button", "act": "open-thing", "text": "Open", "data": {"name": "name"}},
    ],
}
ROWS = [
    {"group": "g", "name": "item10", "v": 0.5, "ci": [0.1, 0.9]},
    {"group": "g", "name": "item9", "v": None, "ci": None},
    {"group": "g", "name": "Item2", "v": -0.25, "ci": [-0.5, 0.0]},
    {"group": "g", "name": "item1", "v": 0.1 + 0.2, "ci": [0.2, 0.4]},
]


def _js(expr: str) -> object:
    return _node(f"const SPEC = {json.dumps(SPEC)}; const ROWS = {json.dumps(ROWS)};\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


def _names(expr: str) -> list[str]:
    return [r["name"] for r in _js(expr)]


@pytest.mark.parametrize("v, shown", [(0.5, "0.50"), (0, "0.00"), (None, "N/A"), ("0.5", "N/A"), (True, "N/A")])
def test_fmt_num(v, shown) -> None:
    assert _js(f"fmtNum({json.dumps(v)}, 2)") == shown


def test_fmt_num_shows_a_non_finite_number_as_na() -> None:
    assert _js("[fmtNum(NaN), fmtNum(Infinity)]") == ["N/A", "N/A"]


def test_numbers_sort_numerically_with_missing_values_last_both_ways() -> None:
    assert _names('sortRows(SPEC, ROWS, "v", "asc")') == ["Item2", "item1", "item10", "item9"]
    assert _names('sortRows(SPEC, ROWS, "v", "desc")') == ["item10", "item1", "Item2", "item9"]
    # A range sorts by its lower end.
    assert _names('sortRows(SPEC, ROWS, "ci", "asc")') == ["Item2", "item10", "item1", "item9"]


def test_text_sorts_in_natural_order() -> None:
    assert _names('sortRows(SPEC, ROWS, "name", "asc")') == ["item1", "Item2", "item9", "item10"]


def test_an_unknown_or_button_column_keeps_the_order() -> None:
    assert _names('sortRows(SPEC, ROWS, "open", "asc")') == [r["name"] for r in ROWS]
    assert _names('sortRows(SPEC, ROWS, "nope", "desc")') == [r["name"] for r in ROWS]


def test_the_filter_matches_what_the_viewer_reads() -> None:
    assert _names('filterRows(SPEC, ROWS, "ITEM1")') == ["item10", "item1"]
    assert _names('filterRows(SPEC, ROWS, "n/a")') == ["item9"]
    assert _names('filterRows(SPEC, ROWS, "-0.25")') == ["Item2"]
    # An export-only column is not what the viewer reads.
    assert _names('filterRows(SPEC, ROWS, "g")') == []
    assert _names('filterRows(SPEC, ROWS, "  ")') == [r["name"] for r in ROWS]


@pytest.mark.parametrize(
    "value, cell",
    [
        ("=1+1", "'=1+1"),
        ("+1", "'+1"),
        ("-x", "'-x"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\tx", "'\tx"),
        ("\r=1", "\"'\r=1\""),  # a carriage return also makes the field quoted
        ("a=b", "a=b"),  # only the first character counts
        ("@a,b", "\"'@a,b\""),
        ("a,b", '"a,b"'),
        ('say "hi"', '"say ""hi"""'),
        ("two\nlines", '"two\nlines"'),
        ("plain", "plain"),
        (-0.25, "-0.25"),
        (0.30000000000000004, "0.30000000000000004"),
        (None, ""),
        (True, "true"),
    ],
)
def test_csv_cell(value, cell) -> None:
    assert _js(f"csvCell({json.dumps(value)})") == cell


def test_the_csv_holds_every_data_column_and_reads_back() -> None:
    text = _js("toCsv(SPEC, ROWS)")
    assert text.endswith("\r\n")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    assert rows[0] == ["group", "name", "v", "ci_lo", "ci_hi"]
    assert rows[1:] == [
        ["g", "item10", "0.5", "0.1", "0.9"],
        ["g", "item9", "", "", ""],
        ["g", "Item2", "-0.25", "-0.5", "0"],
        ["g", "item1", "0.30000000000000004", "0.2", "0.4"],
    ]


def test_the_json_is_the_rows_as_served() -> None:
    rows = [{"group": "=g", "name": "-x", "v": 0.1 + 0.2, "ci": [0.2, 0.4]}]
    out = json.loads(_node(f"const SPEC = {json.dumps(SPEC)};\n{_all_functions()}\nprocess.stdout.write(JSON.stringify(toJsonRows(SPEC, {json.dumps(rows)})));"))
    assert out == [{"group": "=g", "name": "-x", "v": 0.30000000000000004, "ci": [0.2, 0.4]}]


def test_the_shown_rows_are_the_exported_rows() -> None:
    shown = _names('viewRows(SPEC, ROWS, {sort: "v", dir: "desc", q: "item1"})')
    assert shown == ["item10", "item1"]


def test_the_body_marks_the_sorted_column_and_escapes_every_string() -> None:
    hostile = [{"group": HOSTILE, "name": HOSTILE, "v": 1, "ci": [0, 1]}]
    spec = {**SPEC, "id": HOSTILE, "caption": HOSTILE}
    html = _node(f"{_all_functions()}\nprocess.stdout.write(JSON.stringify(tableHtml({json.dumps(spec)}, {json.dumps(hostile)}, {{{json.dumps(HOSTILE)}: {{sort: 'v', dir: 'desc', q: {json.dumps(HOSTILE)}}}}})));")
    assert "<img" not in html
    assert html.count(ESCAPED) >= 5  # the id, caption, filter text, the cell and the button's data
    assert '<th aria-sort="descending">' in html and html.count('aria-sort="none"') == 2
    # The hostile filter text matches the hostile cell, so the row is shown.
    assert "0 of 1" not in html


def test_the_count_says_how_many_rows_the_filter_keeps() -> None:
    assert _js("[tableCountText(4, 4), tableCountText(2, 4)]") == ["4 rows", "2 of 4 rows"]


def test_an_empty_filter_result_says_so() -> None:
    html = _js('tableBodyHtml(SPEC, ROWS, {sort: "", dir: "asc", q: "zzz"})')
    assert "No rows match the filter." in html


@pytest.mark.parametrize("name, safe", [("q-1/m:2 · noul", "q-1_m_2_noul"), ("", "table"), ("a..b", "a..b")])
def test_file_safe(name, safe) -> None:
    assert _js(f"fileSafe({json.dumps(name)})") == safe


def test_table_state_ignores_inherited_and_malformed_entries() -> None:
    assert _js('tableState({}, "constructor")') == {"sort": "", "dir": "asc", "q": ""}
    assert _js('tableState({x: {sort: 3, dir: "up", q: null}}, "x")') == {"sort": "", "dir": "asc", "q": ""}
