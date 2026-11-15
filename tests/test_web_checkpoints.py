"""The checkpoint view rendered in node over real /api/checkpoints payloads: the names table, each
checkpoint's notes, step-metric charts and tables, the empty states, and the shell keeping a
response only once it is on screen."""

from __future__ import annotations

import copy
import html as html_lib
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.contracts.schemas import save_policies
from s1ops.demo import generate_workspace
from s1ops.improve.trainrun import load_trainrun
from s1ops.web.app import create_app
from tests.test_checkpoints import build_workspace, policy_entry
from tests.test_trainrun_shown import BASELINE, R1, R2
from tests.test_web_harness import HOSTILE, _mutated
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")


def _get(root: Path, **kwargs) -> dict:
    r = TestClient(create_app(workspace_dir=root, **kwargs)).get("/api/checkpoints")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def payloads(tmp_path_factory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("ws")
    generate_workspace(root / "demo")
    (root / "full").mkdir()
    ws = build_workspace(root / "full")
    r1 = load_trainrun(ws / "trainrun.json").trainrun_id
    save_policies([policy_entry(R1, r1), policy_entry(R2, r1), policy_entry(BASELINE, None)], ws / "policy.json")
    full = _get(ws)
    # The second checkpoint alone, its manifest exported again: no parent record, another manifest.
    manifest = ws / "training_manifest-r2.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    orphan = _get(ws, training_run_paths=[ws / "trainrun-r2.json"])
    (root / "empty").mkdir()
    return {"full": full, "orphan": orphan, "demo": _get(root / "demo"), "empty": _get(root / "empty")}


def _js(expr: str, prelude: str = "") -> object:
    return _node(f"{prelude}\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


def _render(d: dict, ck: str = "") -> str:
    return _js(f"renderCheckpoints({json.dumps(d)}, {json.dumps(ck)}, {{tstate: {{}}}})")


def _cards(html: str) -> dict[str, str]:
    parts = re.split(r'<div class="card">\n            <h3[^>]*>Checkpoint: <span class="badge badge-info">', html)[1:]
    return {part[:part.index("</span>")]: part for part in parts}


def _by_name(d: dict) -> dict[str, dict]:
    return {c["checkpoint"]: c for c in d["checkpoints"]}


def _svgs(card: str) -> dict[str, str]:
    return {m.group(1): m.group(0) for m in re.finditer(r'<svg class="chart"[^>]*><title>(\S+) by step.*?</svg>', card, re.S)}


def _num(v: float) -> str:
    """A number as JavaScript prints it: the shortest text that reads back, a whole one without a point."""
    return str(int(v)) if float(v).is_integer() else repr(v)


def _notes(card: str) -> list[str]:
    head = card[:card.index('<div class="grid-3">')]
    return re.findall(r'<div style="margin:4px 0;"><span class="badge [a-z-]+">([A-Z ]+)</span>', head)


# --- the names table -------------------------------------------------------------------------------


@needs_node
def test_the_table_lists_every_name_with_the_served_counts(payloads) -> None:
    d = payloads["full"]
    rows = _js(f"checkpointsTable({json.dumps(d)}).rows")
    assert [r["checkpoint"] for r in rows] == [c["checkpoint"] for c in d["checkpoints"]] == [R1, R2, BASELINE]
    for r, c in zip(rows, d["checkpoints"]):
        assert (r["n_records"], r["n_trained"], r["n_trained_hits"]) == (c["n_records"], c["n_trained"], c["n_trained_hits"])
        assert r["manifests"] == len(c["manifests"]) and r["policies"] == len(c["policies"])
        assert r["trainrun_id"] == (c["run"]["trainrun_id"] if c["run"] else None)
    assert [r["status"] for r in rows] == ["ON TRAINED ITEMS", "ON TRAINED ITEMS", "TRAINING SET UNKNOWN"]
    assert (rows[0]["parent"], rows[1]["parent"], rows[2]["steps"], rows[0]["steps"]) == (None, R1, None, 5)
    html = _render(d)
    for name in (R1, R2, BASELINE):
        assert f'data-act="checkpoint-show" data-ck="{name}" aria-label="Show: {name}">Show</button>' in html
    # An unknown count is shown as unknown, never as a count of zero.
    assert rows[2]["n_trained_hits"] is None and '<div class="kpi">unknown</div>' in _cards(html)[BASELINE]


@needs_node
@pytest.mark.parametrize(
    "counts, status, rank",
    [
        ({"n_records": 9, "n_trained": 3, "n_trained_hits": 2}, "ON TRAINED ITEMS", 3),
        ({"n_records": 0, "n_trained": 3, "n_trained_hits": 0}, "NO RECORDS", 2),
        ({"n_records": 9, "n_trained": None, "n_trained_hits": None}, "TRAINING SET UNKNOWN", 1),
        ({"n_records": 0, "n_trained": None, "n_trained_hits": None}, "TRAINING SET UNKNOWN", 1),
        ({"n_records": 9, "n_trained": 3, "n_trained_hits": 0}, "HELD OUT", 0),
    ],
)
def test_a_status_is_the_served_counts_in_words(counts, status, rank) -> None:
    st = _js(f"checkpointStatus({json.dumps(counts)})")
    assert (st["text"], st["rank"]) == (status, rank)


@needs_node
def test_status_sorts_by_how_pressing_it_is(payloads) -> None:
    d = copy.deepcopy(payloads["full"])
    d["checkpoints"][0].update(n_trained_hits=0)  # held out
    order = _js(f"(() => {{ const t = checkpointsTable({json.dumps(d)}); return sortRows(t.spec, t.rows, 'status', 'desc').map(r => r.status); }})()")
    assert order == ["ON TRAINED ITEMS", "TRAINING SET UNKNOWN", "HELD OUT"]


# --- one checkpoint or all ---------------------------------------------------------------------------


@needs_node
def test_a_chosen_checkpoint_shows_its_card_under_the_whole_table(payloads) -> None:
    d = payloads["full"]
    assert list(_cards(_render(d))) == [R1, R2, BASELINE] and "tabindex" not in _render(d)
    one = _render(d, R2)
    assert list(_cards(one)) == [R2] and one.count('data-act="checkpoint-show"') == 3
    assert f'<option value="{R2}" selected>{R2}</option>' in one and "All checkpoints (3)" in one
    assert '<h3 tabindex="-1">Checkpoint:' in one
    # The steps table keeps one DOM id whichever cards are shown: the checkpoint's place in the report.
    assert 'id="ckpt-steps-table-1"' in one and 'id="ckpt-steps-table-1"' in _cards(_render(d))[R2]


@needs_node
def test_a_name_the_report_does_not_hold_is_said_not_swapped(payloads) -> None:
    html = _render(payloads["full"], HOSTILE)
    assert "<img" not in html and "No checkpoint &quot;&#39;&gt;&lt;img src=x onerror=alert(1)&gt; in this workspace." in html
    assert _cards(html) == {} and html.count('data-act="checkpoint-show"') == 3


# --- notes -----------------------------------------------------------------------------------------


@needs_node
def test_each_note_follows_its_served_fact(payloads) -> None:
    full, orphan = payloads["full"], payloads["orphan"]
    cards = _cards(_render(full))
    assert _notes(cards[R1]) == ["ON TRAINED ITEMS"]
    assert _notes(cards[R2]) == ["ON TRAINED ITEMS", "POLICY NAMES ANOTHER RUN"]
    assert _notes(cards[BASELINE]) == ["TRAINING SET UNKNOWN"]
    r1, r2 = _by_name(full)[R1], _by_name(full)[R2]
    assert "2 item(s) among its answered records are in its training set, first <span class=\"mono\">d / t0</span> in records_kev.jsonl" in cards[R1]
    assert f"names {r1['run']['trainrun_id']}, and the record here is {r2['run']['trainrun_id']}" in cards[R2]
    # Without the parent's record and with the manifest exported again.
    o = _by_name(orphan)
    assert (o[R2]["run"]["manifest_loaded"], o[R2]["run"]["parent_loaded"], o[R1]["run"]) == (False, False, None)
    ocards = _cards(_render(orphan))
    assert _notes(ocards[R2]) == ["ON TRAINED ITEMS", "MANIFEST DIFFERS", "PARENT NOT HERE", "POLICY NAMES ANOTHER RUN"]
    assert o[R2]["run"]["record"]["manifest"]["sha256"] in ocards[R2]
    assert f"It started from <span class=\"mono\">{R1}</span> ({o[R2]['run']['record']['parent']['trainrun_id']}), whose run record is not in this workspace." in ocards[R2]
    # A policy naming a run with no record here to compare with is no mismatch.
    assert o[R1]["policies"][0]["run_matches"] is None and _notes(ocards[R1]) == ["ON TRAINED ITEMS"]


@needs_node
def test_a_checkpoint_that_answered_nothing_and_one_held_out(payloads) -> None:
    d = copy.deepcopy(payloads["full"])
    c = d["checkpoints"][0]
    c.update(n_records=0, n_trained_hits=0, groups=[])
    assert _notes(_cards(_render(d))[R1]) == ["NO RECORDS"]
    d = copy.deepcopy(payloads["full"])
    c = d["checkpoints"][0]
    c.update(n_trained_hits=0)
    for g in c["groups"]:
        g.update(n_trained_hits=0, first_trained_hit=None)
    card = _cards(_render(d))[R1]
    assert _notes(card) == [] and '<span class="badge badge-ok">HELD OUT</span>' in card and "none on a trained item" in card


# --- step metrics ----------------------------------------------------------------------------------


@needs_node
def test_each_metric_is_a_line_through_its_own_steps(payloads) -> None:
    d = payloads["full"]
    c = _by_name(d)[R1]
    steps = c["run"]["record"]["steps"]
    card = _cards(_render(d))[R1]
    svgs = _svgs(card)
    assert list(svgs) == ["loss", "lr"]
    for metric, svg in svgs.items():
        points = re.search(r'<polyline class="line ser-1" points="([^"]+)"', svg).group(1).split(" ")
        assert len(points) == len(steps) == svg.count('<circle class="dot ser-1"')
        xs = [float(p.split(",")[0]) for p in points]
        assert xs == sorted(xs) and (xs[0], xs[-1]) == (70.0, 440.0)
        # The chart names the steps table of its card, which holds every step.
        target = re.search(r'aria-details="([^"]+)"', svg).group(1)
        assert f'data-table="checkpoints:steps:{R1}" id="{target}"' in card
        entries = json.loads(html_lib.unescape(re.search(r'data-points="([^"]+)"', svg).group(1)))
        assert [tip for _, tip in entries] == [f"step {s['step']}\nloss {_num(s['loss'])}\nlr {_num(s['lr'])}" for s in steps]
    # Heights follow the values: the loss falls, so its line descends on the page (y grows).
    ys = [float(p.split(",")[1]) for p in re.search(r'points="([^"]+)"', svgs["loss"]).group(1).split(" ")]
    assert ys == sorted(ys) and ys[0] == 14.0
    # A constant metric runs through the middle of its plot, not along an edge.
    lr = {float(p.split(",")[1]) for p in re.search(r'points="([^"]+)"', svgs["lr"]).group(1).split(" ")}
    assert lr == {14.0 + 170 / 2}


@needs_node
def test_a_metric_recorded_at_some_steps_draws_only_those(payloads) -> None:
    d = copy.deepcopy(payloads["full"])
    steps = d["checkpoints"][0]["run"]["record"]["steps"]
    steps[1]["eval_loss"] = 0.5
    steps[4]["eval_loss"] = 0.25
    svgs = _svgs(_cards(_render(d))[R1])
    assert list(svgs) == ["loss", "lr", "eval_loss"]
    assert len(re.search(r'points="([^"]+)"', svgs["eval_loss"]).group(1).split(" ")) == 2
    assert "eval_loss at 2 recorded steps, from step 10 to 40." in svgs["eval_loss"]
    rows = _js(f"checkpointStepsTable({json.dumps(d['checkpoints'][0])}).rows")
    assert [r.get("eval_loss") for r in rows] == [None, 0.5, None, None, 0.25]


@needs_node
def test_more_metrics_than_the_limit_are_in_the_table_only(payloads) -> None:
    d = copy.deepcopy(payloads["full"])
    for s in d["checkpoints"][0]["run"]["record"]["steps"]:
        s.update({f"m{i}": float(i + s["step"]) for i in range(7)})
    card = _cards(_render(d))[R1]
    assert len(_svgs(card)) == 6 == _js("stepChartLimit()")
    assert "3 more metrics are in the table only." in card
    spec = _js(f"checkpointStepsTable({json.dumps(d['checkpoints'][0])}).spec")
    assert [c["key"] for c in spec["columns"]] == ["checkpoint", "trainrun_id", "step", "loss", "lr"] + [f"m{i}" for i in range(7)]


@needs_node
def test_a_run_without_steps_says_so_and_a_model_without_a_run_draws_nothing(payloads) -> None:
    d = copy.deepcopy(payloads["full"])
    d["checkpoints"][0]["run"]["record"]["steps"] = []
    cards = _cards(_render(d))
    assert "The run record holds no step metrics" in cards[R1] and _svgs(cards[R1]) == {}
    assert _svgs(cards[BASELINE]) == {} and "no step metrics" not in cards[BASELINE]
    # The two long tables scroll inside the card (97 rows of settings in a real record).
    assert cards[R2].count('<div class="panel-block scroll-table">') == 2 and '<div class="panel-block scroll-table">' not in cards[BASELINE]
    assert f'data-table="checkpoints:steps:{BASELINE}"' not in cards[BASELINE] and f'data-table="checkpoints:run:{BASELINE}"' not in cards[BASELINE]


@needs_node
def test_a_single_step_is_drawn_as_a_point(payloads) -> None:
    d = copy.deepcopy(payloads["full"])
    rec = d["checkpoints"][0]["run"]["record"]
    rec["steps"] = rec["steps"][:1]
    svg = _svgs(_cards(_render(d))[R1])["loss"]
    assert "<polyline" not in svg and svg.count('<circle class="dot ser-1"') == 1 and "NaN" not in svg


# --- tables ----------------------------------------------------------------------------------------


@needs_node
def test_the_tables_hold_the_record_as_served(payloads) -> None:
    d = payloads["full"]
    c = _by_name(d)[R1]
    rec = c["run"]["record"]
    tables = _js(f"Object.fromEntries(Object.entries(checkpointTables({json.dumps(d)})).map(([k, t]) => [k, t.rows]))")
    assert set(tables) == {"checkpoints:all"} | {f"checkpoints:{kind}:{name}" for kind in ("groups", "manifests", "run", "steps", "policies") for name in (R1, R2, BASELINE)}
    run = tables[f"checkpoints:run:{R1}"]
    assert {(r["section"], r["name"]): r["value"] for r in run if r["section"] == "trainer setting"} == {("trainer setting", k): v for k, v in rec["trainer"].items()}
    assert [r["value"] for r in run if r["section"] == "weights file (SHA-256)"] == list(rec["weights_sha256"].values())
    assert {r["name"]: r["value"] for r in run if r["section"] == "manifest recorded"} == rec["manifest"]
    assert {r["name"]: r["value"] for r in run if r["section"] == "data file"} == rec["data"]
    assert all(r["trainrun_id"] == c["run"]["trainrun_id"] for r in run)
    steps = tables[f"checkpoints:steps:{R1}"]
    assert [{k: r[k] for k in ("step", "loss", "lr")} for r in steps] == rec["steps"]
    groups = tables[f"checkpoints:groups:{R1}"]
    assert [(r["file"], r["n_records"], r["n_trained_hits"], r["first_trained_text"]) for r in groups] == [("records.jsonl", 6, 0, ""), ("records_kev.jsonl", 3, 2, "d / t0")]
    assert [(r["first_trained_dataset"], r["first_trained_item_id"]) for r in groups] == [(None, None), ("d", "t0")]
    assert groups[0]["path"] == c["groups"][0]["file"]
    (m,) = tables[f"checkpoints:manifests:{R1}"]
    assert (m["file"], m["sha256"], m["n_rows"]) == ("training_manifest.json", c["manifests"][0]["sha256"], 3)
    policies = {name: tables[f"checkpoints:policies:{name}"][0] for name in (R1, R2, BASELINE)}
    assert [policies[n]["names"] for n in (R1, R2, BASELINE)] == ["the run recorded here", "another run", ""]


@needs_node
def test_a_small_metric_keeps_its_digits_and_a_whole_one_has_none(payloads) -> None:
    c = _by_name(payloads["full"])[R1]
    spec = _js(f"checkpointStepsTable({json.dumps(c)}).spec")
    digits = {col["key"]: col.get("digits") for col in spec["columns"]}
    assert (digits["step"], digits["lr"], digits["loss"]) == (0, 6, 4)
    assert ">0.000200</td>" in _cards(_render(payloads["full"]))[R1]
    assert _js("[metricDigits([1, 2, 3]), metricDigits([0, 0.5]), metricDigits([]), metricDigits([0, 0.0]), metricDigits([1e-12, 1])]") == [0, 3, 0, 0, 8]


@needs_node
def test_a_metric_named_as_an_identifying_column_keeps_its_own_values(payloads) -> None:
    c = copy.deepcopy(_by_name(payloads["full"])[R1])
    for s in c["run"]["record"]["steps"]:
        s["checkpoint"] = 7.5
    t = _js(f"checkpointStepsTable({json.dumps(c)})")
    assert [col["key"] for col in t["spec"]["columns"]] == ["trainrun_id", "step", "loss", "lr", "checkpoint"]
    assert {r["checkpoint"] for r in t["rows"]} == {7.5}
    # Recorded at some steps only: the others have no value there, never the checkpoint's name.
    del c["run"]["record"]["steps"][0]["checkpoint"]
    rows = _js(f"checkpointStepsTable({json.dumps(c)}).rows")
    assert [r.get("checkpoint") for r in rows] == [None, 7.5, 7.5, 7.5, 7.5] and {r["trainrun_id"] for r in rows} == {c["run"]["trainrun_id"]}


@needs_node
def test_small_helpers() -> None:
    assert _js('[baseName("/a/b/c.json"), baseName("c:\\\\x\\\\y.json"), baseName(null), baseName("plain")]') == ["c.json", "y.json", "", "plain"]
    assert _js('[itemText(["d", "t0"]), itemText([null, "t0"]), itemText(null)]') == ["d / t0", "t0", ""]
    assert _js("[niceDomain(0.02, 1), niceDomain(2e-4, 2e-4), niceDomain(-3, -3), niceDomain(0, 0), niceDomain(NaN, 1), niceDomain(3, 97)]") == [
        [0, 1], [0, 4e-4], [-6, 0], [0, 1], [0, 1], [0, 100]]
    assert _js("[extent([3, -1, 2]), extent([5]), extent(Array.from({length: 200000}, (_, i) => i))]") == [[-1, 3], [5, 5], [0, 199999]]


@needs_node
@pytest.mark.parametrize(
    "lo, hi",
    [
        (1234567890126, 1234567890128),  # a span at the thirteenth digit: the rounded ends once passed both values
        (0.3, 0.30000000000000004),  # float jitter around one value
        (1e308, 1e308),  # twice it is not a number
        (-1e308, -1e308),
        (-1e308, 1e308),  # a span that is not a number
        (1e308, 1.7e308),  # its top end rounds up past the largest number
        (0, 1.79769313486e308),  # niceTicks' last tick is Infinity: its loop never ended
        (8.988465674311578e307, 8.988465674311578e307),  # twice it is a number, its rounded end is not
        (-1.5e308, 1e307),  # rounded ends whose span is not a number
        (-5e-324, 5e-324),  # opposite signs whose quarters are both 0, not one value
        (-2.5, 7.25),
        (0.999999999, 1.5),
        (5e-324, 1e-323),
        (-7, -7 + 1e-12),
    ],
)
def test_a_value_axis_holds_its_values_and_has_a_width(lo, hi) -> None:
    d0, d1 = _js(f"niceDomain({lo!r}, {hi!r})")
    assert d0 is not None and d1 is not None  # finite: JSON has no Infinity
    assert d0 <= lo and hi <= d1 and d0 < d1


@needs_node
def test_a_long_or_extreme_series_is_drawn_inside_its_plot(payloads) -> None:
    c = copy.deepcopy(_by_name(payloads["full"])[R1])
    cases = {
        "long": [{"step": i, "loss": 1 / (i + 1)} for i in range(150_000)],
        "narrow": [{"step": i, "loss": 1234567890126 + i} for i in range(3)],
        "jitter": [{"step": 0, "loss": 0.3}, {"step": 1, "loss": 0.30000000000000004}],
        "huge": [{"step": 0, "loss": 1e308}, {"step": 1, "loss": 1e308}],
        "huge span": [{"step": 0, "loss": 1e308}, {"step": 1, "loss": 1.7e308}],
        "near the largest": [{"step": 0, "loss": 1e308}, {"step": 1, "loss": 1.79769313486e308}],
        "span not a number": [{"step": 0, "loss": -1e308}, {"step": 1, "loss": 1e308}],
        "rounded span not a number": [{"step": 0, "loss": -1.5e308}, {"step": 1, "loss": 1e307}],
        "opposite subnormals": [{"step": 0, "loss": -5e-324}, {"step": 1, "loss": 5e-324}],
        "constant near half the largest": [{"step": 0, "loss": 8.5e307}, {"step": 1, "loss": 8.5e307}],
    }
    for name, steps in cases.items():
        svg = _js(f"stepChartHtml(c, 'loss', steps, ['loss'], 'details')", prelude=f"const c = {json.dumps(c)}; const steps = {json.dumps(steps)};")
        points = [p.split(",") for p in re.search(r'<polyline class="line ser-1" points="([^"]+)"', svg).group(1).split(" ")]
        assert len(points) == len(steps), name
        # Inside the plot box: x in [70, 440], y in [14, 184].
        assert all(70 <= float(x) <= 440 and 14 <= float(y) <= 184 for x, y in points), name
        assert "NaN" not in svg and "Infinity" not in svg, name
        # Every y gridline is inside the plot too (a NaN position once drew one at y = 0).
        grid = [float(y) for y in re.findall(r'<line class="grid" x1="70(?:\.0)?" y1="([^"]+)"', svg)]
        assert grid and all(14 <= y <= 184 for y in grid), (name, grid)
        if name in ("huge span", "near the largest", "span not a number", "rounded span not a number", "opposite subnormals"):
            # Two values, two heights: an axis ending on Infinity, or a span that is, drew both at the bottom edge.
            assert len({y for _, y in points}) == 2, (name, points)
        if name == "constant near half the largest":  # a constant runs through the middle, not along the top
            assert {y for _, y in points} == {"99.0"}, points


@needs_node
def test_a_chosen_name_the_picker_does_not_hold_has_its_own_entry(payloads) -> None:
    html = _render(payloads["full"], "tickets-r9")
    assert '<option value="tickets-r9" selected>Not found: tickets-r9</option>' in html and html.count(" selected>") == 1
    # "All" is then another value than the one selected, so choosing it is a change.
    assert '<option value="">All checkpoints (3)</option>' in html
    assert "Not found:" not in _render(payloads["full"], R1) and "Not found:" not in _render(payloads["full"])
    # The question pickers are the same builder.
    drift = _js('questionPickerHtml("drift-question", [{question_id: "q-1"}], "q-9", "All drift questions")')
    assert '<option value="q-9" selected>Not found: q-9</option>' in drift
    hostile = _js(f'pickerHtml("a", "Checkpoint", ["x"], {json.dumps(HOSTILE)}, "All")')
    assert "<img" not in hostile and "Not found: &quot;" in hostile


# --- empty states ------------------------------------------------------------------------------------


@needs_node
def test_a_workspace_without_training_files_names_what_is_read(payloads) -> None:
    demo = _render(payloads["demo"])
    assert "No training manifest or training-run record found." in demo
    assert "<code>training_manifest*.json</code> (written by <code>s1 trainset export</code>) and <code>trainrun*.json</code> (<code>s1 trainset record</code>)" in demo
    assert set(_cards(demo)) == {"jev-1.13.0", "jev-1.14.0-rc1", "kev-1.0"} and "<svg" not in demo
    empty = _render(payloads["empty"])
    assert "No checkpoint in this workspace." in empty and "trainrun*.json" in empty and "<table" not in empty
    assert "No training manifest or training-run record found." not in _render(payloads["full"])


@needs_node
def test_the_summary_names_the_files_read(payloads) -> None:
    html = _render(payloads["full"])
    assert ("3 names · 2 training manifests · 2 training-run records. Read: records.jsonl, records_kev.jsonl, "
            "training_manifest-r2.json, training_manifest.json, trainrun-r2.json, trainrun.json, policy.json.") in html


# --- the shell keeps a response only once it is on screen ---------------------------------------


def _shell_run(steps: str, payload: dict, functions: str | None = None) -> dict:
    js = f"""
let STATE = {{view: "checkpoints"}};
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
STATE = {{...defaultState(), view: "checkpoints"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


RERENDER_STEPS = """
  const cards = () => el("checkpoints-container").innerHTML.split('>Checkpoint: <span class="badge badge-info">').length - 1;
  runView("checkpoints");
  out.url = pending[0].url;
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.first = cards();
  // Choosing a checkpoint re-renders from what is kept (no request), and so does coming back.
  STATE = {...STATE, ck: payload.checkpoints[1].checkpoint};
  runView("checkpoints");
  out.requests = pending.length;
  out.chosen = cards();
  STATE = {...STATE, view: "lint"};
  STATE = {...STATE, view: "checkpoints"};
  runView("checkpoints");
  out.back = cards();
  const t = currentTables()["checkpoints:steps:" + STATE.ck];
  out.steps = t ? t.rows.map(r => r.step) : null;
  out.tables = Object.keys(currentTables()).length;
"""


@needs_node
def test_a_checkpoint_chosen_after_loading_keeps_its_tables(payloads) -> None:
    out = _shell_run(RERENDER_STEPS, payloads["full"])
    assert out == {"url": "/api/checkpoints", "first": 3, "requests": 0, "chosen": 1, "back": 1, "steps": [0, 10, 20, 30, 40], "tables": 16}


FAIL_STEPS = """
  runView("checkpoints");
  pending.shift().reject(Object.assign(new Error("boom"), {status: 400}));
  await tick(); await tick();
  out.failedError = el("checkpoints-container").innerHTML.includes("ERROR</span> boom");
  out.failedKept = "checkpoints" in viewData;
  // A response the renderer throws on (a list of files that is not one): the error card, nothing kept.
  generation += 1;
  loadedSig = {};
  runView("checkpoints");
  pending.shift().resolve({...payload, display: {files: {records: 5}}});
  await tick(); await tick();
  out.error = el("checkpoints-container").innerHTML.includes("ERROR</span>");
  out.kept = "checkpoints" in viewData;
  out.tables = Object.keys(currentTables()).length;
"""


@needs_node
def test_a_failed_or_unrenderable_response_keeps_nothing(payloads) -> None:
    assert _shell_run(FAIL_STEPS, payloads["full"]) == {"failedError": True, "failedKept": False, "error": True, "kept": False, "tables": 0}
    # Kept before rendering, it would stay behind the error card.
    functions = _mutated(
        '      document.getElementById("checkpoints-container").innerHTML = renderCheckpoints(d, ctx.ck ? ctx.ck() : "", ctx.opts ? ctx.opts() : {});\n      if (ctx.keep) ctx.keep(d);\n',
        '      if (ctx.keep) ctx.keep(d);\n      document.getElementById("checkpoints-container").innerHTML = renderCheckpoints(d, ctx.ck ? ctx.ck() : "", ctx.opts ? ctx.opts() : {});\n',
    )
    assert _shell_run(FAIL_STEPS, payloads["full"], functions=functions)["kept"] is True


@needs_node
def test_a_request_a_refresh_outdated_keeps_and_renders_nothing(payloads) -> None:
    steps = """
  runView("checkpoints");
  generation += 1;
  loadedSig = {};
  viewData = {};
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.kept = "checkpoints" in viewData;
  out.rendered = el("checkpoints-container").innerHTML.includes("Checkpoint:");
"""
    assert _shell_run(steps, payloads["full"]) == {"kept": False, "rendered": False}


@needs_node
def test_the_table_sorts_filters_and_exports_what_it_shows(payloads) -> None:
    d = payloads["full"]
    out = _js(
        "(() => { const t = checkpointsTable(d); const rows = viewRows(t.spec, t.rows, {sort: 'n_records', dir: 'asc', q: 'tickets'});"
        " return {names: rows.map(r => r.checkpoint), csv: toCsv(t.spec, rows).split('\\r\\n')[0], json: JSON.parse(toJsonRows(t.spec, rows))[0]}; })()",
        prelude=f"const d = {json.dumps(d)};",
    )
    assert out["names"] == [R1, R2]
    # The status is the page's wording of the counts: shown, sorted and filtered, never exported.
    assert out["csv"] == "checkpoint,n_records,n_trained_hits,n_trained,manifests,trainrun_id,parent,steps,policies"
    assert out["json"]["trainrun_id"] == _by_name(d)[R1]["run"]["trainrun_id"] and "show" not in out["json"] and "status" not in out["json"]
    held = _js("(() => { const t = checkpointsTable(d); return filterRows(t.spec, t.rows, 'unknown').map(r => r.checkpoint); })()", prelude=f"const d = {json.dumps(d)};")
    assert held == [BASELINE]


@needs_node
def test_exports_hold_served_values_not_the_pages_words(payloads) -> None:
    c = _by_name(payloads["full"])[R1]
    out = _js(
        "(() => { const g = checkpointGroupsTable(c), p = checkpointPoliciesTable(c);"
        " return {groups: JSON.parse(toJsonRows(g.spec, g.rows)), groupsCsv: toCsv(g.spec, g.rows).split('\\r\\n')[0],"
        " policies: JSON.parse(toJsonRows(p.spec, p.rows)), shown: shownColumns(g.spec).map(x => x.key)}; })()",
        prelude=f"const c = {json.dumps(c)};",
    )
    assert out["groupsCsv"] == "checkpoint,file,path,question_id,primitive,n_records,n_items,n_trained_hits,first_trained_dataset,first_trained_item_id"
    assert [(g["first_trained_dataset"], g["first_trained_item_id"]) for g in out["groups"]] == [(None, None), ("d", "t0")]
    assert "first_trained_text" in out["shown"] and "first_trained_dataset" not in out["shown"]
    assert out["policies"] == [{"checkpoint": R1, "question_id": c["policies"][0]["question_id"], "primitive": "noul",
                                "trainrun_id": c["run"]["trainrun_id"], "run_matches": True}]


def test_the_view_has_its_tab_and_pane() -> None:
    html = (Path(__file__).parent.parent / "src" / "s1ops" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="nav-checkpoints" data-view="checkpoints" aria-controls="tab-checkpoints"' in html
    assert '<section id="tab-checkpoints" class="tab-pane" role="tabpanel" aria-labelledby="nav-checkpoints">' in html
    assert html.count('role="tab"') == 9
