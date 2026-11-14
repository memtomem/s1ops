"""`s1 slice` and `/api/slices` report each `(question_id, model_served, primitive)` group (#38).

0.5.1 computed one set of slice statistics over every labeled record selected, so a file with two
questions or two models got one number for populations that behave differently. Each group's
report must equal what 0.5.1 printed for that group alone (narrowed with `--question` and
`--model`), which `_old_report` reproduces from 0.5.1's loop, except that a non-finite
`odds_ratio` is now `null` where 0.5.1 printed `Infinity` or `NaN`. 0.5.1 predicted from each
record's `calib` when it had one; since #130 that is `--use-calib` (checked), and the default
predicts from the raw distribution, so `_old_report` takes the distribution to compare with.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records, select
from s1ops.demo import generate_workspace
from s1ops.improve.slices import benjamini_hochberg, mine_error_slices, parse_feature_rows, slice_report
from s1ops.web.app import create_app

STATIC = Path(__file__).resolve().parents[1] / "src" / "s1ops" / "web" / "static" / "index.html"


def _old_report(
    records_path: Path, features_path: Path | None, cmd: str, question=None, model=None, *, use_calib: bool = False
) -> dict:
    """0.5.1's `_cmd_slice`, with its CLI defaults (`--min-n 10`, `--min-support 5`, `--fdr-q 0.10`),
    over the raw distribution, or with `use_calib` over the calibrated one as 0.5.1 read it."""
    records = read_records(records_path, use_calib=use_calib)
    features_by_item, slice_by_item = {}, {}
    if features_path is not None:
        rows = [json.loads(line) for line in features_path.read_text().splitlines() if line.strip()]
        features_by_item, slice_by_item = parse_feature_rows(rows)
    filtered = [
        r for r in records
        if r.model_served is not None and r.label is not None
        and (question is None or r.question_id == question) and (model is None or r.model_served == model)
    ]
    if cmd == "answerable":
        items = []
        for r in filtered:
            pred = r.classes[int(max(range(len(r.probs)), key=lambda i: r.probs[i]))]
            sl = slice_by_item.get(r.item) or slice_by_item.get((None, r.item_id)) or "default"
            items.append({"slice": sl, "predicted": pred, "label": r.label})
        return _json({"slices": [s.as_dict() for s in slice_report(items, min_n=10)]})
    rows = []
    for r in filtered:
        pred = r.classes[int(max(range(len(r.probs)), key=lambda i: r.probs[i]))]
        feats = features_by_item.get(r.item) or features_by_item.get((None, r.item_id)) or {}
        rows.append({"error": pred != r.label, "features": feats})
    return _json({"error_slices": [_finite(asdict(e)) for e in mine_error_slices(rows, min_support=5, fdr_q=0.10)]})


def _finite(finding: dict) -> dict:
    """0.5.1 serialized `asdict(finding)`, printing a non-finite odds ratio as `Infinity` or `NaN`;
    the one intended change is that it is now `null` (`test_an_infinite_odds_ratio_is_null`,
    `test_a_nan_odds_ratio_is_null`)."""
    if not math.isfinite(finding["odds_ratio"]):
        finding["odds_ratio"] = None
    return finding


def _json(report: dict) -> dict:
    """As the CLI prints it (tuples become lists); floats round-trip exactly."""
    return _strict(json.dumps(report))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _cli(capsys, *args) -> tuple[int, str, str]:
    code = cli_main(["slice", *map(str, args)])
    out, err = capsys.readouterr()
    return code, out, err


def _strict(text: str) -> dict:
    def refuse(name):
        raise AssertionError(f"not valid JSON: {name}")
    return json.loads(text, parse_constant=refuse)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> dict:
    """The demo workspace plus features for the categorical items (the demo tags only the noul
    ones), chosen so both groups mine non-empty and different error slices."""
    root = tmp_path_factory.mktemp("demo")
    ws = generate_workspace(root / "ws")
    cat = Path(ws["records_categorical"])
    feats = []
    for i, r in enumerate(read_records(cat, use_calib=True)):
        pred = r.classes[max(range(len(r.probs)), key=lambda k: r.probs[k])]
        hard = (pred != r.label or i % 7 == 0) if r.primitive == "choice" else i % 3 == 0
        feats.append({"dataset": r.dataset, "item_id": r.item_id, "slice": f"s{i % 2}",
                      "features": {"hard": hard, "tier": "a" if i % 4 else "b"}})
    cat_features = _write(root / "cat_features.jsonl", feats)
    return {"ws": ws, "records": Path(ws["records"]), "features": Path(ws["features"]),
            "categorical": cat, "cat_features": cat_features}


def _groups(path: Path) -> list[tuple[str, str, str]]:
    return [g.key for g in select(read_records(path), allow_model_mismatch=True).groups]


@pytest.mark.parametrize("cmd, key", [("answerable", "slices"), ("mine", "error_slices")])
@pytest.mark.parametrize("which", ["noul", "categorical"])
@pytest.mark.parametrize("use_calib", [False, True])
def test_each_group_equals_the_old_report_narrowed_to_it(demo, capsys, cmd: str, key: str, which: str, use_calib: bool) -> None:
    path, feats = (demo["records"], demo["features"]) if which == "noul" else (demo["categorical"], demo["cat_features"])
    code, out, err = _cli(capsys, cmd, path, "--features", feats, *(["--use-calib"] if use_calib else []))
    assert (code, err) == (0, "")
    report = _strict(out)
    assert report["distribution"] == ("calibrated" if use_calib else "raw")
    got = report["groups"]
    keys = _groups(path)
    assert [(g["question_id"], g["model_served"], g["primitive"]) for g in got] == keys
    for g, (q, m, p) in zip(got, keys):
        old = _old_report(path, feats, cmd, question=q, model=m, use_calib=use_calib)
        assert g == {"question_id": q, "model_served": m, "primitive": p, "n": g["n"], key: old[key]}
        assert g["n"] == sum(1 for r in read_records(path) if r.key == (q, m, p) and r.label is not None)
        assert g[key], "an empty report would make this comparison vacuous"


def test_categorical_groups_differ_and_are_not_pooled(demo, capsys) -> None:
    code, out, _ = _cli(capsys, "mine", demo["categorical"], "--features", demo["cat_features"])
    score, choice = _strict(out)["groups"]  # sorted by key: the score question's id sorts first
    assert (choice["primitive"], score["primitive"], choice["n"], score["n"]) == ("choice", "score", 150, 150)
    hard = {g["primitive"]: next(e for e in g["error_slices"] if e["feature"] == "hard") for g in (choice, score)}
    assert hard["choice"]["significant"] and not hard["score"]["significant"]
    assert hard["choice"]["rest_n"] + hard["choice"]["slice_n"] == 150
    feats = {(f["dataset"], f["item_id"]): f["features"] for f in _rows(demo["cat_features"])}
    for g in (choice, score):
        records = [r for r in read_records(demo["categorical"], use_calib=True) if r.primitive == g["primitive"]]
        wrong = [r.classes[max(range(len(r.probs)), key=lambda k: r.probs[k])] != r.label for r in records]
        tagged = [bool(feats[r.item]["hard"]) for r in records]
        e = hard[g["primitive"]]
        assert (e["slice_n"], e["slice_errors"]) == (sum(tagged), sum(w for w, h in zip(wrong, tagged) if h))
        assert (e["rest_n"], e["rest_errors"]) == (len(records) - sum(tagged), sum(w for w, h in zip(wrong, tagged) if not h))
        # The BH family is the group's own features, not every group's.
        assert [x["q_value"] for x in g["error_slices"]] == benjamini_hochberg([x["p_value"] for x in g["error_slices"]])
    pooled = next(e for e in _old_report(demo["categorical"], demo["cat_features"], "mine")["error_slices"] if e["feature"] == "hard")
    assert pooled["slice_n"] + pooled["rest_n"] == 300

    code, out, _ = _cli(capsys, "answerable", demo["categorical"], "--features", demo["cat_features"])
    assert [sum(s["n"] for s in g["slices"]) for g in _strict(out)["groups"]] == [150, 150]


def test_two_models_mirrored_report_complementary_accuracy(demo, tmp_path: Path, capsys) -> None:
    """The issue's file: the demo's noul records plus the same items from a second model whose
    probabilities are mirrored, so it is wrong exactly where the first is right."""
    rows = _rows(demo["records"])
    mirrored = []
    for r in rows:
        m = {**r, "model_requested": "other-model", "model_served": "other-model", "p": 1 - r["p"]}
        m["calib"] = {**r["calib"], "p": 1 - r["calib"]["p"]}
        mirrored.append(m)
    path = _write(tmp_path / "mixed.jsonl", rows + mirrored)
    code, out, err = _cli(capsys, "answerable", path, "--features", demo["features"])
    assert (code, err) == (0, "")
    a, b = _strict(out)["groups"]
    assert {a["model_served"], b["model_served"]} == {"jev-1.13.0", "other-model"}
    for sa, sb in zip(a["slices"], b["slices"]):
        assert (sa["slice"], sa["n"]) == (sb["slice"], sb["n"])
        assert sa["accuracy"] + sb["accuracy"] == pytest.approx(1.0, abs=1e-12)
    pooled = {s["slice"]: s for s in _old_report(path, demo["features"], "answerable")["slices"]}
    assert all(pooled[s["slice"]]["n"] == 2 * s["n"] for s in a["slices"])
    for g in (a, b):
        assert g["slices"] == _old_report(path, demo["features"], "answerable", model=g["model_served"])["slices"]
    code, out, _ = _cli(capsys, "mine", path, "--features", demo["features"])
    mined = {g["model_served"]: g["error_slices"] for g in _strict(out)["groups"]}
    web = {g["model_served"]: g for g in TestClient(create_app(demo["records"].parent, records_path=path)).get("/api/slices").json()["groups"]}
    for model in ("jev-1.13.0", "other-model"):
        old_mine = _old_report(path, demo["features"], "mine", model=model)["error_slices"]
        assert old_mine and mined[model] == old_mine
        assert web[model]["error_slices"] == old_mine
        assert web[model]["answerability"] == _old_report(path, demo["features"], "answerable", model=model)["slices"]

    code, out, _ = _cli(capsys, "answerable", path, "--features", demo["features"], "--model", "other-model")
    (only,) = _strict(out)["groups"]
    assert (code, only) == (0, b if b["model_served"] == "other-model" else a)
    client = TestClient(create_app(demo["records"].parent, records_path=path))
    (web,) = client.get("/api/slices", params={"model_served": "other-model"}).json()["groups"]
    assert (web["model_served"], web["answerability"]) == ("other-model", only["slices"])


def _choice_row(template: dict, item_id: str, probs: dict, label: str) -> dict:
    row = {k: v for k, v in template.items() if k != "calib"}
    return {**row, "item_id": item_id, "probabilities": probs, "choice": next(iter(probs)), "label": label}


def test_each_record_breaks_a_tie_in_its_own_key_order(demo, tmp_path: Path, capsys) -> None:
    template = next(r for r in _rows(demo["categorical"]) if r["primitive"] == "choice")
    rows = [
        _choice_row(template, "t1", {"account": 0.5, "billing": 0.5, "technical": 0.0}, "account"),
        _choice_row(template, "t2", {"billing": 0.5, "account": 0.5, "technical": 0.0}, "account"),
    ]
    path = _write(tmp_path / "tie.jsonl", rows)
    code, out, _ = _cli(capsys, "answerable", path)
    (group,) = _strict(out)["groups"]
    (s,) = group["slices"]
    assert (code, s["n"], s["accuracy"]) == (0, 2, 0.5)
    assert group["slices"] == _old_report(path, None, "answerable")["slices"]


def test_the_prediction_comes_from_the_distribution_asked_for(demo, tmp_path: Path, capsys) -> None:
    """Raw `p` 0.9 answers true, `calib.p` 0.1 answers false; the label is false. Up to 0.10.0 the
    calibrated answer was always the prediction (and a record without `calib` fell back to raw)."""
    row = _rows(demo["records"])[0]
    row = {**row, "p": 0.9, "label": False, "calib": {**row["calib"], "p": 0.1}}
    path = _write(tmp_path / "cal.jsonl", [row])
    for flags, distribution, accuracy in (((), "raw", 0.0), (("--use-calib",), "calibrated", 1.0)):
        code, out, err = _cli(capsys, "answerable", path, *flags)
        report = _strict(out)
        (group,) = report["groups"]
        assert (code, err, report["distribution"], group["slices"][0]["accuracy"]) == (0, "", distribution, accuracy)
    client = _client(demo, path)
    for params, distribution, accuracy in (({}, "raw", 0.0), ({"use_calib": True}, "calibrated", 1.0)):
        r = client.get("/api/slices", params=params)
        (group,) = r.json()["groups"]
        assert (r.status_code, r.json()["distribution"], group["answerability"][0]["accuracy"]) == (200, distribution, accuracy)


def _no_groups_line(path: Path, filters: str) -> str:
    sel = select(read_records(path), allow_model_mismatch=True)
    return (f"s1: no labeled groups matching {filters} in {path} ({len(sel.groups)} group(s) before filters; "
            f"skipped {sel.no_response} without a response, {sel.unlabeled} unlabeled, {sel.other_split} in other splits)\n")


@pytest.mark.parametrize("cmd", ["answerable", "mine"])
def test_no_matching_group_exits_1(demo, capsys, cmd: str) -> None:
    code, out, err = _cli(capsys, cmd, demo["records"], "--features", demo["features"], "--question", "q-nope")
    assert (code, out, err) == (1, "", _no_groups_line(demo["records"], "--question q-nope"))


def _client(demo, records: Path) -> TestClient:
    return TestClient(create_app(demo["records"].parent, records_path=records, features_path=demo["cat_features"]))


def test_web_reports_each_group_as_the_cli(demo) -> None:
    client = _client(demo, demo["categorical"])
    groups = client.get("/api/slices").json()["groups"]
    for g in groups:
        q, m = g["question_id"], g["model_served"]
        assert g["answerability"] == _old_report(demo["categorical"], demo["cat_features"], "answerable", q, m)["slices"]
        assert g["error_slices"] == _old_report(demo["categorical"], demo["cat_features"], "mine", q, m)["error_slices"]
    choice = next(g for g in groups if g["primitive"] == "choice")
    narrowed = client.get("/api/slices", params={"question_id": choice["question_id"]}).json()["groups"]
    assert narrowed == [choice]
    assert client.get("/api/slices", params={"question_id": "q-nope"}).json() == {"distribution": "raw", "groups": []}


def _duplicate_score_row(demo, tmp_path: Path, **change) -> tuple[Path, str, str]:
    rows = _rows(demo["categorical"])
    i = next(k for k, r in enumerate(rows) if r["primitive"] == "score")
    path = _write(tmp_path / "dup.jsonl", rows + [{**rows[i], **change}])
    key = (rows[i]["question_id"], rows[i]["model_served"], "score")
    choice_q = next(r["question_id"] for r in rows if r["primitive"] == "choice")
    message = (f"{path}: line {len(rows) + 1}: item ('{rows[i]['dataset']}', '{rows[i]['item_id']}') (dataset, item_id) "
               f"appears twice in group {key} (first on line {i + 1})")
    return path, choice_q, message


@pytest.mark.parametrize("cmd", ["answerable", "mine"])
@pytest.mark.parametrize("change", [{}, {"split": "other"}], ids=["same-split", "other-split"])
def test_a_duplicate_in_an_excluded_group_is_refused(demo, tmp_path: Path, capsys, cmd: str, change: dict) -> None:
    path, choice_q, message = _duplicate_score_row(demo, tmp_path, **change)
    code, out, err = _cli(capsys, cmd, path, "--features", demo["cat_features"], "--question", choice_q)
    assert (code, out, err) == (1, "", f"s1: {message}\n")
    r = _client(demo, path).get("/api/slices", params={"question_id": choice_q})
    assert (r.status_code, r.json()["detail"]) == (400, message)


@pytest.mark.parametrize("cmd", ["answerable", "mine"])
def test_differing_classes_in_an_excluded_group_are_refused(demo, tmp_path: Path, capsys, cmd: str) -> None:
    rows = _rows(demo["categorical"])
    i = next(k for k, r in enumerate(rows) if r["primitive"] == "score")
    extra = {**rows[i], "item_id": "extra-level", "probabilities": {**rows[i]["probabilities"], "9": 0.0}}
    extra.pop("calib")
    path = _write(tmp_path / "cls.jsonl", rows + [extra])
    choice_q = next(r["question_id"] for r in rows if r["primitive"] == "choice")
    code, out, err = _cli(capsys, cmd, path, "--features", demo["cat_features"], "--question", choice_q)
    first = next(k for k, r in enumerate(rows) if r["primitive"] == "score")
    levels = sorted(rows[i]["probabilities"])
    message = (f"{path}: line {len(rows) + 1}: {rows[i]['question_id']} ({rows[i]['model_served']}): answer classes "
               f"{sorted(levels + ['9'])} differ from {levels} (first on line {first + 1}) for the same question; "
               "the question_id cannot be trusted")
    assert (code, out, err) == (1, "", f"s1: {message}\n")
    r = _client(demo, path).get("/api/slices", params={"question_id": choice_q})
    assert (r.status_code, r.json()["detail"]) == (400, message)


def test_an_infinite_odds_ratio_is_null(demo, tmp_path: Path, capsys) -> None:
    """A feature carried by every error and nothing else separates perfectly: Fisher's odds ratio is
    infinite, which 0.5.1 printed as `Infinity` (not JSON) and the web turned into a `null` the
    UI could not format."""
    rows = read_records(demo["records"], use_calib=True)
    feats = [{"dataset": r.dataset, "item_id": r.item_id,
              "features": {"bad": r.classes[max(range(2), key=lambda k: r.probs[k])] != r.label}} for r in rows]
    feature_path = _write(tmp_path / "f.jsonl", feats)
    code, out, _ = _cli(capsys, "mine", demo["records"], "--features", feature_path)
    (group,) = _strict(out)["groups"]
    bad = next(e for e in group["error_slices"] if e["feature"] == "bad")
    assert (code, bad["odds_ratio"], bad["rest_errors"]) == (0, None, 0)
    client = TestClient(create_app(demo["records"].parent, features_path=feature_path))
    (web,) = client.get("/api/slices").json()["groups"]
    assert next(e for e in web["error_slices"] if e["feature"] == "bad")["odds_ratio"] is None


def test_a_nan_odds_ratio_is_null() -> None:
    """No error anywhere in the group: Fisher's table `[[0, 5], [0, 1]]` has a `nan` odds ratio,
    which 0.5.1 printed as `NaN`."""
    rows = [{"error": False, "features": {"f": i < 5}} for i in range(6)]
    (finding,) = mine_error_slices(rows, min_support=5)
    assert math.isnan(finding.odds_ratio)
    assert finding.as_dict()["odds_ratio"] is None


def test_the_ui_formats_odds_ratio_only_when_it_is_a_number() -> None:
    html = STATIC.read_text(encoding="utf-8")
    uses = re.findall(r"\$\{[^}]*odds_ratio[^}]*\}", html)
    assert uses == ["${e.odds_ratio !== null ? e.odds_ratio.toFixed(2) : 'N/A'}"]
