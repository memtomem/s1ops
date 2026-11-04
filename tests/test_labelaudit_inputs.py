"""`labelaudit`: every fit method ranks, and a bad fit, states file or key is one error line, not a traceback."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import load_fits, transform
from s1ops.improve.labelaudit import (
    LabelAuditInputError,
    build_sheet,
    check_key,
    disagreement,
    rank_with_skips,
)
from s1ops.web.app import create_app


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _error_line(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("s1: ") and err.count("\n") == 1
    assert "Traceback" not in err
    return err


def _fit(ws: dict[str, str], tmp_path: Path, method: str) -> str:
    path = str(tmp_path / f"fits_{method}.json")
    assert cli_main(["fit", ws["records_train"], "-o", path, "--method", method]) == 0
    return path


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_non_temperature_fit_ranks_through_transform(ws, tmp_path: Path, method: str) -> None:
    fits = load_fits(_fit(ws, tmp_path, method))
    (fit,) = fits
    records = [r for r in read_records(ws["records"]) if r.model_served is not None and r.label is not None]
    for r in records[:40]:
        flag = disagreement(r.raw, fits)
        P = np.array([[r.probs[r.classes.index(c)] for c in fit.classes]])
        q = transform(fit.method, fit.primitive, fit.params, P)[0]
        probs = {c: float(q[i]) for i, c in enumerate(fit.classes)}
        answer = r.raw["p"] >= 0.5
        expected = 0.0 if answer == r.label else max(0.0, probs[answer] - probs[r.label])
        assert flag.score == expected


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_cli_rank_accepts_non_temperature_fits(ws, tmp_path: Path, capsys, method: str) -> None:
    fits = _fit(ws, tmp_path, method)
    capsys.readouterr()
    assert cli_main(["labelaudit", "rank", ws["records"], "--fit", fits, "--top", "3"]) == 0
    assert len(json.loads(capsys.readouterr().out)["top"]) == 3


def _other_model_records(ws: dict[str, str], tmp_path: Path) -> str:
    rows = _rows(ws["records"])
    for i, row in enumerate(rows):
        if i % 2:
            row["model_served"] = "other-model-9"
    return _write_rows(tmp_path / "records_other.jsonl", rows)


def test_fit_for_another_model_is_refused(ws, tmp_path: Path, capsys) -> None:
    records = _other_model_records(ws, tmp_path)
    with pytest.raises(LabelAuditInputError, match="other-model-9"):
        rank_with_skips(_rows(records), load_fits(ws["fits"]))
    assert cli_main(["labelaudit", "rank", records, "--fit", ws["fits"]]) == 1
    assert "fit is for model_served" in _error_line(capsys)


def test_temperature_mapping_without_the_model_is_refused(ws) -> None:
    with pytest.raises(LabelAuditInputError, match="no temperature fitted"):
        rank_with_skips(_rows(ws["records"]), {("q-0000000000000000", "jev-1.13.0"): 1.5})


def test_fit_for_another_primitive_is_not_used(ws) -> None:
    # The categorical fits share no group key with the noul records.
    with pytest.raises(LabelAuditInputError, match="no fit for question_id"):
        rank_with_skips(_rows(ws["records"]), load_fits(ws["fits_categorical"]))


def test_web_labelaudit_ranks_platt_and_refuses_another_model(ws, tmp_path: Path) -> None:
    workspace = Path(ws["records"]).parent
    platt = _fit(ws, tmp_path, "platt")
    Path(ws["fits"]).write_text(Path(platt).read_text())
    client = TestClient(create_app(workspace_dir=workspace))
    assert client.get("/api/labelaudit?top_k=5").status_code == 200

    Path(ws["records"]).write_text(Path(_other_model_records(ws, tmp_path)).read_text())
    r = client.get("/api/labelaudit?top_k=5")
    assert r.status_code == 400
    assert "records.jsonl vs fits.json" in r.json()["detail"]


def _sheet_args(ws: dict[str, str], tmp_path: Path, records: str, states: str) -> list[str]:
    return [
        "labelaudit", "sheet", records, "--states", states, "--fit", ws["fits"],
        "-o", str(tmp_path / "sheet.md"), "-k", str(tmp_path / "key.json"), "--top", "3", "--controls", "3",
    ]


def test_sheet_with_a_missing_state_is_one_line(ws, tmp_path: Path, capsys) -> None:
    short = _write_rows(tmp_path / "states_short.jsonl", _rows(ws["states_local"])[:3])
    assert cli_main(_sheet_args(ws, tmp_path, ws["records"], short)) == 1
    err = _error_line(capsys)
    assert "no state for" in err and "states file" in err


def test_build_sheet_names_the_missing_item_never_a_state(ws) -> None:
    flags, _ = rank_with_skips(_rows(ws["records"]), load_fits(ws["fits"]))
    text_of = {(f.dataset, f.item_id): "SECRET STATE" for f in flags[1:]}
    with pytest.raises(ValueError, match="no state for") as e:
        build_sheet(read_records(ws["records"]), text_of, 3, 3, fit=load_fits(ws["fits"]))
    assert "SECRET" not in str(e.value)


def test_states_rows_are_checked_like_probe_inputs(ws, tmp_path: Path, capsys) -> None:
    rows = _rows(ws["states_local"])
    rows[1]["item_id"] = True
    bad = _write_rows(tmp_path / "states_bad.jsonl", rows)
    assert cli_main(_sheet_args(ws, tmp_path, ws["records"], bad)) == 1
    err = _error_line(capsys)
    assert "states_bad.jsonl: row 2: 'item_id'" in err


def test_states_without_dataset_match_records_probe_wrote_as_default(ws, tmp_path: Path, capsys) -> None:
    # `s1 probe` writes `dataset: "default"` for a state row without one; the sheet must find it.
    records = _rows(ws["records"])
    for row in records:
        row["dataset"] = "default"
    states = [{k: v for k, v in row.items() if k != "dataset"} for row in _rows(ws["states_local"])]
    args = _sheet_args(
        ws, tmp_path, _write_rows(tmp_path / "records_default.jsonl", records), _write_rows(tmp_path / "states_nods.jsonl", states)
    )
    # The fit was recorded for the original dataset; ranking only needs the group key.
    assert cli_main(args) == 0
    assert json.loads(capsys.readouterr().out)["n_items"] == 6


def test_explicit_null_dataset_matches_records_without_dataset(ws, tmp_path: Path, capsys) -> None:
    records = [{k: v for k, v in row.items() if k != "dataset"} for row in _rows(ws["records"])]
    states = [{**row, "dataset": None} for row in _rows(ws["states_local"])]
    args = _sheet_args(
        ws, tmp_path, _write_rows(tmp_path / "records_nods.jsonl", records), _write_rows(tmp_path / "states_null.jsonl", states)
    )
    assert cli_main(args) == 0
    capsys.readouterr()


_ITEM = {"n": 1, "arm": "flag", "label": "True", "predicted": "False"}


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ({}, "options: Field required"),
        ([1, 2], "must be a JSON object, got list"),
        ({"options": [], "items": [_ITEM]}, "options: List should have at least 1 item"),
        ({"options": ["True"], "items": [{**_ITEM, "arm": "both"}]}, "items.0.arm"),
        ({"options": ["True"], "items": [_ITEM]}, "no control item"),
        ({"options": ["True"], "items": [_ITEM, {**_ITEM, "arm": "control"}]}, "items.1.n: item 1 appears more than once"),
    ],
)
def test_check_key_refuses_what_score_sheet_cannot_score(key, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_key(key)


def test_estimate_with_a_one_arm_key_is_one_line(ws, tmp_path: Path, capsys) -> None:
    assert cli_main(_sheet_args(ws, tmp_path, ws["records"], ws["states_local"])) == 0
    capsys.readouterr()
    key = json.loads((tmp_path / "key.json").read_text())
    key["items"] = [item for item in key["items"] if item["arm"] == "flag"]
    (tmp_path / "key_flag.json").write_text(json.dumps(key))
    assert cli_main(["labelaudit", "estimate", str(tmp_path / "sheet.md"), "--key", str(tmp_path / "key_flag.json")]) == 1
    assert "key_flag.json: items: no control item" in _error_line(capsys)


def test_temperature_fit_keeps_labelaudit_arithmetic(ws) -> None:
    # A temperature fit ranks with `labelaudit.calibrate`, as before fits of other methods were
    # accepted: at T == 1 that is the identity, so an endpoint p scores exactly 1.0, where
    # `calibrate.transform` would clip it to 1 - EPS.
    (fit,) = load_fits(ws["fits"])
    unit = [replace(fit, params={"temperature": 1.0})]
    row = {**_rows(ws["records"])[0], "p": 1.0, "label": False}
    assert disagreement(row, unit).score == disagreement(row, None).score == 1.0


def _empty_fits(ws: dict[str, str]) -> None:
    doc = json.loads(Path(ws["fits"]).read_text())
    Path(ws["fits"]).write_text(json.dumps({**doc, "fits": []}))


def test_an_empty_fits_file_refuses_rather_than_ranking_raw(ws, capsys) -> None:
    _empty_fits(ws)
    assert load_fits(ws["fits"]) == []
    assert cli_main(["labelaudit", "rank", ws["records"], "--fit", ws["fits"]]) == 1
    assert "no fit for question_id" in _error_line(capsys)
    r = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get("/api/labelaudit")
    assert r.status_code == 400
    assert "no fit for question_id" in r.json()["detail"]


def test_without_a_fits_file_the_web_ranks_raw(ws) -> None:
    Path(ws["fits"]).unlink()
    r = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get("/api/labelaudit")
    assert r.status_code == 200
    assert r.json()["top_flags"]


def test_same_question_and_model_but_another_primitive_is_refused(ws) -> None:
    (fit,) = load_fits(ws["fits"])
    other = [replace(fit, primitive="choice")]
    with pytest.raises(LabelAuditInputError) as e:
        rank_with_skips(_rows(ws["records"]), other)
    assert f"fit for {fit.question_id} is for primitive choice, not noul" in str(e.value)


def test_same_group_with_other_classes_is_refused(ws) -> None:
    (fit,) = load_fits(ws["fits_categorical"])[:1]
    other = [replace(fit, classes=[*fit.classes[:-1], "zzz"])]
    rows = [r for r in _rows(ws["records_categorical"]) if r["question_id"] == fit.question_id]
    with pytest.raises(LabelAuditInputError, match="answer classes .* differ from the fit's"):
        rank_with_skips(rows, other)


def test_choice_scores_do_not_depend_on_option_order(ws) -> None:
    fits = load_fits(ws["fits_categorical"])
    rows = [r for r in _rows(ws["records_categorical"]) if r["primitive"] == "choice" and r.get("label") is not None]
    reordered = [{**r, "probabilities": dict(reversed(list(r["probabilities"].items())))} for r in rows]
    assert [disagreement(r, fits).score for r in rows] == [disagreement(r, fits).score for r in reordered]


def test_a_non_object_key_item_names_the_field_not_a_class() -> None:
    with pytest.raises(ValueError) as e:
        check_key({"options": ["True"], "items": [1]})
    assert str(e.value) == "items.0: must be an object"
