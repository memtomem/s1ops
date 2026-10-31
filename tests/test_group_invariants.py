"""`compare` and `drift` never pool a question's answers across groups they cannot report as one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from s1ops.eval.compare import CompareInputError, compare_records
from s1ops.monitor.drift import DriftInputError, audit_drift
from s1ops.web.app import create_app


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path: str | Path, rows: list[dict]) -> str:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _error(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("s1: ") and err.count("\n") == 1
    assert "Traceback" not in err
    return err.rstrip("\n")


def _two_models(path: str) -> list[dict]:
    rows = _rows(path)
    for i, row in enumerate(rows):
        if i % 2:
            row["model_served"] = "other-model-9"
    return rows


def _qid(ws: dict[str, str]) -> str:
    return _rows(ws["records"])[0]["question_id"]


@pytest.mark.parametrize("side", ["primary", "shadow"])
def test_compare_refuses_two_models_for_one_question(ws, tmp_path: Path, capsys, side: str) -> None:
    mixed = _write(tmp_path / "mixed.jsonl", _two_models(ws["records"] if side == "primary" else ws["records_kev"]))
    primary, shadow = (mixed, ws["records_kev"]) if side == "primary" else (ws["records"], mixed)
    model = _rows(ws["records"] if side == "primary" else ws["records_kev"])[0]["model_served"]
    expected = (
        f"{side} records hold more than one model_served for question {_qid(ws)}: "
        f"{sorted([model, 'other-model-9'])}; compare one model per file (split it by model_served)"
    )
    with pytest.raises(CompareInputError) as e:
        compare_records(primary, shadow)
    assert str(e.value) == expected
    assert cli_main(["compare", primary, shadow]) == 1
    assert _error(capsys) == f"s1: {expected}"


def test_web_compare_refuses_two_models_for_one_question(ws) -> None:
    _write(ws["records"], _two_models(ws["records"]))
    r = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get("/api/compare")
    assert r.status_code == 400
    assert r.json()["detail"].startswith("records.jsonl vs records_kev.jsonl: primary records hold more than one model_served")


def _choice_under(rows: list[dict], categorical: list[dict], at: int) -> list[dict]:
    """A choice answer carrying a noul question's id and model: only a hand edit makes one."""
    choice = next(r for r in categorical if r["primitive"] == "choice")
    choice = {k: v for k, v in choice.items() if k != "calib"}
    choice.update(question_id=rows[0]["question_id"], model_served=rows[0]["model_served"], item_id="zz-1")
    return [*rows[:at], choice, *rows[at:]]


def test_compare_refuses_two_primitives_for_one_question(ws) -> None:
    # The pair check already refuses a primitive that differs across sides; within one side,
    # a question's rows must agree too.
    categorical = _rows(ws["records_categorical"])
    primary = _choice_under(_rows(ws["records"]), categorical, 0)
    shadow = _choice_under(_rows(ws["records_kev"]), categorical, 0)
    shadow[0]["model_served"] = shadow[1]["model_served"]
    with pytest.raises(CompareInputError, match="primary records answer question .* as more than one primitive"):
        compare_records(primary, shadow)


@pytest.mark.parametrize("side", ["reference", "current"])
def test_drift_refuses_two_primitives_for_one_question(ws, tmp_path: Path, capsys, side: str) -> None:
    categorical = _rows(ws["records_categorical"])
    mixed = _write(tmp_path / "mixed.jsonl", _choice_under(_rows(ws["records_canary"]), categorical, 3))
    ref, cur = (mixed, ws["records"]) if side == "reference" else (ws["records"], mixed)
    with pytest.raises(DriftInputError) as e:
        audit_drift(ref, cur)
    assert e.value.side == side
    # Reference first: on that side the 4th line is the odd one; as current, every earlier
    # reference record set the question's primitive to noul, so line 4 is still the first conflict.
    assert e.value.line == 4
    assert "is answered as choice here and as noul elsewhere" in e.value.message
    assert cli_main(["drift", ref, cur]) == 1
    assert _error(capsys).startswith(f"s1: {side} records: line 4: question ")


def test_web_drift_refuses_two_primitives_for_one_question(ws) -> None:
    categorical = _rows(ws["records_categorical"])
    _write(ws["records_canary"], _choice_under(_rows(ws["records_canary"]), categorical, 3))
    r = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get("/api/drift")
    assert r.status_code == 400
    assert r.json()["detail"].startswith("records_canary.jsonl: line 4: question ")


def test_drift_still_pools_a_questions_models(ws) -> None:
    # The documented exception: a model change within one primitive is reported, not refused.
    (report,) = audit_drift(ws["records"], ws["records_canary"])["questions"]
    assert report["model_drift_detected"] is True
    assert report["ref_models"] != report["cur_models"]


def test_demo_compare_and_drift_unchanged(ws) -> None:
    assert compare_records(ws["records"], ws["records_kev"])["questions"]
    assert audit_drift(ws["records"], ws["records_canary"])["questions"]


def test_drift_refuses_sides_that_each_hold_one_other_primitive(ws, tmp_path: Path) -> None:
    # Each side is internally consistent; only comparing across sides finds the conflict.
    noul = _rows(ws["records"])
    choice = next(r for r in _rows(ws["records_categorical"]) if r["primitive"] == "choice")
    choice = {k: v for k, v in choice.items() if k != "calib"}
    choice.update(question_id=noul[0]["question_id"], model_served=noul[0]["model_served"], item_id="zz-1")
    cur = _write(tmp_path / "cur.jsonl", [choice])
    with pytest.raises(DriftInputError) as e:
        audit_drift(ws["records"], cur)
    assert (e.value.side, e.value.line) == ("current", 1)
    assert e.value.message.startswith(f"question {noul[0]['question_id']} is answered as choice here and as noul elsewhere")


def test_cli_and_web_compare_refuse_two_primitives_for_one_question(ws, capsys) -> None:
    categorical = _rows(ws["records_categorical"])
    primary = _choice_under(_rows(ws["records"]), categorical, 0)
    shadow = _choice_under(_rows(ws["records_kev"]), categorical, 0)
    shadow[0]["model_served"] = shadow[1]["model_served"]
    _write(ws["records"], primary)
    _write(ws["records_kev"], shadow)
    expected = (
        f"primary records answer question {_qid(ws)} as more than one primitive ['choice', 'noul']; "
        "a question_id names one wire question, so one primitive: the file is hand-edited or mislabeled"
    )
    assert cli_main(["compare", ws["records"], ws["records_kev"]]) == 1
    assert _error(capsys) == f"s1: {expected}"
    r = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get("/api/compare")
    assert r.status_code == 400
    assert r.json()["detail"] == f"records.jsonl vs records_kev.jsonl: {expected}"
