"""`s1 apply --pass-through-unfitted` makes "run `s1 apply`" followable on a mixed-model file (#21).

Without it, a canary served by a model the fit file has no fit for stops `s1 apply` with exit 3,
while a calibrated policy in `s1 drift` / `s1 export` refuses records without `calib`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import load_fits

FIT_MODEL, NEW_MODEL = "jev-1.13.0", "jev-1.14.0-rc1"


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _stripped(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "calib"} for r in rows]


def _apply(records: str, fits: str, out: Path, *flags: str) -> int:
    return cli_main(["apply", records, "--fit", fits, "-o", str(out), *flags])


def test_the_demo_canary_is_rebuilt_by_apply(ws, tmp_path: Path, capsys) -> None:
    canary = _rows(ws["records_canary"])
    raw = _write(tmp_path / "raw.jsonl", _stripped(canary))
    out = tmp_path / "applied.jsonl"

    assert _apply(raw, ws["fits"], out) == 3
    assert not out.exists()
    capsys.readouterr()

    assert _apply(raw, ws["fits"], out, "--pass-through-unfitted") == 0
    # The demo built its canary with apply_to_record on the same fit: same calib on the 80
    # jev-1.13.0 rows, the 40 upgraded rows unchanged.
    assert _rows(out) == canary
    (fit,) = load_fits(ws["fits"])
    assert json.loads(capsys.readouterr().out) == {
        "calibrated": 80,
        "passed_through": [{
            "question_id": fit.question_id, "model_served": NEW_MODEL, "primitive": "noul", "n": 40,
            "reason": f"fit is for model_served {FIT_MODEL}; this answer was served by {NEW_MODEL}",
        }],
    }


def test_drift_and_export_run_on_the_applied_canary(ws, tmp_path: Path, capsys) -> None:
    raw = _write(tmp_path / "raw.jsonl", _stripped(_rows(ws["records_canary"])))
    for args in (["drift", ws["records"], raw, "--policy", ws["policy"]],
                 ["export", "phoenix", raw, "-o", str(tmp_path / "ph0"), "--policy", ws["policy"]]):
        assert cli_main(args) == 1
        assert "`--pass-through-unfitted` keeps records of models it has no fit for" in capsys.readouterr().err

    out = tmp_path / "applied.jsonl"
    assert _apply(raw, ws["fits"], out, "--pass-through-unfitted") == 0
    capsys.readouterr()
    rates = {}
    for name, cur in (("applied", str(out)), ("demo", ws["records_canary"])):
        assert cli_main(["drift", ws["records"], cur, "--policy", ws["policy"]]) == 0
        (q,) = json.loads(capsys.readouterr().out)["questions"]
        rates[name] = (q["escalation_rate_ref"], q["escalation_rate_cur"])
    assert rates["applied"] == rates["demo"]
    assert rates["applied"][1] is not None
    assert cli_main(["export", "phoenix", str(out), "-o", str(tmp_path / "ph"), "--policy", ws["policy"]]) == 0


def test_groups_are_listed_in_first_seen_order_with_their_counts(ws, tmp_path: Path, capsys) -> None:
    (fit,) = load_fits(ws["fits"])
    canary = _stripped(_rows(ws["records_canary"]))
    first_new = next(i for i, r in enumerate(canary) if r["model_served"] == NEW_MODEL)
    base = next(r for r in canary if r["model_served"] == FIT_MODEL)
    unknown = {**base, "question_id": "q-unknown", "item_id": "u"}
    as_choice = {k: v for k, v in base.items() if k != "p"}
    as_choice.update({"primitive": "choice", "choice": "A", "confidence": 0.7,
                      "probabilities": {"A": 0.7, "B": 0.3}, "item_id": "c"})
    # unknown question before the first upgraded record, then another primitive, then more of both.
    rows = canary[:first_new] + [unknown] + canary[first_new:first_new + 1] + [as_choice, unknown] + canary[first_new + 1:] + [as_choice]
    out = tmp_path / "applied.jsonl"
    assert _apply(_write(tmp_path / "mixed.jsonl", rows), ws["fits"], out, "--pass-through-unfitted") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["calibrated"] == 80
    assert [(g["question_id"], g["model_served"], g["primitive"], g["n"], g["reason"]) for g in summary["passed_through"]] == [
        ("q-unknown", FIT_MODEL, "noul", 2, "no fit for question_id q-unknown"),
        (fit.question_id, NEW_MODEL, "noul", 40, f"fit is for model_served {FIT_MODEL}; this answer was served by {NEW_MODEL}"),
        (fit.question_id, FIT_MODEL, "choice", 2, f"fit for {fit.question_id} is for primitive noul, not choice"),
    ]
    assert [r for r in _rows(out) if "calib" not in r] == [r for r in rows if r["model_served"] != FIT_MODEL or r["question_id"] == "q-unknown" or r["primitive"] == "choice"]


def test_nothing_fitted_still_succeeds_and_says_so(ws, tmp_path: Path, capsys) -> None:
    rows = [r for r in _stripped(_rows(ws["records_canary"])) if r["model_served"] == NEW_MODEL]
    out = tmp_path / "applied.jsonl"
    assert _apply(_write(tmp_path / "new.jsonl", rows), ws["fits"], out, "--pass-through-unfitted") == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["calibrated"], [g["n"] for g in summary["passed_through"]]) == (0, [40])
    assert _rows(out) == rows


def test_a_passed_through_record_keeps_the_calib_it_carried(ws, tmp_path: Path, capsys) -> None:
    rows = _stripped(_rows(ws["records_canary"]))
    n = next(i for i, r in enumerate(rows) if r["model_served"] == NEW_MODEL)
    rows[n]["calib"] = {"method": "temperature", "fit_split": "cal", "fit_n": 9, "fitted_at": "2026-01-01T00:00:00Z", "p": 0.25}
    out = tmp_path / "applied.jsonl"
    assert _apply(_write(tmp_path / "r.jsonl", rows), ws["fits"], out, "--pass-through-unfitted") == 0
    assert _rows(out)[n] == rows[n]


def test_fitted_groups_are_still_refused_as_before(ws, tmp_path: Path, capsys) -> None:
    out = tmp_path / "o.jsonl"
    # In-sample items: exit 2, or calibrated and marked with --allow-in-sample.
    assert _apply(ws["records_train"], ws["fits"], out, "--pass-through-unfitted") == 2
    assert "in-sample" in capsys.readouterr().err
    assert _apply(ws["records_train"], ws["fits"], out, "--pass-through-unfitted", "--allow-in-sample") == 0
    assert json.loads(capsys.readouterr().out) == {"calibrated": 120, "passed_through": []}
    assert all(r["calib"]["in_fit_sample"] is True for r in _rows(out))

    # A record of a fitted group whose classes differ from the fit's: exit 1.
    (choice_fit, _) = sorted(load_fits(ws["fits_categorical"]), key=lambda f: f.primitive)
    rows = _rows(ws["records_categorical"])
    n = next(i for i, r in enumerate(rows) if r["primitive"] == "choice")
    probs = rows[n]["probabilities"]
    first, *rest = probs
    rows[n] = {**rows[n], "probabilities": {"not-an-option": probs[first], **{k: probs[k] for k in rest}}}
    rows[n]["choice"] = max(rows[n]["probabilities"], key=rows[n]["probabilities"].get)
    rows[n].pop("label", None)
    assert choice_fit.primitive == "choice"
    capsys.readouterr()
    assert _apply(_write(tmp_path / "cat.jsonl", _stripped(rows)), ws["fits_categorical"], out, "--pass-through-unfitted") == 1
    assert "differ from the fit's" in capsys.readouterr().err


def test_failed_records_are_written_unchanged(ws, tmp_path: Path, capsys) -> None:
    rows = _stripped(_rows(ws["records_canary"]))
    failed = {k: v for k, v in rows[0].items() if k != "p"} | {"model_served": None, "error_code": "timeout", "item_id": "f"}
    rows.insert(1, failed)
    out = tmp_path / "applied.jsonl"
    assert _apply(_write(tmp_path / "r.jsonl", rows), ws["fits"], out, "--pass-through-unfitted") == 0
    assert _rows(out)[1] == failed
    assert json.loads(capsys.readouterr().out)["calibrated"] == 80


def test_without_the_flag_apply_prints_nothing(ws, tmp_path: Path, capsys) -> None:
    assert _apply(ws["records"], ws["fits"], tmp_path / "o.jsonl") == 0
    assert capsys.readouterr().out == ""
