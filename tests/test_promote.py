"""`s1 promote`: README's promotion check for a fine-tuned candidate as one command (#89)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import load_policies
from s1ops.improve.trainset import TrainedItemError

CURRENT, CANDIDATE = "current-1", "tickets-r1"
WALL_CLOCK = ("created_at", "fitted_at")


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _without_wall_clock(value):
    if isinstance(value, dict):
        return {k: _without_wall_clock(v) for k, v in value.items() if k not in WALL_CLOCK}
    if isinstance(value, list):
        return [_without_wall_clock(v) for v in value]
    return value


@pytest.fixture
def ws(tmp_path: Path, capsys) -> dict:
    """The files README's sequence has before step 4, probed on the offline mock."""
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"urgent": {"type": "noul", "instructions": "Is this urgent?"}}), encoding="utf-8")
    states = [
        {"dataset": "d", "item_id": f"i{i}", "state": f"body {i}", "split": split, "label": i % 3 != 0}
        for split, span in (("train", range(0, 60)), ("cal", range(60, 120)), ("eval", range(120, 180)))
        for i in span
    ]
    files = {split: _write(tmp_path / f"states_{split}.jsonl", [s for s in states if s["split"] == split]) for split in ("train", "cal", "eval")}
    ledger = str(tmp_path / "ledger.jsonl")

    def probe(model: str, split: str, out: str) -> str:
        path = str(tmp_path / out)
        argv = ["probe", str(questions), "--no-expand-probes", "--model", model, "--states", files[split], "-o", path, "--ledger", ledger]
        assert cli_main(argv) == 0
        return path

    out = {
        "train": probe(CURRENT, "train", "current_train.jsonl"),
        "current": probe(CURRENT, "eval", "current_eval.jsonl"),
        "cal": probe(CANDIDATE, "cal", "candidate_cal.jsonl"),
        "eval": probe(CANDIDATE, "eval", "candidate_eval.jsonl"),
        "manifest": str(tmp_path / "training_manifest.json"),
        "dir": tmp_path,
    }
    assert cli_main(["trainset", "export", out["train"], "--split", "train", "--checkpoint", CANDIDATE, "-o", out["manifest"]]) == 0
    capsys.readouterr()
    return out


def _promote(ws: dict, out: Path, *extra: str, **replace: str) -> int:
    files = {**ws, **replace}
    argv = ["promote", "--current", files["current"], "--cal", files["cal"], "--eval", files["eval"], "--training-manifest", files["manifest"], "-o", str(out)]
    return cli_main([*argv, *extra])


def _json_out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_promote_is_the_documented_commands(ws: dict, tmp_path: Path, capsys) -> None:
    out = tmp_path / "promotion"
    assert _promote(ws, out) == 0
    report = _json_out(capsys)
    assert json.loads((out / "promotion.json").read_text()) == report
    assert sorted(p.name for p in out.iterdir()) == ["candidate_eval_calibrated.jsonl", "fits.json", "policy.json", "promotion.json"]
    assert (report["schema"], report["checkpoint"], report["current_models"]) == ("s1-promotion/1", CANDIDATE, [CURRENT])
    assert report["output"] == {
        "fits": str(out / "fits.json"),
        "calibrated": str(out / "candidate_eval_calibrated.jsonl"),
        "policy": str(out / "policy.json"),
        "report": str(out / "promotion.json"),
    }
    assert "trainrun_id" not in report

    # README step 4, by hand, into another directory.
    hand = tmp_path / "hand"
    hand.mkdir()
    flag = ["--training-manifest", ws["manifest"]]
    assert cli_main(["compare", ws["current"], ws["eval"], *flag]) == 0
    compared = _json_out(capsys)
    assert cli_main(["fit", ws["cal"], *flag, "-o", str(hand / "fits.json")]) == 0
    fitted = _json_out(capsys)
    assert cli_main(["apply", ws["eval"], "--fit", str(hand / "fits.json"), "-o", str(hand / "candidate_eval_calibrated.jsonl")]) == 0
    assert capsys.readouterr().out == ""
    assert cli_main(["policy", "tune", str(hand / "candidate_eval_calibrated.jsonl"), "--use-calib", *flag, "-o", str(hand / "policy.json")]) == 0
    tuned = _json_out(capsys)
    assert cli_main(["drift", ws["current"], ws["eval"]]) == 0
    drift = _json_out(capsys)

    def placed(value):
        """A hand-made summary as the promotion's directory would have it."""
        return json.loads(json.dumps(_without_wall_clock(value)).replace(str(hand), str(out)))

    assert report["compare"] == compared
    assert report["drift"] == drift
    assert report["fit"] == placed(fitted)
    assert report["apply"] == {"calibrated": 60}
    assert _without_wall_clock(report["policy"]) == placed(tuned)
    assert report["replaceable"] is all(q["replaceable"] for q in compared["questions"])
    for name in ("fits.json", "policy.json"):
        assert _without_wall_clock(json.loads((out / name).read_text())) == _without_wall_clock(json.loads((hand / name).read_text()))
    assert _without_wall_clock(_rows(str(out / "candidate_eval_calibrated.jsonl"))) == _without_wall_clock(_rows(str(hand / "candidate_eval_calibrated.jsonl")))


def test_the_tune_and_fit_options_are_the_commands_own(ws: dict, tmp_path: Path, capsys) -> None:
    out = tmp_path / "promotion"
    options = ["--error-budget", "0.02", "--noul-routing", "threshold", "--use-cp-upper", "--cp-alpha", "0.2"]
    assert _promote(ws, out, "--method", "platt", *options) == 0
    report = _json_out(capsys)
    assert {g["method"] for g in report["fit"]["groups"]} == {"platt"}
    hand = tmp_path / "p.json"
    assert cli_main(["policy", "tune", str(out / "candidate_eval_calibrated.jsonl"), "--use-calib", *options, "-o", str(hand)]) == 0
    assert _without_wall_clock(json.loads(hand.read_text())) == _without_wall_clock(json.loads((out / "policy.json").read_text()))
    (policy,) = load_policies(out / "policy.json")
    assert (policy.mode, policy.error_budget, policy.calibrated) == ("error_budget", 0.02, True)


def test_a_bad_tune_option_is_refused_as_policy_tune_refuses_it(ws: dict, tmp_path: Path, capsys) -> None:
    assert _promote(ws, tmp_path / "promotion", "--error-budget", "2") == 1
    assert capsys.readouterr().err == "s1: promote: policy tune: --error-budget must be a number in [0, 1], got 2.0\n"


def test_the_training_run_reaches_the_report_and_the_policy_file(ws: dict, tmp_path: Path, capsys) -> None:
    train = _write(tmp_path / "train.jsonl", [{"dataset": r["dataset"], "item_id": r["item_id"]} for r in _rows(ws["train"])])
    (tmp_path / "head.pt").write_bytes(b"w")
    run = tmp_path / "trainrun.json"
    assert cli_main(["trainset", "record", ws["manifest"], "--data", train, "--weights", str(tmp_path / "head.pt"), "-o", str(run)]) == 0
    trainrun_id = _json_out(capsys)["trainrun_id"]
    out = tmp_path / "promotion"
    assert _promote(ws, out, "--training-run", str(run)) == 0
    report = _json_out(capsys)
    assert report["trainrun_id"] == trainrun_id
    assert [(q["primary_trainrun_id"], q["shadow_trainrun_id"]) for q in report["compare"]["questions"]] == [(None, trainrun_id)]
    assert [p.trainrun_id for p in load_policies(out / "policy.json")] == [trainrun_id]


# ---- the manifest is on every step that takes one ----


def _relabeled(path: str, out: Path, model: str) -> str:
    return _write(out, [{**r, "model_served": model, "model_requested": model} for r in _rows(path)])


def test_an_evaluation_item_the_candidate_trained_on_stops_at_compare_with_nothing_written(ws: dict, tmp_path: Path, capsys) -> None:
    # The candidate "evaluated" on its training rows, with the model in place answering them too.
    trained_eval = _relabeled(ws["train"], tmp_path / "leaky_eval.jsonl", CANDIDATE)
    out = tmp_path / "promotion"
    assert _promote(ws, out, current=ws["train"], eval=trained_eval) == 2
    err = capsys.readouterr().err
    # `compare` checks pair by pair, so it names the first one.
    assert err.startswith(f"s1: promote: compare: {trained_eval}: 1 item(s)") and f"training set of '{CANDIDATE}'" in err
    assert not out.exists()


def test_a_calibration_item_the_candidate_trained_on_stops_at_fit(ws: dict, tmp_path: Path, capsys) -> None:
    trained_cal = _relabeled(ws["train"], tmp_path / "leaky_cal.jsonl", CANDIDATE)
    out = tmp_path / "promotion"
    assert _promote(ws, out, cal=trained_cal) == 2
    assert capsys.readouterr().err.startswith(f"s1: promote: fit: {trained_cal}: 60 item(s)")
    assert list(out.iterdir()) == []


def test_calibrating_on_the_evaluation_fold_stops_at_apply(ws: dict, tmp_path: Path, capsys) -> None:
    out = tmp_path / "promotion"
    assert _promote(ws, out, cal=ws["eval"]) == 2
    assert capsys.readouterr().err.startswith("s1: promote: apply: ")
    # The step before it had finished: its file stays, as it would after the commands by hand.
    assert [p.name for p in out.iterdir()] == ["fits.json"]


def test_debug_mode_raises_the_library_error(ws: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("S1_DEBUG", "1")
    with pytest.raises(TrainedItemError):
        _promote(ws, tmp_path / "promotion", cal=_relabeled(ws["train"], tmp_path / "leaky_cal.jsonl", CANDIDATE))


# ---- what a promotion is of ----


def test_no_manifest_for_the_candidate_is_refused(ws: dict, tmp_path: Path, capsys) -> None:
    other = tmp_path / "other.json"
    assert cli_main(["trainset", "export", ws["train"], "--split", "train", "--checkpoint", "tickets-r0", "-o", str(other)]) == 0
    capsys.readouterr()
    out = tmp_path / "promotion"
    assert _promote(ws, out, manifest=str(other)) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"s1: no --training-manifest is for '{CANDIDATE}'") and "['tickets-r0']" in err
    assert not out.exists()


def test_a_manifest_for_the_model_in_place_goes_to_compare_alone(ws: dict, tmp_path: Path, capsys) -> None:
    """The model in place was fine-tuned too: its manifest names no record of the candidate's files,
    which `fit` and `policy tune` would refuse, and `compare` checks its side with it."""
    old = _write(tmp_path / "old.jsonl", [{**r, "item_id": f"old-{r['item_id']}"} for r in _rows(ws["train"])])
    current_manifest = tmp_path / "current_manifest.json"
    assert cli_main(["trainset", "export", old, "--split", "train", "--checkpoint", CURRENT, "-o", str(current_manifest)]) == 0
    capsys.readouterr()
    assert _promote(ws, tmp_path / "promotion", "--training-manifest", str(current_manifest)) == 0
    # ...and it is checked: the model in place answering its own training items is refused.
    leaky = tmp_path / "leaky_manifest.json"
    assert cli_main(["trainset", "export", _relabeled(ws["current"], tmp_path / "c.jsonl", "base-0"), "--split", "eval", "--checkpoint", CURRENT, "-o", str(leaky)]) == 0
    capsys.readouterr()
    assert _promote(ws, tmp_path / "promotion2", "--training-manifest", str(leaky)) == 2
    assert capsys.readouterr().err.startswith(f"s1: promote: compare: {ws['current']}: ")


def test_candidate_files_name_one_model_and_the_same_one(ws: dict, tmp_path: Path, capsys) -> None:
    mixed = _write(tmp_path / "mixed.jsonl", [*_rows(ws["cal"]), *_rows(ws["current"])])
    assert _promote(ws, tmp_path / "p", cal=mixed) == 1
    assert capsys.readouterr().err.startswith(f"s1: {mixed}: a candidate's records are answered by one model, and these name ['{CURRENT}', '{CANDIDATE}']")
    other = _relabeled(ws["eval"], tmp_path / "other_eval.jsonl", "tickets-r2")
    assert _promote(ws, tmp_path / "p", eval=other) == 1
    assert capsys.readouterr().err.startswith(f"s1: {other}: answered by ['tickets-r2'], and {ws['cal']} by '{CANDIDATE}'")
    assert _promote(ws, tmp_path / "p", current=ws["eval"]) == 1
    assert capsys.readouterr().err.startswith(f"s1: {ws['eval']}: holds answers of the candidate '{CANDIDATE}'")
    assert not (tmp_path / "p").exists()


def test_nothing_shared_with_the_model_in_place_is_refused(ws: dict, tmp_path: Path, capsys) -> None:
    out = tmp_path / "promotion"
    assert _promote(ws, out, current=ws["train"]) == 1
    assert capsys.readouterr().err == f"s1: promote: compare: no item of {ws['eval']} was answered in {ws['train']}; nothing would be compared\n"
    assert not out.exists()


# ---- inputs and outputs ----


def test_a_missing_input_is_named(ws: dict, tmp_path: Path, capsys) -> None:
    assert _promote(ws, tmp_path / "p", cal=str(tmp_path / "nope.jsonl")) == 1
    assert capsys.readouterr().err == f"s1: candidate calibration records file not found: {tmp_path / 'nope.jsonl'}\n"
    assert _promote(ws, tmp_path / "p", manifest=str(tmp_path / "nope.json")) == 1
    assert capsys.readouterr().err == f"s1: training manifest not found: {tmp_path / 'nope.json'} (run `s1 trainset export` first)\n"


def test_an_input_that_can_be_read_once_is_refused(ws: dict, tmp_path: Path, capsys) -> None:
    fifo = tmp_path / "eval.fifo"
    os.mkfifo(fifo)
    assert _promote(ws, tmp_path / "p", eval=str(fifo)) == 1
    assert capsys.readouterr().err == f"s1: {fifo}: more than one step reads each input, so it must be a regular file\n"


def test_the_output_directory_cannot_hold_an_input_under_an_output_name(ws: dict, tmp_path: Path, capsys) -> None:
    out = tmp_path / "promotion"
    out.mkdir()
    clash = out / "candidate_eval_calibrated.jsonl"
    clash.write_text(Path(ws["eval"]).read_text())
    assert _promote(ws, out, eval=str(clash)) == 1
    assert capsys.readouterr().err == f"s1: --output {out} would write {clash}, which is the input {clash}\n"
    assert clash.read_text() == Path(ws["eval"]).read_text()


def test_a_second_run_into_the_directory_replaces_the_first(ws: dict, tmp_path: Path, capsys) -> None:
    out = tmp_path / "promotion"
    assert _promote(ws, out) == 0
    first = _json_out(capsys)
    assert _promote(ws, out, "--error-budget", "0.02", "--noul-routing", "threshold") == 0
    second = _json_out(capsys)
    assert first["compare"] == second["compare"] and first["policy"] != second["policy"]
    assert json.loads((out / "promotion.json").read_text()) == second
