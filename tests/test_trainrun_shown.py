"""`--training-run`: reports and the Langfuse export say which recorded run produced a model (#89)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from s1ops.bridges.langfuse import build_langfuse_events
from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records
from s1ops.contracts.schemas import load_policies
from s1ops.eval.compare import compare_records
from s1ops.improve.trainrun import TrainRun, TrainRunError, _trainrun_id, load_trainrun, runs_by_checkpoint

QID = "q-aaaaaaaaaaaaaaaa"
BASELINE, R1, R2 = "current-1", "tickets-r1", "tickets-r2"
SECRET = "SECRET-STATE-TEXT"


def _row(item: str, split: str | None, model: str, *, label: bool = True, p: float = 0.8) -> dict:
    row = {
        "schema": "s1-record/1",
        "item_id": item,
        "question_id": QID,
        "primitive": "noul",
        "model_requested": model,
        "model_served": model,
        "p": p,
        "dataset": "d",
        "label": label,
    }
    if split is not None:
        row["split"] = split
    return row


def _spread(prefix: str, n: int, split: str | None, model: str) -> list[dict]:
    return [_row(f"{prefix}{i}", split, model, label=i % 2 == 0, p=0.2 + 0.6 * (i % 2 == 0) + 0.01 * i) for i in range(n)]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _recorded(tmp_path: Path, checkpoint: str, *, weights: bytes = b"\x00w", name: str | None = None) -> Path:
    """The record of a run that trained `checkpoint` on three items, written by the CLI."""
    d = tmp_path / (name or checkpoint)
    d.mkdir()
    source = _write(d / "source.jsonl", _spread("t", 3, "train", BASELINE))
    manifest = d / "manifest.json"
    assert cli_main(["trainset", "export", source, "--split", "train", "--checkpoint", checkpoint, "-o", str(manifest)]) == 0
    data = _write(d / "train.jsonl", [{"dataset": "d", "item_id": f"t{i}", "state": f"{SECRET} {i}", "label": "yes"} for i in range(3)])
    (d / "head.pt").write_bytes(weights)
    trainer = d / "trainer.json"
    trainer.write_text(json.dumps({"epochs": 3, "lr": 2e-4}))
    out = d / "trainrun.json"
    argv = ["trainset", "record", str(manifest), "--data", data, "--weights", str(d / "head.pt"), "--trainer", str(trainer), "-o", str(out)]
    assert cli_main(argv) == 0
    return out


@pytest.fixture
def run(tmp_path: Path, capsys) -> Path:
    out = _recorded(tmp_path, R1)
    capsys.readouterr()
    return out


def _id(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["trainrun_id"]


def _printed(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


# ---- compare ----


def test_compare_names_the_run_of_each_side(tmp_path: Path, run: Path, capsys) -> None:
    primary = _write(tmp_path / "p.jsonl", _spread("h", 6, None, BASELINE))
    shadow = _write(tmp_path / "s.jsonl", _spread("h", 6, None, R1))
    assert cli_main(["compare", primary, shadow]) == 0
    plain = _printed(capsys)
    assert cli_main(["compare", primary, shadow, "--training-run", str(run)]) == 0
    shown = _printed(capsys)
    (question,) = shown["questions"]
    assert (question["primary_trainrun_id"], question["shadow_trainrun_id"]) == (None, _id(run))
    # Nothing else moves, and without the option the report has neither key.
    for key in ("primary_trainrun_id", "shadow_trainrun_id"):
        assert key not in plain["questions"][0]
        del question[key]
    assert shown == plain


def test_compare_takes_one_record_per_side(tmp_path: Path, run: Path, capsys) -> None:
    other = _recorded(tmp_path, R2, weights=b"\x01w")
    capsys.readouterr()
    primary = _write(tmp_path / "p.jsonl", _spread("h", 6, None, R1))
    shadow = _write(tmp_path / "s.jsonl", _spread("h", 6, None, R2))
    assert cli_main(["compare", primary, shadow, "--training-run", str(other), "--training-run", str(run)]) == 0
    (question,) = _printed(capsys)["questions"]
    assert (question["primary_trainrun_id"], question["shadow_trainrun_id"]) == (_id(run), _id(other))
    assert _id(run) != _id(other)


def test_compare_refuses_a_record_whose_checkpoint_answered_nothing(tmp_path: Path, run: Path, capsys) -> None:
    primary = _write(tmp_path / "p.jsonl", _spread("h", 6, None, BASELINE))
    shadow = _write(tmp_path / "s.jsonl", _spread("h", 6, None, "another-model"))
    assert cli_main(["compare", primary, shadow, "--training-run", str(run)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(f"s1: {run}: checkpoint '{R1}' answered no record in {primary} or {shadow}")
    assert "['another-model', 'current-1']" in err


def test_compare_records_takes_records_from_the_sdk(tmp_path: Path, run: Path) -> None:
    primary, shadow = _spread("h", 6, None, BASELINE), _spread("h", 6, None, R1)
    (question,) = compare_records(primary, shadow, training_runs=[run])["questions"]
    assert question["shadow_trainrun_id"] == _id(run)
    assert "shadow_trainrun_id" not in compare_records(primary, shadow)["questions"][0]
    with pytest.raises(TrainRunError, match="answered no record in primary records or shadow records"):
        compare_records(primary, primary, training_runs=[run])


# ---- policy ----


def test_policy_tune_prints_the_run_and_writes_the_same_policy_file(tmp_path: Path, run: Path, capsys) -> None:
    records = _write(tmp_path / "e.jsonl", [*_spread("h", 12, None, R1), *_spread("h", 12, None, BASELINE)])
    assert cli_main(["policy", "tune", records, "-o", str(tmp_path / "plain.json")]) == 0
    plain = _printed(capsys)
    assert cli_main(["policy", "tune", records, "-o", str(tmp_path / "shown.json"), "--training-run", str(run)]) == 0
    shown = _printed(capsys)
    assert {p["model_served"]: p["trainrun_id"] for p in shown["policies"]} == {R1: _id(run), BASELINE: None}
    assert all("trainrun_id" not in p for p in plain["policies"])

    def rows(path: Path) -> list[dict]:
        return [{k: v for k, v in p.items() if k != "created_at"} for p in json.loads(path.read_text())["policies"]]

    # `s1-policy/1` has no such field: the file is what it was, and every reader still takes it.
    assert rows(tmp_path / "shown.json") == rows(tmp_path / "plain.json")
    assert "trainrun_id" not in (tmp_path / "shown.json").read_text()
    assert len(load_policies(tmp_path / "shown.json")) == 2


def test_policy_curve_prints_the_run(tmp_path: Path, run: Path, capsys) -> None:
    records = _write(tmp_path / "e.jsonl", [*_spread("h", 12, None, R1), *_spread("h", 12, None, BASELINE)])
    assert cli_main(["policy", "curve", records]) == 0
    plain = _printed(capsys)
    assert cli_main(["policy", "curve", records, "--training-run", str(run)]) == 0
    shown = _printed(capsys)
    assert {g["model_served"]: g["trainrun_id"] for g in shown["groups"]} == {R1: _id(run), BASELINE: None}
    for group in shown["groups"]:
        del group["trainrun_id"]
    assert shown == plain


@pytest.mark.parametrize("command", ["tune", "curve"])
def test_a_record_is_bound_to_the_file_not_to_the_selected_groups(tmp_path: Path, run: Path, capsys, command: str) -> None:
    records = _write(tmp_path / "e.jsonl", [*_spread("h", 12, None, R1), *_spread("h", 12, None, BASELINE)])
    argv = ["policy", command, records, "--model", BASELINE, "--training-run", str(run)]
    if command == "tune":
        argv += ["-o", str(tmp_path / "p.json")]
    assert cli_main(argv) == 0
    (row,) = _printed(capsys)["policies" if command == "tune" else "groups"]
    assert (row["model_served"], row["trainrun_id"]) == (BASELINE, None)


@pytest.mark.parametrize("command", ["tune", "curve"])
def test_policy_refuses_a_record_whose_checkpoint_answered_nothing(tmp_path: Path, run: Path, capsys, command: str) -> None:
    records = _write(tmp_path / "e.jsonl", _spread("h", 12, None, BASELINE))
    out = tmp_path / "p.json"
    argv = ["policy", command, records, "--training-run", str(run)]
    if command == "tune":
        argv += ["-o", str(out)]
    assert cli_main(argv) == 1
    assert capsys.readouterr().err.startswith(f"s1: {run}: checkpoint '{R1}' answered no record in {records}")
    assert not out.exists()


# ---- which records are taken ----


def test_two_runs_for_one_checkpoint_are_refused(tmp_path: Path, run: Path, capsys) -> None:
    again = _recorded(tmp_path, R1, weights=b"\x02other", name="retrained")
    capsys.readouterr()
    records = _write(tmp_path / "e.jsonl", _spread("h", 12, None, R1))
    assert cli_main(["policy", "curve", records, "--training-run", str(run), "--training-run", str(again)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"s1: {again}: {_id(again)} is a second run for checkpoint '{R1}', after {_id(run)} in {run}")


def test_one_run_given_twice_is_one_run(tmp_path: Path, run: Path, capsys) -> None:
    copy = tmp_path / "copy.json"
    shutil.copy(run, copy)
    records = _write(tmp_path / "e.jsonl", _spread("h", 12, None, R1))
    assert cli_main(["policy", "curve", records, "--training-run", str(run), "--training-run", str(copy)]) == 0
    assert _printed(capsys)["groups"][0]["trainrun_id"] == _id(run)


def test_an_edited_record_is_refused_by_name(tmp_path: Path, run: Path, capsys) -> None:
    doc = json.loads(run.read_text())
    doc["weights_sha256"]["head.pt"] = "0" * 64
    run.write_text(json.dumps(doc))
    records = _write(tmp_path / "e.jsonl", _spread("h", 12, None, R1))
    for argv in (
        ["policy", "curve", records],
        ["compare", records, records],
        ["export", "langfuse", records, "-o", str(tmp_path / "lf.json")],
    ):
        assert cli_main([*argv, "--training-run", str(run)]) == 1
        assert capsys.readouterr().err.startswith(f"s1: {run}: trainrun_id {doc['trainrun_id']} does not match the record's contents")
    assert not (tmp_path / "lf.json").exists()


def test_a_missing_record_names_the_command_that_writes_it(tmp_path: Path, capsys) -> None:
    records = _write(tmp_path / "e.jsonl", _spread("h", 12, None, R1))
    missing = tmp_path / "nope.json"
    assert cli_main(["policy", "curve", records, "--training-run", str(missing)]) == 1
    assert capsys.readouterr().err == f"s1: training-run record not found: {missing} (run `s1 trainset record` first)\n"


def test_a_manifest_is_not_a_record(tmp_path: Path, run: Path, capsys) -> None:
    manifest = run.parent / "manifest.json"
    records = _write(tmp_path / "e.jsonl", _spread("h", 12, None, R1))
    assert cli_main(["policy", "curve", records, "--training-run", str(manifest)]) == 1
    assert capsys.readouterr().err == f"s1: {manifest}: not a s1-trainrun/1 file\n"


# ---- Langfuse export ----


def test_the_langfuse_export_carries_the_record_and_names_it_on_its_observations(tmp_path: Path, run: Path, capsys) -> None:
    records = _write(tmp_path / "e.jsonl", [*_spread("h", 4, None, R1), *_spread("h", 4, None, BASELINE)])
    assert cli_main(["export", "langfuse", records, "-o", str(tmp_path / "plain.json")]) == 0
    capsys.readouterr()
    assert cli_main(["export", "langfuse", records, "-o", str(tmp_path / "shown.json"), "--training-run", str(run)]) == 0
    assert _printed(capsys)["n_training_runs"] == 1
    text = (tmp_path / "shown.json").read_text()
    shown, plain = json.loads(text), json.loads((tmp_path / "plain.json").read_text())
    assert shown["training_runs"] == [json.loads(run.read_text())]
    assert shown["summary"]["n_training_runs"] == 1
    by_model = {o["model"]: o["metadata"].get("trainrun_id") for o in shown["observations"]}
    assert by_model == {R1: _id(run), BASELINE: None}
    assert all("trainrun_id" not in o["metadata"] for o in shown["observations"] if o["model"] == BASELINE)
    assert SECRET not in text and '"state"' not in text
    # Without the option the file is what it was.
    assert "training_runs" not in plain and "n_training_runs" not in plain["summary"]
    del shown["training_runs"], shown["summary"]["n_training_runs"]
    for o in shown["observations"]:
        o["metadata"].pop("trainrun_id", None)
    assert shown == plain


def test_the_langfuse_export_lists_records_by_checkpoint(tmp_path: Path, run: Path, capsys) -> None:
    other = _recorded(tmp_path, R2, weights=b"\x01w")
    records = _write(tmp_path / "e.jsonl", [*_spread("h", 4, None, R2), *_spread("h", 4, None, R1)])
    out = tmp_path / "lf.json"
    assert cli_main(["export", "langfuse", records, "-o", str(out), "--training-run", str(other), "--training-run", str(run)]) == 0
    assert [r["checkpoint"] for r in json.loads(out.read_text())["training_runs"]] == [R1, R2]


def test_the_langfuse_export_refuses_a_record_whose_checkpoint_answered_nothing(tmp_path: Path, run: Path, capsys) -> None:
    records = _write(tmp_path / "e.jsonl", _spread("h", 4, None, BASELINE))
    out = tmp_path / "lf.json"
    assert cli_main(["export", "langfuse", records, "-o", str(out), "--training-run", str(run)]) == 1
    assert capsys.readouterr().err.startswith(f"s1: {run}: checkpoint '{R1}' answered no record in {records}")
    assert not out.exists()


def test_the_phoenix_export_takes_no_record(tmp_path: Path, run: Path, capsys) -> None:
    records = _write(tmp_path / "e.jsonl", _spread("h", 4, None, R1))
    with pytest.raises(SystemExit) as e:
        cli_main(["export", "phoenix", records, "-o", str(tmp_path / "s.jsonl"), "--training-run", str(run)])
    assert e.value.code == 1


# ---- the SDK ----


def _records(tmp_path: Path, model: str = R1) -> list:
    return read_records(_write(tmp_path / "sdk.jsonl", _spread("h", 4, None, model)))


def test_the_builder_takes_loaded_records(tmp_path: Path, run: Path) -> None:
    loaded = load_trainrun(run)
    assert loaded.path == run
    payload = build_langfuse_events(_records(tmp_path), training_runs=[loaded])
    assert payload["training_runs"] == [loaded.doc]
    assert {o["metadata"]["trainrun_id"] for o in payload["observations"]} == {loaded.trainrun_id}
    assert "training_runs" not in build_langfuse_events(_records(tmp_path), training_runs=[])
    with pytest.raises(TrainRunError, match="answered no record in the exported records"):
        build_langfuse_events(_records(tmp_path, BASELINE), training_runs=[loaded])


@pytest.mark.parametrize(
    ("edit", "rehash", "message"),
    [
        (lambda d: {**d, "state": SECRET}, True, "state: Extra inputs are not permitted"),
        (lambda d: {**d, "trainer": {"note": {"state": SECRET}}}, True, "trainer setting 'note' is a dict"),
        (lambda d: {**d, "split": "cal"}, False, "does not match the record's contents"),
        (lambda d: [d], False, "not a s1-trainrun/1 file"),
    ],
)
def test_a_record_built_by_hand_is_checked_before_it_is_exported(tmp_path: Path, run: Path, edit, rehash: bool, message: str) -> None:
    """`TrainRun` is a plain dataclass: the export attaches `doc`, so it is checked as a file is."""
    good = load_trainrun(run)
    doc = edit(dict(good.doc))
    if rehash:  # the id matches, so only the rule for a record's contents refuses it
        doc["trainrun_id"] = _trainrun_id(doc)
    forged = TrainRun(trainrun_id=good.trainrun_id, checkpoint=R1, doc=doc)
    with pytest.raises(TrainRunError) as e:
        build_langfuse_events(_records(tmp_path), training_runs=[forged])
    assert message in str(e.value) and SECRET not in str(e.value)
    assert str(e.value).startswith("the training-run record: ")


def test_a_run_given_under_another_name_than_its_record_is_refused(tmp_path: Path, run: Path) -> None:
    good = load_trainrun(run)
    renamed = TrainRun(trainrun_id=good.trainrun_id, checkpoint=BASELINE, doc=good.doc)
    with pytest.raises(TrainRunError, match=f"given as {good.trainrun_id} for '{BASELINE}', but its record is {good.trainrun_id} for '{R1}'"):
        runs_by_checkpoint([renamed], [BASELINE], "the records")
