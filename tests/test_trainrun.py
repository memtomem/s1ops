"""`s1 trainset record`: a finished training run, recorded next to its manifest, without state or labels."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.improve.trainrun import TrainingDataError, TrainRunError, _trainrun_id, build_trainrun, load_trainrun, save_trainrun
from s1ops.improve.trainset import TrainedItemError, load_manifest

QID = "q-aaaaaaaaaaaaaaaa"
CHECKPOINT, CHILD = "tickets-r1", "tickets-r2"
SECRET = "SECRET-STATE-TEXT"


def _row(item: str, split: str, *, label: bool = True) -> dict:
    return {
        "schema": "s1-record/1",
        "item_id": item,
        "question_id": QID,
        "primitive": "noul",
        "model_requested": "base-1",
        "model_served": "base-1",
        "p": 0.7,
        "dataset": "d",
        "label": label,
        "split": split,
    }


def _jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _manifest(tmp_path: Path, items: list[str], *, checkpoint: str = CHECKPOINT, parent: Path | None = None, name: str = "m.json") -> Path:
    records = _jsonl(tmp_path / f"{name}.records.jsonl", [_row(i, "train", label=n % 2 == 0) for n, i in enumerate(items)] + [_row("e0", "eval")])
    out = tmp_path / name
    flags = ["--parent", str(parent)] if parent else []
    assert cli_main(["trainset", "export", records, "--split", "train", "--checkpoint", checkpoint, "-o", str(out), *flags]) == 0
    return out


def _data(tmp_path: Path, items: list[tuple[str | None, str]], name: str = "train.jsonl") -> str:
    """Trainer rows: state text and label sit beside the ids, as a trainer's own file has them."""
    rows = []
    for dataset, item in items:
        row = {"item_id": item, "state": f"{SECRET} {item}", "label": "yes", "text": SECRET}
        if dataset is not None:
            row["dataset"] = dataset
        rows.append(row)
    return _jsonl(tmp_path / name, rows)


def _weights(tmp_path: Path) -> list[str]:
    (tmp_path / "w").mkdir(exist_ok=True)
    (tmp_path / "w" / "adapter.safetensors").write_bytes(b"\x00adapter")
    (tmp_path / "w" / "head.pt").write_bytes(b"\x01head")
    return [str(tmp_path / "w" / "adapter.safetensors"), str(tmp_path / "w" / "head.pt")]


def _record(manifest: Path, data: str, weights: list[str], out: Path, *flags: str) -> int:
    args = ["trainset", "record", str(manifest), "--data", data, "-o", str(out)]
    for w in weights:
        args += ["--weights", w]
    return cli_main([*args, *flags])


@pytest.fixture
def run(tmp_path: Path) -> dict:
    manifest = _manifest(tmp_path, ["t0", "t1", "t2"])
    # t1 twice: two questions, or two epochs, of one item
    data = _data(tmp_path, [("d", "t0"), ("d", "t1"), ("d", "t1"), ("d", "t2")])
    trainer = tmp_path / "trainer.json"
    trainer.write_text(json.dumps({"epochs": 3, "lr": 2e-4, "lora.rank": 16, "base": "kev-4b", "bf16": True, "seed": None}))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"wall_seconds": 308.5, "peak_device_bytes": 19542646784, "temperature": 2.41}))
    steps = _jsonl(tmp_path / "steps.jsonl", [{"step": 0, "epoch": 0, "loss": 0.69}, {"step": 10, "epoch": 1, "loss": 0.21}])
    return {"manifest": manifest, "data": data, "weights": _weights(tmp_path), "trainer": str(trainer), "summary": str(summary), "steps": steps}


def _full(run: dict, out: Path, *extra: str) -> int:
    return _record(run["manifest"], run["data"], run["weights"], out, "--trainer", run["trainer"], "--summary", run["summary"], "--steps", run["steps"], *extra)


def test_record_holds_digests_counts_and_numbers(tmp_path: Path, run: dict, capsys) -> None:
    out = tmp_path / "run.json"
    assert _full(run, out) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "s1-trainrun/1"
    assert (doc["checkpoint"], doc["split"]) == (CHECKPOINT, "train")
    assert doc["manifest"]["sha256"] == hashlib.sha256(run["manifest"].read_bytes()).hexdigest()
    assert doc["manifest"]["source_sha256"] == load_manifest(run["manifest"]).source_sha256
    assert (doc["manifest"]["items"], doc["manifest"]["inherited_items"]) == (3, 0)
    assert doc["data"] == {"sha256": hashlib.sha256(Path(run["data"]).read_bytes()).hexdigest(), "rows": 4, "items": 3}
    assert doc["weights_sha256"] == {
        "adapter.safetensors": hashlib.sha256(b"\x00adapter").hexdigest(),
        "head.pt": hashlib.sha256(b"\x01head").hexdigest(),
    }
    assert doc["trainer"]["lora.rank"] == 16 and doc["trainer"]["seed"] is None
    assert doc["summary"]["temperature"] == 2.41
    assert [s["step"] for s in doc["steps"]] == [0, 10]
    assert doc["parent"] is None
    assert doc["trainrun_id"].startswith("tr-") and len(doc["trainrun_id"]) == 19
    printed = json.loads(capsys.readouterr().out)
    assert (printed["trainrun_id"], printed["steps"], printed["manifest_items"]) == (doc["trainrun_id"], 2, 3)
    assert load_trainrun(out).trainrun_id == doc["trainrun_id"]


def test_record_holds_no_state_labels_or_paths(tmp_path: Path, run: dict) -> None:
    out = tmp_path / "run.json"
    assert _full(run, out) == 0
    written = out.read_text(encoding="utf-8")
    assert SECRET not in written and '"yes"' not in written
    assert str(tmp_path) not in written  # weights are keyed by file name


def test_same_inputs_give_the_same_id(tmp_path: Path, run: dict) -> None:
    assert _full(run, tmp_path / "a.json") == 0
    assert _full(run, tmp_path / "b.json") == 0
    a, b = (json.loads((tmp_path / n).read_text()) for n in ("a.json", "b.json"))
    assert a["trainrun_id"] == b["trainrun_id"]
    Path(run["weights"][1]).write_bytes(b"\x01other head")
    assert _full(run, tmp_path / "c.json") == 0
    assert json.loads((tmp_path / "c.json").read_text())["trainrun_id"] != a["trainrun_id"]


def test_data_with_an_unlisted_item_is_exit_2_and_writes_nothing(tmp_path: Path, run: dict, capsys) -> None:
    data = _data(tmp_path, [("d", "t0"), ("d", "t1"), ("d", "t2"), ("d", "x9")], "extra.jsonl")
    out = tmp_path / "run.json"
    assert _record(run["manifest"], data, run["weights"], out) == 2
    err = capsys.readouterr().err
    assert "1 item(s) are not in" in err and "('d', 'x9')" in err and "export the manifest from every row" in err
    assert not out.exists()


def test_the_unlisted_item_error_is_a_trained_item_error() -> None:
    assert issubclass(TrainingDataError, TrainedItemError)
    assert issubclass(TrainRunError, ValueError)


def test_a_dataset_differs_from_no_dataset(tmp_path: Path, run: dict, capsys) -> None:
    data = _data(tmp_path, [("d", "t0"), ("d", "t1"), (None, "t2")], "nodataset.jsonl")
    assert _record(run["manifest"], data, run["weights"], tmp_path / "run.json") == 2
    assert "(None, 't2')" in capsys.readouterr().err


def test_a_listed_item_without_a_row_is_exit_1(tmp_path: Path, run: dict, capsys) -> None:
    data = _data(tmp_path, [("d", "t0"), ("d", "t1")], "short.jsonl")
    assert _record(run["manifest"], data, run["weights"], tmp_path / "run.json") == 1
    err = capsys.readouterr().err
    assert "1 item(s) have no row in" in err and "('d', 't2')" in err


def test_inherited_items_may_appear_in_the_data(tmp_path: Path, run: dict) -> None:
    child = _manifest(tmp_path, ["n0", "n1"], checkpoint=CHILD, parent=run["manifest"], name="child.json")
    replay = _data(tmp_path, [("d", "n0"), ("d", "n1"), ("d", "t0")], "replay.jsonl")
    assert _record(child, replay, run["weights"], tmp_path / "child-run.json") == 0
    doc = json.loads((tmp_path / "child-run.json").read_text())
    assert (doc["manifest"]["items"], doc["manifest"]["inherited_items"], doc["data"]["items"]) == (2, 3, 3)


def test_parent_run_is_linked_by_id(tmp_path: Path, run: dict, capsys) -> None:
    parent = tmp_path / "parent-run.json"
    assert _full(run, parent) == 0
    child = _manifest(tmp_path, ["n0"], checkpoint=CHILD, parent=run["manifest"], name="child.json")
    data = _data(tmp_path, [("d", "n0")], "child.jsonl")
    out = tmp_path / "child-run.json"
    assert _record(child, data, run["weights"], out, "--parent-run", str(parent)) == 0
    linked = json.loads(out.read_text())["parent"]
    assert linked == {"trainrun_id": json.loads(parent.read_text())["trainrun_id"], "checkpoint": CHECKPOINT}

    capsys.readouterr()
    assert _full(run, tmp_path / "again.json", "--parent-run", str(parent)) == 1
    assert "is for the same checkpoint" in capsys.readouterr().err


def test_an_edited_parent_run_is_refused(tmp_path: Path, run: dict, capsys) -> None:
    parent = tmp_path / "parent-run.json"
    assert _full(run, parent) == 0
    doc = json.loads(parent.read_text())
    doc["summary"]["temperature"] = 1.0
    parent.write_text(json.dumps(doc))
    child = _manifest(tmp_path, ["n0"], checkpoint=CHILD, parent=run["manifest"], name="child.json")
    data = _data(tmp_path, [("d", "n0")], "child.jsonl")
    assert _record(child, data, run["weights"], tmp_path / "c.json", "--parent-run", str(parent)) == 1
    assert "does not match the record's contents" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"state": SECRET}], "line 1: missing item_id"),
        ([{"item_id": ""}], "line 1: item_id must be a non-empty string"),
        ([{"item_id": "t0", "dataset": 3}], "line 1: dataset must be a non-empty string, got a value of type int"),
        ([{"item_id": "t\udc80"}], "lone surrogate (U+DC80)"),
        ([], "no data rows"),
    ],
)
def test_malformed_data_rows_are_exit_1_naming_the_line(tmp_path: Path, run: dict, capsys, rows, message) -> None:
    data = tmp_path / "bad.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert _record(run["manifest"], str(data), run["weights"], tmp_path / "run.json") == 1
    err = capsys.readouterr().err
    assert str(data) in err and message in err
    assert SECRET not in err


def test_data_that_is_not_utf8_or_json_names_the_line(tmp_path: Path, run: dict, capsys) -> None:
    data = tmp_path / "bad.jsonl"
    data.write_bytes(b'{"item_id": "t0"}\n{"item_id": "\xff"}\n')
    assert _record(run["manifest"], str(data), run["weights"], tmp_path / "r.json") == 1
    assert "line 2: not valid UTF-8" in capsys.readouterr().err
    data.write_bytes(b'{"item_id": "t0"}\n[1]\n{"item_id": NaN}\n')
    assert _record(run["manifest"], str(data), run["weights"], tmp_path / "r.json") == 1
    assert "line 2: expected a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"lora": {"rank": 16}}', "'lora' is a dict"),
        ('{"layers": [1, 2]}', "'layers' is a list"),
        ('{"lr": NaN}', "non-finite number NaN"),
        ('{"lr": 1e400}', "'lr' must be finite"),
        ("[1]", "must be a JSON object"),
    ],
)
def test_trainer_settings_are_flat_scalars(tmp_path: Path, run: dict, capsys, content, message) -> None:
    trainer = tmp_path / "bad-trainer.json"
    trainer.write_text(content)
    assert _record(run["manifest"], run["data"], run["weights"], tmp_path / "r.json", "--trainer", str(trainer)) == 1
    err = capsys.readouterr().err
    assert str(trainer) in err and message in err


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"loss": 0.5}], "line 1: step must be a non-negative integer"),
        ([{"step": True}], "line 1: step must be a non-negative integer"),
        ([{"step": 2}, {"step": 2}], "line 2: step 2 does not follow step 2"),
        ([{"step": 0, "phase": "warmup"}], "line 1: metric 'phase' must be a finite number"),
    ],
)
def test_step_metrics_are_increasing_numbers(tmp_path: Path, run: dict, capsys, rows, message) -> None:
    steps = _jsonl(tmp_path / "bad-steps.jsonl", rows)
    assert _record(run["manifest"], run["data"], run["weights"], tmp_path / "r.json", "--steps", steps) == 1
    assert message in capsys.readouterr().err


def test_two_weights_with_one_name_are_refused(tmp_path: Path, run: dict, capsys) -> None:
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "head.pt").write_bytes(b"x")
    assert _record(run["manifest"], run["data"], [*run["weights"], str(tmp_path / "other" / "head.pt")], tmp_path / "r.json") == 1
    assert "two weights files are named 'head.pt'" in capsys.readouterr().err


def test_missing_inputs_name_the_producer(tmp_path: Path, run: dict, capsys) -> None:
    assert _record(tmp_path / "nope.json", run["data"], run["weights"], tmp_path / "r.json") == 1
    err = capsys.readouterr().err
    assert "training manifest not found" in err and "run `s1 trainset export` first" in err
    assert _record(run["manifest"], run["data"], run["weights"], tmp_path / "r.json", "--parent-run", str(tmp_path / "nope-run.json")) == 1
    assert "run `s1 trainset record` first" in capsys.readouterr().err


def test_the_output_must_not_be_an_input(tmp_path: Path, run: dict, capsys) -> None:
    before = Path(run["data"]).read_bytes()
    assert _record(run["manifest"], run["data"], run["weights"], Path(run["data"])) == 1
    assert "is the same file as" in capsys.readouterr().err
    assert Path(run["data"]).read_bytes() == before


def test_a_manifest_from_a_fifo_is_read_and_hashed_once(tmp_path: Path, run: dict) -> None:
    fifo = tmp_path / "manifest.fifo"
    os.mkfifo(fifo)
    data = run["manifest"].read_bytes()
    done = threading.Event()

    def serve() -> None:
        fifo.write_bytes(data)
        while not done.is_set():  # an empty stream to any later open, as `<(cat f)` gives
            try:
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                done.wait(0.01)

    threading.Thread(target=serve, daemon=True).start()
    try:
        assert _record(fifo, run["data"], run["weights"], tmp_path / "r.json") == 0
    finally:
        done.set()
    assert json.loads((tmp_path / "r.json").read_text())["manifest"]["sha256"] == hashlib.sha256(data).hexdigest()


# ---- a record is read as strictly as it is written (review of #149) ----


def _valid_doc(tmp_path: Path, run: dict) -> dict:
    out = tmp_path / "valid-run.json"
    assert _full(run, out) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _rehashed(doc: dict) -> dict:
    """`doc` with the id its contents hash to, as an edit that also recomputes the id would leave it."""
    return {**doc, "trainrun_id": _trainrun_id(doc)}


def _set(doc: dict, path: str, value) -> dict:
    out = json.loads(json.dumps(doc))
    at = out
    *parents, last = path.split(".")
    for key in parents:
        at = at[key]
    at[last] = value
    return out


def _without(doc: dict, key: str) -> dict:
    return {k: v for k, v in doc.items() if k != key}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: {**d, "state": SECRET}, "state: Extra inputs are not permitted"),
        (lambda d: _set(d, "checkpoint", "two words"), "checkpoint 'two words' is not a model name"),
        (lambda d: _without(d, "manifest"), "manifest: Field required"),
        (lambda d: _without(d, "steps"), "steps: Field required"),
        (lambda d: _set(d, "manifest.sha256", "nope"), "manifest.sha256: String should match pattern"),
        (lambda d: _set(d, "data.rows", 2.5), "data.rows: Input should be a valid integer"),
        (lambda d: _set(d, "data.items", 99), "data.items 99 exceeds data.rows 4"),
        (lambda d: _set(d, "weights_sha256", {}), "weights_sha256: Dictionary should have at least 1 item"),
        (lambda d: _set(d, "trainer.lora", {"rank": 16}), "trainer: trainer setting 'lora' is a dict"),
        (lambda d: _set(d, "trainer.state", SECRET), "trainer: a trainer setting cannot be named 'state'"),
        (lambda d: _set(d, "summary.state", SECRET), "summary: a summary metric cannot be named 'state'"),
        (lambda d: _set(d, "steps", [{"loss": 0.5}]), "steps.0: step must be a non-negative integer"),
        (lambda d: _set(d, "parent", {"trainrun_id": "tr-" + "0" * 16, "checkpoint": "two words"}), "parent.checkpoint 'two words' is not a model name"),
        (lambda d: _set(d, "parent", {"trainrun_id": "tr-" + "0" * 16, "checkpoint": CHECKPOINT}), "is for the same checkpoint"),
        (lambda d: _set(d, "trainrun_id", "tr-0"), "trainrun_id: String should match pattern"),
    ],
)
def test_a_record_whose_id_matches_is_still_checked_field_by_field(tmp_path: Path, run: dict, edit, message) -> None:
    doc = edit(_valid_doc(tmp_path, run))
    if "tr-0" != doc.get("trainrun_id"):
        doc = _rehashed(doc)
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(TrainRunError) as e:
        load_trainrun(path)
    assert str(e.value).startswith(f"{path}: ") and message in str(e.value)
    assert SECRET not in str(e.value)


@pytest.mark.parametrize("field", ["checkpoint", "created_at", "split"])
def test_a_lone_surrogate_in_a_record_is_a_one_line_refusal(tmp_path: Path, run: dict, capsys, field) -> None:
    doc = _set(_valid_doc(tmp_path, run), field, "x\ud800")
    parent = tmp_path / "surrogate.json"
    parent.write_text(json.dumps(doc), encoding="utf-8")  # ensure_ascii keeps it a \ud800 escape
    child = _manifest(tmp_path, ["n0"], checkpoint=CHILD, name="child.json")
    data = _data(tmp_path, [("d", "n0")], "child.jsonl")
    capsys.readouterr()
    assert _record(child, data, run["weights"], tmp_path / "c.json", "--parent-run", str(parent)) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"s1: {parent}: ") and err.count("\n") == 1
    assert not (tmp_path / "c.json").exists()


def test_an_overflowing_number_in_a_record_is_refused(tmp_path: Path, run: dict) -> None:
    text = json.dumps(_valid_doc(tmp_path, run)).replace('"temperature": 2.41', '"temperature": 1e400')
    assert "1e400" in text
    path = tmp_path / "inf.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TrainRunError, match="summary: summary metric 'temperature' must be finite"):
        load_trainrun(path)


def _build(tmp_path: Path, run: dict, **kw):
    manifest = load_manifest(run["manifest"])
    args = {"manifest_sha256": "ab" * 32, "data": run["data"], "weights": run["weights"], **kw}
    return build_trainrun(manifest, **args)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"summary": {"loss": float("nan")}}, "summary: summary metric 'loss' must be finite"),
        ({"summary": {"loss": float("inf")}}, "summary: summary metric 'loss' must be finite"),
        ({"trainer": {"lora": {"rank": 16}}}, "trainer: trainer setting 'lora' is a dict"),
        ({"trainer": {"note": "x\ud800"}}, "lone surrogate (U+D800)"),
        ({"trainer": {1: 2}}, "trainer.1.[key]: Input should be a valid string"),
        ({"trainer": {"state": SECRET}}, "trainer: a trainer setting cannot be named 'state'"),
        ({"summary": {"state": 1}}, "summary: a summary metric cannot be named 'state'"),
        ({"trainer": {"max_state": 4096, "State": "x"}, "weights": 5}, "weights must be a sequence of paths, got a int"),
        ({"weights": "head.pt"}, "weights must be a sequence of paths, got a str"),
        ({"weights": Path("head.pt")}, "weights must be a sequence of paths, got a PosixPath"),
        ({"weights": [5]}, "a weights file must be a path, got a int"),
        ({"weights": iter(())}, "a training run records at least one weights file"),
        ({"trainer": [1]}, "trainer: Input should be a valid dictionary"),
        ({"steps": [{"loss": 0.5}]}, "steps.0: step must be a non-negative integer"),
        ({"steps": [{"step": 1}, {"step": 1}]}, "steps.1: step 1 does not follow step 1"),
        ({"steps": ["x"]}, "steps.0: Input should be a valid dictionary"),
        ({"steps": 5}, "steps: Input should be a valid list"),
        ({"steps": None}, "steps: Input should be a valid list"),
        ({"steps": "abc"}, "steps.0: Input should be a valid dictionary"),
        ({"steps": {"step": 0}}, "steps.0: Input should be a valid dictionary"),
        ({"manifest_sha256": "nope"}, "manifest.sha256: String should match pattern"),
        ({"manifest_sha256": None}, "manifest.sha256: Input should be a valid string"),
    ],
)
def test_the_library_refuses_what_the_reader_would(tmp_path: Path, run: dict, kw, message) -> None:
    with pytest.raises(TrainRunError) as e:
        _build(tmp_path, run, **kw)
    assert message in str(e.value) and SECRET not in str(e.value)


def test_only_the_name_state_is_refused_and_weights_may_be_any_iterable_of_paths(tmp_path: Path, run: dict) -> None:
    built = _build(tmp_path, run, trainer={"max_state": 4096, "State": "x", "state_dict": "y"}, weights=(Path(w) for w in run["weights"]))
    assert sorted(built.doc["trainer"]) == ["State", "max_state", "state_dict"]
    assert sorted(built.doc["weights_sha256"]) == ["adapter.safetensors", "head.pt"]


@pytest.mark.parametrize(("flag", "what"), [("--trainer", "trainer setting"), ("--summary", "summary metric")])
def test_a_setting_named_state_is_refused_before_anything_is_written(tmp_path: Path, run: dict, capsys, flag: str, what: str) -> None:
    """A trainer row's `state` passed along by mistake: the Langfuse export copies these values out."""
    settings = tmp_path / "with-state.json"
    settings.write_text(json.dumps({"epochs": 3, "state": SECRET}))
    out = tmp_path / "r.json"
    assert _record(run["manifest"], run["data"], run["weights"], out, flag, str(settings)) == 1
    err = capsys.readouterr().err
    assert err == f"s1: {settings}: a {what} cannot be named 'state' (safe-v1: item text is never recorded)\n"
    assert not out.exists()


def test_steps_may_come_from_any_iterable(tmp_path: Path, run: dict) -> None:
    rows = [{"step": 0, "loss": 0.7}, {"step": 5, "loss": 0.3}]
    assert _build(tmp_path, run, steps=iter(rows)).doc["steps"] == rows
    assert _build(tmp_path, run, steps=iter(rows)).trainrun_id == _build(tmp_path, run, steps=rows).trainrun_id


def test_a_weights_name_that_is_not_utf8_is_refused_by_code_point(tmp_path: Path, run: dict) -> None:
    odd = os.fsdecode(os.fsencode(tmp_path) + b"/head-\xff.pt")  # surrogateescape: the byte becomes U+DCFF
    try:
        Path(odd).write_bytes(b"w")
    except OSError:
        pytest.skip("this filesystem refuses a file name that is not UTF-8")
    with pytest.raises(TrainRunError, match=r"a weights file name holds a lone surrogate \(U\+DCFF\)"):
        _build(tmp_path, run, weights=[odd])


def test_whatever_the_library_builds_it_reads_back(tmp_path: Path, run: dict) -> None:
    built = _build(tmp_path, run, trainer={"lr": 2e-4, "seed": None, "bf16": True}, summary={"t": 2.41}, steps=[{"step": 0, "loss": 0.7}])
    path = tmp_path / "sdk.json"
    save_trainrun(built, path)
    loaded = load_trainrun(path)
    assert (loaded.trainrun_id, loaded.checkpoint) == (built.trainrun_id, CHECKPOINT)
    assert loaded.doc == built.doc


def test_an_integer_past_the_float_range_is_not_a_setting(tmp_path: Path, run: dict, capsys) -> None:
    trainer = tmp_path / "big.json"
    trainer.write_text('{"seed": 1' + "0" * 400 + "}")
    assert _record(run["manifest"], run["data"], run["weights"], tmp_path / "r.json", "--trainer", str(trainer)) == 1
    err = capsys.readouterr().err
    assert str(trainer) in err and "trainer setting 'seed' must be finite" in err


def test_an_empty_id_is_called_empty(tmp_path: Path, run: dict, capsys) -> None:
    data = _jsonl(tmp_path / "empty-id.jsonl", [{"item_id": ""}])
    assert _record(run["manifest"], data, run["weights"], tmp_path / "r.json") == 1
    assert "item_id must be a non-empty string, got an empty string" in capsys.readouterr().err
