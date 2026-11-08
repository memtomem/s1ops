"""`s1 trainset export` and `--training-manifest`: what a checkpoint trained on stays out of fitting and evaluation (#89)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records
from s1ops.eval.compare import compare_records
from s1ops.improve.trainset import TrainedItemError, load_manifest
from s1ops.monitor.runner import check_states

QID = "q-aaaaaaaaaaaaaaaa"
OTHER_QID = "q-bbbbbbbbbbbbbbbb"
CANDIDATE, BASELINE = "tickets-r1", "current-1"
SECRET = "SECRET-STATE-TEXT"


def _row(item: str, split: str | None, model: str = BASELINE, *, dataset: str | None = "d", qid: str = QID, label=True, p: float = 0.8) -> dict:
    row = {
        "schema": "s1-record/1",
        "item_id": item,
        "question_id": qid,
        "primitive": "noul",
        "model_requested": model,
        "model_served": model,
        "p": p,
        "dataset": dataset,
    }
    if label is not None:
        row["label"] = label
    if split is not None:
        row["split"] = split
    return row


def _spread(prefix: str, n: int, split: str | None, model: str = BASELINE, **kw) -> list[dict]:
    return [_row(f"{prefix}{i}", split, model, label=i % 2 == 0, p=0.2 + 0.6 * (i % 2 == 0) + 0.01 * i, **kw) for i in range(n)]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _export(records: str, out: Path, *flags: str, split: str = "train", checkpoint: str = CANDIDATE) -> int:
    return cli_main(["trainset", "export", records, "--split", split, "--checkpoint", checkpoint, "-o", str(out), *flags])


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    out = tmp_path / "m.json"
    assert _export(_write(tmp_path / "base.jsonl", _spread("t", 6, "train")), out) == 0
    return out


# ---- export ----


def test_export_writes_the_split_labels_and_fingerprint(tmp_path: Path, capsys) -> None:
    rows = _spread("t", 4, "train") + _spread("c", 2, "cal") + [_row("u", "train", label=None)]
    source = _write(tmp_path / "r.jsonl", rows)
    assert _export(source, tmp_path / "m.json") == 0
    m = load_manifest(tmp_path / "m.json")
    assert [(i.dataset, i.item_id, i.question_id, i.label) for i in m.items] == [
        ("d", "t0", QID, True),
        ("d", "t1", QID, False),
        ("d", "t2", QID, True),
        ("d", "t3", QID, False),
    ]
    assert (m.checkpoint, m.split, m.inherited_items) == (CANDIDATE, "train", ())
    assert m.source_sha256 == hashlib.sha256(Path(source).read_bytes()).hexdigest()
    summary = json.loads(capsys.readouterr().out)
    assert summary["skipped"] == {"no_response": 0, "unlabeled": 1, "other_split": 2}
    assert (summary["items"], summary["questions"]) == (4, 1)
    assert "also_in_other_splits" not in summary and "answered_as_checkpoint" not in summary


def test_label_types_survive_the_file(tmp_path: Path) -> None:
    choice = {**_row("c", "train", qid="q-c"), "primitive": "choice", "choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}, "label": "b"}
    score = {**_row("s", "train", qid="q-s"), "primitive": "score", "score": 1.0, "confidence": 0.9, "probabilities": {"0": 0.1, "1": 0.9}, "label": 1}
    assert _export(_write(tmp_path / "r.jsonl", [choice, score, _row("n", "train", label=False)]), tmp_path / "m.json") == 0
    labels = {i.item_id: i.label for i in load_manifest(tmp_path / "m.json").items}
    assert labels == {"c": "b", "s": 1, "n": False}
    assert type(labels["s"]) is int and type(labels["n"]) is bool


def test_a_manifest_holds_no_state_or_wording(tmp_path: Path) -> None:
    row = {**_row("t0", "train"), "state": SECRET, "instructions": SECRET, "note": SECRET, "state_sha256": SECRET}
    assert _export(_write(tmp_path / "r.jsonl", [row]), tmp_path / "m.json") == 0
    assert SECRET not in (tmp_path / "m.json").read_text(encoding="utf-8")
    assert set(json.loads((tmp_path / "m.json").read_text())["items"][0]) == {"dataset", "item_id", "question_id", "label"}


def test_one_item_answered_by_two_models_is_one_row(tmp_path: Path) -> None:
    rows = [_row("t0", "train", "a"), _row("t0", "train", "b")]
    assert _export(_write(tmp_path / "r.jsonl", rows), tmp_path / "m.json") == 0
    assert len(load_manifest(tmp_path / "m.json").items) == 1


def test_two_labels_for_one_item_and_question_are_refused(tmp_path: Path, capsys) -> None:
    source = _write(tmp_path / "r.jsonl", [_row("t0", "train", "a", label=True), _row("t0", "train", "b", label=False)])
    assert _export(source, tmp_path / "m.json") == 1
    err = capsys.readouterr().err
    assert source in err and "line 2" in err and "line 1" in err
    assert not (tmp_path / "m.json").exists()


def test_true_and_one_are_two_labels(tmp_path: Path, capsys) -> None:
    score = {**_row("t0", "train", "b"), "primitive": "score", "score": 1.0, "confidence": 0.9, "probabilities": {"0": 0.1, "1": 0.9}, "label": 1}
    source = _write(tmp_path / "r.jsonl", [_row("t0", "train", "a", label=True), score])
    assert _export(source, tmp_path / "m.json") == 1
    assert "labeled" in capsys.readouterr().err


def test_a_failed_probe_gives_no_row(tmp_path: Path, capsys) -> None:
    failed = {**_row("t0", "train"), "model_served": None, "error_code": "timeout"}
    source = _write(tmp_path / "r.jsonl", [failed, _row("t1", "train")])
    assert _export(source, tmp_path / "m.json") == 0
    assert [i.item_id for i in load_manifest(tmp_path / "m.json").items] == ["t1"]
    assert json.loads(capsys.readouterr().out)["skipped"]["no_response"] == 1


def test_no_rows_is_an_error_that_says_what_was_skipped(tmp_path: Path, capsys) -> None:
    assert _export(_write(tmp_path / "r.jsonl", _spread("c", 2, "cal")), tmp_path / "m.json") == 1
    assert "2 in other splits" in capsys.readouterr().err


def test_a_name_no_server_could_report_is_refused(tmp_path: Path, capsys) -> None:
    source = _write(tmp_path / "r.jsonl", [_row("t0", "train")])
    assert _export(source, tmp_path / "m.json", checkpoint="two words") == 1
    err = capsys.readouterr().err
    assert "two words" in err and "1-128 characters" in err


def test_reused_name_and_other_splits_are_reported_not_refused(tmp_path: Path, capsys) -> None:
    rows = [_row("t0", "train"), _row("t0", "cal"), _row("t1", "train", CANDIDATE)]
    assert _export(_write(tmp_path / "r.jsonl", rows), tmp_path / "m.json") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["answered_as_checkpoint"] == 1
    assert summary["also_in_other_splits"] == {"n": 1, "first": ["d", "t0"]}


def test_a_fifo_source_is_refused(tmp_path: Path) -> None:
    source = _write(tmp_path / "r.jsonl", [_row("t0", "train")])
    script = f"s1=({_python_s1()}); \"${{s1[@]}}\" trainset export <(cat '{source}') --split train --checkpoint c -o '{tmp_path / 'm.json'}'"
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and "regular file" in done.stderr
    assert not (tmp_path / "m.json").exists()


def _python_s1() -> str:
    return f"'{sys.executable}' -c 'from s1ops.cli import main; import sys; sys.exit(main())'"


def test_output_cannot_replace_an_input(tmp_path: Path, capsys) -> None:
    source = tmp_path / "r.jsonl"
    before = _write(source, [_row("t0", "train")]) and source.read_bytes()
    assert _export(str(source), source) == 1
    assert source.read_bytes() == before
    assert "same file" in capsys.readouterr().err


def test_parent_items_are_inherited_through_a_chain(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    assert _export(_write(tmp_path / "ra.jsonl", [_row("x", "train"), _row("shared", "train")]), a, checkpoint="r1") == 0
    assert _export(_write(tmp_path / "rb.jsonl", [_row("y", "train"), _row("shared", "train")]), b, "--parent", str(a), checkpoint="r2") == 0
    assert _export(_write(tmp_path / "rc.jsonl", [_row("z", "train")]), c, "--parent", str(b), checkpoint="r3") == 0
    mb, mc = load_manifest(b), load_manifest(c)
    assert mb.inherited_items == (("d", "x"),)
    assert mc.inherited_items == (("d", "shared"), ("d", "x"), ("d", "y"))
    assert mc.trained == {("d", "x"), ("d", "y"), ("d", "z"), ("d", "shared")}


def test_a_parent_of_the_same_name_is_refused(tmp_path: Path, manifest: Path, capsys) -> None:
    source = _write(tmp_path / "r.jsonl", [_row("y", "train")])
    assert _export(source, tmp_path / "m2.json", "--parent", str(manifest)) == 1
    assert "same checkpoint" in capsys.readouterr().err


# ---- loading ----


def _good() -> dict:
    return {
        "schema": "s1-training-manifest/1",
        "checkpoint": CANDIDATE,
        "split": "train",
        "source_sha256": "0" * 64,
        "created_at": "2026-11-08T00:00:00+00:00",
        "items": [{"dataset": None, "item_id": "t0", "question_id": QID, "label": True}],
        "inherited_items": [],
    }


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        (lambda d: d.update(checkpoint=1), "checkpoint"),
        (lambda d: d.update(checkpoint="two words"), "two words"),
        (lambda d: d.update(source_sha256="abc"), "source_sha256"),
        (lambda d: d.update(items=[]), "items"),
        (lambda d: d["items"][0].update(item_id=7), "items.0.item_id"),
        (lambda d: d["items"][0].update(dataset=""), "items.0.dataset"),
        (lambda d: d["items"][0].pop("label"), "items.0.label"),
        (lambda d: d["items"][0].update(label=None), "items.0.label"),
        (lambda d: d["items"][0].update(state=SECRET), "items.0.state"),
        (lambda d: d.update(state=SECRET), "state"),
        (lambda d: d.update(inherited_items=[["d", "x"]]), "inherited_items.0"),
        (lambda d: d["items"].append(dict(d["items"][0])), "appears twice"),
        (lambda d: d.update(schema="other/1"), "not a s1-training-manifest/1"),
    ],
)
def test_a_malformed_manifest_names_its_file_and_field(tmp_path: Path, capsys, edit, named: str) -> None:
    doc = _good()
    edit(doc)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    source = _write(tmp_path / "r.jsonl", _spread("c", 4, "cal", CANDIDATE))
    assert cli_main(["fit", source, "-o", str(tmp_path / "f.json"), "--training-manifest", str(bad)]) == 1
    err = capsys.readouterr().err
    assert str(bad) in err and named in err
    assert len(err.strip().splitlines()) == 1


def test_a_missing_manifest_names_the_command_that_writes_it(tmp_path: Path, capsys) -> None:
    source = _write(tmp_path / "r.jsonl", _spread("c", 4, "cal", CANDIDATE))
    assert cli_main(["fit", source, "-o", str(tmp_path / "f.json"), "--training-manifest", str(tmp_path / "none.json")]) == 1
    assert "s1 trainset export" in capsys.readouterr().err


# ---- the refusal ----


def _commands(records: str, tmp_path: Path) -> dict[str, list[str]]:
    return {
        "fit": ["fit", records, "-o", str(tmp_path / "f.json")],
        "audit": ["audit", records, "--boot", "5"],
        "tune": ["policy", "tune", records, "-o", str(tmp_path / "p.json")],
        "curve": ["policy", "curve", records],
    }


@pytest.mark.parametrize("command", ["fit", "audit", "tune", "curve"])
def test_each_command_refuses_trained_items_with_exit_2(tmp_path: Path, manifest: Path, capsys, command: str) -> None:
    records = _write(tmp_path / "c.jsonl", _spread("t", 6, None, CANDIDATE))
    argv = _commands(records, tmp_path)[command]
    assert cli_main(argv) == 0  # without the flag, nothing is checked
    capsys.readouterr()
    assert cli_main([*argv, "--training-manifest", str(manifest)]) == 2
    err = capsys.readouterr().err
    assert records in err and CANDIDATE in err and "6 item(s)" in err and "'t0'" in err


@pytest.mark.parametrize("command", ["fit", "audit", "tune", "curve"])
def test_each_command_passes_held_out_items(tmp_path: Path, manifest: Path, capsys, command: str) -> None:
    records = _write(tmp_path / "c.jsonl", _spread("h", 6, None, CANDIDATE))
    assert cli_main([*_commands(records, tmp_path)[command], "--training-manifest", str(manifest)]) == 0


def test_one_trained_item_among_held_out_ones_is_enough(tmp_path: Path, manifest: Path, capsys) -> None:
    records = _write(tmp_path / "c.jsonl", [*_spread("h", 6, None, CANDIDATE), _row("t3", None, CANDIDATE)])
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(manifest)]) == 2
    assert "1 item(s)" in capsys.readouterr().err


def test_another_models_answers_on_those_items_pass(tmp_path: Path, manifest: Path) -> None:
    rows = [*_spread("t", 6, None, BASELINE), *_spread("h", 6, None, CANDIDATE)]
    records = _write(tmp_path / "c.jsonl", rows)
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(manifest)]) == 0


def test_a_filter_decides_which_rows_are_checked(tmp_path: Path, manifest: Path) -> None:
    rows = [*_spread("t", 6, "train", CANDIDATE), *_spread("h", 6, "cal", CANDIDATE)]
    records = _write(tmp_path / "c.jsonl", rows)
    flag = ["--training-manifest", str(manifest)]
    assert cli_main(["fit", records, "--split", "cal", "-o", str(tmp_path / "f.json"), *flag]) == 0
    assert cli_main(["fit", records, "--split", "train", "-o", str(tmp_path / "f.json"), *flag]) == 2
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), *flag]) == 2


@pytest.mark.parametrize("command", ["fit", "audit", "tune", "curve"])
def test_selecting_the_baseline_of_a_mixed_file_checks_no_candidate_row(tmp_path: Path, manifest: Path, command: str) -> None:
    rows = [*_spread("t", 6, None, BASELINE), *_spread("t", 6, None, CANDIDATE)]
    argv = _commands(_write(tmp_path / "c.jsonl", rows), tmp_path)[command]
    flag = ["--training-manifest", str(manifest)]
    assert cli_main([*argv, "--model", BASELINE, *flag]) == 0
    assert cli_main([*argv, *flag]) == 2


@pytest.mark.parametrize("command", ["fit", "audit", "tune", "curve"])
def test_selecting_another_question_checks_no_row_of_the_trained_one(tmp_path: Path, manifest: Path, command: str) -> None:
    rows = [*_spread("t", 6, None, CANDIDATE), *_spread("h", 6, None, CANDIDATE, qid=OTHER_QID)]
    argv = _commands(_write(tmp_path / "c.jsonl", rows), tmp_path)[command]
    flag = ["--training-manifest", str(manifest)]
    assert cli_main([*argv, "--question", OTHER_QID, *flag]) == 0
    assert cli_main([*argv, "--question", QID, *flag]) == 2


def test_a_reworded_question_on_a_trained_item_is_still_refused(tmp_path: Path, manifest: Path, capsys) -> None:
    records = _write(tmp_path / "c.jsonl", _spread("t", 6, None, CANDIDATE, qid=OTHER_QID))
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(manifest)]) == 2
    assert OTHER_QID in capsys.readouterr().err


def test_a_null_dataset_is_not_a_named_one(tmp_path: Path, manifest: Path) -> None:
    records = _write(tmp_path / "c.jsonl", _spread("t", 6, None, CANDIDATE, dataset=None))
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(manifest)]) == 0
    other = tmp_path / "null.json"
    assert _export(_write(tmp_path / "n.jsonl", _spread("t", 6, "train", dataset=None)), other) == 0
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(other)]) == 2


def test_inherited_items_are_refused(tmp_path: Path, manifest: Path) -> None:
    parent = tmp_path / "parent.json"
    assert _export(_write(tmp_path / "p.jsonl", _spread("old", 6, "train")), parent, checkpoint="r0") == 0
    child = tmp_path / "child.json"
    assert _export(_write(tmp_path / "ch.jsonl", _spread("new", 6, "train")), child, "--parent", str(parent)) == 0
    records = _write(tmp_path / "c.jsonl", _spread("old", 6, None, CANDIDATE))
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(child)]) == 2


def test_a_manifest_naming_no_model_in_the_file_is_exit_1(tmp_path: Path, manifest: Path, capsys) -> None:
    records = _write(tmp_path / "c.jsonl", _spread("t", 6, None, "tickets_r1"))
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(manifest)]) == 1
    err = capsys.readouterr().err
    assert str(manifest) in err and records in err and "tickets_r1" in err


def test_an_all_failed_file_names_no_model(tmp_path: Path, manifest: Path, capsys) -> None:
    failed = {**_row("t0", None), "model_served": None, "error_code": "timeout", "model_requested": CANDIDATE}
    records = _write(tmp_path / "c.jsonl", [failed])
    assert cli_main(["audit", records, "--training-manifest", str(manifest)]) == 1
    assert "answered no record" in capsys.readouterr().err


@pytest.fixture
def two_manifests(tmp_path: Path, manifest: Path) -> tuple[Path, Path]:
    second = tmp_path / "m2.json"
    assert _export(_write(tmp_path / "b2.jsonl", _spread("u", 6, "train")), second) == 0
    return manifest, second


@pytest.mark.parametrize("prefix", ["t", "u"])
@pytest.mark.parametrize("reverse", [False, True])
def test_two_manifests_for_a_checkpoint_add_up(tmp_path: Path, two_manifests, prefix: str, reverse: bool) -> None:
    first, second = reversed(two_manifests) if reverse else two_manifests
    flags = ["--training-manifest", str(first), "--training-manifest", str(second)]
    clean = _write(tmp_path / "clean.jsonl", _spread("h", 6, None, CANDIDATE))
    assert cli_main(["fit", clean, "-o", str(tmp_path / "f.json"), *flags]) == 0
    hit = _write(tmp_path / "hit.jsonl", _spread(prefix, 6, None, CANDIDATE))
    assert cli_main(["fit", hit, "-o", str(tmp_path / "f.json"), *flags]) == 2
    baseline = _write(tmp_path / "base.jsonl", _spread(prefix, 6, None, BASELINE))
    assert cli_main(["compare", baseline, hit, *flags]) == 2


def test_a_refused_fit_writes_nothing(tmp_path: Path, manifest: Path) -> None:
    records = _write(tmp_path / "c.jsonl", _spread("t", 6, None, CANDIDATE))
    assert cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(manifest)]) == 2
    assert not (tmp_path / "f.json").exists()


def test_debug_mode_raises_the_library_error(tmp_path: Path, manifest: Path, monkeypatch) -> None:
    monkeypatch.setenv("S1_DEBUG", "1")
    records = _write(tmp_path / "c.jsonl", _spread("t", 6, None, CANDIDATE))
    with pytest.raises(TrainedItemError):
        cli_main(["fit", records, "-o", str(tmp_path / "f.json"), "--training-manifest", str(manifest)])


# ---- compare ----


def test_compare_refuses_a_pair_on_a_trained_item(tmp_path: Path, manifest: Path, capsys) -> None:
    primary = _write(tmp_path / "p.jsonl", _spread("t", 6, None, BASELINE))
    shadow = _write(tmp_path / "s.jsonl", _spread("t", 6, None, CANDIDATE))
    assert cli_main(["compare", primary, shadow]) == 0
    capsys.readouterr()
    assert cli_main(["compare", primary, shadow, "--training-manifest", str(manifest)]) == 2
    assert shadow in capsys.readouterr().err


def test_compare_checks_the_primary_side_too(tmp_path: Path, manifest: Path, capsys) -> None:
    primary = _write(tmp_path / "p.jsonl", _spread("t", 6, None, CANDIDATE))
    shadow = _write(tmp_path / "s.jsonl", _spread("t", 6, None, BASELINE))
    assert cli_main(["compare", primary, shadow, "--training-manifest", str(manifest)]) == 2
    assert primary in capsys.readouterr().err


def test_compare_does_not_refuse_an_unshared_trained_item(tmp_path: Path, manifest: Path, capsys) -> None:
    primary = _write(tmp_path / "p.jsonl", _spread("h", 6, None, BASELINE))
    shadow = _write(tmp_path / "s.jsonl", [*_spread("h", 6, None, CANDIDATE), *_spread("t", 6, None, CANDIDATE)])
    assert cli_main(["compare", primary, shadow, "--training-manifest", str(manifest)]) == 0
    assert json.loads(capsys.readouterr().out)["shared_pairs"] == 6


def test_compare_needs_one_side_to_name_the_checkpoint(tmp_path: Path, manifest: Path, capsys) -> None:
    primary = _write(tmp_path / "p.jsonl", _spread("t", 6, None, BASELINE))
    shadow = _write(tmp_path / "s.jsonl", _spread("t", 6, None, "another-model"))
    assert cli_main(["compare", primary, shadow, "--training-manifest", str(manifest)]) == 1
    err = capsys.readouterr().err
    assert primary in err and shadow in err


def test_compare_takes_one_manifest_per_side(tmp_path: Path, manifest: Path) -> None:
    other = tmp_path / "other.json"
    assert _export(_write(tmp_path / "o.jsonl", _spread("x", 6, "train")), other, checkpoint="tickets-r2") == 0
    primary = _write(tmp_path / "p.jsonl", _spread("h", 6, None, CANDIDATE))
    shadow = _write(tmp_path / "s.jsonl", _spread("h", 6, None, "tickets-r2"))
    assert cli_main(["compare", primary, shadow, "--training-manifest", str(manifest), "--training-manifest", str(other)]) == 0


def test_compare_records_takes_manifests_from_the_sdk(tmp_path: Path, manifest: Path) -> None:
    primary = _spread("t", 6, None, BASELINE)
    shadow = _spread("t", 6, None, CANDIDATE)
    with pytest.raises(TrainedItemError, match="shadow records"):
        compare_records(primary, shadow, training_manifests=[manifest])
    assert compare_records(primary, shadow)["shared_pairs"] == 6


# ---- probe copies split ----


def test_a_states_row_may_carry_a_split(tmp_path: Path, capsys) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"urgent": {"type": "noul", "instructions": "Is this urgent?"}}), encoding="utf-8")
    states = tmp_path / "s.jsonl"
    _write(states, [
        {"item_id": "a", "state": SECRET, "split": "train", "label": True},
        {"item_id": "b", "state": SECRET},
    ])
    out = tmp_path / "r.jsonl"
    assert cli_main(["probe", str(questions), "--no-expand-probes", "--states", str(states), "-o", str(out), "--ledger", str(tmp_path / "l.jsonl")]) == 0
    rows = [r for r in read_records(out)]
    assert [(r.item_id, r.split) for r in rows] == [("a", "train"), ("b", None)]
    assert "split" not in rows[1].raw


@pytest.mark.parametrize("bad", ["", 3, ["train"], True])
def test_a_malformed_split_is_refused_before_anything_is_sent(bad) -> None:
    with pytest.raises(ValueError, match="row 1: 'split'"):
        check_states([{"item_id": "a", "state": "x", "split": bad}])


# ---- the documented sequence ----


@pytest.mark.parametrize("wrapped", [False, True])
def test_the_readme_recipe_builds_rows_from_the_manifest(tmp_path: Path, monkeypatch, wrapped: bool) -> None:
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Fine-Tuning Around an External Trainer", 1)[1]
    recipe = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    question = {"type": "noul", "instructions": "Is this urgent?", "x-s1-lint": {"ignore": ["JL012"]}}
    questions = {"urgent": question}
    (tmp_path / "questions.json").write_text(json.dumps({"questions": questions} if wrapped else questions), encoding="utf-8")
    states = tmp_path / "states_local.jsonl"
    _write(states, [
        {"item_id": "a", "state": "state of a", "split": "train", "label": True},
        {"item_id": "b", "dataset": None, "state": "state of b", "split": "train", "label": False},
        {"item_id": "c", "state": "state of c", "split": "eval", "label": False},
    ])
    probed = tmp_path / "current.jsonl"
    ledger = tmp_path / "l.jsonl"
    assert cli_main(["probe", str(tmp_path / "questions.json"), "--no-expand-probes", "--states", str(states), "-o", str(probed), "--ledger", str(ledger)]) == 0
    assert _export(str(probed), tmp_path / "training_manifest.json") == 0
    monkeypatch.chdir(tmp_path)
    exec(compile(recipe, "README recipe", "exec"), {})
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    qid = read_records(probed)[0].question_id
    assert sorted(rows, key=lambda r: r["state"]) == [
        {"state": "state of a", "questions": {qid: {"type": "noul", "instructions": "Is this urgent?", "label": True}}},
        {"state": "state of b", "questions": {qid: {"type": "noul", "instructions": "Is this urgent?", "label": False}}},
    ]


def test_the_documented_promotion_sequence_runs(tmp_path: Path, capsys) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"urgent": {"type": "noul", "instructions": "Is this urgent?"}}), encoding="utf-8")
    rows = [
        {"dataset": "d", "item_id": f"i{i}", "state": f"body {i}", "split": split, "label": i % 3 != 0}
        for split, span in (("train", range(0, 60)), ("cal", range(60, 120)), ("eval", range(120, 180)))
        for i in span
    ]
    every = _write(tmp_path / "states.jsonl", rows)
    cal = _write(tmp_path / "states_cal.jsonl", [r for r in rows if r["split"] == "cal"])
    held = _write(tmp_path / "states_eval.jsonl", [r for r in rows if r["split"] == "eval"])
    ledger = str(tmp_path / "ledger.jsonl")

    def probe(model: str, states: str, out: str) -> str:
        path = str(tmp_path / out)
        assert cli_main(["probe", str(questions), "--no-expand-probes", "--model", model, "--states", states, "-o", path, "--ledger", ledger]) == 0
        return path

    current = probe(BASELINE, every, "current.jsonl")
    current_eval = probe(BASELINE, held, "current_eval.jsonl")
    m = str(tmp_path / "training_manifest.json")
    assert _export(current, Path(m)) == 0
    assert [i.item_id for i in load_manifest(m).items][:1] == ["i0"] and len(load_manifest(m).items) == 60
    candidate_cal = probe(CANDIDATE, cal, "candidate_cal.jsonl")
    candidate_eval = probe(CANDIDATE, held, "candidate_eval.jsonl")
    flag = ["--training-manifest", m]

    assert cli_main(["compare", current, candidate_eval, *flag]) == 0
    fits = str(tmp_path / "fits.json")
    assert cli_main(["fit", candidate_cal, *flag, "-o", fits]) == 0
    applied = str(tmp_path / "candidate_eval_calibrated.jsonl")
    assert cli_main(["apply", candidate_eval, "--fit", fits, "-o", applied]) == 0
    assert cli_main(["policy", "tune", applied, "--use-calib", *flag, "-o", str(tmp_path / "policy.json")]) == 0

    current_fits = str(tmp_path / "current_fits.json")
    assert cli_main(["fit", current, "--split", "cal", "-o", current_fits]) == 0
    assert cli_main(["apply", candidate_eval, "--fit", current_fits, "-o", str(tmp_path / "x.jsonl")]) == 3

    capsys.readouterr()
    assert cli_main(["drift", current_eval, candidate_eval]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [q["status"] for q in report["questions"]] == ["CRITICAL"]
    assert "model_served changed" in report["questions"][0]["alerts"][0]
    assert cli_main(["drift", current_eval, candidate_eval, "--fail-on-critical"]) == 4
