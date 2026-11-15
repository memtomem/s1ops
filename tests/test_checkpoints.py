"""`GET /api/checkpoints`: what a workspace knows about each checkpoint (#89, the read-only view).

Manifests and training-run records are found by name, each checkpoint's records are counted against
what it trained on by the rule `--training-manifest` refuses by, and nothing is refused or written."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records
from s1ops.contracts.schemas import PolicyEntry, save_policies
from s1ops.demo import generate_workspace
from s1ops.improve.checkpoints import checkpoints_report
from s1ops.improve.trainrun import load_trainrun, runs_by_checkpoint
from s1ops.improve.trainset import ManifestError, TrainedItemError, check_not_trained, load_manifest, load_manifest_hashed, trained_hits
from s1ops.web.app import MANIFEST_PATTERN, TRAINRUN_PATTERN, create_app
from tests.test_trainrun_shown import BASELINE, QID, R1, R2, SECRET, _row, _spread, _write

LABEL = "LABEL-ONLY-THE-RECORDS-HOLD"


def _export(source: str, out: Path, checkpoint: str, *flags: str) -> None:
    assert cli_main(["trainset", "export", source, "--split", "train", "--checkpoint", checkpoint, "-o", str(out), *flags]) == 0


def _record(manifest: Path, items: list[str], out: Path, *flags: str, weights: bytes = b"\x00w") -> None:
    d = out.parent / (out.stem + "-inputs")
    d.mkdir()
    data = _write(d / "train.jsonl", [{"dataset": "d", "item_id": i, "state": f"{SECRET} {i}", "label": LABEL} for i in items])
    (d / "head.pt").write_bytes(weights)
    (d / "trainer.json").write_text(json.dumps({"epochs": 3, "lr": 2e-4, "base": "kev-4b"}))
    (d / "summary.json").write_text(json.dumps({"wall_seconds": 61.5, "final_loss": 0.21}))
    _write(d / "steps.jsonl", [{"step": s, "loss": 1.0 / (s + 1), "lr": 2e-4} for s in range(0, 50, 10)])
    argv = ["trainset", "record", str(manifest), "--data", data, "--weights", str(d / "head.pt"), "--trainer", str(d / "trainer.json"),
            "--summary", str(d / "summary.json"), "--steps", str(d / "steps.jsonl"), "-o", str(out), *flags]
    assert cli_main(argv) == 0


def build_workspace(tmp_path: Path) -> Path:
    """Two checkpoints, the second started from the first, and records of both and of the model in place.

    `tickets-r1` trained on t0..t2; `tickets-r2` on u0..u1 and inherits r1's. `records.jsonl` holds
    held-out answers of all three models; `records_kev.jsonl` holds r1 on t0 and t1 (trained) and r2
    on t2 (inherited) and u0 (its own)."""
    root = tmp_path / "ws"
    root.mkdir()
    src1 = _write(tmp_path / "source1.jsonl", _spread("t", 3, "train", BASELINE))
    src2 = _write(tmp_path / "source2.jsonl", _spread("u", 2, "train", BASELINE))
    _export(src1, root / "training_manifest.json", R1)
    _export(src2, root / "training_manifest-r2.json", R2, "--parent", str(root / "training_manifest.json"))
    _record(root / "training_manifest.json", ["t0", "t1", "t2"], root / "trainrun.json")
    _record(root / "training_manifest-r2.json", ["u0", "u1"], root / "trainrun-r2.json", "--parent-run", str(root / "trainrun.json"), weights=b"\x00w2")
    held_out = [row for model in (BASELINE, R1, R2) for row in _spread("e", 6, "eval", model)]
    _write(root / "records.jsonl", held_out + [{**_row("e9", "eval", R1), "model_served": None, "error_code": "timeout"}])
    _write(root / "records_kev.jsonl", [_row("t0", "train", R1), _row("t1", "train", R1), _row("e0", "eval", R1),
                                        _row("t2", "train", R2), _row("u0", "train", R2), _row("e1", "eval", R2)])
    return root


def policy_entry(model: str, trainrun_id: str | None) -> PolicyEntry:
    return PolicyEntry(question_id=QID, model_served=model, primitive="noul", mode="error_budget", split="eval", n=6,
                       threshold=0.7, error_budget=0.05, trainrun_id=trainrun_id)


@pytest.fixture
def ws(tmp_path: Path, capsys) -> Path:
    root = build_workspace(tmp_path)
    capsys.readouterr()
    return root


def _get(root: Path, **kwargs) -> dict:
    r = TestClient(create_app(workspace_dir=root, **kwargs)).get("/api/checkpoints")
    assert r.status_code == 200, r.text
    return r.json()


def _by_name(d: dict) -> dict[str, dict]:
    return {c["checkpoint"]: c for c in d["checkpoints"]}


def _refused(root: Path) -> TestClient:
    return TestClient(create_app(workspace_dir=root), raise_server_exceptions=False)


# --- what is listed ----------------------------------------------------------------------------


def test_every_name_is_listed_trained_checkpoints_first(ws: Path) -> None:
    d = _get(ws)
    assert [c["checkpoint"] for c in d["checkpoints"]] == [R1, R2, BASELINE]
    assert d["summary"] == {"n_checkpoints": 3, "n_manifests": 2, "n_training_runs": 2}
    assert d["schema"] == "s1-checkpoints/1"
    files = d["display"]["files"]
    assert [Path(f).name for f in files["training_manifests"]] == ["training_manifest-r2.json", "training_manifest.json"]
    assert [Path(f).name for f in files["training_runs"]] == ["trainrun-r2.json", "trainrun.json"]
    assert [Path(f).name for f in files["records"]] == ["records.jsonl", "records_kev.jsonl"]
    assert files["policy"] is None
    assert d["display"]["patterns"] == {"training_manifests": MANIFEST_PATTERN, "training_runs": TRAINRUN_PATTERN}


def test_a_model_without_a_manifest_has_unknown_counts_not_zero(ws: Path) -> None:
    base = _by_name(_get(ws))[BASELINE]
    assert base["manifests"] == [] and base["run"] is None
    assert base["n_trained"] is None and base["n_trained_hits"] is None
    assert [g["n_trained_hits"] for g in base["groups"]] == [None]
    assert base["n_records"] == 6


def test_the_demo_workspace_lists_its_models(tmp_path: Path) -> None:
    generate_workspace(tmp_path / "w")
    d = _get(tmp_path / "w")
    assert d["summary"] == {"n_checkpoints": 3, "n_manifests": 0, "n_training_runs": 0}
    assert all(c["n_trained"] is None and c["run"] is None for c in d["checkpoints"])
    # The canary's silent upgrade is a name of its own.
    assert {c["checkpoint"] for c in d["checkpoints"]} == {"jev-1.13.0", "jev-1.14.0-rc1", "kev-1.0"}


def test_an_empty_workspace_is_an_empty_report(tmp_path: Path) -> None:
    d = _get(tmp_path)
    assert d["checkpoints"] == [] and d["summary"]["n_checkpoints"] == 0


def test_a_checkpoint_that_answered_nothing_is_listed_not_refused(ws: Path) -> None:
    # `--training-manifest` and `--training-run` exit 1 here; the view is where it shows.
    (ws / "records.jsonl").unlink()
    (ws / "records_kev.jsonl").unlink()
    r1 = _by_name(_get(ws))[R1]
    assert r1["n_records"] == 0 and r1["groups"] == [] and r1["n_trained"] == 3 and r1["n_trained_hits"] == 0


def test_failed_probes_are_no_ones_records(ws: Path) -> None:
    r1 = _by_name(_get(ws))[R1]
    by_file = {Path(g["file"]).name: g for g in r1["groups"]}
    assert by_file["records.jsonl"]["n_records"] == 6  # the failed e9 has no model
    assert r1["n_records"] == 9


# --- trained items -----------------------------------------------------------------------------


def test_the_counts_are_the_items_the_cli_refuses(ws: Path, capsys) -> None:
    r1 = _by_name(_get(ws))[R1]
    by_file = {Path(g["file"]).name: g for g in r1["groups"]}
    assert by_file["records.jsonl"]["n_trained_hits"] == 0 and by_file["records.jsonl"]["first_trained_hit"] is None
    kev = by_file["records_kev.jsonl"]
    assert (kev["question_id"], kev["primitive"], kev["n_records"], kev["n_items"]) == (QID, "noul", 3, 3)
    assert r1["n_trained"] == 3 and r1["n_trained_hits"] == kev["n_trained_hits"] == 2
    # `s1 audit --training-manifest` on the same file names the same count and first item.
    argv = ["audit", str(ws / "records_kev.jsonl"), "--model", R1, "--boot", "20", "--training-manifest", str(ws / "training_manifest.json")]
    assert cli_main(argv) == 2
    said = capsys.readouterr().err
    n, dataset, item_id = re.search(r": (\d+) item\(s\) of group .* first \('([^']+)', '([^']+)'\)", said).groups()
    assert int(n) == kev["n_trained_hits"] and kev["first_trained_hit"] == [dataset, item_id] == ["d", "t0"]
    # The held-out file passes the CLI, as the view's 0 says.
    assert cli_main(["audit", str(ws / "records.jsonl"), "--model", R1, "--boot", "20", "--training-manifest", str(ws / "training_manifest.json")]) == 0


def test_inherited_items_count_as_trained(ws: Path) -> None:
    r2 = _by_name(_get(ws))[R2]
    assert r2["n_trained"] == 5  # u0, u1 and r1's three
    kev = next(g for g in r2["groups"] if Path(g["file"]).name == "records_kev.jsonl")
    assert kev["n_trained_hits"] == 2 and kev["first_trained_hit"] == ["d", "t2"]
    (m,) = r2["manifests"]
    assert (m["n_rows"], m["n_items"], m["n_inherited"], m["split"]) == (2, 2, 3, "train")
    assert m["questions"] == [{"question_id": QID, "n_rows": 2}]


def test_two_manifests_for_a_checkpoint_add_up(ws: Path, tmp_path: Path, capsys) -> None:
    more = _write(tmp_path / "more.jsonl", _spread("e", 1, "train", BASELINE))
    _export(more, ws / "training_manifest-r1-more.json", R1)
    r1 = _by_name(_get(ws))[R1]
    assert len(r1["manifests"]) == 2 and r1["n_trained"] == 4
    held = next(g for g in r1["groups"] if Path(g["file"]).name == "records.jsonl")
    assert held["n_trained_hits"] == 1 and held["first_trained_hit"] == ["d", "e0"]


def test_a_checkpoints_count_is_distinct_items_not_the_sum_of_its_groups(ws: Path) -> None:
    # t0 and t1 again in the canary, and t0 under a second question in the shadow file.
    _write(ws / "records_canary.jsonl", [_row("t0", "prod", R1), _row("t1", "prod", R1)])
    other = {**_row("t0", "train", R1), "question_id": "q-bbbbbbbbbbbbbbbb"}
    with open(ws / "records_kev.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(other) + "\n")
    r1 = _by_name(_get(ws))[R1]
    assert [g["n_trained_hits"] for g in r1["groups"]] == [0, 2, 1, 2]
    assert r1["n_trained_hits"] == 2 <= r1["n_trained"] == 3
    # Every checkpoint of the report: never more hits than trained items.
    assert all(c["n_trained_hits"] <= c["n_trained"] for c in _get(ws)["checkpoints"] if c["n_trained"] is not None)


def test_trained_hits_is_the_refusals_rule() -> None:
    trained = {R1: frozenset({("d", "a"), (None, "b")})}
    items = [("d", "a"), ("d", "a"), (None, "b"), ("d", "b"), ("d", "c")]
    assert trained_hits((QID, R1, "noul"), items, trained) == [(None, "b"), ("d", "a")]
    assert trained_hits((QID, BASELINE, "noul"), items, trained) == []
    with pytest.raises(TrainedItemError, match=r"2 item\(s\) of group .* first \(None, 'b'\)"):
        check_not_trained((QID, R1, "noul"), items, trained, "records")
    check_not_trained((QID, BASELINE, "noul"), items, trained, "records")


# --- the run -----------------------------------------------------------------------------------


def test_the_run_is_the_record_as_it_is_on_disk(ws: Path) -> None:
    by = _by_name(_get(ws))
    run = by[R1]["run"]
    on_disk = json.loads((ws / "trainrun.json").read_text(encoding="utf-8"))
    assert run["record"] == on_disk and run["trainrun_id"] == on_disk["trainrun_id"]
    assert Path(run["path"]).name == "trainrun.json"
    assert run["manifest_loaded"] is True and run["parent_loaded"] is None
    assert [s["step"] for s in run["record"]["steps"]] == [0, 10, 20, 30, 40]
    child = by[R2]["run"]
    assert child["record"]["parent"] == {"trainrun_id": run["trainrun_id"], "checkpoint": R1}
    assert child["parent_loaded"] is True


def test_a_parent_whose_record_is_not_here_is_said(ws: Path) -> None:
    (ws / "trainrun.json").unlink()
    by = _by_name(_get(ws))
    assert by[R2]["run"]["parent_loaded"] is False
    assert by[R1]["run"] is None and by[R1]["n_trained"] == 3  # still a trained checkpoint, by its manifest


def test_a_manifest_exported_again_is_not_the_one_the_run_recorded(ws: Path, tmp_path: Path, capsys) -> None:
    again = _write(tmp_path / "again.jsonl", _spread("t", 3, "train", BASELINE) + _spread("x", 1, "eval", BASELINE))
    _export(again, ws / "training_manifest.json", R1)
    run = _by_name(_get(ws))[R1]["run"]
    assert run["manifest_loaded"] is False
    assert run["record"]["manifest"]["sha256"] != load_manifest_hashed(ws / "training_manifest.json")[1]


def test_a_run_without_a_manifest_is_shown_with_unknown_counts(ws: Path) -> None:
    (ws / "training_manifest.json").unlink()
    r1 = _by_name(_get(ws))[R1]
    assert r1["run"]["manifest_loaded"] is False and r1["manifests"] == []
    assert r1["n_trained"] is None and r1["n_trained_hits"] is None


# --- policies ----------------------------------------------------------------------------------


_policy = policy_entry


def test_a_policy_says_whether_it_names_the_recorded_run(ws: Path) -> None:
    r1_id = load_trainrun(ws / "trainrun.json").trainrun_id
    r2_id = load_trainrun(ws / "trainrun-r2.json").trainrun_id
    save_policies([_policy(R1, r1_id), _policy(R2, r1_id), _policy(BASELINE, None)], ws / "policy.json")
    d = _get(ws)
    by = _by_name(d)
    assert by[R1]["policies"] == [{"question_id": QID, "primitive": "noul", "trainrun_id": r1_id, "run_matches": True}]
    assert by[R2]["policies"][0]["run_matches"] is False and r1_id != r2_id
    assert by[BASELINE]["policies"][0] == {"question_id": QID, "primitive": "noul", "trainrun_id": None, "run_matches": None}
    assert Path(d["display"]["files"]["policy"]).name == "policy.json"


def test_a_policy_tuned_without_the_record_matches_nothing(ws: Path) -> None:
    save_policies([_policy(R1, None)], ws / "policy.json")
    assert _by_name(_get(ws))[R1]["policies"][0]["run_matches"] is None
    (ws / "trainrun.json").unlink()
    (ws / "trainrun-r2.json").unlink()
    save_policies([_policy(R1, "tr-0123456789abcdef")], ws / "policy.json")
    assert _by_name(_get(ws))[R1]["policies"][0]["run_matches"] is None  # no record to compare with


# --- what is read ------------------------------------------------------------------------------


def test_only_the_named_files_are_read(ws: Path) -> None:
    # Neither a backup nor another JSON file is taken for a manifest or a record.
    (ws / "training_manifest.json.bak").write_text("not json")
    (ws / "trainrun.json~").write_text("not json")
    (ws / "my_trainrun.json").write_text("not json")
    (ws / "TRAINRUN.json").write_text("not json")
    assert _get(ws)["summary"] == {"n_checkpoints": 3, "n_manifests": 2, "n_training_runs": 2}


def test_a_file_written_after_the_server_started_shows(ws: Path, tmp_path: Path, capsys) -> None:
    client = TestClient(create_app(workspace_dir=ws))
    assert client.get("/api/checkpoints").json()["summary"]["n_manifests"] == 2
    _export(_write(tmp_path / "s3.jsonl", _spread("v", 2, "train", BASELINE)), ws / "training_manifest-r3.json", "tickets-r3")
    d = client.get("/api/checkpoints").json()
    assert d["summary"]["n_manifests"] == 3 and "tickets-r3" in _by_name(d)


def test_paths_given_to_create_app_replace_the_search(ws: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.rename(ws / "training_manifest-r2.json", elsewhere / "m.json")
    os.rename(ws / "trainrun-r2.json", elsewhere / "r.json")
    d = _get(ws, training_manifest_paths=[elsewhere / "m.json", elsewhere / "absent.json"], training_run_paths=[elsewhere / "r.json"])
    assert d["summary"] == {"n_checkpoints": 3, "n_manifests": 1, "n_training_runs": 1}
    by = _by_name(d)
    assert by[R2]["run"] is not None and by[R1]["run"] is None and by[R1]["manifests"] == []
    assert d["display"]["files"]["training_manifests"] == [str(elsewhere / "m.json")]


def test_the_examples_folder_is_searched_when_the_workspace_has_none(ws: Path, tmp_path: Path) -> None:
    # The fallback every other workspace file has (`_resolve`), per kind of file.
    root = tmp_path / "repo"
    (root / "examples").mkdir(parents=True)
    os.rename(ws, root / "examples" / "data")
    d = _get(root)
    assert d["summary"] == {"n_checkpoints": 3, "n_manifests": 2, "n_training_runs": 2}
    os.rename(root / "examples" / "data" / "trainrun.json", root / "trainrun.json")
    d = _get(root)
    assert d["display"]["files"]["training_runs"] == [str(root / "trainrun.json")] and d["summary"]["n_manifests"] == 2


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root lists any directory")
def test_a_workspace_that_cannot_be_listed_is_400_naming_it(ws: Path) -> None:
    client = _refused(ws)
    os.chmod(ws, 0o300)
    try:
        r = client.get("/api/checkpoints")
    finally:
        os.chmod(ws, 0o700)
    assert r.status_code == 400 and r.json()["detail"].startswith(f"{ws}: cannot read: ")


def test_one_records_file_under_two_roles_is_read_once(ws: Path) -> None:
    d = _get(ws, compare_records_path=ws / "records.jsonl")
    assert [Path(f).name for f in d["display"]["files"]["records"]] == ["records.jsonl"]
    assert _by_name(d)[BASELINE]["n_records"] == 6


def test_the_canary_is_a_records_file_too(ws: Path) -> None:
    _write(ws / "records_canary.jsonl", [_row("t0", "prod", R1), _row("p0", "prod", R1)])
    r1 = _by_name(_get(ws))[R1]
    canary = next(g for g in r1["groups"] if Path(g["file"]).name == "records_canary.jsonl")
    assert (canary["n_records"], canary["n_trained_hits"]) == (2, 1)
    assert [Path(g["file"]).name for g in r1["groups"]] == ["records.jsonl", "records_kev.jsonl", "records_canary.jsonl"]


def test_the_report_holds_no_state_label_or_item_list(ws: Path) -> None:
    d = _get(ws)
    text = json.dumps(d)
    assert SECRET not in text and LABEL not in text and '"label"' not in text
    # A manifest's rows (ids and labels) stay in the manifest: the report counts them.
    for c in d["checkpoints"]:
        for m in c["manifests"]:
            assert set(m) == {"path", "sha256", "split", "source_sha256", "created_at", "n_rows", "n_items", "n_inherited", "questions"}
    # The only item ids are a group's first trained hit.
    ids = set(re.findall(r'"([tue]\d)"', text))
    assert ids == {c_id for c in d["checkpoints"] for g in c["groups"] if g["first_trained_hit"] for c_id in g["first_trained_hit"][1:]}


def test_the_route_is_read_only(ws: Path) -> None:
    app = create_app(workspace_dir=ws)
    before = {p.name: p.read_bytes() for p in ws.iterdir() if p.is_file()}
    client = TestClient(app)
    assert client.get("/api/checkpoints").status_code == 200
    assert client.post("/api/checkpoints").status_code == 405
    assert {p.name: p.read_bytes() for p in ws.iterdir() if p.is_file()} == before
    methods = {m for route in app.routes if getattr(route, "path", "").startswith("/api/") for m in route.methods}
    assert methods == {"GET"}


def test_the_report_is_the_librarys(ws: Path) -> None:
    manifests = [load_manifest_hashed(p) for p in sorted(ws.glob("training_manifest*.json"))]
    loaded = [load_trainrun(p) for p in sorted(ws.glob("trainrun*.json"))]
    runs = runs_by_checkpoint(loaded, [r.checkpoint for r in loaded], "the workspace")
    records = {str(ws / n): read_records(ws / n) for n in ("records.jsonl", "records_kev.jsonl")}
    served = _get(ws)
    served.pop("display")
    assert served == json.loads(json.dumps(checkpoints_report(manifests, runs, records)))


# --- refusals ----------------------------------------------------------------------------------


def test_a_malformed_manifest_is_400_naming_the_file_once(ws: Path) -> None:
    (ws / "training_manifest-r2.json").write_text('{"schema": "s1-training-manifest/1", "checkpoint": 7}')
    r = _refused(ws).get("/api/checkpoints")
    detail = r.json()["detail"]
    assert r.status_code == 400 and detail.count("training_manifest-r2.json") == 1 and "checkpoint" in detail


def _with_surrogate(doc: dict, field: str) -> None:
    if field in ("question_id", "item_id", "dataset"):
        doc["items"][0][field] = "x\ud800"
    elif field == "inherited_items":
        doc["inherited_items"][0]["item_id"] = "x\ud800"
    else:
        doc[field] = "x\ud800"


@pytest.mark.parametrize("field", ["created_at", "split", "checkpoint", "question_id", "item_id", "dataset", "inherited_items"])
def test_an_escaped_lone_surrogate_in_a_manifest_is_400_by_code_point(ws: Path, field: str) -> None:
    # `created_at` has no constraint, so the row model let it through and the response failed to encode (a 500).
    path = ws / "training_manifest-r2.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    _with_surrogate(doc, field)
    path.write_text(json.dumps(doc), encoding="ascii")  # the surrogate as a JSON escape
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400
    assert r.json()["detail"] == f"{path}: a string holds a lone surrogate (U+D800), which is not a character"
    with pytest.raises(ManifestError, match=r"lone surrogate \(U\+D800\)"):
        load_manifest(path)


def test_a_matching_file_name_that_is_not_utf8_is_400(ws: Path) -> None:
    os.symlink("trainrun.json", os.fsencode(ws) + b"/trainrun-\xff.json")
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and r.json()["detail"] == f"{ws}: a file matching trainrun*.json has a name that is not UTF-8"
    os.unlink(os.fsencode(ws) + b"/trainrun-\xff.json")
    # A name the view does not read may be anything.
    os.symlink("trainrun.json", os.fsencode(ws) + b"/notes-\xff.json")
    assert _refused(ws).get("/api/checkpoints").status_code == 200


def test_a_manifest_under_a_records_name_is_400(ws: Path) -> None:
    (ws / "trainrun-wrong.json").write_bytes((ws / "training_manifest.json").read_bytes())
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and r.json()["detail"].endswith("trainrun-wrong.json: not a s1-trainrun/1 file")


def test_an_edited_record_is_400(ws: Path) -> None:
    doc = json.loads((ws / "trainrun.json").read_text())
    doc["summary"]["final_loss"] = 0.01
    (ws / "trainrun.json").write_text(json.dumps(doc))
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and "does not match the record's contents" in r.json()["detail"]
    assert r.json()["detail"].count("trainrun.json") == 1


def test_two_runs_for_one_checkpoint_are_400_naming_the_second(ws: Path, capsys) -> None:
    _record(ws / "training_manifest.json", ["t0", "t1", "t2"], ws / "trainrun-other.json", weights=b"other weights")
    r = _refused(ws).get("/api/checkpoints")
    detail = r.json()["detail"]
    assert r.status_code == 400 and "is a second run for checkpoint 'tickets-r1'" in detail
    assert detail.startswith(str(ws / "trainrun.json")) and str(ws / "trainrun-other.json") in detail


def test_one_run_under_two_names_is_one_run(ws: Path) -> None:
    (ws / "trainrun-copy.json").write_bytes((ws / "trainrun.json").read_bytes())
    d = _get(ws)
    assert d["summary"]["n_training_runs"] == 2 and len(d["display"]["files"]["training_runs"]) == 3


@pytest.mark.parametrize("name", ["training_manifest.json", "trainrun.json"])
def test_a_directory_under_a_name_is_400(ws: Path, name: str) -> None:
    (ws / name).unlink()
    (ws / name).mkdir()
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and name in r.json()["detail"] and "cannot read" in r.json()["detail"]


@pytest.mark.parametrize("name", ["training_manifest-x.json", "trainrun-x.json"])
def test_a_dangling_link_is_400_and_a_fifo_is_400(ws: Path, name: str) -> None:
    (ws / name).symlink_to(ws / "nowhere.json")
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and r.json()["detail"] == f"{ws / name}: cannot read: dangling symbolic link"
    (ws / name).unlink()
    os.mkfifo(ws / name)
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and r.json()["detail"] == f"{ws / name}: not a regular file"


def test_a_malformed_records_file_or_policy_is_400_naming_it(ws: Path) -> None:
    (ws / "records_kev.jsonl").write_text("{not json}\n")
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and "records_kev.jsonl: line 1" in r.json()["detail"]
    (ws / "records_kev.jsonl").unlink()
    (ws / "policy.json").write_text("{}")
    r = _refused(ws).get("/api/checkpoints")
    assert r.status_code == 400 and "policy.json" in r.json()["detail"]
