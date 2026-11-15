"""What a workspace knows about each checkpoint: its manifests, its training-run record, the records
it answered and how many of them are on items it trained on, and the policies tuned for it.

Read-only, for the workbench's checkpoint view (`GET /api/checkpoints`): it refuses nothing and
starts no training. A checkpoint is a name: `model_served` in records, `checkpoint` in a manifest or
a record of a run. The trained-item count of a group is `trainset.trained_hits`, the rule
`--training-manifest` refuses by, so the view counts exactly the items `s1 fit`, `audit`, `policy`
and `compare` would name. Like a manifest and a run record, the report holds ids, counts, digests
and the trainer's numbers; never `state` text, labels or question wording.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from s1ops.contracts.records import ItemKey, Record
from s1ops.contracts.schemas import PolicyEntry
from s1ops.improve.trainrun import TrainRuns
from s1ops.improve.trainset import TrainingManifest, trained_hits

CHECKPOINTS_SCHEMA = "s1-checkpoints/1"


def _path(path: str | Path | None) -> str | None:
    return None if path is None else str(path)


def _manifest_row(manifest: TrainingManifest, sha256: str) -> dict:
    per_question: dict[str, int] = {}
    for item in manifest.items:
        per_question[item.question_id] = per_question.get(item.question_id, 0) + 1
    return {
        "path": _path(manifest.path),
        "sha256": sha256,
        "split": manifest.split,
        "source_sha256": manifest.source_sha256,
        "created_at": manifest.created_at,
        "n_rows": len(manifest.items),
        "n_items": len({i.item for i in manifest.items}),
        "n_inherited": len(manifest.inherited_items),
        "questions": [{"question_id": q, "n_rows": n} for q, n in sorted(per_question.items())],
    }


def checkpoints_report(
    manifests: Sequence[tuple[TrainingManifest, str]],
    runs: TrainRuns,
    records: Mapping[str, Sequence[Record]],
    policies: Sequence[PolicyEntry] = (),
) -> dict:
    """One entry per checkpoint name the inputs hold.

    `manifests` are `(manifest, SHA-256 of its file)` pairs (`load_manifest_hashed`); several for one
    checkpoint add up, as under `--training-manifest`. `runs` is `runs_by_checkpoint`'s result, so
    one run per checkpoint. `records` maps a records file's name to its records, in the order the
    report lists them. A checkpoint with a manifest or a run comes first; a model that only answered
    records follows, with no manifest: what it trained on is unknown here, so its counts of trained
    items are `None`, never 0.
    """
    by_checkpoint: dict[str, list[tuple[TrainingManifest, str]]] = {}
    for manifest, sha256 in manifests:
        by_checkpoint.setdefault(manifest.checkpoint, []).append((manifest, sha256))
    trained = {
        name: frozenset().union(*(m.trained for m, _ in pairs)) for name, pairs in by_checkpoint.items()
    }
    run_ids = {run.trainrun_id for run in runs.values()}

    # Answered records per (model, file, question, primitive), in the files' order.
    groups: dict[str, dict[tuple[str, str, str], list[ItemKey]]] = {}
    for source, rows in records.items():
        for r in rows:
            if r.model_served is None:
                continue
            groups.setdefault(r.model_served, {}).setdefault((source, r.question_id, r.primitive), []).append(r.item)

    named = sorted(set(by_checkpoint) | set(runs))
    names = named + sorted(set(groups) - set(named))
    out = []
    for name in names:
        known = name in trained
        group_rows = []
        for (source, question_id, primitive), items in groups.get(name, {}).items():
            hits = trained_hits((question_id, name, primitive), items, trained) if known else None
            group_rows.append(
                {
                    "file": source,
                    "question_id": question_id,
                    "primitive": primitive,
                    "n_records": len(items),
                    "n_items": len(set(items)),
                    "n_trained_hits": None if hits is None else len(hits),
                    "first_trained_hit": list(hits[0]) if hits else None,
                }
            )
        run = runs.get(name)
        run_row = None
        if run is not None:
            parent = run.doc["parent"]
            recorded_against = run.doc["manifest"]["sha256"]
            run_row = {
                "path": _path(run.path),
                "trainrun_id": run.trainrun_id,
                # Whether one of this checkpoint's manifests is the file the run was recorded
                # against, by digest: false when it was exported again since, or is not here.
                "manifest_loaded": any(sha256 == recorded_against for _, sha256 in by_checkpoint.get(name, [])),
                # Whether the record of the run this one started from is here too; None without a parent.
                "parent_loaded": None if parent is None else parent["trainrun_id"] in run_ids,
                "record": dict(run.doc),
            }
        policy_rows = [
            {
                "question_id": p.question_id,
                "primitive": p.primitive,
                "trainrun_id": p.trainrun_id,
                # Whether the policy names this checkpoint's recorded run; None when either is absent.
                "run_matches": None if run is None or p.trainrun_id is None else p.trainrun_id == run.trainrun_id,
            }
            for p in policies
            if p.model_served == name
        ]
        out.append(
            {
                "checkpoint": name,
                "n_records": sum(g["n_records"] for g in group_rows),
                "n_trained": len(trained[name]) if known else None,
                "n_trained_hits": sum(g["n_trained_hits"] for g in group_rows) if known else None,
                "manifests": [_manifest_row(m, sha256) for m, sha256 in by_checkpoint.get(name, [])],
                "run": run_row,
                "groups": group_rows,
                "policies": policy_rows,
            }
        )
    return {
        "schema": CHECKPOINTS_SCHEMA,
        "summary": {
            "n_checkpoints": len(out),
            "n_manifests": len(manifests),
            "n_training_runs": len(runs),
        },
        "checkpoints": out,
    }
