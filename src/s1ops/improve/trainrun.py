"""Training-run records: how a checkpoint was trained, next to the manifest of what it trained on.

s1ops starts no training. After an external trainer has trained the checkpoint a manifest names,
`build_trainrun` writes what the run did as one `s1-trainrun/1` record: the manifest's digest, a check
that the trainer's data file holds exactly the manifest's items, the weights' digests, and the
trainer's settings, summary metrics and per-step metrics as the caller passes them. Like a manifest,
the record holds ids, counts, digests and numbers; never `state` text, labels or question wording.

The check is what makes the manifest's leakage refusal hold. `--training-manifest` refuses only the
items a manifest lists, so a data row the manifest does not list is a trained item that `fit`,
`audit`, `policy` and `compare` would accept; `build_trainrun` refuses it (`TrainingDataError`, CLI
exit 2). A listed item with no data row is refused too (`ManifestError`, exit 1), since the
manifest would then claim the checkpoint trained on what it never saw.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from s1ops.contracts._validation import UNDECODABLE, read_json_file
from s1ops.contracts.records import ItemKey, surrogate_problem
from s1ops.improve.trainset import ManifestError, TrainedItemError, TrainingManifest, _item_order

TRAINRUN_SCHEMA = "s1-trainrun/1"

Scalar = str | int | float | bool | None


class TrainRunError(ValueError):
    """A training-run input or record that is malformed or cannot be used as asked."""


class TrainingDataError(TrainedItemError):
    """The trainer's data holds an item its manifest does not list (CLI exit 2): the leakage refusal
    would not know the checkpoint trained on it."""


@dataclass(frozen=True)
class DataCheck:
    sha256: str
    rows: int
    items: int
    """Distinct `(dataset, item_id)` among the rows; an item asked several questions, or repeated
    for several epochs, is one."""


@dataclass(frozen=True)
class TrainRun:
    """One `s1-trainrun/1` record. `doc` is the record as written, `trainrun_id` included."""

    trainrun_id: str
    checkpoint: str
    doc: Mapping[str, object]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _reject_constant(name: str) -> float:
    raise ValueError(f"non-finite number {name} is not valid JSON")


def _jsonl_rows(path: str | Path) -> tuple[list[tuple[int, dict]], str]:
    """The object on each non-blank line of `path`, with its 1-based line number, and the SHA-256 of
    the bytes read. One read, so a FIFO works; a malformed line raises `TrainRunError` naming it."""
    data = Path(path).read_bytes()
    text = data.decode("utf-8", errors="surrogateescape").replace("\r\n", "\n").replace("\r", "\n")
    rows: list[tuple[int, dict]] = []
    for n, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if UNDECODABLE.search(line):
            raise TrainRunError(f"{path}: line {n}: not valid UTF-8")
        try:
            row = json.loads(line, parse_constant=_reject_constant)
        except RecursionError:
            raise TrainRunError(f"{path}: line {n}: not valid JSON: nested too deeply") from None
        except ValueError as e:
            raise TrainRunError(f"{path}: line {n}: not valid JSON: {e}") from None
        if not isinstance(row, dict):
            raise TrainRunError(f"{path}: line {n}: expected a JSON object")
        rows.append((n, row))
    return rows, _sha256_bytes(data)


def _id_string(value: object, what: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainRunError(f"{where}: {what} must be a non-empty string, got a value of type {type(value).__name__}")
    problem = surrogate_problem(what, value)
    if problem is not None:
        raise TrainRunError(f"{where}: {problem}")
    return value


def read_data_items(path: str | Path) -> tuple[list[ItemKey], DataCheck]:
    """The `(dataset, item_id)` of every row of the trainer's data file, in file order.

    Each row is a JSON object with a non-empty string `item_id` and, when its items belong to a
    dataset, a non-empty string `dataset` (absent or null: none). Every other field, the state text
    and label among them, is the trainer's and is neither read nor kept."""
    rows, digest = _jsonl_rows(path)
    if not rows:
        raise TrainRunError(f"{path}: no data rows")
    items: list[ItemKey] = []
    for n, row in rows:
        where = f"{path}: line {n}"
        if "item_id" not in row:
            raise TrainRunError(f"{where}: missing item_id; each data row names the manifest item it trains on")
        item_id = _id_string(row["item_id"], "item_id", where)
        dataset = row.get("dataset")
        if dataset is not None:
            dataset = _id_string(dataset, "dataset", where)
        items.append((dataset, item_id))
    return items, DataCheck(sha256=digest, rows=len(items), items=len(set(items)))


def check_data(manifest: TrainingManifest, items: Iterable[ItemKey], *, data: str | Path = "the data") -> None:
    """Refuse data whose items differ from the manifest's own items.

    An item the manifest inherits is accepted in the data as well: the checkpoint trained on it
    either way, and the manifest already lists it."""
    have = set(items)
    where = manifest.path or "the manifest"
    unlisted = sorted(have - manifest.trained, key=_item_order)
    if unlisted:
        raise TrainingDataError(
            f"{data}: {len(unlisted)} item(s) are not in {where} for {manifest.checkpoint!r}, first {unlisted[0]}; "
            "--training-manifest would not refuse them, so export the manifest from every row the trainer read"
        )
    missing = sorted({i.item for i in manifest.items} - have, key=_item_order)
    if missing:
        raise ManifestError(
            f"{where}: {len(missing)} item(s) have no row in {data}, first {missing[0]}; "
            "the manifest would claim the checkpoint trained on them"
        )


def read_scalars(path: str | Path, what: str) -> dict[str, Scalar]:
    """A JSON object of names to strings, finite numbers, booleans or null, such as trainer settings
    or summary metrics. Nested values are refused: flatten them (`lora.rank`) before recording."""
    doc = read_json_file(path, TrainRunError)
    if not isinstance(doc, dict):
        raise TrainRunError(f"{path}: {what} must be a JSON object")
    out: dict[str, Scalar] = {}
    for key, value in doc.items():
        name = _id_string(key, f"a {what} name", str(path))
        if isinstance(value, str):
            problem = surrogate_problem(f"{what} {name!r}", value)
            if problem is not None:
                raise TrainRunError(f"{path}: {problem}")
        elif isinstance(value, float) and not math.isfinite(value):  # an integer literal past the float range
            raise TrainRunError(f"{path}: {what} {name!r} must be finite")
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise TrainRunError(
                f"{path}: {what} {name!r} is a {type(value).__name__}; values are strings, numbers, booleans or null"
            )
        out[name] = value
    return out


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer beyond the float range
        return False


def read_steps(path: str | Path) -> list[dict[str, float | int]]:
    """Per-step metrics, one JSON object per line: a non-negative integer `step` and any other
    finite numbers (`epoch`, `loss`, `lr`, ...). Steps must increase down the file."""
    rows, _ = _jsonl_rows(path)
    steps: list[dict[str, float | int]] = []
    last = -1
    for n, row in rows:
        where = f"{path}: line {n}"
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TrainRunError(f"{where}: step must be a non-negative integer")
        if step <= last:
            raise TrainRunError(f"{where}: step {step} does not follow step {last}")
        last = step
        for key, value in row.items():
            name = _id_string(key, "a metric name", where)
            if not _finite(value):
                raise TrainRunError(f"{where}: metric {name!r} must be a finite number")
        steps.append(dict(row))
    return steps


def _trainrun_id(doc: Mapping[str, object]) -> str:
    body = {k: v for k, v in doc.items() if k not in ("trainrun_id", "created_at")}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "tr-" + _sha256_bytes(canonical.encode("utf-8"))[:16]


def build_trainrun(
    manifest: TrainingManifest,
    *,
    manifest_sha256: str,
    data: str | Path,
    weights: Sequence[str | Path],
    trainer: Mapping[str, Scalar] | None = None,
    summary: Mapping[str, Scalar] | None = None,
    steps: Sequence[Mapping[str, float | int]] = (),
    parent: TrainRun | None = None,
) -> TrainRun:
    """The `s1-trainrun/1` record of training `manifest.checkpoint` on the data file `data`.

    `weights` are the files the run produced (adapter, head, full model); each is recorded by its
    file name and SHA-256, so names must differ. `parent` is the record of the run this checkpoint
    started from. `trainrun_id` hashes everything but `created_at`: the same inputs give the same id.
    """
    if not weights:
        raise TrainRunError("a training run records at least one weights file")
    items, check = read_data_items(data)
    check_data(manifest, items, data=data)
    digests: dict[str, str] = {}
    for w in weights:
        name = Path(w).name
        if name in digests:
            raise TrainRunError(f"{w}: two weights files are named {name!r}; the record keys them by file name")
        digests[name] = _file_sha256(w)
    if parent is not None and parent.checkpoint == manifest.checkpoint:
        raise TrainRunError(f"the parent run {parent.trainrun_id} is for the same checkpoint {manifest.checkpoint!r}")
    doc: dict[str, object] = {
        "schema": TRAINRUN_SCHEMA,
        "checkpoint": manifest.checkpoint,
        "split": manifest.split,
        "manifest": {
            "sha256": manifest_sha256,
            "source_sha256": manifest.source_sha256,
            "items": len(manifest.items),
            "inherited_items": len(manifest.inherited_items),
        },
        "data": {"sha256": check.sha256, "rows": check.rows, "items": check.items},
        "weights_sha256": dict(sorted(digests.items())),
        "trainer": dict(trainer or {}),
        "summary": dict(summary or {}),
        "steps": [dict(s) for s in steps],
        "parent": None if parent is None else {"trainrun_id": parent.trainrun_id, "checkpoint": parent.checkpoint},
    }
    trainrun_id = _trainrun_id(doc)
    doc = {"schema": TRAINRUN_SCHEMA, "trainrun_id": trainrun_id, **{k: v for k, v in doc.items() if k != "schema"}}
    doc["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return TrainRun(trainrun_id=trainrun_id, checkpoint=manifest.checkpoint, doc=doc)


def save_trainrun(run: TrainRun, path: str | Path) -> None:
    # Encoded before the file is opened, so a value that cannot be written leaves the old file in place.
    data = (json.dumps(run.doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)


def load_trainrun(path: str | Path) -> TrainRun:
    """A record `save_trainrun` wrote. Its `trainrun_id` must match its contents."""
    doc = read_json_file(path, TrainRunError)
    if not isinstance(doc, dict) or doc.get("schema") != TRAINRUN_SCHEMA:
        raise TrainRunError(f"{path}: not a {TRAINRUN_SCHEMA} file")
    trainrun_id, checkpoint = doc.get("trainrun_id"), doc.get("checkpoint")
    if not isinstance(trainrun_id, str) or not isinstance(checkpoint, str):
        raise TrainRunError(f"{path}: trainrun_id and checkpoint must be strings")
    if _trainrun_id(doc) != trainrun_id:
        raise TrainRunError(f"{path}: trainrun_id {trainrun_id} does not match the record's contents")
    return TrainRun(trainrun_id=trainrun_id, checkpoint=checkpoint, doc=doc)
