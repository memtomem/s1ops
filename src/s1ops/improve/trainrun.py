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
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s1ops.contracts._validation import UNDECODABLE, pydantic_errors, read_json_file
from s1ops.contracts.records import MODEL_NAME, ItemKey, _row_surrogate_problem, surrogate_problem
from s1ops.improve.trainset import _NAME_RULE, ManifestError, TrainedItemError, TrainingManifest, _item_order

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
    path: str | Path | None = field(default=None, kw_only=True, repr=False, compare=False)
    """The file `load_trainrun` read this from; `None` for one just built. Errors about it name it."""


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
        got = "an empty string" if value == "" else f"a value of type {type(value).__name__}"
        raise TrainRunError(f"{where}: {what} must be a non-empty string, got {got}")
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


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer beyond the float range, which JSON decodes as an int
        return False


def _check_scalars(values: object, what: str, where: str) -> dict[str, Scalar]:
    """`values` as names to strings, finite numbers, booleans or null, or a `TrainRunError` at `where`."""
    if not isinstance(values, Mapping):
        raise TrainRunError(f"{where}: {what}s must be a JSON object")
    out: dict[str, Scalar] = {}
    for key, value in values.items():
        name = _id_string(key, f"a {what} name", where)
        if name == "state":
            # As the ledger refuses the key in its metadata: reports and the Langfuse export copy
            # these values out, and a row's `state` passed along by mistake would go with them.
            raise TrainRunError(f"{where}: a {what} cannot be named 'state' (safe-v1: item text is never recorded)")
        if isinstance(value, str):
            problem = surrogate_problem(f"{what} {name!r}", value)
            if problem is not None:
                raise TrainRunError(f"{where}: {problem}")
        elif value is None or isinstance(value, bool):
            pass
        elif isinstance(value, (int, float)):
            if not _finite(value):
                raise TrainRunError(f"{where}: {what} {name!r} must be finite")
        else:
            raise TrainRunError(
                f"{where}: {what} {name!r} is a {type(value).__name__}; values are strings, numbers, booleans or null"
            )
        out[name] = value
    return out


def _check_steps(rows: Iterable[tuple[str, object]]) -> list[dict[str, float | int]]:
    """Step rows, each with the place an error names: an increasing integer `step` and finite numbers."""
    steps: list[dict[str, float | int]] = []
    last = -1
    for where, row in rows:
        if not isinstance(row, Mapping):
            raise TrainRunError(f"{where}: expected a JSON object")
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0 or not _finite(step):
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


def read_scalars(path: str | Path, what: str) -> dict[str, Scalar]:
    """A JSON object of names to strings, finite numbers, booleans or null, such as trainer settings
    or summary metrics. Nested values are refused: flatten them (`lora.rank`) before recording."""
    return _check_scalars(read_json_file(path, TrainRunError), what, str(path))


def read_steps(path: str | Path) -> list[dict[str, float | int]]:
    """Per-step metrics, one JSON object per line: a non-negative integer `step` and any other
    finite numbers (`epoch`, `loss`, `lr`, ...). Steps must increase down the file."""
    rows, _ = _jsonl_rows(path)
    return _check_steps((f"{path}: line {n}", row) for n, row in rows)


_SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Count = Annotated[int, Field(ge=0, le=2**53)]
_TRAINRUN_ID = r"^tr-[0-9a-f]{16}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ManifestRef(_Strict):
    sha256: _SHA256
    source_sha256: _SHA256
    items: Annotated[int, Field(ge=1, le=2**53)]
    inherited_items: _Count


class _DataRef(_Strict):
    sha256: _SHA256
    rows: Annotated[int, Field(ge=1, le=2**53)]
    items: Annotated[int, Field(ge=1, le=2**53)]


class _ParentRef(_Strict):
    trainrun_id: str = Field(pattern=_TRAINRUN_ID)
    checkpoint: str


class _Body(_Strict):
    """A record before its id is known: what `build_trainrun` hashes."""

    schema_: str = Field(alias="schema")
    checkpoint: str
    split: str = Field(min_length=1)
    manifest: _ManifestRef
    data: _DataRef
    weights_sha256: dict[Annotated[str, Field(min_length=1)], _SHA256] = Field(min_length=1)
    trainer: dict[str, Any]
    summary: dict[str, Any]
    steps: list[dict[str, Any]]
    parent: _ParentRef | None
    created_at: str


class _Record(_Body):
    trainrun_id: str = Field(pattern=_TRAINRUN_ID)


def _check_doc(doc: object, model: type[_Body], where: str | Path | None) -> None:
    """The one rule for a record's contents: `build_trainrun` writes only what `load_trainrun` reads.

    Extra keys are refused, so a record cannot carry a `state`. A `TrainRunError` starts with `where`."""
    prefix = "" if where is None else f"{where}: "
    # First, and by code point: a validation message would otherwise show the string it refuses.
    problem = _row_surrogate_problem(doc)
    if problem is not None:
        raise TrainRunError(prefix + problem)
    try:
        parsed = model.model_validate(doc)
    except ValidationError as e:
        raise TrainRunError(prefix + "; ".join(pydantic_errors(e))) from None
    if not MODEL_NAME.fullmatch(parsed.checkpoint):
        raise TrainRunError(f"{prefix}checkpoint {parsed.checkpoint!r} is not a model name: use {_NAME_RULE}")
    if parsed.parent is not None:
        if not MODEL_NAME.fullmatch(parsed.parent.checkpoint):
            raise TrainRunError(f"{prefix}parent.checkpoint {parsed.parent.checkpoint!r} is not a model name: use {_NAME_RULE}")
        if parsed.parent.checkpoint == parsed.checkpoint:
            raise TrainRunError(
                f"{prefix}the parent run {parsed.parent.trainrun_id} is for the same checkpoint {parsed.checkpoint!r}"
            )
    if parsed.data.items > parsed.data.rows:
        raise TrainRunError(f"{prefix}data.items {parsed.data.items} exceeds data.rows {parsed.data.rows}")
    _check_scalars(parsed.trainer, "trainer setting", f"{prefix}trainer")
    _check_scalars(parsed.summary, "summary metric", f"{prefix}summary")
    _check_steps((f"{prefix}steps.{i}", row) for i, row in enumerate(parsed.steps))


def _trainrun_id(doc: Mapping[str, object]) -> str:
    body = {k: v for k, v in doc.items() if k not in ("trainrun_id", "created_at")}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "tr-" + _sha256_bytes(canonical.encode("utf-8"))[:16]


def build_trainrun(
    manifest: TrainingManifest,
    *,
    manifest_sha256: str,
    data: str | Path,
    weights: Iterable[str | Path],
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
    # A lone path would be read character by character, and each character taken for a file; a
    # mapping, such as a record's own `weights_sha256`, would be read as its keys.
    if isinstance(weights, (str, bytes, os.PathLike, Mapping)) or not isinstance(weights, Iterable):
        raise TrainRunError(f"weights must be a sequence of paths, got a {type(weights).__name__}")
    weights = list(weights)
    if not weights:
        raise TrainRunError("a training run records at least one weights file")
    for w in weights:
        if not isinstance(w, (str, os.PathLike)):
            raise TrainRunError(f"a weights file must be a path, got a {type(w).__name__}")
    if parent is not None and parent.checkpoint == manifest.checkpoint:
        raise TrainRunError(f"the parent run {parent.trainrun_id} is for the same checkpoint {manifest.checkpoint!r}")
    items, check = read_data_items(data)
    check_data(manifest, items, data=data)
    digests: dict[str, str] = {}
    for w in weights:
        name = Path(w).name
        problem = surrogate_problem("a weights file name", name)  # a name that is not UTF-8, read with surrogateescape
        if problem is not None:
            raise TrainRunError(problem)
        if name in digests:
            raise TrainRunError(f"{w}: two weights files are named {name!r}; the record keys them by file name")
        digests[name] = _file_sha256(w)
    body: dict[str, object] = {
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
        "trainer": _plain(trainer),
        "summary": _plain(summary),
        "steps": _plain_rows(steps),
        "parent": None if parent is None else {"trainrun_id": parent.trainrun_id, "checkpoint": parent.checkpoint},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Before the id is hashed: a value the reader would refuse is refused here, with nothing written.
    _check_doc(body, _Body, None)
    trainrun_id = _trainrun_id(body)
    doc = {"schema": TRAINRUN_SCHEMA, "trainrun_id": trainrun_id, **{k: v for k, v in body.items() if k != "schema"}}
    return TrainRun(trainrun_id=trainrun_id, checkpoint=manifest.checkpoint, doc=doc)


def _plain(value: object) -> object:
    """A caller's mapping as a dict, `None` as an empty one; anything else is left for `_check_doc` to refuse."""
    if value is None:
        return {}
    return dict(value) if isinstance(value, Mapping) else value


def _plain_rows(rows: object) -> object:
    """A caller's step rows as a list; a value that cannot be iterated is left for `_check_doc` to refuse."""
    return [_plain(row) for row in rows] if isinstance(rows, Iterable) else rows


def save_trainrun(run: TrainRun, path: str | Path) -> None:
    # Encoded before the file is opened, so a value that cannot be written leaves the old file in place.
    data = (json.dumps(run.doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)


def load_trainrun(path: str | Path) -> TrainRun:
    """A record `save_trainrun` wrote: the fields `build_trainrun` writes and no others, with a
    `trainrun_id` that matches them. A malformed one raises `TrainRunError` naming `path`."""
    doc = read_json_file(path, TrainRunError)
    _check_record(doc, path)
    return TrainRun(trainrun_id=doc["trainrun_id"], checkpoint=doc["checkpoint"], doc=doc, path=path)


def _check_record(doc: object, where: str | Path) -> None:
    """A whole record: the schema, `_check_doc`, and a `trainrun_id` that matches the contents."""
    if not isinstance(doc, dict) or doc.get("schema") != TRAINRUN_SCHEMA:
        raise TrainRunError(f"{where}: not a {TRAINRUN_SCHEMA} file")
    _check_doc(doc, _Record, where)
    if _trainrun_id(doc) != doc["trainrun_id"]:
        raise TrainRunError(f"{where}: trainrun_id {doc['trainrun_id']} does not match the record's contents")


TrainRuns = Mapping[str, TrainRun]
"""`checkpoint -> the record of the run that produced it`."""


def runs_by_checkpoint(runs: Iterable[TrainRun], served: Iterable[str | None], where: str) -> TrainRuns:
    """`runs` keyed by the checkpoint each produced, for a report or export that names them.

    Each is checked as `load_trainrun` checks a file, so one built by hand cannot put into an export
    what a record may not hold. `served` holds the `model_served` of every row in the input named
    `where`, before any filter: a record whose checkpoint is not among them is refused, as a
    manifest is (`load_trained`), since no row would show its id. Two records for one checkpoint
    are refused unless they are the same run: a checkpoint name names one set of weights."""
    names = set(served)
    out: dict[str, TrainRun] = {}
    for run in runs:
        source = run.path if run.path is not None else "the training-run record"
        doc = run.doc
        _check_record(doc, source)
        if (run.trainrun_id, run.checkpoint) != (doc["trainrun_id"], doc["checkpoint"]):
            raise TrainRunError(
                f"{source}: given as {run.trainrun_id} for {run.checkpoint!r}, "
                f"but its record is {doc['trainrun_id']} for {doc['checkpoint']!r}"
            )
        if run.checkpoint not in names:
            raise TrainRunError(
                f"{source}: checkpoint {run.checkpoint!r} answered no record in {where}; "
                f"its answered models are {sorted(n for n in names if n is not None)}"
            )
        first = out.setdefault(run.checkpoint, run)
        if first.trainrun_id != run.trainrun_id:
            other = first.path if first.path is not None else "another record"
            raise TrainRunError(
                f"{source}: {run.trainrun_id} is a second run for checkpoint {run.checkpoint!r}, "
                f"after {first.trainrun_id} in {other}; name each checkpoint once"
            )
    return dict(sorted(out.items()))


def load_trainruns(paths: Sequence[str | Path], served: Iterable[str | None], where: str) -> TrainRuns:
    """`runs_by_checkpoint` over the records at `paths`."""
    return runs_by_checkpoint([load_trainrun(path) for path in paths], served, where)
