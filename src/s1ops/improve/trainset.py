"""Training-set manifests: what an external trainer gets, and what a checkpoint must not be judged on.

s1ops starts no training. `build_manifest` turns labeled records into a manifest that names the
checkpoint to be trained and its items, and `check_not_trained` refuses to fit or evaluate that
checkpoint on them. A manifest holds ids, question ids and labels, which records already hold; never
`state` text or question wording, which the trainer joins from its own local files.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s1ops.contracts._validation import pydantic_errors, read_json_bytes, read_json_file
from s1ops.contracts.records import MODEL_NAME, GroupKey, ItemKey, Record, RecordError, _row_surrogate_problem

MANIFEST_SCHEMA = "s1-training-manifest/1"

Trained = Mapping[str, frozenset[ItemKey]]
"""`checkpoint -> (dataset, item_id)` of every item it was trained on, inherited ones included."""


_NAME_RULE = "1-128 characters, an ASCII letter or digit and then ASCII letters, digits or ._:/@+-"


class ManifestError(ValueError):
    """A training manifest that is malformed or cannot be built or used as asked."""


class TrainedItemError(ValueError):
    """A checkpoint would be fitted or evaluated on an item it was trained on (CLI exit 2)."""


@dataclass(frozen=True)
class TrainingItem:
    dataset: str | None
    item_id: str
    question_id: str
    label: bool | int | str

    @property
    def item(self) -> ItemKey:
        return (self.dataset, self.item_id)


@dataclass(frozen=True)
class TrainingManifest:
    checkpoint: str
    split: str
    source_sha256: str
    created_at: str
    items: tuple[TrainingItem, ...]
    inherited_items: tuple[ItemKey, ...] = ()
    """What the parent checkpoints trained on, which this one inherits; not part of its training data."""
    path: str | Path | None = field(default=None, kw_only=True, repr=False, compare=False)
    """The file `load_manifest` read this from; `None` for one just built. Errors about it name it."""

    @property
    def trained(self) -> frozenset[ItemKey]:
        return frozenset(i.item for i in self.items) | frozenset(self.inherited_items)


@dataclass(frozen=True)
class ManifestBuild:
    manifest: TrainingManifest
    no_response: int
    unlabeled: int
    other_split: int
    also_in_other_splits: tuple[ItemKey, ...]
    answered_as_checkpoint: int


def _item_order(item: ItemKey) -> tuple:
    dataset, item_id = item
    return (dataset is not None, dataset or "", item_id)


def build_manifest(
    records: Iterable[Record],
    *,
    split: str,
    checkpoint: str,
    source_sha256: str,
    parent: TrainingManifest | None = None,
    source: str | Path = "the records",
) -> ManifestBuild:
    """The manifest of `checkpoint`, trained on the answered, labeled records of `split`.

    One item and question answered by several models is one row, and its labels must agree:
    labels are compared with their primitive, since `True == 1`. `parent` is the manifest of the
    checkpoint this one starts from; what it trained on is recorded as inherited. `source` names the
    records in an error, since an empty input has no record to take a path from.
    """
    if not MODEL_NAME.fullmatch(checkpoint):
        raise ManifestError(f"checkpoint {checkpoint!r} is not a model name: use {_NAME_RULE}")
    if parent is not None and parent.checkpoint == checkpoint:
        where = parent.path or "the parent manifest"
        raise ManifestError(f"{where}: is for the same checkpoint {checkpoint!r}; name the new one differently")
    records = list(records)
    no_response = unlabeled = other_split = 0
    chosen: dict[tuple[ItemKey, str], Record] = {}
    for r in records:
        if r.model_served is None:
            no_response += 1
        elif r.split != split:
            other_split += 1
        elif r.label is None:
            unlabeled += 1
        else:
            first = chosen.setdefault((r.item, r.question_id), r)
            if (first.primitive, first.label) != (r.primitive, r.label):
                raise RecordError(
                    r.line,
                    f"item {r.item} of question {r.question_id} is a {r.primitive} labeled {r.label!r}, "
                    f"and a {first.primitive} labeled {first.label!r} on line {first.line}",
                    r.path,
                )
    if not chosen:
        raise ManifestError(
            f"{source}: no answered, labeled record has split {split!r} "
            f"({no_response} without a response, {unlabeled} unlabeled, {other_split} in other splits)"
        )
    ordered = sorted(chosen.values(), key=lambda r: (_item_order(r.item), r.question_id))
    items = tuple(TrainingItem(r.dataset, r.item_id, r.question_id, r.label) for r in ordered)
    own = {i.item for i in items}
    inherited = sorted((parent.trained - own) if parent is not None else (), key=_item_order)
    elsewhere = {r.item for r in records if r.split != split} & own
    return ManifestBuild(
        TrainingManifest(
            checkpoint=checkpoint,
            split=split,
            source_sha256=source_sha256,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            items=items,
            inherited_items=tuple(inherited),
        ),
        no_response=no_response,
        unlabeled=unlabeled,
        other_split=other_split,
        also_in_other_splits=tuple(sorted(elsewhere, key=_item_order)),
        answered_as_checkpoint=sum(r.model_served == checkpoint for r in records),
    )


def save_manifest(manifest: TrainingManifest, path: str | Path) -> None:
    doc = {
        "schema": MANIFEST_SCHEMA,
        "checkpoint": manifest.checkpoint,
        "split": manifest.split,
        "source_sha256": manifest.source_sha256,
        "created_at": manifest.created_at,
        "items": [
            {"dataset": i.dataset, "item_id": i.item_id, "question_id": i.question_id, "label": i.label}
            for i in manifest.items
        ],
        "inherited_items": [{"dataset": d, "item_id": i} for d, i in manifest.inherited_items],
    }
    # Encoded before the file is opened, so an id that cannot be written leaves the old file in place.
    data = (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)


class _Identity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset: str | None = Field(min_length=1)
    item_id: str = Field(min_length=1)


class _Item(_Identity):
    question_id: str = Field(min_length=1)
    label: bool | int | str


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_: str = Field(alias="schema")
    checkpoint: str = Field(min_length=1)
    split: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str
    items: list[_Item] = Field(min_length=1)
    inherited_items: list[_Identity]


def load_manifest(path: str | Path) -> TrainingManifest:
    return _manifest_from_doc(read_json_file(path, ManifestError), path)


def load_manifest_hashed(path: str | Path) -> tuple[TrainingManifest, str]:
    """`load_manifest`, plus the SHA-256 of the bytes it parsed, from one read: a FIFO has only one."""
    data = Path(path).read_bytes()
    return _manifest_from_doc(read_json_bytes(data, path, ManifestError), path), hashlib.sha256(data).hexdigest()


def _manifest_from_doc(doc: object, path: str | Path) -> TrainingManifest:
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"{path}: not a {MANIFEST_SCHEMA} file")
    # First, and by code point, as a run record is checked: an escaped lone surrogate passes an
    # unconstrained field (`created_at`), and whatever copies the manifest's strings out (the
    # workbench's response, an error naming the checkpoint) then fails to encode.
    problem = _row_surrogate_problem(doc)
    if problem is not None:
        raise ManifestError(f"{path}: {problem}")
    try:
        parsed = _Manifest.model_validate(doc)
    except ValidationError as e:
        raise ManifestError(f"{path}: " + "; ".join(pydantic_errors(e))) from None
    if not MODEL_NAME.fullmatch(parsed.checkpoint):
        raise ManifestError(f"{path}: checkpoint {parsed.checkpoint!r} is not a model name: use {_NAME_RULE}")
    items = tuple(TrainingItem(i.dataset, i.item_id, i.question_id, i.label) for i in parsed.items)
    seen: set[tuple[ItemKey, str]] = set()
    for i in items:
        if (i.item, i.question_id) in seen:
            raise ManifestError(f"{path}: item {i.item} of question {i.question_id} appears twice")
        seen.add((i.item, i.question_id))
    return TrainingManifest(
        checkpoint=parsed.checkpoint,
        split=parsed.split,
        source_sha256=parsed.source_sha256,
        created_at=parsed.created_at,
        items=items,
        inherited_items=tuple((i.dataset, i.item_id) for i in parsed.inherited_items),
        path=path,
    )


def load_trained(paths: Sequence[str | Path], served: Iterable[str | None], where: str) -> Trained:
    """What the manifests at `paths` say each checkpoint trained on; several for one checkpoint add up.

    `served` holds the `model_served` of every row in the input named `where`, before any filter.
    A manifest whose checkpoint is not among them is refused: it would be a misspelled name, or a
    probe run under another one, and nothing would be checked. A name that is present does not
    promise the rows a command then selects include that model."""
    names = set(served)
    trained: dict[str, set[ItemKey]] = {}
    for path in paths:
        manifest = load_manifest(path)
        if manifest.checkpoint not in names:
            raise ManifestError(
                f"{path}: checkpoint {manifest.checkpoint!r} answered no record in {where}; "
                f"its answered models are {sorted(n for n in names if n is not None)}"
            )
        trained.setdefault(manifest.checkpoint, set()).update(manifest.trained)
    return {checkpoint: frozenset(items) for checkpoint, items in trained.items()}


def trained_hits(key: GroupKey, items: Iterable[ItemKey], trained: Trained) -> list[ItemKey]:
    """The items of group `key` that its model was trained on, each once, in item order. Identity is
    `(dataset, item_id)` across questions: the model saw the state, and rewording a question changes
    its `question_id`. A group of another model has none, since that model never saw these items."""
    known = trained.get(key[1])
    if not known:
        return []
    return sorted({i for i in items if i in known}, key=_item_order)


def check_not_trained(key: GroupKey, items: Iterable[ItemKey], trained: Trained, where: str) -> None:
    """Refuse the items of group `key` that its model was trained on (`trained_hits`)."""
    hits = trained_hits(key, items, trained)
    if hits:
        raise TrainedItemError(
            f"{where}: {len(hits)} item(s) of group {key} are in the training set of {key[1]!r}, first {hits[0]}; "
            "a checkpoint cannot be fitted or evaluated on what it trained on"
        )
