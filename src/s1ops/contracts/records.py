"""Read, validate and group `jev-record/1` and `s1-record/1` decision records.

Every record is checked before anything is computed: a calibration or threshold number built on a
stringly-typed label, a NaN probability, or a distribution that sums to 0.7 is worse than no number.
The first invalid line stops the run with its 1-based line number.

Records are grouped strictly by `(question_id, model_served, primitive)`. Nothing is ever pooled
across groups: a different wording, option order or served model is a different probability
distribution.
"""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMAS = ("jev-record/1", "s1-record/1")
DEFAULT_SCHEMA = "jev-record/1"
PRIMITIVES = ("noul", "choice", "score")
REQUIRED = ("schema", "item_id", "question_id", "primitive", "model_requested", "model_served")
ANSWER_KEYS = {
    "noul": ("p",),
    "choice": ("choice", "probabilities", "confidence"),
    "score": ("score", "probabilities", "confidence"),
}
DEFAULT_SUM_TOL = 0.02


class RecordError(ValueError):
    """An input line is not a valid decision record. Carries the 1-based line number."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line


GroupKey = tuple[str, str, str]
"""`(question_id, model_served, primitive)`."""

ItemKey = tuple[str | None, str]
"""`(dataset, item_id)`: an item's identity. `item_id` is unique only within `dataset`."""


@dataclass(frozen=True)
class Record:
    line: int
    item_id: str
    question_id: str
    primitive: str
    model_requested: str
    model_served: str | None
    split: str | None
    dataset: str | None
    label: bool | str | int | None
    classes: tuple
    """Answer classes in canonical order: `(False, True)` for Noul, option labels in the record's
    order for Choice, integer levels ascending for Score."""
    probs: tuple[float, ...]
    """Probabilities aligned with `classes`, renormalized to sum to 1."""
    raw: dict = field(repr=False, compare=False)

    @property
    def key(self) -> GroupKey:
        return (self.question_id, str(self.model_served), self.primitive)

    @property
    def item(self) -> ItemKey:
        return (self.dataset, self.item_id)

    @property
    def model_mismatch(self) -> bool:
        return self.model_served != self.model_requested


def _reject_constant(name: str) -> float:
    raise ValueError(f"non-finite number {name} is not valid JSON")


def _number(value: object, what: str, line: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(line, f"{what} must be a number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        # A JSON integer beyond the float range, such as 1 followed by 400 zeros.
        finite = False
    if not finite:
        raise RecordError(line, f"{what} must be finite, got {value!r:.40}")
    return float(value)


def _probability(value: object, what: str, line: int) -> float:
    x = _number(value, what, line)
    if not 0.0 <= x <= 1.0:
        raise RecordError(line, f"{what} must be in [0, 1], got {x!r}")
    return x


def _string(value: object, what: str, line: int) -> str:
    if not isinstance(value, str) or not value:
        raise RecordError(line, f"{what} must be a non-empty string, got {value!r}")
    return value


def _level(key: str, line: int) -> int:
    if not (isinstance(key, str) and key.isascii() and key.isdigit()):
        raise RecordError(line, f"score level keys must be non-negative integers as strings, got {key!r}")
    return int(key)


def probability_total(values: Iterable[float]) -> float:
    """The sum a distribution is checked against and divided by.

    `math.fsum` is correctly rounded, so the total does not depend on key order. Plain `sum` is
    naive on Python 3.11, where the order a backend or `s1 apply` (`sort_keys=True`) wrote the keys
    would change the last bit, and with it which side of a tuned threshold an answer lands on.
    """
    return math.fsum(values)


def _distribution(value: object, line: int, sum_tol: float) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise RecordError(line, f"probabilities must be a non-empty object, got {value!r}")
    probs = {k: _probability(v, f"probabilities[{k!r}]", line) for k, v in value.items()}
    total = probability_total(probs.values())
    if abs(total - 1.0) > sum_tol:
        raise RecordError(line, f"probabilities sum to {total:.6g}, not 1 within {sum_tol}")
    return probs


def parse_record(
    row: object,
    line: int = 1,
    *,
    sum_tol: float = DEFAULT_SUM_TOL,
    require_answer: bool = True,
    use_calib: bool = False,
) -> Record:
    """Validate one decoded record dict.

    When `use_calib=True` and the record carries a `calib` object from `s1 apply`, the calibrated
    probabilities (`calib.p` or `calib.probabilities`) are used as `probs`.
    """
    if not isinstance(row, dict):
        raise RecordError(line, f"expected a JSON object, got {type(row).__name__}")
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise RecordError(line, f"missing required keys {missing}")
    if row["schema"] not in SCHEMAS:
        raise RecordError(line, f"schema must be one of {SCHEMAS}, got {row['schema']!r}")
    primitive = row["primitive"]
    if primitive not in PRIMITIVES:
        raise RecordError(line, f"primitive must be one of {PRIMITIVES}, got {primitive!r}")
    item_id = _string(row["item_id"], "item_id", line)
    question_id = _string(row["question_id"], "question_id", line)
    model_requested = _string(row["model_requested"], "model_requested", line)
    served = row["model_served"]
    if served is not None:
        served = _string(served, "model_served", line)
    split = row.get("split")
    if split is not None:
        split = _string(split, "split", line)
    dataset = row.get("dataset")
    if dataset is not None:
        dataset = _string(dataset, "dataset", line)

    common = {
        "line": line,
        "item_id": item_id,
        "question_id": question_id,
        "primitive": primitive,
        "model_requested": model_requested,
        "model_served": served,
        "split": split,
        "dataset": dataset,
        "raw": row,
    }
    if served is None or not require_answer:
        return Record(label=None, classes=(), probs=(), **common)

    missing = [k for k in ANSWER_KEYS[primitive] if k not in row]
    if missing:
        raise RecordError(line, f"{primitive} record is missing answer keys {missing}")
    label = row.get("label")
    calib = row.get("calib") if use_calib and isinstance(row.get("calib"), dict) else None

    if primitive == "noul":
        p_raw = _probability(row["p"], "p", line)
        p = _probability(calib["p"], "calib.p", line) if calib and "p" in calib else p_raw
        if label is not None and not isinstance(label, bool):
            raise RecordError(line, f"noul label must be a JSON boolean, got {label!r}")
        return Record(label=label, classes=(False, True), probs=(1.0 - p, p), **common)

    _probability(row["confidence"], "confidence", line)
    dist_raw = _distribution(row["probabilities"], line, sum_tol)
    dist = _distribution(calib["probabilities"], line, sum_tol) if calib and "probabilities" in calib else dist_raw
    total = probability_total(dist.values())
    if primitive == "choice":
        choice = _string(row["choice"], "choice", line)
        if choice not in dist_raw:
            raise RecordError(line, f"choice {choice!r} is not one of the probability keys {sorted(dist_raw)}")
        if label is not None and not isinstance(label, str):
            raise RecordError(line, f"choice label must be a string, got {label!r}")
        if label is not None and label not in dist_raw:
            raise RecordError(line, f"choice label {label!r} is not one of the options {list(dist_raw)}")
        classes: tuple = tuple(dist_raw)
        probs = tuple(dist[c] / total for c in classes)
    else:
        levels = {_level(k, line): v for k, v in dist.items()}
        _number(row["score"], "score", line)
        if label is not None and (isinstance(label, bool) or not isinstance(label, int)):
            raise RecordError(line, f"score label must be an integer level, got {label!r}")
        if label is not None and label not in levels:
            raise RecordError(line, f"score label {label} is not one of the levels {sorted(levels)}")
        classes = tuple(sorted(levels))
        probs = tuple(levels[c] / total for c in classes)
    return Record(label=label, classes=classes, probs=probs, **common)


def read_records(
    path: str | Path,
    *,
    sum_tol: float = DEFAULT_SUM_TOL,
    require_answer: bool = True,
    use_calib: bool = False,
) -> list[Record]:
    records = []
    with open(path, encoding="utf-8") as f:
        for n, text in enumerate(f, start=1):
            if not text.strip():
                continue
            try:
                row = json.loads(text, parse_constant=_reject_constant)
            except ValueError as e:
                raise RecordError(n, f"not valid JSON: {e}") from None
            records.append(
                parse_record(row, n, sum_tol=sum_tol, require_answer=require_answer, use_calib=use_calib)
            )
    return records


@dataclass(frozen=True)
class Group:
    """All labeled answers to one question from one served model."""

    key: GroupKey
    classes: tuple
    P: np.ndarray
    """`n × K` probabilities aligned with `classes`."""
    y: np.ndarray
    """`n` indices into `classes`."""
    items: tuple[ItemKey, ...]
    """`(dataset, item_id)` per row."""
    splits: tuple[str | None, ...]

    @property
    def question_id(self) -> str:
        return self.key[0]

    @property
    def model_served(self) -> str:
        return self.key[1]

    @property
    def primitive(self) -> str:
        return self.key[2]

    @property
    def n(self) -> int:
        return len(self.y)


@dataclass
class Selection:
    """What was kept, and why the rest was not."""

    groups: list[Group]
    no_response: int = 0
    unlabeled: int = 0
    other_split: int = 0
    model_mismatch: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "no_response": self.no_response,
            "unlabeled": self.unlabeled,
            "other_split": self.other_split,
            "model_mismatch": [
                {"question_id": q, "model_requested": req, "model_served": srv, "n": n}
                for (q, req, srv), n in sorted(self.model_mismatch.items())
            ],
        }


def check_classes(key: GroupKey, expected: Sequence, record: Record) -> None:
    if set(record.classes) != set(expected):
        raise RecordError(
            record.line,
            f"{key[0]} ({key[1]}): answer classes {sorted(map(str, record.classes))} differ from "
            f"{sorted(map(str, expected))} seen earlier for the same question; the question_id cannot be trusted",
        )


def select(
    records: Iterable[Record],
    *,
    split: str | None = None,
    allow_model_mismatch: bool = False,
) -> Selection:
    sel = Selection(groups=[])
    buckets: dict[GroupKey, list[Record]] = defaultdict(list)
    for r in records:
        if r.model_served is None:
            sel.no_response += 1
        elif split is not None and r.split != split:
            sel.other_split += 1
        elif r.label is None:
            sel.unlabeled += 1
        elif r.model_mismatch and not allow_model_mismatch:
            sel.model_mismatch[(r.question_id, r.model_requested, r.model_served)] += 1
        else:
            buckets[r.key].append(r)
    for key in sorted(buckets):
        rows = buckets[key]
        classes = rows[0].classes
        seen: dict[ItemKey, int] = {}
        for r in rows:
            check_classes(key, classes, r)
            if r.item in seen:
                raise RecordError(
                    r.line,
                    f"item {r.item} (dataset, item_id) appears twice in group {key} (first on line {seen[r.item]})",
                )
            seen[r.item] = r.line
        index = {c: i for i, c in enumerate(classes)}
        P = np.array([[dict(zip(r.classes, r.probs))[c] for c in classes] for r in rows], dtype=float)
        y = np.array([index[r.label] for r in rows], dtype=int)
        sel.groups.append(
            Group(key, classes, P, y, tuple(r.item for r in rows), tuple(r.split for r in rows))
        )
    return sel
