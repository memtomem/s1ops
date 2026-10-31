"""Blind label noise auditor: calibrated disagreement ranking, blind sheet with random control, and Wilson lift."""

import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s1ops.contracts._validation import pydantic_errors
from s1ops.contracts.records import Record, parse_record
from s1ops.improve.calibrate import Fit, fit_for, refusal, transform
from s1ops.improve.slices import wilson

EPS = 1e-6
ANSWER_RE = re.compile(r"^## (\d+)[ \t]*\n(.*?)(?=^## \d+[ \t]*$|\Z)", re.M | re.S)

FitKey = tuple[str, str | None]
Identity = tuple[str | None, str]
Calibration = float | Mapping[FitKey, float] | Sequence[Fit] | None


class LabelAuditInputError(ValueError):
    """A record the ranking cannot calibrate: no fit for its group, or a fit for other classes.

    Detectable only while ranking, so the web catches it explicitly rather than at a file read.
    """


@dataclass(frozen=True)
class Flag:
    dataset: str | None
    item_id: str
    question_id: str
    label: str
    predicted: str
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Skipped:
    no_response: int = 0
    unlabeled: int = 0


def calibrate(probs: Mapping, temperature: float | None) -> dict:
    if temperature is None:
        return dict(probs)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    if temperature == 1.0:
        return dict(probs)
    logits = {k: math.log(max(v, EPS)) / temperature for k, v in probs.items()}
    top = max(logits.values())
    weights = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def temperatures_from_fits(fits: Iterable[Fit]) -> dict[FitKey, float]:
    """Extract `(question_id, model_served) -> temperature` from a loaded `CalibFit` list."""
    out: dict[FitKey, float] = {}
    for f in fits:
        if f.method == "temperature" and "temperature" in f.params:
            out[(f.question_id, f.model_served)] = float(f.params["temperature"])
    return out


def _calibrated(fit: Calibration, record: Record) -> dict:
    """`record`'s distribution `{class: p}` under `fit`.

    A float is one temperature for every record; a mapping is `(question_id, model_served) -> T`;
    a `Fit` sequence is looked up by the full group key, so a fit for another model or primitive
    is never used. A temperature keeps `calibrate`'s arithmetic; platt and isotonic go through
    `calibrate.transform`, as `s1 apply` does.
    """
    probs = dict(zip(record.classes, record.probs))
    if fit is None:
        return probs
    if isinstance(fit, (int, float)) and not isinstance(fit, bool):
        return calibrate(probs, float(fit))
    where = f"item {record.item} of group {record.key}"
    if isinstance(fit, Sequence) and not isinstance(fit, (str, bytes)):
        fits = list(fit)
        found = fit_for(fits, record.key)
        problem = refusal(fits, record.key, record.classes)
        if found is None or problem:
            raise LabelAuditInputError(f"{where}: {problem}; refusing to rank on another group's calibration")
        if found.method == "temperature":
            return calibrate(probs, float(found.params["temperature"]))
        index = {c: i for i, c in enumerate(found.classes)}
        Q = transform(found.method, found.primitive, found.params, np.array([[probs[c] for c in found.classes]]))[0]
        return {c: float(Q[index[c]]) for c in record.classes}
    if isinstance(fit, Mapping):
        key = (record.question_id, record.model_served)
        if key not in fit:
            raise LabelAuditInputError(f"{where}: no temperature fitted for {key}; refusing to rank on another model's calibration")
        return calibrate(probs, fit[key])
    raise TypeError(f"unsupported temperature specification {type(fit).__name__}")


def _answer(record: Record):
    raw = record.raw
    if record.primitive == "noul":
        return raw["p"] >= 0.5
    if record.primitive == "choice":
        return raw["choice"]
    return max(zip(record.classes, record.probs), key=lambda cp: cp[1])[0]


def disagreement(
    row: Mapping,
    temperature: Calibration = None,
    line: int = 0,
) -> Flag | None:
    record = parse_record(row, line)
    if record.model_served is None or record.label is None:
        return None
    probs = _calibrated(temperature, record)
    answer = _answer(record)
    score = 0.0 if answer == record.label else max(0.0, probs[answer] - probs[record.label])
    return Flag(record.dataset, record.item_id, record.question_id, str(record.label), str(answer), score)


def rank_with_skips(
    records: Iterable[Mapping],
    temperature: Calibration = None,
    balance: bool = True,
) -> tuple[list[Flag], Skipped]:
    flags: list[Flag] = []
    no_response = unlabeled = 0
    for n, row in enumerate(records, start=1):
        f = disagreement(row, temperature, n)
        if f is None:
            if row.get("model_served") is None:
                no_response += 1
            else:
                unlabeled += 1
            continue
        flags.append(f)
    skipped = Skipped(no_response, unlabeled)
    ordered = sorted(flags, key=lambda f: (-f.score, f.item_id))
    if not balance:
        return ordered, skipped
    by_class: dict[str, list[Flag]] = {}
    for f in ordered:
        by_class.setdefault(f.label, []).append(f)
    out: list[Flag] = []
    queues = [by_class[k] for k in sorted(by_class)]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out, skipped


def rank(
    records: Iterable[Mapping],
    temperature: Calibration = None,
    balance: bool = True,
) -> list[Flag]:
    return rank_with_skips(records, temperature, balance)[0]


@dataclass(frozen=True)
class Sheet:
    form: str
    key: dict


def quote(text: str) -> str:
    return "\n".join("> " + line for line in (text.splitlines() or [""]))


def blind_sheet(
    ranked: Sequence[Flag],
    text_of: Mapping[Identity, str],
    k: int,
    control_k: int,
    options: Sequence[str],
    seed: int = 0,
    title: str = "라벨 블라인드 재검수",
) -> Sheet:
    if k <= 0 or control_k <= 0:
        raise ValueError("k and control_k must be positive")
    unique: list[Flag] = []
    seen: set[Identity] = set()
    for f in ranked:
        if (f.dataset, f.item_id) not in seen:
            seen.add((f.dataset, f.item_id))
            unique.append(f)
    if len(unique) < k + control_k:
        raise ValueError(f"need at least {k + control_k} distinct items, got {len(unique)}")
    rng = random.Random(seed)
    flags = [(f, "flag") for f in unique[:k]]
    controls = [(f, "control") for f in rng.sample(unique[k:], control_k)]
    items = flags + controls
    missing = [(f.dataset, f.item_id) for f, _ in items if (f.dataset, f.item_id) not in text_of]
    if missing:
        raise ValueError(
            f"no state for {len(missing)} of the {len(items)} items under review, first {missing[0]!r}; "
            "the states file must hold every flagged and control item"
        )
    rng.shuffle(items)
    lines = [
        f"# {title}",
        "",
        f"각 항목의 올바른 답을 하나 고르세요: {', '.join(options)}. `답:` 뒤에 적어 주세요.",
        "",
    ]
    key = {"seed": seed, "k": k, "control_k": control_k, "options": list(options), "items": []}
    for n, (f, arm) in enumerate(items, 1):
        key["items"].append(
            {
                "n": n,
                "item_id": f.item_id,
                "dataset": f.dataset,
                "question_id": f.question_id,
                "arm": arm,
                "label": f.label,
                "predicted": f.predicted,
                "score": f.score,
            }
        )
        lines += [f"## {n}", "", quote(text_of[(f.dataset, f.item_id)]), "", "답: ", ""]
    return Sheet("\n".join(lines), key)


def parse_answers(form: str, options: Sequence[str]) -> dict[str, str]:
    allowed = set(options)
    out: dict[str, str] = {}
    for n, body in ANSWER_RE.findall(form):
        parts = re.split(r"^답:", body, flags=re.M)
        if len(parts) != 2:
            raise ValueError(f"item {n}: expected one line starting with 답:")
        answer = " ".join(parts[1].split())
        if answer not in allowed:
            raise ValueError(f"item {n}: answer {parts[1].strip()!r} is not one of {sorted(allowed)}")
        if n in out:
            raise ValueError(f"item {n} is answered more than once; keep one `## {n}` section")
        out[n] = answer
    return out


class _KeyItem(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    n: int = Field(ge=1)
    arm: Literal["flag", "control"]
    label: str
    predicted: str


class _Key(BaseModel):
    """The parts of `blind_sheet`'s secret key that `score_sheet` reads."""

    model_config = ConfigDict(extra="allow", strict=True)

    options: list[str] = Field(min_length=1)
    items: list[_KeyItem]


def check_key(key: object) -> None:
    """Refuse a secret key `score_sheet` cannot score: a wrong shape, a repeated item number
    (one answer would count twice), or an arm with no items (its rate would divide by zero)."""
    if not isinstance(key, Mapping):
        raise ValueError(f"the key must be a JSON object, got {type(key).__name__}")
    try:
        doc = _Key.model_validate(key)
    except ValidationError as e:
        raise ValueError("; ".join(pydantic_errors(e))) from None
    seen: set[int] = set()
    for i, item in enumerate(doc.items):
        if item.n in seen:
            raise ValueError(f"items.{i}.n: item {item.n} appears more than once")
        seen.add(item.n)
    for arm in ("flag", "control"):
        if not any(item.arm == arm for item in doc.items):
            raise ValueError(f"items: no {arm} item; a key from `s1 labelaudit sheet` has both arms")


def score_sheet(form: str, key: Mapping) -> dict:
    check_key(key)
    answers = parse_answers(form, key["options"])
    out = {
        arm: {"n": 0, "reviewer_disagrees_with_label": 0, "reviewer_agrees_with_model": 0}
        for arm in ("flag", "control")
    }
    for item in key["items"]:
        a = answers.get(str(item["n"]))
        if a is None:
            raise ValueError(f"item {item['n']}: no answer")
        arm = out[item["arm"]]
        arm["n"] += 1
        arm["reviewer_disagrees_with_label"] += a != item["label"]
        arm["reviewer_agrees_with_model"] += a == item["predicted"]
    return out


def precision_at_k(hits: int, k: int) -> tuple[float, tuple[float, float]]:
    return hits / k, wilson(hits, k)


@dataclass(frozen=True)
class Estimate:
    flag_precision: float
    flag_ci: tuple[float, float]
    control_rate: float
    control_ci: tuple[float, float]
    lift: float
    conservative_lift: float

    def outcome(self, factor: float = 2.0) -> str:
        if self.lift < factor:
            return "fail"
        return "pass" if self.conservative_lift >= 1.0 else "inconclusive"

    def passes(self, factor: float = 2.0) -> bool:
        return self.outcome(factor) == "pass"

    def as_dict(self, factor: float = 2.0) -> dict:
        return {
            **asdict(self),
            "flag_ci": list(self.flag_ci),
            "control_ci": list(self.control_ci),
            "outcome": self.outcome(factor),
            "passes": self.passes(factor),
        }


def estimate(flag_hits: int, flag_n: int, control_hits: int, control_n: int) -> Estimate:
    fp, fci = precision_at_k(flag_hits, flag_n)
    cr, cci = precision_at_k(control_hits, control_n)
    lift = math.inf if cr == 0 else fp / cr
    conservative = math.inf if cci[1] == 0 else fci[0] / cci[1]
    return Estimate(fp, fci, cr, cci, lift, conservative)
