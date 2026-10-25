"""Blind label noise auditor: calibrated disagreement ranking, blind sheet with random control, and Wilson lift."""

import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from s1ops.contracts.records import Record, parse_record
from s1ops.improve.calibrate import Fit
from s1ops.improve.slices import wilson

EPS = 1e-6
ANSWER_RE = re.compile(r"^## (\d+)[ \t]*\n(.*?)(?=^## \d+[ \t]*$|\Z)", re.M | re.S)

FitKey = tuple[str, str | None]
Identity = tuple[str | None, str]


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


def _temperature(fit: float | Mapping[FitKey, float] | Sequence[Fit] | None, record: Record) -> float | None:
    if fit is None:
        return None
    if isinstance(fit, (int, float)) and not isinstance(fit, bool):
        return float(fit)
    if isinstance(fit, Sequence) and not isinstance(fit, (str, bytes)):
        fit = temperatures_from_fits(fit)
    if isinstance(fit, Mapping):
        key = (record.question_id, record.model_served)
        if key not in fit:
            raise KeyError(f"no temperature fitted for {key}; refusing to rank on another model's calibration")
        return fit[key]
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
    temperature: float | Mapping[FitKey, float] | Sequence[Fit] | None = None,
    line: int = 0,
) -> Flag | None:
    record = parse_record(row, line)
    if record.model_served is None or record.label is None:
        return None
    probs = calibrate(dict(zip(record.classes, record.probs)), _temperature(temperature, record))
    answer = _answer(record)
    score = 0.0 if answer == record.label else max(0.0, probs[answer] - probs[record.label])
    return Flag(record.dataset, record.item_id, record.question_id, str(record.label), str(answer), score)


def rank_with_skips(
    records: Iterable[Mapping],
    temperature: float | Mapping[FitKey, float] | Sequence[Fit] | None = None,
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
    temperature: float | Mapping[FitKey, float] | Sequence[Fit] | None = None,
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


def score_sheet(form: str, key: Mapping) -> dict:
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
