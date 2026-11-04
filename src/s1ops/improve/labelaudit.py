"""Blind label noise auditor: calibrated disagreement ranking, blind sheet with random control, and Wilson lift."""

import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s1ops.contracts._validation import pydantic_errors
from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, QuestionModel, ScoreQuestion, parse_wire_question
from s1ops.contracts.questions import question_id as question_id_of
from s1ops.contracts.records import Record, parse_record, probability_total
from s1ops.improve.calibrate import Fit, fit_for, refusal, transform
from s1ops.improve.slices import wilson

EPS = 1e-6
SHEET_TITLE = "라벨 블라인드 재검수"
ANSWER_RE = re.compile(r"^## (\d+)[ \t]*\n(.*?)(?=^## \d+[ \t]*$|\Z)", re.M | re.S)

FitKey = tuple[str, str | None]
Identity = tuple[str | None, str]
Calibration = float | Mapping[FitKey, float] | Sequence[Fit] | None


class SheetInputError(ValueError):
    """`build_sheet` cannot make a sheet a reviewer can answer for one question. `source` says
    where the problem is: `"records"` or `"question"` (the question passed in); `many_questions`
    is set when the records hold more than one question, so a caller can say how to pick one."""

    def __init__(self, message: str, source: str = "records", many_questions: bool = False) -> None:
        super().__init__(message)
        self.source, self.many_questions = source, many_questions


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
    total = probability_total(weights.values())
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


def _normalized(answer: str) -> str:
    """What `parse_answers` makes of a reviewer's answer."""
    return " ".join(answer.split())


def check_sheet_records(records: Sequence[Record]) -> None:
    """Refuse answered, labeled records of one question that one review sheet cannot ask about
    consistently: more than one primitive, different class sets, or one item labeled differently
    by two records (a sheet shows each item once, so it would keep whichever label ranked first)."""
    if not records:
        return
    primitives = sorted({r.primitive for r in records})
    if len(primitives) > 1:
        raise ValueError(f"question {records[0].question_id} is answered as {', '.join(primitives)}; a sheet asks one kind of question")
    first = records[0]
    labels: dict[Identity, Record] = {}
    for r in records:
        # Primitive first: noul (False, True) and score (0, 1) compare equal as sets.
        if set(r.classes) != set(first.classes):
            raise ValueError(
                f"item {r.item} of model {r.model_served!r} has classes {list(r.classes)}, "
                f"item {first.item} of model {first.model_served!r} has {list(first.classes)}"
            )
        seen = labels.setdefault(r.item, r)
        if seen.label != r.label:
            raise ValueError(
                f"item {r.item} is labeled {seen.label!r} in a record of model {seen.model_served!r} "
                f"and {r.label!r} in one of model {r.model_served!r}"
            )


def sheet_options(records: Sequence[Record], given: Sequence[str] | None = None) -> list[str]:
    """The answers a sheet offers for one question, as `Flag.label` writes labels into its key:
    `str()` of each class, noul as `True, False`. Given options must be exactly those classes, in
    any order (`check_options`)."""
    first = records[0]
    classes = [str(c) for c in first.classes]
    derived = ["True", "False"] if first.primitive == "noul" else classes
    options = list(given) if given is not None else derived
    check_options(options, classes, first.question_id)
    return options


def check_options(options: Sequence[str], labels: Iterable[str], question_id: str) -> None:
    """Refuse answers a reviewer could not give back as written: a non-string or empty option, one
    that `parse_answers`' whitespace normalization changes (which also refuses a newline, so an
    option cannot open an answer section), a repeat, or options that are not exactly `labels`, the
    question's answers (a missing one cannot be given; an extra one is no label)."""
    for o in options:
        if not isinstance(o, str) or not o or _normalized(o) != o:
            raise ValueError(
                f"option {o!r} cannot be answered: a reviewer's answer is read with its spaces collapsed and trimmed"
            )
    dup = sorted({o for o in options if list(options).count(o) > 1})
    if dup:
        raise ValueError(f"options repeat {dup}")
    labels = set(labels)
    missing = sorted(lab for lab in labels if lab not in options)
    if missing:
        raise ValueError(f"options {list(options)} cannot answer labels {missing} of question {question_id}")
    extra = [o for o in options if o not in labels]
    if extra:
        raise ValueError(f"options {extra} are not answers of question {question_id} (its answers: {sorted(labels)})")


def _shown(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def check_question(question: QuestionModel, primitive: str, classes: Sequence) -> None:
    """Refuse a question that is not the one the records answer: another primitive, choice keys
    that differ, or score levels past its criteria."""
    if question.type != primitive:
        raise ValueError(f"the question is a {question.type} question, the records answer a {primitive} one")
    if question.type == "choice" and set(question.criteria) != {str(c) for c in classes}:
        raise ValueError(f"the question's options {sorted(question.criteria)} differ from the records' {sorted(map(str, classes))}")
    if question.type == "score" and any(c >= len(question.criteria) for c in classes):
        raise ValueError(f"the records' levels {list(classes)} go past the question's {len(question.criteria)} levels")


def question_lines(question_id: str, question: QuestionModel | None) -> list[str]:
    """The block printed under a sheet's title. Every line of the question's own text is quoted,
    so wording that holds `## 1` or `답:` cannot open an answer section; so is the id, which the
    record parser accepts with newlines in it."""
    lines = ["질문 ID:", "", quote(question_id), ""]
    if question is None:
        return lines
    lines += ["질문:", "", quote(_shown(question.instructions)), ""]
    if question.type == "noul":
        criteria = [(o, (question.criteria or {}).get(k)) for o, k in (("True", "true"), ("False", "false"))]
        criteria = [(o, c) for o, c in criteria if c is not None]
    elif question.type == "choice":
        criteria = list(question.criteria.items())
    else:
        criteria = [(str(i), c) for i, c in enumerate(question.criteria)]
    if criteria:
        lines += ["선택지:", "", quote("\n".join(f"- {o}: {_shown(c)}" for o, c in criteria)), ""]
    return lines


def _render_sheet(
    ranked: Sequence[Flag],
    text_of: Mapping[Identity, str],
    k: int,
    control_k: int,
    options: Sequence[str],
    seed: int,
    question_id: str,
    question: QuestionModel | None,
) -> Sheet:
    """Render a sheet and its key from inputs `build_sheet` has validated: flags of one question,
    options that are exactly its answers, the matching question or none. Only the options reach
    the form unquoted, and `check_options` keeps them to single normalized lines."""
    header = question_lines(question_id, question)
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
        f"# {SHEET_TITLE}",
        "",
        f"각 항목의 올바른 답을 하나 고르세요: {', '.join(options)}. `답:` 뒤에 적어 주세요.",
        "",
        *header,
    ]
    key = {"seed": seed, "k": k, "control_k": control_k, "options": list(options), "question_id": question_id, "items": []}
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


def build_sheet(
    records: Sequence[Record],
    text_of: Mapping[Identity, str],
    k: int,
    control_k: int,
    *,
    options: Sequence[str] | None = None,
    question: QuestionModel | Mapping | None = None,
    fit: Calibration = None,
    seed: int = 0,
    balance: bool = True,
) -> Sheet:
    """A blind review sheet and its secret key for the records of one question.

    Every check the sheet depends on runs here, on the records: one question among the answered,
    labeled records (`SheetInputError.many_questions` otherwise), one primitive, class set and
    label per item across models (`check_sheet_records`), options that are exactly the question's
    answers (`sheet_options`; derived when not given), and a `question` (parsed or its wire
    mapping) that is the records' question in id, primitive and classes (`check_question`).
    The records are then ranked under `fit` (`rank_with_skips`, class-balanced unless `balance`
    is false) and the sheet rendered: `k`
    flagged items and `control_k` random controls, with their states from `text_of`."""
    eligible = [r for r in records if r.model_served is not None and r.label is not None]
    if not eligible:
        raise SheetInputError("no answered, labeled record to put on a sheet")
    question_ids = sorted({r.question_id for r in eligible})
    if len(question_ids) > 1:
        shown = ", ".join(question_ids[:5]) + (", ..." if len(question_ids) > 5 else "")
        raise SheetInputError(
            f"records hold {len(question_ids)} questions ({shown}); a review sheet asks one question",
            many_questions=True,
        )
    question_id = question_ids[0]
    try:
        check_sheet_records(eligible)
        chosen = sheet_options(eligible, options)
    except ValueError as e:
        raise SheetInputError(str(e)) from None
    if question is not None:
        if not isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
            try:
                question = parse_wire_question(dict(question))
            except ValueError as e:
                raise SheetInputError(f"question: {e}", source="question") from None
        if question_id_of(question) != question_id:
            raise SheetInputError(f"the question is {question_id_of(question)}, the records are for {question_id}", source="question")
        try:
            check_question(question, eligible[0].primitive, eligible[0].classes)
        except ValueError as e:
            raise SheetInputError(f"question {question_id}: {e}", source="question") from None
    flags, _ = rank_with_skips([r.raw for r in records], temperature=fit, balance=balance)
    return _render_sheet(flags, text_of, k, control_k, chosen, seed, question_id, question)


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


def _key_errors(error: ValidationError) -> list[str]:
    # pydantic names the model class for a non-object item ("instance of _KeyItem"); say "object".
    out = []
    for e, text in zip(error.errors(include_url=False), pydantic_errors(error)):
        if e["type"] in ("model_type", "dict_type"):
            text = f"{'.'.join(str(part) for part in e['loc']) or '(root)'}: must be an object"
        out.append(text)
    return out


def check_key(key: object) -> None:
    """Refuse a secret key `score_sheet` cannot score: a wrong shape, a repeated item number
    (one answer would count twice), or an arm with no items (its rate would divide by zero)."""
    if not isinstance(key, Mapping):
        raise ValueError(f"the key must be a JSON object, got {type(key).__name__}")
    try:
        doc = _Key.model_validate(key)
    except ValidationError as e:
        raise ValueError("; ".join(_key_errors(e))) from None
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
