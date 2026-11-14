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
from s1ops.contracts.records import Record, RecordError, label_problem, parse_record, probability_total, reparse, without_state
from s1ops.improve.calibrate import Fit, fit_for, refusal, transform
from s1ops.improve.slices import wilson

EPS = 1e-6
SHEET_TITLE = "Label blind review"
ANSWER_RE = re.compile(r"^## (\d+)[ \t]*\n(.*?)(?=^## \d+[ \t]*$|\Z)", re.M | re.S)

FitKey = tuple[str, str | None]
Identity = tuple[str | None, str]
Calibration = float | Mapping[FitKey, float] | Sequence[Fit] | None


class SheetInputError(ValueError):
    """`build_sheet` cannot make a sheet a reviewer can answer for one question. `source` says
    where the problem is: `"records"`, `"question"` (the question passed in) or `"states"` (the
    state text passed in); `many_questions` is set when the records hold more than one question,
    so a caller can say how to pick one."""

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
    so wording that holds `## 1` or `Answer:` cannot open an answer section; so is the id, which the
    record parser accepts with newlines in it."""
    lines = ["Question ID:", "", quote(question_id), ""]
    if question is None:
        return lines
    lines += ["Question:", "", quote(_shown(question.instructions)), ""]
    if question.type == "noul":
        criteria = [(o, (question.criteria or {}).get(k)) for o, k in (("True", "true"), ("False", "false"))]
        criteria = [(o, c) for o, c in criteria if c is not None]
    elif question.type == "choice":
        criteria = list(question.criteria.items())
    else:
        criteria = [(str(i), c) for i, c in enumerate(question.criteria)]
    if criteria:
        lines += ["Options:", "", quote("\n".join(f"- {o}: {_shown(c)}" for o, c in criteria)), ""]
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
        raise SheetInputError(f"need at least {k + control_k} distinct items, got {len(unique)}")
    rng = random.Random(seed)
    flags = [(f, "flag") for f in unique[:k]]
    controls = [(f, "control") for f in rng.sample(unique[k:], control_k)]
    items = flags + controls
    missing = [(f.dataset, f.item_id) for f, _ in items if (f.dataset, f.item_id) not in text_of]
    if missing:
        raise SheetInputError(
            f"no state for {len(missing)} of the {len(items)} items under review, first {missing[0]!r}; "
            "the states file must hold every flagged and control item",
            source="states",
        )
    rng.shuffle(items)
    lines = [
        f"# {SHEET_TITLE}",
        "",
        f"Choose one answer for each item: {', '.join(options)}. Write it after `Answer:`.",
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
        lines += [f"## {n}", "", quote(text_of[(f.dataset, f.item_id)]), "", "Answer: ", ""]
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
    flagged items and `control_k` random controls, with their states from `text_of`.

    The records' raw rows are the source: each is parsed again (`reparse`, which keeps its file in
    errors) before anything is checked, and the same rows are ranked, so a `Record` edited after
    parsing, or parsed with looser options, is read as its row says."""
    records = [reparse(r) for r in records]
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
    flags, _ = rank_with_skips([r.raw for r in eligible], temperature=fit, balance=balance)
    return _render_sheet(flags, text_of, k, control_k, chosen, seed, question_id, question)


# `답:` is what sheets printed by s1ops 0.6.0 and earlier say, and one may still be with a reviewer.
ANSWER_MARKER = re.compile(r"^(?:Answer:|답:)", re.M)


def parse_answers(form: str, options: Sequence[str]) -> dict[str, str]:
    allowed = set(options)
    out: dict[str, str] = {}
    for n, body in ANSWER_RE.findall(form):
        parts = ANSWER_MARKER.split(body)
        if len(parts) != 2:
            raise ValueError(f"item {n}: expected one line starting with Answer:")
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


# `질문 ID:` is what sheets printed by s1ops 0.6.0 and earlier say.
_SHEET_QUESTION = re.compile(r"^(?:Question ID:|질문 ID:)[ \t]*\n[ \t]*\n((?:>.*(?:\n|\Z))+)", re.M)
_FIRST_ITEM = re.compile(r"^## \d+[ \t]*$", re.M)


def sheet_question_id(form: str) -> str | None:
    """The question a sheet says it asks: the quoted block under `Question ID:`, above the first
    item. `None` when the sheet has no such block."""
    first = _FIRST_ITEM.search(form)
    found = _SHEET_QUESTION.search(form[: first.start()] if first else form)
    if found is None:
        return None
    lines = found.group(1).rstrip("\n").split("\n")
    return "\n".join(line[2:] if line.startswith("> ") else line[1:] for line in lines)


class LabelPatchError(ValueError):
    """A finished review that cannot be applied to these records. `source` says which input is at
    fault: the `key`, the `sheet` or the `records`."""

    def __init__(self, message: str, source: str = "records") -> None:
        super().__init__(message)
        self.source = source


@dataclass(frozen=True)
class Correction:
    """One reviewed item whose reviewer answered otherwise than its label."""

    dataset: str | None
    item_id: str
    question_id: str
    arm: str
    label_from: bool | int | str
    label_to: bool | int | str

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "item_id": self.item_id, "question_id": self.question_id, "arm": self.arm, "from": self.label_from, "to": self.label_to}


@dataclass(frozen=True)
class LabelPatch:
    rows: list[dict]
    """Every input row in order, without `state`; a corrected item's rows carry the reviewer's answer."""
    reviewed: int
    corrections: tuple[Correction, ...]
    records_patched: int
    records_already: int
    """Rows of a corrected item that carried the reviewer's answer already: the patch was applied before."""


class _PatchKeyItem(_KeyItem):
    item_id: str = Field(min_length=1)
    dataset: str | None
    question_id: str = Field(min_length=1)


class _PatchKey(_Key):
    """The parts of the secret key `patch_labels` reads, beyond what scoring does."""

    question_id: str = Field(min_length=1)
    items: list[_PatchKeyItem]


def patch_labels(records: Sequence[Record], form: str, key: Mapping) -> LabelPatch:
    """Apply a finished blind review to the labels of the records its sheet was built from.

    Where the reviewer's answer differs from the label the key holds, every labeled row of that
    item and question takes the answer, whichever model answered it, and a row whose probe failed
    as well: a label is the item's. The
    answer is one of the sheet's options, which are the question's classes written out, so it
    becomes the class it names (a noul boolean, a choice option, a score level).

    The records must be the sheet's: every reviewed item is in them, and a row of a corrected item
    carries either the label the reviewer was shown to disagree with or, when the patch was applied
    before, the answer. Anything else is refused (`LabelPatchError`) rather than overwritten.
    Rows are copied as `s1 apply` copies them, without a top-level `state`."""
    try:
        check_key(key)
        doc = _PatchKey.model_validate(key)
    except ValidationError as e:
        raise LabelPatchError("; ".join(_key_errors(e)), "key") from None
    except ValueError as e:
        raise LabelPatchError(str(e), "key") from None
    try:
        answers = parse_answers(form, doc.options)
    except ValueError as e:
        raise LabelPatchError(str(e), "sheet") from None
    # The sheet and the key must be of one review: two sheets number their items alike, so the
    # answers of one would otherwise be applied through the key of the other.
    on_sheet = sheet_question_id(form)
    if on_sheet is None:
        raise LabelPatchError("no `Question ID:` block above the first item, so the sheet cannot be matched to the key", "sheet")
    if on_sheet != "\n".join(doc.question_id.splitlines() or [""]):
        raise LabelPatchError(f"the sheet asks question {on_sheet}, and the key is for {doc.question_id}", "sheet")
    extra = sorted(set(answers) - {str(item.n) for item in doc.items}, key=int)
    if extra:
        raise LabelPatchError(
            f"item {extra[0]} is answered on the sheet and the key has no such item ({len(extra)} in all); they are not of one review",
            "sheet",
        )
    records = [reparse(r) for r in records]
    asked = [r for r in records if r.question_id == doc.question_id and r.model_served is not None and r.label is not None]
    if not asked:
        raise LabelPatchError(f"no answered, labeled record is for question {doc.question_id}, the one the sheet asked")
    try:
        check_sheet_records(asked)
        sheet_options(asked, doc.options)
    except ValueError as e:
        raise LabelPatchError(str(e)) from None
    primitive, classes = asked[0].primitive, asked[0].classes
    named = {str(c): c for c in classes}

    reviewed: dict[tuple[Identity, str], _PatchKeyItem] = {}
    for item in doc.items:
        if item.question_id != doc.question_id:
            raise LabelPatchError(f"item {item.n} is for question {item.question_id}, the key for {doc.question_id}", "key")
        if str(item.n) not in answers:
            raise LabelPatchError(f"item {item.n}: no answer", "sheet")
        if item.label not in named:
            raise LabelPatchError(f"item {item.n}: label {item.label!r} is not one of the options {doc.options}", "key")
        at = ((item.dataset, item.item_id), item.question_id)
        if at in reviewed:
            raise LabelPatchError(f"items {reviewed[at].n} and {item.n} are both {at[0]!r}; a sheet shows each item once", "key")
        reviewed[at] = item

    seen: set[tuple[Identity, str]] = set()
    corrections: dict[tuple[Identity, str], Correction] = {}
    rows: list[dict] = []
    patched = already = 0
    for r in records:
        row = without_state(r.raw)
        at = (r.item, r.question_id)
        item = reviewed.get(at)
        # A probe that failed still wrote the item's label, which the parser does not read.
        label = r.label if r.model_served is not None else r.raw.get("label")
        if item is not None and label is not None:
            if r.model_served is not None:
                seen.add(at)
            else:
                problem = label_problem(primitive, label, classes)
                if problem is not None:
                    raise RecordError(r.line, problem, r.path)
            answer = answers[str(item.n)]
            old, new = named[item.label], named[answer]
            if str(label) not in (item.label, answer):
                raise RecordError(
                    r.line,
                    f"item {r.item} is labeled {label!r}, and the review was of the label {old!r} (item {item.n} of the key); "
                    "these are not the records the sheet was built from",
                    r.path,
                )
            if answer != item.label:
                if str(label) == answer:
                    already += 1
                else:
                    problem = label_problem(primitive, new, classes)
                    if problem is not None:  # the options are the classes, so only a hand-made key gets here
                        raise LabelPatchError(f"item {item.n}: {problem}", "key")
                    row["label"] = new
                    patched += 1
                corrections.setdefault(at, Correction(item.dataset, item.item_id, item.question_id, item.arm, old, new))
        rows.append(row)
    missing = [item for at, item in reviewed.items() if at not in seen]
    if missing:
        first = missing[0]
        raise LabelPatchError(
            f"{len(missing)} of the {len(reviewed)} reviewed items have no labeled record, first {(first.dataset, first.item_id)!r} "
            f"(item {first.n} of the key); these are not the records the sheet was built from"
        )
    ordered = sorted(corrections.values(), key=lambda c: (c.dataset is not None, c.dataset or "", c.item_id))
    return LabelPatch(rows, len(reviewed), tuple(ordered), patched, already)


def precision_at_k(hits: int, k: int) -> tuple[float, tuple[float, float]]:
    return hits / k, wilson(hits, k)


def _finite_or_none(value: float) -> float | None:
    """Infinity is not JSON; `Estimate` keeps it for `outcome`, and only its dict says null."""
    return value if math.isfinite(value) else None


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
            "lift": _finite_or_none(self.lift),
            "conservative_lift": _finite_or_none(self.conservative_lift),
            "lift_note": None if math.isfinite(self.lift) else "control rate is 0",
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
