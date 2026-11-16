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
SHEET_INSTRUCTION = ("Choose one answer for each item: ", ". Write it after `Answer:`.")
"""What the line under a sheet's title says before and after its options, which it joins with `", "`."""
# What sheets printed by s1ops 0.6.0 and earlier say (e072369 made them English).
_OLD_SHEET_TITLE = "라벨 블라인드 재검수"
_OLD_SHEET_INSTRUCTION = ("각 항목의 올바른 답을 하나 고르세요: ", ". `답:` 뒤에 적어 주세요.")
ANSWER_RE = re.compile(r"^## (\d+)[ \t]*\n(.*?)(?=^## \d+[ \t]*$|\Z)", re.M | re.S)
_ITEM_HEADING = re.compile(r"^## \d+[ \t]*$", re.M)

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
        f"{SHEET_INSTRUCTION[0]}{', '.join(options)}{SHEET_INSTRUCTION[1]}",
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
        joined = [o for o in chosen if ", " in o]
        if joined:
            # `read_sheet` splits the instruction line back into the options; `patch` and `pull` take
            # the options from the key or the records, so only a sheet refuses this.
            raise ValueError(f"option {joined[0]!r} cannot be put on a sheet: its instruction line joins the options with ', '")
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


@dataclass(frozen=True)
class _Section:
    """One `## <n>` section: `n` as written, its body's span, and every answer marker in the body."""

    n: str
    start: int
    end: int
    markers: tuple[re.Match, ...]


def _sections(form: str) -> list[_Section]:
    """The answer sections both sheet readers see: `parse_answers` and `read_sheet` cannot disagree on
    where an item or its answer is. A marker counts only at a line start, so a quoted `> Answer:` in a
    state does not."""
    return [
        _Section(m.group(1), m.start(2), m.end(2), tuple(ANSWER_MARKER.finditer(form, m.start(2), m.end(2))))
        for m in ANSWER_RE.finditer(form)
    ]


def parse_answers(form: str, options: Sequence[str]) -> dict[str, str]:
    allowed = set(options)
    out: dict[str, str] = {}
    for sec in _sections(form):
        n = sec.n
        if len(sec.markers) != 1:
            raise ValueError(f"item {n}: expected one line starting with Answer:")
        given = form[sec.markers[0].end() : sec.end]
        answer = " ".join(given.split())
        if answer not in allowed:
            raise ValueError(f"item {n}: answer {given.strip()!r} is not one of {sorted(allowed)}")
        if n in out:
            raise ValueError(f"item {n} is answered more than once; keep one `## {n}` section")
        out[n] = answer
    return out


class SheetFormatError(ValueError):
    """`read_sheet` cannot read a review sheet in the form `s1 labelaudit sheet` writes. The message
    names an item number or a kind of line, never the sheet's text."""


@dataclass(frozen=True)
class SheetItem:
    n: int
    state: str
    answer: str | None


@dataclass(frozen=True)
class SheetForm:
    """A review sheet as the web workbench answers it. `pieces` is the sheet cut where each item's
    answer goes, the answers taken out (N items, N+1 pieces, each but the last ending with the marker
    and one space), so `fill` writes answers into the sheet without rendering it again."""

    question_id: str | None
    header: str
    options: tuple[str, ...]
    items: tuple[SheetItem, ...]
    pieces: tuple[str, ...]

    def fill(self, answers: Mapping[int, str | None]) -> str:
        """The sheet with `answers` by item number written in, `None` clearing one; an item not given
        keeps the sheet's own answer. `fill({})` is the sheet."""
        for k, v in answers.items():
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(self.items):
                raise ValueError(f"the sheet has no item {k!r}")
            if v is not None and v not in self.options:
                raise ValueError(f"item {k}: the answer is not one of the sheet's options")
        out = [self.pieces[0]]
        for item, piece in zip(self.items, self.pieces[1:]):
            answer = answers[item.n] if item.n in answers else item.answer
            out += [answer or "", piece]
        return "".join(out)


def _first_lines(form: str, count: int) -> list[tuple[int, str]]:
    """The first `count` non-blank lines as (offset after the line, the line without trailing spaces)."""
    out, pos = [], 0
    for line in form.split("\n"):
        pos += len(line) + 1
        if line.strip():
            out.append((pos, line.rstrip(" \t")))
            if len(out) == count:
                break
    return out


def _unquoted(text: str) -> str:
    """A state as `quote` wrote it: the blank lines around the quoted block dropped first, so a
    state's own empty first or last line (`> `) is kept."""
    lines = text.split("\n")
    while lines and not lines[0].strip(" \t"):
        lines.pop(0)
    while lines and not lines[-1].strip(" \t"):
        lines.pop()
    return "\n".join(line[2:] if line.startswith("> ") else "" if line == ">" else line for line in lines)


def read_sheet(form: str) -> SheetForm:
    """Read a review sheet in the form `s1 labelaudit sheet` writes it (0.6.0's Korean one too), for the
    web workbench to answer: the title, the instruction line naming the options, items `## 1` .. `## N`
    in order, and under each one marker line that says `Answer: ` with nothing or one option after it
    and nothing below it. Anything else is `SheetFormatError` naming the item; `s1 labelaudit estimate`
    still scores an answer moved to the next line, though not a note under the answer."""
    first = _first_lines(form, 2)
    titles = {f"# {SHEET_TITLE}": SHEET_INSTRUCTION, f"# {_OLD_SHEET_TITLE}": _OLD_SHEET_INSTRUCTION}
    if not first or first[0][1] not in titles:
        raise SheetFormatError(f"the first line is not the title `# {SHEET_TITLE}`")
    prefix, suffix = titles[first[0][1]]
    if len(first) < 2 or not (first[1][1].startswith(prefix) and first[1][1].endswith(suffix)) or len(first[1][1]) < len(prefix) + len(suffix):
        raise SheetFormatError("the line after the title is not the instruction line naming the options")
    header_start, line = first[1]
    options = line[len(prefix) : len(line) - len(suffix)].split(", ")
    if any(not o or _normalized(o) != o for o in options) or len(set(options)) != len(options):
        raise SheetFormatError("the instruction line's options cannot be read: one is empty, repeated, or has extra spaces")
    sections = _sections(form)
    if not sections:
        raise SheetFormatError("the sheet has no item")
    starts = {m.start() for m in ANSWER_RE.finditer(form)}
    for heading in _ITEM_HEADING.finditer(form):
        if heading.start() not in starts:
            line_no = form.count("\n", 0, heading.start()) + 1
            raise SheetFormatError(f"line {line_no} is an item heading that opens no item (one at the end of the file needs a line after it)")
    first_heading = min(starts)
    if first_heading < header_start:
        raise SheetFormatError("the line after the title is not the instruction line naming the options")
    items, pieces, pos = [], [], 0
    for i, sec in enumerate(sections, 1):
        if sec.n != str(i):
            raise SheetFormatError(f"the heading of item {i} is not `## {i}`: items are numbered 1, 2, ... in order")
        if len(sec.markers) != 1:
            raise SheetFormatError(f"item {i}: expected one line starting with Answer:")
        marker = sec.markers[0]
        answer_line, _, below = form[marker.end() : sec.end].partition("\n")
        given = answer_line[1:]
        if not answer_line.startswith(" ") or (given and given not in options) or below.strip(" \t\n"):
            raise SheetFormatError(
                f'item {i}: the web reads an answer only as "Answer: <option>" on its own line; '
                "fix the item by hand, or score the sheet with s1 labelaudit estimate"
            )
        cut = marker.end() + 1
        pieces.append(form[pos:cut])
        pos = cut + len(given)
        items.append(SheetItem(i, _unquoted(form[sec.start : marker.start()]), given or None))
    pieces.append(form[pos:])
    return SheetForm(
        question_id=sheet_question_id(form),
        header=form[header_start:first_heading].strip("\n"),
        options=tuple(options),
        items=tuple(items),
        pieces=tuple(pieces),
    )


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
    arm: str | None
    """`flag` or `control` for a sheet's item; None for a review pulled from Langfuse, which has no arms."""
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

    reviewed: dict[tuple[Identity, str], _Review] = {}
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
        reviewed[at] = _Review(item.label, answers[str(item.n)], named, primitive, classes, item.arm, f"item {item.n} of the key", "key", item.n)
    return _relabel(records, reviewed, "the sheet")


@dataclass(frozen=True)
class _Review:
    """One reviewed item of one question: the label the review was of, the reviewer's answer (both
    as the options write them), and what turns them into the question's classes."""

    old: str
    answer: str
    named: Mapping[str, bool | int | str]
    primitive: str
    classes: tuple
    arm: str | None
    where: str
    """Where the review is, for an error: `item 3 of the key`, `score <id>`."""
    source: str
    """The input a bad answer belongs to, as `LabelPatchError.source` names it."""
    n: int = 0


def _relabel(records: Sequence[Record], reviewed: Mapping[tuple[Identity, str], _Review], made_from: str) -> LabelPatch:
    """The one rule for applying reviews to records (`patch_labels`, `pull_labels`). Where an
    answer differs from the label the review was of, every labeled row of that item and question
    takes it, whichever model answered it, and a row whose probe failed as well. A row labeled
    neither the reviewed label nor the answer, or a reviewed item with no labeled record, is
    refused: these are not the records the review was of. Rows are copied without `state`."""
    seen: set[tuple[Identity, str]] = set()
    corrections: dict[tuple[Identity, str], Correction] = {}
    rows: list[dict] = []
    patched = already = 0
    for r in records:
        row = without_state(r.raw)
        at = (r.item, r.question_id)
        review = reviewed.get(at)
        # A probe that failed still wrote the item's label, which the parser does not read.
        label = r.label if r.model_served is not None else r.raw.get("label")
        if review is not None and label is not None:
            if r.model_served is not None:
                seen.add(at)
            else:
                problem = label_problem(review.primitive, label, review.classes)
                if problem is not None:
                    raise RecordError(r.line, problem, r.path)
            old, new = review.named[review.old], review.named[review.answer]
            if str(label) not in (review.old, review.answer):
                raise RecordError(
                    r.line,
                    f"item {r.item} is labeled {label!r}, and the review was of the label {old!r} ({review.where}); "
                    f"these are not the records {made_from} was built from",
                    r.path,
                )
            if review.answer != review.old:
                if str(label) == review.answer:
                    already += 1
                else:
                    problem = label_problem(review.primitive, new, review.classes)
                    if problem is not None:  # the options are the classes, so only a hand-made input gets here
                        raise LabelPatchError(f"{review.where}: {problem}", review.source)
                    row["label"] = new
                    patched += 1
                corrections.setdefault(at, Correction(at[0][0], at[0][1], at[1], review.arm, old, new))
        rows.append(row)
    missing = [(at, review) for at, review in reviewed.items() if at not in seen]
    if missing:
        at, first = missing[0]
        raise LabelPatchError(
            f"{len(missing)} of the {len(reviewed)} reviewed items have no labeled record, first {at[0]!r} "
            f"({first.where}); these are not the records {made_from} was built from"
        )
    ordered = sorted(corrections.values(), key=lambda c: (c.question_id, c.dataset is not None, c.dataset or "", c.item_id))
    return LabelPatch(rows, len(reviewed), tuple(ordered), patched, already)


LABEL_SCORE = re.compile(r"\As1\.(q-[0-9a-f]{16})\.label\Z")
"""The name of a Langfuse score that carries a reviewer's answer: `s1.<question_id>.label`, as the
export names its other scores (`s1.<question_id>.brier`)."""


class _ExportInput(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    dataset: str | None
    item_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)


class _ExportObservation(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    id: str = Field(min_length=1)
    traceId: str = Field(min_length=1)
    input: _ExportInput


class _Export(BaseModel):
    """The parts of an `s1 export langfuse` file `pull_labels` reads."""

    model_config = ConfigDict(extra="allow", strict=True)

    schema_: Literal["s1-langfuse-export/1"] = Field(alias="schema")
    redaction: Literal["safe-v1"]
    observations: list[_ExportObservation]


class _Subject(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    kind: str
    id: str = Field(min_length=1)
    traceId: str | None = None


class _Score(BaseModel):
    """A Scores API v3 row (`GET /api/public/v3/scores?fields=subject`): one typed `value`."""

    model_config = ConfigDict(extra="allow", strict=True)

    id: str | None = None
    name: str
    dataType: str
    value: bool | int | float | str | None = None
    subject: _Subject | None = None


@dataclass(frozen=True)
class LabelPull(LabelPatch):
    scores: int
    """`s1.<question_id>.label` scores on this export's observations and traces."""
    scores_elsewhere: int
    """Label scores on observations or traces the export does not hold (another run's): left out."""


def _score_answer(score: _Score, primitive: str, options: Sequence[str]) -> str | None:
    """The answer a label score gives, written as the options write it, or None when its type and
    value cannot answer this question: noul a BOOLEAN, choice a CATEGORICAL option, score a level
    as a whole NUMERIC or a CATEGORICAL."""
    v, kind = score.value, score.dataType.upper()
    answer: str | None = None
    if primitive == "noul" and kind == "BOOLEAN" and isinstance(v, bool):
        answer = "True" if v else "False"
    elif primitive in ("choice", "score") and kind == "CATEGORICAL" and isinstance(v, str):
        answer = v
    elif primitive == "score" and kind == "NUMERIC" and isinstance(v, (int, float)) and not isinstance(v, bool):
        if math.isfinite(v) and float(v).is_integer():
            answer = str(int(v))
    return answer if answer in options else None


_TAKES = {"noul": "a BOOLEAN", "choice": "a CATEGORICAL naming an option", "score": "a whole NUMERIC level or a CATEGORICAL naming one"}


def pull_labels(records: Sequence[Record], export: object, scores: object) -> LabelPull:
    """Apply reviews made in Langfuse to the labels of the records an export was made from.

    `export` is the `s1 export langfuse` file whose observations were reviewed; `scores` is a
    Scores API v3 response (`{"data": [...]}`, fetched with `fields=subject`) or its rows. A
    review is a score named `s1.<question_id>.label` on an exported observation, or on its trace
    (a trace is one item), whose value is the reviewer's answer. Other scores are not read, and a
    label score on an observation or trace the export does not hold is left out and counted.
    Answers become labels as `patch_labels` applies a sheet's: the label the review was of is the
    item's label in the records. Two answers for one item and question that differ are refused,
    as is anything `patch_labels` refuses (`LabelPatchError`, `.source` `export`, `scores` or
    `records`). s1ops makes no network call: fetching the scores is the caller's."""
    try:
        doc = _Export.model_validate(export)
    except ValidationError as e:
        raise LabelPatchError("; ".join(_key_errors(e)), "export") from None
    by_obs = {o.id: o for o in doc.observations}
    by_trace: dict[str, list[_ExportObservation]] = {}
    for o in doc.observations:
        by_trace.setdefault(o.traceId, []).append(o)

    rows = scores.get("data") if isinstance(scores, Mapping) else scores
    if not isinstance(rows, list):
        raise LabelPatchError("must be a Scores API v3 response (an object with a `data` list) or a list of its rows", "scores")
    answers: dict[tuple[Identity, str], list[tuple[str, str, _Score]]] = {}
    used = elsewhere = 0
    for i, raw in enumerate(rows):
        name = raw.get("name") if isinstance(raw, Mapping) else None
        m = LABEL_SCORE.match(name) if isinstance(name, str) else None
        if m is None:
            continue
        try:
            score = _Score.model_validate(raw)
        except ValidationError as e:
            raise LabelPatchError(f"data.{i}: " + "; ".join(_key_errors(e)), "scores") from None
        ref = f"score {score.id}" if score.id else f"score data.{i}"
        qid, subject = m.group(1), score.subject
        if subject is None:
            raise LabelPatchError(f"{ref} ({name}) has no `subject`; fetch the scores with `fields=subject`", "scores")
        if subject.kind == "observation":
            obs = by_obs.get(subject.id)
            if obs is None:
                elsewhere += 1
                continue
            if obs.input.question_id != qid:
                raise LabelPatchError(
                    f"{ref} is named {name} and is on observation {obs.id}, which asked question {obs.input.question_id}", "scores"
                )
            item = (obs.input.dataset, obs.input.item_id)
        elif subject.kind == "trace":
            on_trace = by_trace.get(subject.id)
            if on_trace is None:
                elsewhere += 1
                continue
            asked = [o for o in on_trace if o.input.question_id == qid]
            if not asked:
                raise LabelPatchError(f"{ref} is named {name} and is on trace {subject.id}, which did not ask question {qid}", "scores")
            item = (asked[0].input.dataset, asked[0].input.item_id)
        else:
            raise LabelPatchError(f"{ref} ({name}) is on a {subject.kind}; a label score is on an observation or a trace", "scores")
        used += 1
        answers.setdefault((item, qid), []).append((ref, name, score))
    if not answers:
        raise LabelPatchError(
            f"no `s1.<question_id>.label` score is on an observation or trace of this export ({elsewhere} on others)", "scores"
        )

    records = [reparse(r) for r in records]
    by_question: dict[str, list[Record]] = {}
    for r in records:
        if r.model_served is not None and r.label is not None:
            by_question.setdefault(r.question_id, []).append(r)
    reviewed: dict[tuple[Identity, str], _Review] = {}
    for qid in sorted({at[1] for at in answers}):
        asked = by_question.get(qid)
        if not asked:
            raise LabelPatchError(f"no answered, labeled record is for question {qid}, which the scores label")
        try:
            check_sheet_records(asked)
            options = sheet_options(asked)
        except ValueError as e:
            raise LabelPatchError(str(e)) from None
        primitive, classes = asked[0].primitive, asked[0].classes
        named = {str(c): c for c in classes}
        labels = {r.item: str(r.label) for r in asked}
        for (item, q), given in sorted(answers.items(), key=lambda kv: (kv[0][1], kv[0][0][0] or "", kv[0][0][1])):
            if q != qid:
                continue
            chosen: tuple[str, str] | None = None
            for ref, name, score in given:
                answer = _score_answer(score, primitive, options)
                if answer is None:
                    raise LabelPatchError(
                        f"{ref} ({name}) is {score.dataType} {score.value!r}; a {primitive} question takes {_TAKES[primitive]} "
                        f"of {options}",
                        "scores",
                    )
                if chosen is not None and chosen[1] != answer:
                    raise LabelPatchError(f"{chosen[0]} and {ref} answer item {item!r} {chosen[1]!r} and {answer!r}; keep one", "scores")
                chosen = (ref, answer)
            assert chosen is not None
            if item not in labels:
                raise LabelPatchError(
                    f"item {item!r} of question {qid} ({chosen[0]}) has no answered, labeled record; "
                    "these are not the records the export was made from"
                )
            reviewed[(item, qid)] = _Review(labels[item], chosen[1], named, primitive, classes, None, chosen[0], "scores")
    patch = _relabel(records, reviewed, "the export")
    return LabelPull(patch.rows, patch.reviewed, patch.corrections, patch.records_patched, patch.records_already, used, elsewhere)


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
