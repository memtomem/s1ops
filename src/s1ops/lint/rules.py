"""Static lint rules JL000-JL060 for System 1 decision questions."""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from s1ops.lint import text as T
from s1ops.lint.loader import QuestionSpec

Severity = Literal["error", "warning", "info"]

MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = (2, 10)
_ANCHORS = {"R§3": "r3", "R§6": "r6", "R§8": "r8", "R§11": "r11", "R-arch 3": "ra", "R-new 7": "rn"}


@dataclass(frozen=True)
class Rule:
    id: str
    severity: Severity
    title: str
    fix: str
    evidence: str | None

    @property
    def evidence_url(self) -> str | None:
        return None if self.evidence is None else f"REPORT.md#{_ANCHORS[self.evidence]}"


RULES: dict[str, Rule] = {
    r.id: r
    for r in [
        Rule("JL000", "error", "file or question does not parse", "Fix the shape so the question schema accepts it.", None),
        Rule("JL001", "error", "Choice option count outside 2..255", "Keep 2 to 255 options: split a larger set into rounds of <= 254 plus a catch-all and merge winners in code.", "R§3"),
        Rule("JL002", "warning", "no room for a reserved catch-all slot", "Keep options at 255 minus the slots your caller appends (e.g. none_of_the_above), or set --reserved-slots.", "R§8"),
        Rule("JL003", "error", "Score levels outside 2..10", "Use 2 to 10 ordered levels; put finer distinctions in code.", "R§3"),
        Rule("JL010", "warning", "negation inside a Noul", "Ask a Choice with the negated case as its own option (e.g. asserted / negated / hypothetical).", "R§6"),
        Rule("JL011", "warning", "double negation", "Rephrase positively; if the negated case matters, make it an explicit Choice option.", "R§11"),
        Rule("JL012", "warning", "counting or arithmetic asked of the model", "Count, sum or compare numbers in code; ask the model only the per-item judgment.", "R§11"),
        Rule("JL013", "warning", "date comparison or ordering", "Parse and compare dates in code; pass the result or ask about content only.", "R§11"),
        Rule("JL014", "warning", "more than one judgment in one question", "Split into one question per judgment in the same fan-out request, and combine the answers in code.", "R-arch 3"),
        Rule("JL015", "warning", "asks the model to generate text", "System 1 models return a typed decision, not text. Use an LLM to generate, or ask a closed question.", "R§11"),
        Rule("JL016", "warning", "indirect or positional reference", "Resolve which item is meant in code and pass that item explicitly.", "R§11"),
        Rule("JL017", "info", "plan / intent question as a Noul", "Negated or hypothetical mentions (\"has no plans to...\") are the Lyon trap: ask a Choice with asserted / negated / hypothetical options.", "R§6"),
        Rule("JL020", "error", "duplicate Choice label", "JSON keeps only the last duplicate key; rename or merge the options.", "R§3"),
        Rule("JL021", "warning", "Choice labels differ only in case, spacing or separators", "Merge them or make the difference meaningful; near-identical labels split probability mass.", "R§3"),
        Rule("JL022", "info", "no Choice option is described", "Describe at least the options whose names could be read more than one way.", "R§11"),
        Rule("JL023", "info", "no catch-all Choice option", "Add an option such as \"other\" or \"none_of_the_above\" so an item that fits nothing is not forced into a label.", "R§8"),
        Rule("JL030", "warning", "Score levels mix dimensions", "Measure one dimension per Score; ask a second Score for the other dimension.", "R-arch 3"),
        Rule("JL031", "warning", "Score levels are numeric quantities", "The answer is a probability-weighted level position, not a count; count in code.", "R§3"),
        Rule("JL040", "warning", "inverse Noul pair", "Ask one of the two and derive the other as 1 - p; separate questions are not held to p + q = 1.", "R§3"),
        Rule("JL050", "warning", "instructions and criteria contradict", "Make the criteria describe the same polarity and range as the instructions.", "R§11"),
        Rule("JL051", "warning", "question type does not match the instructions", "Use Noul for yes/no, Choice for which-one, Score for how-much on a rubric.", "R§11"),
        Rule("JL060", "info", "mostly non-ASCII text", "Non-English text is handled \"but not equally well\": validate this question on labeled data in its language.", "R-new 7"),
    ]
}

UNSUPPRESSIBLE = frozenset({"JL000", "JL001", "JL003", "JL020"})

Hit = tuple[str, str]


@dataclass(frozen=True)
class LintConfig:
    reserved_slots: int = 1


def _instructions(q) -> list[str]:
    return list(T.strings(q.instructions))


def _noul_side(q: NoulQuestion, side: str) -> list[str]:
    return list(T.strings((q.criteria or {}).get(side)))


def _descriptions(q) -> list[str]:
    if isinstance(q, NoulQuestion):
        return _noul_side(q, "true") + _noul_side(q, "false")
    if isinstance(q, ChoiceQuestion):
        return [s for d in q.criteria.values() for s in T.strings(d)]
    return [s for level in q.criteria for s in T.strings(level)]


def _all_text(q) -> list[str]:
    labels = list(q.criteria) if isinstance(q, ChoiceQuestion) else []
    return _instructions(q) + _descriptions(q) + labels


def _first(pattern: re.Pattern, texts: list[str]) -> str | None:
    for t in texts:
        m = pattern.search(T.normalize(t))
        if m:
            return m.group(0)
    return None


_NUM = (
    r"(?:\$?\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|"
    r"twenty|thirty|forty|fifty|hundred|thousand|million|half|a dozen|a hundred|a thousand)"
)
_COUNTING = re.compile(
    "|".join(
        [
            r"\bhow (?:many|much)\b",
            r"\bcount(?:s|ed|ing)?\b(?! as\b)",
            r"\b(?:the|total) number of\b",
            r"\b(?:sum|total) (?:of|amount|number|value|cost|price)\b",
            r"\b(?:average|median)\b",
            r"\badd(?:s|ed)? up\b",
            r"\bin total\b",
            r"\b(?:what|which) (?:percentage|share|fraction|proportion)\b",
            r"\b(?:percentage|per ?cent) of\b",
            rf"\b(?:more|fewer|less|greater|higher|lower|larger|smaller|bigger|older|younger|longer|shorter) than {_NUM}\b",
            rf"\b(?:over|under|above|below|exceed(?:s|ed|ing)?) {_NUM}\b",
            rf"\bat (?:least|most) {_NUM}\b",
            rf"(?:^|\s){_NUM} or (?:more|fewer|less|greater|higher|lower)\b",
            rf"\bbetween {_NUM} and {_NUM}\b",
        ]
    )
)

_MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
_MONTH_ABBR = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
_UNIT = r"(?:second|minute|hour|day|week|month|quarter|year)s?"
_DATE_LITERAL = re.compile(
    "|".join(
        [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            rf"\b(?:{_MONTHS}|{_MONTH_ABBR}\.?) \d{{1,2}}(?:st|nd|rd|th)?\b",
            rf"\b\d{{1,2}}(?:st|nd|rd|th)? (?:of )?{_MONTHS}\b",
            rf"\b{_MONTHS},? \d{{4}}\b",
            r"\bq[1-4] \d{4}\b",
        ]
    )
)
_DATE_BOUND = re.compile(
    r"\b(?:by|on or before|on or after|no later than|no earlier than|between)\s+(?:"
    + _DATE_LITERAL.pattern
    + ")"
)
_DATE_WINDOW = re.compile(
    rf"\b(?:within|in the (?:last|past|next)) {_NUM} {_UNIT}\b|\b{_NUM} {_UNIT} (?:ago|old|before|after|from|since|late|overdue)\b"
    r"|\b(?:overdue|expired|expires|past due)\b"
)
_ORDERING = re.compile(r"\b(?:before|after|earlier|later|prior to|since|until|till|older|newer|oldest|newest|earliest|latest)\b")
_TIME_WORD = re.compile(
    r"\b(?:dates?|dated|deadlines?|timestamps?|expiry|expiration|today|yesterday|tomorrow|seconds?|minutes?|hours?|days?|weeks?|months?|quarters?|years?)\b"
)

_AUX = r"(?:is|are|was|were|does|do|did|has|have|had|will|would|can|could|should|may|might|must)"
_PRONOUN = r"(?:it|they|he|she|we|you)"
_PRED_VERB = (
    r"(?:mentions?|asks?|contains?|includes?|requests?|says?|states?|refers?|uses?|expresses|express|"
    r"threatens?|complains?|wants?|needs?|shows?|indicates?|describes?|provides?|offers?|demands?|"
    r"reports?|claims?|promises?|admits?|agrees?)"
)
_TWO_PREDICATES = re.compile(
    rf"\b(?:and|or|but)\s+(?:also\s+)?(?:{_AUX}|{_PRONOUN}\s+{_AUX}|{_PRED_VERB}|whether|if|what|which|who|how|why|when|where)\b"
    r"|\bboth\b.+\band\b|\beither\b.+\bor\b|\band/or\b"
)

_GEN_VERBS = (
    r"(?:write|summari[sz]e|explain|draft|list|describe|generate|rewrite|translate|compose|outline|"
    r"paraphrase|suggest|extract|produce|create)"
)
_PREAMBLE = r"(?:(?:please|kindly|now|then|also|(?:can|could|would|will) you)\s+)*"
_GEN_START = re.compile(rf"^{_PREAMBLE}{_GEN_VERBS}\b")
_GEN_CHAINED = re.compile(rf"\b(?:and|then),?\s+(?:please\s+)?{_GEN_VERBS}\b")
_QUESTION_START = re.compile(rf"^(?:{_AUX}|what|which|who|whom|whose|when|where|why|how|whether|if)\b")

_INDIRECTION = re.compile(
    "|".join(
        [
            r"\bthe (?:former|latter)\b",
            r"\b(?:second|third|\d+(?:st|nd|rd|th))[- ](?:to[- ]last|from[- ](?:the[- ])?(?:end|last|top|bottom))\b",
            r"\bpenultimate\b",
            r"\bthe (?:first|second|third|fourth|fifth|last|next|previous|preceding|following|\d+(?:st|nd|rd|th)) "
            r"(?:one|item|message|entry|element|row|record|sentence|paragraph|line|email|turn|reply|comment|word|field|value|step|tool call|call)\b",
            r"\bthe (?:one|item|message|person|entity|value|company) (?:mentioned|referenced|referred to|described|listed) "
            r"(?:above|earlier|before|previously)\b",
            r"\b(?:above|afore)-?mentioned\b",
            r"\bthe opposite of\b",
        ]
    )
)

_MODALITY = re.compile(
    r"\b(?:plan(?:s|ned|ning)?|intend(?:s|ed)?|expect(?:s|ed)?|propose(?:s|d)?|consider(?:s|ed|ing)?|(?:is|are) going) to\b"
    r"|\bannounce(?:s|d)?\b"
)

_CLAUSE_BREAK = re.compile(r",|\b(?:and|or|but)\b")
_CONTRAST = re.compile(r"\b(?:but|however|although|though|whereas|despite)\b|(?<!not )\byet\b")
_CONJUNCTION = re.compile(r"\b(?:and|as well as|plus)\b|&")
_NUMERIC_LEVEL = re.compile(r"\d|\b(?:none|zero|one|two|three|four|five|six|seven|eight|nine|ten|dozens?|hundreds?)\b")
_SCALE = re.compile(r"\b(?:from\s+)?(\d+)\s*(?:-|–|to|through)\s*(\d+)\b|\bout of (\d+)\b")
_QUOTED = re.compile(r"`([^`]+)`|\"([^\"]+)\"|'([^'\s][^']*)'")

_CATCH_ALL_WORDS = frozenset(
    {"other", "none", "neither", "unknown", "unclear", "unsure", "na", "unrelated", "undetermined", "abstain", "uncertain", "irrelevant", "nota"}
)
_CATCH_ALL_PHRASES = ("none of the above", "not applicable", "no match", "not sure", "cannot tell", "can't tell", "cannot determine")


def _jl001_jl002(q, config: LintConfig) -> Iterator[Hit]:
    if not isinstance(q, ChoiceQuestion):
        return
    n = len(q.criteria)
    if n > MAX_CHOICE_OPTIONS:
        yield "JL001", f"{n} options; Choice accepts at most {MAX_CHOICE_OPTIONS}"
    elif n < 2:
        yield "JL001", f"{n} option{'s' if n != 1 else ''}; a Choice needs at least 2 to decide anything"
    elif config.reserved_slots and n + config.reserved_slots > MAX_CHOICE_OPTIONS:
        yield "JL002", f"{n} options + {config.reserved_slots} reserved slot(s) > {MAX_CHOICE_OPTIONS}"


def _jl003(q, config) -> Iterator[Hit]:
    if isinstance(q, ScoreQuestion) and not SCORE_LEVELS[0] <= len(q.criteria) <= SCORE_LEVELS[1]:
        yield "JL003", f"{len(q.criteria)} levels; Score accepts {SCORE_LEVELS[0]} to {SCORE_LEVELS[1]}"


def _jl010(q, config) -> Iterator[Hit]:
    if not isinstance(q, NoulQuestion):
        return
    for where, texts in (("instructions", _instructions(q)), ("criteria.true", _noul_side(q, "true"))):
        for t in texts:
            found = T.negations(t)
            if found:
                yield "JL010", f"negation \"{T.negation_phrase(t, found[0])}\" in {where}"
                return


def _jl011(q, config) -> Iterator[Hit]:
    for t in _instructions(q) + _descriptions(q):
        for sentence in T.sentences(t):
            for clause in _CLAUSE_BREAK.split(sentence):
                found = T.negations(clause, negative_words=True)
                if "neither" in found and "nor" in found:
                    found.remove("nor")
                if len(found) >= 2:
                    yield "JL011", f"{len(found)} negations in one clause: {', '.join(found)}"
                    return


def _jl012(q, config) -> Iterator[Hit]:
    texts = _instructions(q) + ([] if isinstance(q, ScoreQuestion) else _descriptions(q))
    match = _first(_COUNTING, texts)
    if match:
        yield "JL012", f"\"{match.strip()}\""


def _jl013(q, config) -> Iterator[Hit]:
    texts = _instructions(q) + _descriptions(q)
    match = _first(_DATE_BOUND, texts) or _first(_DATE_WINDOW, texts)
    if match:
        yield "JL013", f"\"{match}\""
        return
    for t in texts:
        for sentence in T.sentences(t):
            order = _ORDERING.search(sentence)
            time = _DATE_LITERAL.search(sentence) or _TIME_WORD.search(sentence)
            if order and time:
                yield "JL013", f"\"{order.group(0)}\" with \"{time.group(0)}\""
                return


def _jl014(q, config) -> Iterator[Hit]:
    texts = _instructions(q) + (_noul_side(q, "true") if isinstance(q, NoulQuestion) else [])
    for t in texts:
        marks = t.count("?")
        if marks >= 2:
            yield "JL014", f"{marks} question marks"
            return
    match = _first(_TWO_PREDICATES, texts)
    if match:
        yield "JL014", f"\"{match}\" joins two predicates"


def _jl015(q, config) -> Iterator[Hit]:
    for t in _instructions(q):
        for sentence in T.sentences(t):
            start = _GEN_START.search(sentence)
            if start:
                yield "JL015", f"\"{start.group(0)}\""
                return
            if not _QUESTION_START.search(sentence):
                chained = _GEN_CHAINED.search(sentence)
                if chained:
                    yield "JL015", f"\"{chained.group(0)}\""
                    return


def _jl016(q, config) -> Iterator[Hit]:
    match = _first(_INDIRECTION, _instructions(q) + _descriptions(q))
    if match:
        yield "JL016", f"\"{match}\""


def _jl017(q, config) -> Iterator[Hit]:
    if isinstance(q, NoulQuestion):
        match = _first(_MODALITY, _instructions(q) + _noul_side(q, "true"))
        if match:
            yield "JL017", f"\"{match}\""


def _jl020_to_23(spec: QuestionSpec) -> Iterator[Hit]:
    q = spec.question
    if not isinstance(q, ChoiceQuestion):
        return
    for label in spec.duplicate_labels:
        yield "JL020", f"label {label!r} appears more than once"
    groups: dict[str, list[str]] = {}
    for label in q.criteria:
        groups.setdefault(T.identifier_words(label), []).append(label)
    for labels in groups.values():
        if len(labels) > 1:
            yield "JL021", f"labels {labels} are the same after normalizing"
    if len(q.criteria) >= 2 and not any(list(T.strings(d)) and T.joined(d).strip() for d in q.criteria.values()):
        yield "JL022", f"all {len(q.criteria)} options have no description"
    if len(q.criteria) >= 2 and not any(_is_catch_all(label) for label in q.criteria):
        yield "JL023", "no option for items that fit none of the labels"


def _is_catch_all(label: str) -> bool:
    words = T.identifier_words(label)
    return bool(set(words.split()) & _CATCH_ALL_WORDS) or any(p in words for p in _CATCH_ALL_PHRASES)


def _jl030_jl031(q, config) -> Iterator[Hit]:
    if not isinstance(q, ScoreQuestion):
        return
    levels = [T.normalize(T.joined(level)) for level in q.criteria]
    contrast = [i for i, level in enumerate(levels) if _CONTRAST.search(level)]
    joined = [i for i, level in enumerate(levels) if _CONJUNCTION.search(level)]
    if contrast:
        yield "JL030", f"level(s) {contrast} contrast two qualities (\"{_CONTRAST.search(levels[contrast[0]]).group(0)}\")"
    elif len(joined) >= 2:
        yield "JL030", f"levels {joined} each combine two qualities with \"and\""
    numeric = [i for i, level in enumerate(levels) if _NUMERIC_LEVEL.search(level)]
    if levels and len(numeric) * 2 > len(levels):
        yield "JL031", f"{len(numeric)} of {len(levels)} levels are numbers or counts"


def _jl050(q, config) -> Iterator[Hit]:
    instructions = T.joined(q.instructions)
    if isinstance(q, NoulQuestion) and q.criteria:
        yes, no = T.joined(q.criteria.get("true")), T.joined(q.criteria.get("false"))
        if yes.strip() and no.strip() and T.normalize(yes).strip() == T.normalize(no).strip():
            yield "JL050", "criteria.true and criteria.false are the same text"
            return
        if instructions.strip():
            (ci, ni), (cy, ny), (cn, nn) = T.core(instructions), T.core(yes), T.core(no)
            if yes.strip() and cy == ci and ny % 2 != ni % 2:
                yield "JL050", "criteria.true is the negation of the instructions (polarity flipped)"
            elif no.strip() and cn == ci and nn % 2 == ni % 2:
                yield "JL050", "criteria.false restates the instructions (polarity flipped)"
    elif isinstance(q, ScoreQuestion):
        m = _SCALE.search(T.normalize(instructions))
        if m:
            if m.group(3):
                low, high = 0, int(m.group(3))
                expected = high + 1 if f"0 out of {high}" in T.normalize(instructions) else high
            else:
                low, high = int(m.group(1)), int(m.group(2))
                expected = high - low + 1
            n = len(q.criteria)
            if 0 < expected <= 100 and expected != n:
                yield "JL050", f"instructions give a scale \"{m.group(0)}\" ({expected} values) but criteria has {n} levels"
            elif low != 0 and expected == n:
                yield "JL050", f"instructions say \"{m.group(0)}\" but Score levels are numbered from 0 (answers will read 0..{n - 1})"
    elif isinstance(q, ChoiceQuestion):
        labels = {T.identifier_words(label) for label in q.criteria}
        quoted = [next(g for g in m.groups() if g) for m in _QUOTED.finditer(instructions)]
        named = [w for w in quoted if T.identifier_words(w) in labels]
        missing = [w for w in quoted if T.identifier_words(w) not in labels]
        if named and missing:
            yield "JL050", f"instructions name option(s) {missing} that are not in criteria"


def _jl051(q, config) -> Iterator[Hit]:
    first = next((s for t in _instructions(q) for s in T.sentences(t)), "")
    if isinstance(q, NoulQuestion):
        wh = re.match(r"(what|which|who|whom|whose|when|where|why|how)\b", first)
        if wh and not re.match(r"how (?:likely|probable)\b", first):
            yield "JL051", f"a Noul answers yes/no, but the instructions start with \"{wh.group(1)}\""
        elif re.search(r"\bon a scale\b|\brate (?:the|this|how)\b", first):
            yield "JL051", "a Noul answers yes/no, but the instructions ask for a rating"
    elif isinstance(q, ScoreQuestion):
        yn = re.match(r"(is|are|was|were|does|do|did|has|have|can|will|should)\b", first)
        if yn:
            yield "JL051", f"a Score rates on a rubric, but the instructions ask a yes/no question (\"{yn.group(1)} ...\")"


def _jl060(q, config) -> Iterator[Hit]:
    share, letters = T.non_ascii_letter_share("\n".join(_all_text(q)))
    if letters >= 3 and share > 0.5:
        yield "JL060", f"{share:.0%} of letters are non-ASCII"


_QUESTION_RULES: list[Callable[[object, LintConfig], Iterator[Hit]]] = [
    _jl001_jl002, _jl003, _jl010, _jl011, _jl012, _jl013, _jl014, _jl015, _jl016, _jl017,
    _jl030_jl031, _jl050, _jl051, _jl060,
]


def check_question(spec: QuestionSpec, config: LintConfig) -> list[Hit]:
    if spec.question is None:
        return []
    hits: list[Hit] = []
    for rule in _QUESTION_RULES:
        hits.extend(rule(spec.question, config))
    hits.extend(_jl020_to_23(spec))
    return hits


def check_pairs(specs: list[QuestionSpec]) -> list[tuple[str, Hit]]:
    nouls = [s for s in specs if isinstance(s.question, NoulQuestion)]
    hits: list[tuple[str, Hit]] = []
    for i, a in enumerate(nouls):
        for b in nouls[i + 1 :]:
            why = _inverse_reason(a, b)
            if why:
                hits.append((a.name, ("JL040", f"{why} of {b.name!r}; p({a.name}) + p({b.name}) will not sum to 1")))
                hits.append((b.name, ("JL040", f"{why} of {a.name!r}; p({a.name}) + p({b.name}) will not sum to 1")))
    return hits


def _inverse_reason(a: QuestionSpec, b: QuestionSpec) -> str | None:
    ta, tb = T.joined(a.question.instructions), T.joined(b.question.instructions)
    if ta.strip() and tb.strip():
        (ca, na), (cb, nb) = T.core(ta), T.core(tb)
        if ca and ca == cb and na % 2 != nb % 2:
            return "instructions are the negation"
    (ca, na), (cb, nb) = T.core(T.identifier_words(a.name)), T.core(T.identifier_words(b.name))
    if ca and ca == cb and na % 2 != nb % 2:
        return "name is the negation"
    return None
