"""Text helpers shared by static lint rules: string extraction, word tokens, negation, sentences."""

import re
from collections import Counter
from collections.abc import Iterator

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;:])\s+|\n+")

EXPLICIT_NEGATORS = frozenset(
    {"not", "no", "never", "neither", "nor", "none", "nobody", "nothing", "nowhere", "without", "cannot", "non"}
)
IMPLICIT_NEGATORS = frozenset(
    {
        "lack", "lacks", "lacked", "lacking", "absent", "absence",
        "fail", "fails", "failed", "failing", "refuse", "refuses", "refused",
        "deny", "denies", "denied", "neglect", "neglects", "neglected",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "unlikely", "uncommon", "unusual", "unclear", "unknown", "unaware", "unhappy", "unable",
        "unwilling", "unreasonable", "unrelated", "unsuccessful", "uncertain", "untrue", "unsafe",
        "invalid", "incorrect", "inaccurate", "incomplete", "impossible", "improper", "illegal",
        "irrelevant", "dishonest", "disagree", "disagrees", "dissatisfied", "false",
    }
)

_NOT_NEGATION = [
    (re.compile(r"\byes\s*(?:,|or|/)\s*no\b|\bno\s*(?:,|or|/)\s*yes\b"), "yesno"),
    (re.compile(r"\btrue\s*(?:,|or|/)\s*false\b"), "truefalse"),
    (re.compile(r"\bnot only\b"), "only"),
    (re.compile(r"\bnothing but\b"), "only"),
    (re.compile(r"\bnone other than\b"), "exactly"),
]

_CONTRACTION_BASES = {"won": "will", "can": "can", "ca": "can", "shan": "shall"}
_ARTICLES_AND_DO = frozenset({"a", "an", "the", "do", "does", "did"})


def normalize(text: str) -> str:
    return text.translate(_APOSTROPHES).lower()


def strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value)


def joined(node: object) -> str:
    return "\n".join(strings(node))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BREAK.split(normalize(text)) if s.strip()]


def _scrub(text: str) -> str:
    text = normalize(text)
    for pattern, replacement in _NOT_NEGATION:
        text = pattern.sub(replacement, text)
    return text


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def is_negator(token: str) -> bool:
    return token in EXPLICIT_NEGATORS or token.endswith("n't")


def negations(text: str, *, implicit: bool = True, negative_words: bool = False) -> list[str]:
    found = []
    for token in _WORD.findall(_scrub(text)):
        if (
            is_negator(token)
            or (implicit and token in IMPLICIT_NEGATORS)
            or (negative_words and token in NEGATIVE_WORDS)
        ):
            found.append(token)
    return found


def negation_phrase(text: str, token: str) -> str:
    match = re.search(rf"\b{re.escape(token)}\b(?:\s+[\w'-]+)?", _scrub(text))
    return match.group(0) if match else token


def _stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def core(text: str) -> tuple[Counter, int]:
    content: Counter = Counter()
    negators = 0
    for token in _WORD.findall(_scrub(text)):
        if token.endswith("n't"):
            negators += 1
            base = token[:-3]
            token = _CONTRACTION_BASES.get(base, base)
        elif token == "cannot":
            negators += 1
            token = "can"
        elif token in EXPLICIT_NEGATORS:
            negators += 1
            continue
        if token in _ARTICLES_AND_DO:
            continue
        content[_stem(token)] += 1
    return content, negators


def identifier_words(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"[_\-.\s]+", " ", spaced).strip().lower()


def non_ascii_letter_share(text: str) -> tuple[float, int]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0, 0
    return sum(1 for c in letters if not c.isascii()) / len(letters), len(letters)
