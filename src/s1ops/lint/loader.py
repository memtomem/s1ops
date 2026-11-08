"""Read a questions file mapping question name to a `/v1/systemone` or OpenAI `/v1/decisions` question.

Supports both `x-s1-lint` and `x-jev-lint` extension keys (one per question) for rule
suppression (`ignore`) and inverse phrasing (`inverse`, the wording `s1 probe` sends as the
inverse probe).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from s1ops.contracts._validation import pydantic_errors
from s1ops.contracts.questions import (
    NoulQuestion,
    QuestionModel,
    parse_wire_question,
    question_id,
)

EXTENSION_KEYS = ("x-s1-lint", "x-jev-lint")
_EXTENSION_FIELDS = {"ignore", "inverse"}
MAX_NESTING = 100


@dataclass
class QuestionSpec:
    name: str
    question: QuestionModel | None
    ignore: frozenset[str] = frozenset()
    inverse: NoulQuestion | None = None
    errors: list[str] = field(default_factory=list)
    duplicate_labels: list[str] = field(default_factory=list)

    @property
    def qid(self) -> str | None:
        return None if self.question is None else question_id(self.question)


@dataclass
class QuestionsFile:
    specs: list[QuestionSpec]
    errors: list[str]


class _Obj(dict):
    duplicates: list[str]


def _pairs(pairs: list[tuple[str, object]]) -> _Obj:
    obj = _Obj()
    obj.duplicates = []
    for key, value in pairs:
        if key in obj:
            obj.duplicates.append(key)
        obj[key] = value
    return obj


def _plain(node: object) -> object:
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_plain(v) for v in node]
    return node


def parse_question(name: str, raw: object, known_rules: frozenset[str]) -> QuestionSpec:
    if not isinstance(raw, dict):
        return QuestionSpec(name, None, errors=["a question must be a JSON object"])
    duplicates = list(getattr(raw.get("criteria"), "duplicates", [])) if raw.get("type") == "choice" else []
    present = [k for k in EXTENSION_KEYS if k in raw]
    ext_key = present[0] if present else EXTENSION_KEYS[0]
    body = _plain({k: v for k, v in raw.items() if k not in EXTENSION_KEYS})
    extension = _plain(raw.get(ext_key, {}))
    errors: list[str] = []
    if len(present) > 1:
        errors.append(f"use one of {' or '.join(present)}, not both")

    ignore: frozenset[str] = frozenset()
    inverse: NoulQuestion | None = None
    if not isinstance(extension, dict):
        errors.append(f"{ext_key} must be an object")
        extension = {}
    unknown = sorted(set(extension) - _EXTENSION_FIELDS)
    if unknown:
        errors.append(f"{ext_key}: unknown keys {unknown}; allowed: {sorted(_EXTENSION_FIELDS)}")
    if "ignore" in extension:
        ids = extension["ignore"]
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            errors.append(f"{ext_key}.ignore must be a list of rule ids")
        else:
            bad = sorted(set(ids) - known_rules)
            if bad:
                errors.append(f"{ext_key}.ignore: unknown rule ids {bad}")
            ignore = frozenset(ids)
    if "inverse" in extension:
        if body.get("type") not in ("noul", "predicate"):
            errors.append(f"{ext_key}.inverse is only supported on noul/predicate questions")
        elif not isinstance(extension["inverse"], dict):
            errors.append(f"{ext_key}.inverse must be a noul question object")
        else:
            try:
                inverse = NoulQuestion.model_validate({"type": "noul", **extension["inverse"]})
            except ValidationError as error:
                errors.extend(f"{ext_key}.inverse.{e}" for e in pydantic_errors(error))

    question: QuestionModel | None = None
    try:
        question = parse_wire_question(body)
    except ValidationError as error:
        errors.extend(pydantic_errors(error))
    except ValueError as error:
        errors.append(str(error))
    return QuestionSpec(name, question, ignore, inverse, errors, duplicates)


def _nested_too_deep(document: object) -> bool:
    """Whether containers nest past `MAX_NESTING`. Iterative, since every later reader recurses."""
    level = [document]
    for _ in range(MAX_NESTING):
        level = [
            child
            for node in level
            if isinstance(node, (dict, list))
            for child in (node.values() if isinstance(node, dict) else node)
        ]
        if not level:
            return False
    return any(isinstance(node, (dict, list)) for node in level)


def _is_wrapper(document: dict) -> bool:
    """`{"questions": {...}}`. A question's `type` is a string; a question named `type` is an object there."""
    inner = document.get("questions")
    return len(document) == 1 and isinstance(inner, dict) and not isinstance(inner.get("type"), str)


def _repeated(entries: _Obj) -> list[str]:
    return [f"question name {name!r} appears more than once" for name in entries.duplicates]


def read_question_entries(path: str | Path, data: bytes | None = None) -> tuple[dict[str, object], list[str]]:
    """The one reader of a questions file: its raw entries by name, and what is wrong with the file.

    The file maps names to questions, or is `{"questions": <that mapping>}`.
    """
    try:
        text = Path(path).read_text(encoding="utf-8") if data is None else data.decode("utf-8")
    except OSError as error:
        return {}, [f"cannot read {path}: {error.strerror}"]
    except UnicodeDecodeError as error:
        return {}, [f"not valid UTF-8: {error}"]
    try:
        parsed = json.loads(text, object_pairs_hook=_pairs)
    except ValueError as error:  # JSONDecodeError, or the decoder's integer-digit limit
        return {}, [f"not valid JSON: {error}"]
    except RecursionError:
        return {}, [f"nested more than {MAX_NESTING} levels"]
    if _nested_too_deep(parsed):
        return {}, [f"nested more than {MAX_NESTING} levels"]
    if not isinstance(parsed, dict):
        return {}, ["the file must be a JSON object mapping question names to questions"]
    errors = _repeated(parsed)
    entries = parsed["questions"] if _is_wrapper(parsed) else parsed
    if entries is not parsed:
        errors += _repeated(entries)
    if not entries:
        errors.append("the file has no questions")
    return entries, errors


def load_questions(path: str | Path, known_rules: frozenset[str], data: bytes | None = None) -> QuestionsFile:
    entries, errors = read_question_entries(path, data)
    return QuestionsFile([parse_question(name, raw, known_rules) for name, raw in entries.items()], errors)
