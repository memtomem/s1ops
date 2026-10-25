"""Read a questions file mapping question name to a `/v1/systemone` or OpenAI `/v1/decisions` question.

Supports both `x-s1-lint` and `x-jev-lint` extension keys for rule suppression (`ignore`) and
inverse phrasing (`inverse`).
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
    ext_key = next((k for k in EXTENSION_KEYS if k in raw), EXTENSION_KEYS[0])
    body = _plain({k: v for k, v in raw.items() if k not in EXTENSION_KEYS})
    extension = _plain(raw.get(ext_key, {}))
    errors: list[str] = []

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


def load_questions(path: str | Path, known_rules: frozenset[str], data: bytes | None = None) -> QuestionsFile:
    try:
        text = Path(path).read_text(encoding="utf-8") if data is None else data.decode("utf-8")
    except OSError as error:
        return QuestionsFile([], [f"cannot read {path}: {error.strerror}"])
    try:
        parsed = json.loads(text, object_pairs_hook=_pairs)
    except json.JSONDecodeError as error:
        return QuestionsFile([], [f"not valid JSON: {error}"])
    if not isinstance(parsed, dict):
        return QuestionsFile([], ["the file must be a JSON object mapping question names to questions"])
    errors = [f"question name {name!r} appears more than once; only the last one is kept" for name in parsed.duplicates]
    if not parsed:
        errors.append("the file has no questions")
    return QuestionsFile([parse_question(name, raw, known_rules) for name, raw in parsed.items()], errors)
