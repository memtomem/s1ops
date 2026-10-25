"""Vendor-neutral question models and canonical wire-form `question_id` hashing.

Supports:
1. `/v1/systemone` wire format (`noul`, `choice`, `score` — TypeSafe Jev, Kev, Laya, Von)
2. OpenAI Decisions `/v1/decisions` format (`predicate`, `choice`, `score` — e.g. `gpt-6-luna`)

`question_id` is `"q-"` + the first 16 hex characters of the SHA-256 of the canonical
`/v1/systemone` wire JSON (keys in wire order, Choice option order preserved, unset fields
omitted, no extra whitespace). Reordering Choice options or editing a single word changes
the hash on purpose.
"""

import hashlib
import json
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: str | dict[str, Any] | list[Any]
    criteria: dict[Literal["true", "false"], Any] | None = None

    def wire_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str | dict[str, Any] | list[Any]
    criteria: dict[str, Any] = Field(...)

    def wire_dict(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str | dict[str, Any] | list[Any]
    criteria: list[Any] = Field(...)

    def wire_dict(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


QuestionModel = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]

QUESTION_TYPES: dict[str, type[QuestionModel]] = {
    "noul": NoulQuestion,
    "choice": ChoiceQuestion,
    "score": ScoreQuestion,
}


def question_id(question: QuestionModel | dict[str, Any]) -> str:
    """Compute the canonical `q-<16hex>` identity of a question."""
    if isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
        wire = question.wire_dict()
    elif isinstance(question, dict):
        parsed = parse_wire_question(question)
        wire = parsed.wire_dict()
    else:
        wire = question.model_dump(mode="json", exclude_none=True)
    canonical = json.dumps(wire, separators=(",", ":"), ensure_ascii=False)
    return "q-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def parse_wire_question(raw: dict[str, Any]) -> QuestionModel:
    """Parse either `/v1/systemone` (`noul`/`choice`/`score`) or OpenAI Decisions (`predicate`/`choice`)."""
    if not isinstance(raw, dict):
        raise ValueError("question must be a JSON object")
    kind = raw.get("type")
    if kind == "predicate":
        extra = set(raw) - {"type", "name", "instructions"}
        if extra:
            raise ValueError(f"unknown keys in predicate question: {sorted(extra)}")
        return NoulQuestion(type="noul", instructions=raw["instructions"])
    if kind == "choice" and "choices" in raw and "criteria" not in raw:
        extra = set(raw) - {"type", "name", "instructions", "choices"}
        if extra:
            raise ValueError(f"unknown keys in OpenAI choice question: {sorted(extra)}")
        choices = raw["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("choices must be a non-empty list")
        criteria: dict[str, Any] = {}
        for entry in choices:
            if not isinstance(entry, dict) or not isinstance(entry.get("value"), str) or not entry["value"]:
                raise ValueError("each choice entry must have a non-empty string 'value'")
            if entry["value"] in criteria:
                raise ValueError(f"duplicate choice value {entry['value']!r}")
            criteria[entry["value"]] = entry.get("description")
        return ChoiceQuestion(type="choice", instructions=raw["instructions"], criteria=criteria)
    if kind == "score" and "levels" in raw and "criteria" not in raw:
        extra = set(raw) - {"type", "name", "instructions", "levels"}
        if extra:
            raise ValueError(f"unknown keys in OpenAI score question: {sorted(extra)}")
        levels = raw["levels"]
        if not isinstance(levels, list) or not levels:
            raise ValueError("levels must be a non-empty list")
        criteria_levels: list[Any] = []
        for entry in levels:
            if not isinstance(entry, dict) or not isinstance(entry.get("label"), str) or set(entry) - {"label", "description"}:
                raise ValueError("each level must be an object with a string 'label' and an optional 'description'")
            criteria_levels.append(entry["label"] if entry.get("description") is None else dict(entry))
        return ScoreQuestion(type="score", instructions=raw["instructions"], criteria=criteria_levels)
    model_cls = QUESTION_TYPES.get(kind) if isinstance(kind, str) else None
    if model_cls is None:
        raise ValueError(f"type must be one of {sorted(QUESTION_TYPES)}, got {kind!r}")
    return model_cls.model_validate(raw)


CHOICES = (2, 255)
LEVELS = (2, 10)


def _openai_level(level: Any) -> dict[str, Any]:
    if isinstance(level, str):
        return {"label": level}
    if (
        isinstance(level, dict)
        and isinstance(level.get("label"), str)
        and not set(level) - {"label", "description"}
        and isinstance(level.get("description", ""), str)
    ):
        return dict(level)
    raise ValueError("an OpenAI score level is a string or {'label': str, 'description'?: str}")


def to_openai_decision(name: str, question: QuestionModel) -> dict[str, Any]:
    """Convert a question to OpenAI `/v1/decisions` wire format, or raise `ValueError` if it has none.

    Nothing is dropped: a Noul with `criteria` (predicates have none), structured `instructions`, or
    counts outside the API's limits are refused, since sending a different question under the same
    `question_id` would pool answers to two questions.
    """
    if not isinstance(question.instructions, str):
        raise ValueError("OpenAI Decisions takes string instructions")
    if isinstance(question, NoulQuestion):
        if question.criteria is not None:
            raise ValueError("an OpenAI predicate has no criteria; fold them into the instructions")
        return {"type": "predicate", "name": name, "instructions": question.instructions}
    if isinstance(question, ChoiceQuestion):
        if not CHOICES[0] <= len(question.criteria) <= CHOICES[1]:
            raise ValueError(f"OpenAI choice takes {CHOICES[0]} to {CHOICES[1]} options, got {len(question.criteria)}")
        if not all(d is None or isinstance(d, str) for d in question.criteria.values()):
            raise ValueError("OpenAI choice descriptions must be strings")
        choices = [
            {"value": k, "description": d} if d is not None else {"value": k}
            for k, d in question.criteria.items()
        ]
        return {"type": "choice", "name": name, "instructions": question.instructions, "choices": choices}
    if isinstance(question, ScoreQuestion):
        if not LEVELS[0] <= len(question.criteria) <= LEVELS[1]:
            raise ValueError(f"OpenAI score takes {LEVELS[0]} to {LEVELS[1]} levels, got {len(question.criteria)}")
        levels = [_openai_level(level) for level in question.criteria]
        return {"type": "score", "name": name, "instructions": question.instructions, "levels": levels}
    raise ValueError(f"unsupported question model: {type(question).__name__}")
