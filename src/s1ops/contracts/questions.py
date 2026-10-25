"""Vendor-neutral question models and canonical wire-form `question_id` hashing.

Supports:
1. `/v1/systemone` wire format (`noul`, `choice`, `score` — TypeSafe Jev, Kev, Laya, Von)
2. OpenAI Decisions `/v1/decisions` format (`predicate`, `choice` — e.g. `gpt-6-luna`)

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
            criteria[entry["value"]] = entry.get("description")
        return ChoiceQuestion(type="choice", instructions=raw["instructions"], criteria=criteria)
    model_cls = QUESTION_TYPES.get(kind) if isinstance(kind, str) else None
    if model_cls is None:
        raise ValueError(f"type must be one of {sorted(QUESTION_TYPES)}, got {kind!r}")
    return model_cls.model_validate(raw)


def to_openai_decision(name: str, question: QuestionModel) -> dict[str, Any]:
    """Convert a `NoulQuestion` or `ChoiceQuestion` to OpenAI `/v1/decisions` wire format."""
    if isinstance(question, NoulQuestion):
        return {"type": "predicate", "name": name, "instructions": question.instructions}
    if isinstance(question, ChoiceQuestion):
        choices = [
            {"value": k, "description": d} if d is not None else {"value": k}
            for k, d in question.criteria.items()
        ]
        return {"type": "choice", "name": name, "instructions": question.instructions, "choices": choices}
    raise ValueError("OpenAI Decisions API supports only 'noul' (predicate) and 'choice' questions")
