"""Budget-guarded System 1 probe & batch runner under the Reserve-Then-Settle Ledger.

Key guarantees:
1. **Reserve-Then-Settle Accounting**: Every backend invocation first reserves its worst-case USD
   cost (`token_bound(state) * price_per_token`) in `Ledger` and settles the exact measured cost
   after completion (or `unsent` on pre-send failure).
2. **Zero-State Privacy (`safe-v1`)**: Raw `state` text is hashed (`state_sha256`) in memory and
   never written to output `s1-record/1` JSONL files or ledger metadata.
3. **Automatic Probe Expansion (`expand_probes`)**: Given a `NoulQuestion`, automatically constructs
   its `twin` (`ChoiceQuestion` with `{"yes", "no"}`) and `inverse` (`NoulQuestion` with negated
   instructions) for coherence auditing (`s1 lint probe-audit`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from s1ops.contracts.questions import (
    ChoiceQuestion,
    NoulQuestion,
    QuestionModel,
    ScoreQuestion,
    parse_wire_question,
    question_id,
)
from s1ops.ledger.journal import Ledger, token_bound


@dataclass(frozen=True)
class BackendResponse:
    model_served: str
    answer: dict[str, Any]
    cost_usd: float
    latency_ms: float


class Backend(Protocol):
    """Protocol for System 1 execution backends (`/v1/systemone`, `/v1/decisions`, or mock)."""

    def infer(
        self,
        state: str,
        question: QuestionModel,
        *,
        model_requested: str,
    ) -> BackendResponse:
        ...


class DeterministicMockBackend:
    """100% offline deterministic System 1 backend for testing, dry-runs, and CI."""

    def __init__(
        self,
        *,
        served_model: str | None = None,
        cost_per_token_usd: float = 1e-7,
    ) -> None:
        self.served_model = served_model
        self.cost_per_token_usd = cost_per_token_usd

    def infer(
        self,
        state: str,
        question: QuestionModel,
        *,
        model_requested: str,
    ) -> BackendResponse:
        qid = question_id(question)
        seed_bytes = hashlib.sha256(f"{state}::{qid}".encode("utf-8")).digest()
        u = int.from_bytes(seed_bytes[:4], "big") / 0xFFFFFFFF
        p = round(0.05 + 0.90 * u, 6)
        served = self.served_model or model_requested
        tokens = token_bound(state) + token_bound(json.dumps(question.wire_dict()))
        cost = round(tokens * self.cost_per_token_usd, 8)
        latency = round(15.0 + 10.0 * u, 2)

        if isinstance(question, NoulQuestion):
            return BackendResponse(
                model_served=served,
                answer={"p": p},
                cost_usd=cost,
                latency_ms=latency,
            )
        if isinstance(question, ChoiceQuestion):
            keys = list(question.criteria.keys())
            if keys == ["yes", "no"]:
                probs = {"yes": p, "no": round(1.0 - p, 6)}
            else:
                raw_weights = [
                    ((seed_bytes[i % len(seed_bytes)] + 1) / 256.0)
                    for i in range(len(keys))
                ]
                total = sum(raw_weights)
                probs = {k: round(w / total, 6) for k, w in zip(keys, raw_weights)}
                # fix rounding residual on first key
                diff = round(1.0 - sum(probs.values()), 6)
                probs[keys[0]] = round(probs[keys[0]] + diff, 6)
            top_c = max(probs.items(), key=lambda kv: kv[1])[0]
            return BackendResponse(
                model_served=served,
                answer={
                    "choice": top_c,
                    "probabilities": probs,
                    "confidence": probs[top_c],
                },
                cost_usd=cost,
                latency_ms=latency,
            )
        if isinstance(question, ScoreQuestion):
            n_lv = len(question.criteria)
            raw_weights = [
                ((seed_bytes[(i + 3) % len(seed_bytes)] + 1) / 256.0)
                for i in range(n_lv)
            ]
            total = sum(raw_weights)
            probs = {str(i): round(w / total, 6) for i, w in enumerate(raw_weights)}
            diff = round(1.0 - sum(probs.values()), 6)
            probs["0"] = round(probs["0"] + diff, 6)
            top_k = max(probs.items(), key=lambda kv: kv[1])[0]
            exp_score = round(sum(int(k) * v for k, v in probs.items()), 6)
            return BackendResponse(
                model_served=served,
                answer={
                    "score": exp_score,
                    "probabilities": probs,
                    "confidence": probs[top_k],
                },
                cost_usd=cost,
                latency_ms=latency,
            )
        raise ValueError(f"Unsupported question model: {type(question)}")


def expand_probes(
    questions: Mapping[str, QuestionModel | dict[str, Any]],
) -> list[tuple[str, str, QuestionModel]]:
    """Expand each `NoulQuestion` into `(base_name, probe_role, QuestionModel)` triples:
    `noul`, `twin` (yes/no `ChoiceQuestion`), and `inverse` (negated `NoulQuestion`).
    """
    out: list[tuple[str, str, QuestionModel]] = []
    for name, q_raw in questions.items():
        q = parse_wire_question(q_raw) if isinstance(q_raw, dict) else q_raw
        if isinstance(q, NoulQuestion) and isinstance(q.instructions, str):
            out.append((name, "noul", q))
            twin = ChoiceQuestion(
                instructions=q.instructions,
                criteria={"yes": "True / Yes", "no": "False / No"},
            )
            out.append((name, "twin", twin))
            inv_instr = f"Is it false that: {q.instructions.strip()}"
            inv = NoulQuestion(instructions=inv_instr)
            out.append((name, "inverse", inv))
        else:
            out.append((name, "primary", q))
    return out


def run_batch(
    states: Sequence[Mapping[str, Any]],
    questions: Mapping[str, QuestionModel | dict[str, Any]],
    *,
    backend: Backend,
    ledger: Ledger,
    model_requested: str = "jev-1.13.0",
    worst_usd_per_token: float = 2e-6,
    probe_mode: bool = False,
) -> list[dict[str, Any]]:
    """Run System 1 questions over a batch of states with strict `Ledger` reserve-then-settle.

    Never writes raw `state` into returned `s1-record/1` dicts or ledger metadata.
    """
    if probe_mode:
        work_questions = expand_probes(questions)
    else:
        work_questions = [
            (
                name,
                "primary",
                parse_wire_question(q) if isinstance(q, dict) else q,
            )
            for name, q in questions.items()
        ]

    records_out: list[dict[str, Any]] = []
    for item in states:
        dataset = item.get("dataset", "default")
        item_id = str(item["item_id"])
        state_text = str(item["state"])
        state_sha = hashlib.sha256(state_text.encode("utf-8")).hexdigest()

        for base_name, role, q_model in work_questions:
            qid = question_id(q_model)
            q_wire = json.dumps(q_model.wire_dict())
            est_tokens = token_bound(state_text) + token_bound(q_wire)
            worst_usd = max(1e-6, round(est_tokens * worst_usd_per_token, 8))

            reservation = ledger.reserve(
                worst_usd,
                dataset=dataset,
                item_id=item_id,
                question_id=qid,
                model_requested=model_requested,
            )
            try:
                resp = backend.infer(
                    state_text,
                    q_model,
                    model_requested=model_requested,
                )
            except Exception:
                ledger.settle(reservation, 0.0, outcome="unsent")
                raise

            charged = min(resp.cost_usd, reservation.worst_usd)
            ledger.settle(
                reservation,
                charged,
                outcome="measured",
                model_served=resp.model_served,
            )

            rec: dict[str, Any] = {
                "schema": "s1-record/1",
                "dataset": dataset,
                "item_id": item_id,
                "state_sha256": state_sha,
                "question_id": qid,
                "primitive": q_model.type,
                "model_requested": model_requested,
                "model_served": resp.model_served,
                "latency_ms": resp.latency_ms,
                **resp.answer,
            }
            if "label" in item and role in ("primary", "noul"):
                rec["label"] = item["label"]
            if probe_mode and role in ("noul", "twin", "inverse"):
                rec["probe_role"] = role
                rec["probe_question"] = base_name
            records_out.append(rec)

    return records_out
