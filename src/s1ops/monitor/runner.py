"""Budget-guarded System 1 probe & batch runner under the Reserve-Then-Settle Ledger.

Key guarantees:
1. **Reserve-Then-Settle Accounting**: Every backend attempt first reserves its worst-case USD
   cost (`token_bound(state) * price_per_token`) in `Ledger` and settles it as `measured` when the
   backend reports a cost, `unsent` when the request provably never left, and `unknown` (charged at
   the reservation) otherwise. A measured cost above its reservation stops the run.
2. **Zero-State Privacy (`safe-v1`)**: Raw `state` text is hashed (`state_sha256`) in memory and
   never written to output `s1-record/1` JSONL files or ledger metadata.
3. **Automatic Probe Expansion (`expand_probes`)**: Given a `NoulQuestion`, automatically constructs
   its `twin` (`ChoiceQuestion` with `{"yes", "no"}`) and `inverse` (`NoulQuestion` with negated
   instructions) for coherence auditing (`s1 lint probe-audit`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Protocol

from s1ops.contracts.questions import (
    ChoiceQuestion,
    NoulQuestion,
    QuestionModel,
    ScoreQuestion,
    parse_wire_question,
    question_id,
)
from s1ops.ledger.journal import BudgetExceeded, Ledger, LedgerUnusable, Reservation, token_bound


@dataclass(frozen=True)
class BackendResponse:
    model_served: str
    answer: dict[str, Any]
    cost_usd: float
    latency_ms: float


class BackendError(RuntimeError):
    """A backend attempt that produced no usable answer.

    The message is `error_code` plus a fixed phrase, never server-supplied text: a server may echo
    the request, and the request carries the raw `state`.
    """

    def __init__(
        self,
        error_code: str,
        *,
        sent: bool | None,
        retryable: bool = False,
        retry_after_s: float | None = None,
        usage_usd: float | None = None,
    ) -> None:
        super().__init__(f"{error_code}: the backend returned no usable answer")
        self.error_code = error_code
        self.sent = sent
        """`False` only when the request provably never left; `None` when that is unknown."""
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        """The server's requested wait before retrying, in seconds."""
        self.usage_usd = usage_usd
        """Measured cost when a response arrived with valid usage but an unusable answer."""


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


def check_questions(work_questions: Sequence[tuple[str, str, QuestionModel]], backend: Backend | None = None) -> None:
    """Reject questions no backend can answer, or this one cannot, before anything is reserved.

    A backend may define `check_question(question)`, raising `ValueError` for a question its API
    cannot take.
    """
    check = getattr(backend, "check_question", None)
    for name, role, q in work_questions:
        if isinstance(q, (ChoiceQuestion, ScoreQuestion)) and not q.criteria:
            raise ValueError(f"question {name!r}: {q.type} criteria is empty")
        if check is not None:
            try:
                check(q)
            except ValueError as e:
                probe = "" if role in ("primary", "noul") else f" ({role} probe)"
                raise ValueError(f"question {name!r}{probe}: {e}") from None


def check_reservation(worst_usd_per_token: float, reserve_overhead_tokens: float) -> None:
    """Reject reservation inputs that would collapse every reservation to the $1e-6 floor."""
    for name, value, low in (
        ("worst_usd_per_token", worst_usd_per_token, "> 0"),
        ("reserve_overhead_tokens", reserve_overhead_tokens, ">= 0"),
    ):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if not ok or value < 0 or (low == "> 0" and value == 0):
            raise ValueError(f"{name} must be a finite number {low}, got {value!r}")


def _settle_failure(ledger: Ledger, reservation: Reservation, error: BackendError) -> None:
    if error.usage_usd is not None:
        ledger.settle(reservation, error.usage_usd, outcome="measured")
    elif error.sent is False:
        ledger.settle(reservation, 0.0, outcome="unsent")
    else:
        ledger.settle(reservation, outcome="unknown")


def _backoff_s(attempt: int, initial: float, maximum: float) -> float:
    return min(maximum, initial * 2**attempt)


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
    reserve_overhead_tokens: float | None = None,
    probe_mode: bool = False,
    max_retries: int = 0,
    backoff_initial_s: float = 0.5,
    backoff_max_s: float = 8.0,
    max_retry_after_s: float = 60.0,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Run System 1 questions over a batch of states with strict `Ledger` reserve-then-settle.

    Each attempt reserves `(token_bound(state) + token_bound(question) + reserve_overhead_tokens) *
    worst_usd_per_token`. `reserve_overhead_tokens` defaults to the backend's own
    `reserve_overhead_tokens` attribute (the server-side prompt tokens it bills per request), or 0.
    A retryable `BackendError` is retried up to `max_retries` times, waiting the
    server's `Retry-After` or an exponential backoff; a requested wait above `max_retry_after_s` ends
    the retries instead. An item that still fails gets a no-answer record: `model_served: null` and its
    `error_code`. Each record is passed to `on_record` as soon as it exists, so a run that stops early
    keeps what it already paid for. A measured cost above its reservation raises `LedgerUnusable` after
    that record is passed on.

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

    check_questions(work_questions, backend)
    if reserve_overhead_tokens is None:
        reserve_overhead_tokens = getattr(backend, "reserve_overhead_tokens", 0.0)
    check_reservation(worst_usd_per_token, reserve_overhead_tokens)

    records_out: list[dict[str, Any]] = []
    for item in states:
        dataset = item.get("dataset", "default")
        item_id = str(item["item_id"])
        state_text = str(item["state"])
        state_sha = hashlib.sha256(state_text.encode("utf-8")).hexdigest()

        for base_name, role, q_model in work_questions:
            qid = question_id(q_model)
            q_wire = json.dumps(q_model.wire_dict())
            est_tokens = token_bound(state_text) + token_bound(q_wire) + reserve_overhead_tokens
            worst_usd = max(1e-6, round(est_tokens * worst_usd_per_token, 8))

            attempt = 0
            refused: BudgetExceeded | None = None
            while True:
                try:
                    reservation = ledger.reserve(
                        worst_usd,
                        dataset=dataset,
                        item_id=item_id,
                        question_id=qid,
                        model_requested=model_requested,
                    )
                except BudgetExceeded as e:
                    if attempt == 0:
                        raise
                    # A paid attempt already failed: record it before stopping.
                    refused = e
                    break
                settling = False
                try:
                    try:
                        resp = backend.infer(
                            state_text,
                            q_model,
                            model_requested=model_requested,
                        )
                        error: BackendError | None = None
                    except BackendError as e:
                        error = e
                    if error is None:
                        cost_usd, model_served = resp.cost_usd, resp.model_served
                    # From here on a failure is the settlement's own; settling again could charge twice.
                    settling = True
                    if error is None:
                        ledger.settle(
                            reservation,
                            cost_usd,
                            outcome="measured",
                            model_served=model_served,
                        )
                    else:
                        _settle_failure(ledger, reservation, error)
                except BaseException:
                    if not settling:
                        # The request may have left: charge the reservation rather than nothing.
                        ledger.settle(reservation, outcome="unknown")
                    raise
                if error is None or ledger.overrun or not error.retryable or attempt >= max_retries:
                    break
                wait = error.retry_after_s
                if wait is None:
                    wait = _backoff_s(attempt, backoff_initial_s, backoff_max_s)
                elif wait > max_retry_after_s:
                    break
                sleep(wait)
                attempt += 1

            rec: dict[str, Any] = {
                "schema": "s1-record/1",
                "dataset": dataset,
                "item_id": item_id,
                "state_sha256": state_sha,
                "question_id": qid,
                "primitive": q_model.type,
                "model_requested": model_requested,
            }
            if error is None:
                rec["model_served"] = resp.model_served
                rec["latency_ms"] = resp.latency_ms
                rec.update(resp.answer)
            else:
                rec["model_served"] = None
                rec["error_code"] = error.error_code
            if "label" in item and role in ("primary", "noul"):
                rec["label"] = item["label"]
            if probe_mode and role in ("noul", "twin", "inverse"):
                rec["probe_role"] = role
                rec["probe_question"] = base_name
            records_out.append(rec)
            if on_record is not None:
                on_record(rec)
            if ledger.overrun:
                raise LedgerUnusable(ledger.overrun)
            if refused is not None:
                raise refused

    return records_out
