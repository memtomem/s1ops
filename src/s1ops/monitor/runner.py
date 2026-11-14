"""Budget-guarded System 1 probe & batch runner under the Reserve-Then-Settle Ledger.

Key guarantees:
1. **Reserve-Then-Settle Accounting**: Every backend attempt first reserves its worst-case USD
   cost (`token_bound(state) * price_per_token`) in `Ledger` and settles it as `measured` when the
   backend reports a cost, `unsent` when the request provably never left, and `unknown` (charged at
   the reservation) otherwise. A measured cost above its reservation stops the run.
2. **Zero-State Privacy (`safe-v1`)**: Raw `state` text is hashed (`state_sha256`) in memory and
   never written to output `s1-record/1` JSONL files or ledger metadata.
3. **Automatic Probe Expansion (`probe_tasks`)**: Given a `NoulQuestion`, automatically constructs
   its `twin` (`ChoiceQuestion` with `{"yes", "no"}`) and `inverse` (the question's
   `x-s1-lint.inverse` when supplied, else a generated negation) for coherence auditing
   (`s1 lint probe-audit`). Each inverse record says which (`probe_inverse`).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, NamedTuple, Protocol

from s1ops.contracts.questions import (
    ChoiceQuestion,
    NoulQuestion,
    QuestionModel,
    ScoreQuestion,
    question_id,
)
from s1ops.contracts.records import ANSWER_KEYS, label_problem, read_row, surrogate_problem
from s1ops.ledger.journal import BudgetExceeded, Ledger, LedgerUnusable, Reservation, token_bound
from s1ops.lint.loader import parse_question
from s1ops.lint.rules import RULES


@dataclass(frozen=True)
class BackendResponse:
    """What a backend returns for one question. `run_batch` keeps only `answer`'s `ANSWER_KEYS` for the
    question's primitive, and an answer it could not read back as a record is an `invalid_response`."""

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


def check_questions(work_questions: Sequence[ProbeTask | tuple[str, str, QuestionModel]], backend: Backend | None = None) -> None:
    """Reject questions no backend can answer, or this one cannot, before anything is reserved.

    A backend may define `check_question(question)`, raising `ValueError` for a question its API
    cannot take.
    """
    check = getattr(backend, "check_question", None)
    for name, role, q, *_ in work_questions:
        if isinstance(q, (ChoiceQuestion, ScoreQuestion)) and not q.criteria:
            raise ValueError(f"question {name!r}: {q.type} criteria is empty")
        if check is not None:
            try:
                check(q)
            except ValueError as e:
                probe = "" if role in ("primary", "noul") else f" ({role} probe)"
                raise ValueError(f"question {name!r}{probe}: {e}") from None


def check_states(states: Sequence[Any]) -> None:
    """Reject state rows `run_batch` cannot send, before anything is reserved.

    Each row needs `item_id` (a string or integer) and `state` (a string: anything else would be sent
    as its `str()`, such as the text "None"), and no two rows may be the same item as records key it,
    `(row.get("dataset", "default"), str(item_id))` (an explicit `null` dataset stays `None`): both would be paid for, and every reader of the output
    refuses an item that appears twice in a group. Messages name rows, fields and types, never a
    state, since the state is raw text that must not leave the process.
    """
    seen: dict[tuple[Any, str], int] = {}
    for i, row in enumerate(states, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {i} is not an object")
        if "item_id" not in row or "state" not in row:
            raise ValueError(f"row {i} is missing 'item_id' or 'state'")
        item_id = row["item_id"]
        if isinstance(item_id, bool) or not isinstance(item_id, (str, int)) or item_id == "":
            raise ValueError(f"row {i}: 'item_id' must be a non-empty string or an integer, got {type(item_id).__name__}")
        if not isinstance(row["state"], str):
            raise ValueError(f"row {i}: 'state' must be a string, got {type(row['state']).__name__}")
        dataset = row.get("dataset")
        if dataset is not None and (not isinstance(dataset, str) or not dataset):
            raise ValueError(f"row {i}: 'dataset' must be a non-empty string, got {type(dataset).__name__}")
        split = row.get("split")
        if split is not None and (not isinstance(split, str) or not split):
            raise ValueError(f"row {i}: 'split' must be a non-empty string, got {type(split).__name__}")
        # These three are copied into every record of the row, and a reader refuses a lone surrogate.
        for field in ("item_id", "dataset", "split"):
            problem = surrogate_problem(f"'{field}'", row[field]) if isinstance(row.get(field), str) else None
            if problem is not None:
                raise ValueError(f"row {i}: {problem}")
        key = (row.get("dataset", "default"), str(item_id))
        if key in seen:
            raise ValueError(f"row {i} is the same item as row {seen[key]}: (dataset, item_id) = {key!r}")
        seen[key] = i


def _label_options(question: QuestionModel) -> Collection[Any]:
    """The options a label of this question names: what the shipped backends require a response's keys to be."""
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria)
    if isinstance(question, ScoreQuestion):
        return range(len(question.criteria))
    return ()


def check_labels(states: Sequence[Mapping[str, Any]], work_questions: Sequence[ProbeTask | tuple[str, str, QuestionModel]]) -> None:
    """Reject a `label` the records it goes onto could not hold, before anything is reserved.

    Run after `check_states`. One `label` goes onto the record of every question in `LABELED_ROLES`, so it
    has to suit each of them. Refused whether the call would have answered or not: the reader ignores a
    label on a no-answer record, but a run should not start on a file that is wrong.
    """
    for i, row in enumerate(states, start=1):
        for name, role, q, *_ in work_questions:
            if role in LABELED_ROLES:
                problem = label_problem(q.type, row.get("label"), _label_options(q))
                if problem is not None:
                    raise ValueError(f"row {i}: question {name!r}: {problem}")


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


def _readable(record: Mapping[str, Any]) -> bool:
    """Whether a reader would take `record` as `s1 probe` writes it. Over the dumped text, since the parser
    does not look at `latency_ms` or other fields it does not read, and `NaN` only shows there."""
    try:
        read_row(json.dumps(record, sort_keys=True), 1, None)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _answered_record(envelope: Mapping[str, Any], resp: BackendResponse, primitive: str, tail: Mapping[str, Any]) -> dict[str, Any] | None:
    """The record of an answer, or `None` when no reader would take it. Only the answer keys are copied, so
    nothing a backend returns can set `state`, an identity field or a `label`."""
    if not isinstance(resp.answer, Mapping):
        return None
    answer = {key: value for key, value in resp.answer.items() if key in ANSWER_KEYS[primitive]}
    record = {**envelope, "model_served": resp.model_served, "latency_ms": resp.latency_ms, **answer, **tail}
    return record if _readable(record) else None


def _backoff_s(attempt: int, initial: float, maximum: float) -> float:
    return min(maximum, initial * 2**attempt)


_KNOWN_RULES = frozenset(RULES)
LABELED_ROLES = ("primary", "noul")
"""The probe roles a states row's `label` is copied to; a twin or inverse answers another wording."""


class ProbeTask(NamedTuple):
    name: str
    role: str
    question: QuestionModel
    inverse_source: str | None = None


def _read_question(name: str, raw: QuestionModel | dict[str, Any]) -> tuple[QuestionModel, NoulQuestion | None]:
    """A question and its supplied inverse, read as `s1 lint check` reads them."""
    if isinstance(raw, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
        return raw, None
    spec = parse_question(name, raw, _KNOWN_RULES)
    if spec.errors:
        raise ValueError(f"question {name!r}: {'; '.join(spec.errors)}")
    return spec.question, spec.inverse


def _generated_inverse(q: NoulQuestion) -> NoulQuestion:
    criteria = None
    if q.criteria is not None:
        criteria = {k: q.criteria[other] for k, other in (("true", "false"), ("false", "true")) if other in q.criteria}
    return NoulQuestion(instructions=f"Is it false that: {q.instructions.strip()}", criteria=criteria)


def probe_tasks(
    questions: Mapping[str, QuestionModel | dict[str, Any]],
    *,
    probe_mode: bool,
) -> list[ProbeTask]:
    """The questions `run_batch` sends, in order.

    A dict question may carry an `x-s1-lint` / `x-jev-lint` extension, read and stripped as lint
    does, so its `question_id` is lint's. In probe mode a `NoulQuestion` becomes `noul`, `twin`
    (yes/no `ChoiceQuestion`) and `inverse`: the extension's `inverse` when one is supplied,
    otherwise "Is it false that: <instructions>" with the true/false criteria swapped (only for
    string instructions). Everything else is `primary`.

    Raises `ValueError` for a name or a wording that holds a lone surrogate: a probe's name is copied
    into its records, which no reader would then take, and `question_id` cannot hash such a wording.
    Here, so that every question is refused before anything is reserved, not when its turn comes.
    """
    out: list[ProbeTask] = []
    for name, raw in questions.items():
        problem = surrogate_problem("the name", name) if isinstance(name, str) else None
        if problem is not None:
            raise ValueError(f"question {name!r}: {problem}")
        q, supplied = _read_question(name, raw)
        if probe_mode and isinstance(q, NoulQuestion) and (supplied is not None or isinstance(q.instructions, str)):
            out.append(ProbeTask(name, "noul", q))
            twin = ChoiceQuestion(
                instructions=q.instructions,
                criteria={"yes": "True / Yes", "no": "False / No"},
            )
            out.append(ProbeTask(name, "twin", twin))
            if supplied is not None:
                out.append(ProbeTask(name, "inverse", supplied, "supplied"))
            else:
                out.append(ProbeTask(name, "inverse", _generated_inverse(q), "generated"))
        else:
            out.append(ProbeTask(name, "primary", q))
    for task in out:
        try:
            question_id(task.question)
        except UnicodeEncodeError as e:
            raise ValueError(f"question {task.name!r} ({task.role}): {surrogate_problem('the wording', e.object)}") from None
    return out


def expand_probes(
    questions: Mapping[str, QuestionModel | dict[str, Any]],
) -> list[tuple[str, str, QuestionModel]]:
    """`(base_name, probe_role, QuestionModel)` for each probe `run_batch(probe_mode=True)` sends."""
    return [(t.name, t.role, t.question) for t in probe_tasks(questions, probe_mode=True)]


def run_batch(
    states: Sequence[Mapping[str, Any]],
    questions: Mapping[str, QuestionModel | dict[str, Any]],
    *,
    backend: Backend,
    ledger: Ledger,
    model_requested: str,
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

    `model_requested` is the name sent to the backend. It has no default: `model_served` is whatever the
    server reports, and a server that echoes the request reports this name. It must be a non-empty string,
    the rule `read_records` applies to the records written here.

    Each attempt reserves `(token_bound(state) + token_bound(question) + reserve_overhead_tokens) *
    worst_usd_per_token`. `reserve_overhead_tokens` defaults to the backend's own
    `reserve_overhead_tokens` attribute (the server-side prompt tokens it bills per request), or 0.
    A retryable `BackendError` is retried up to `max_retries` times, waiting the
    server's `Retry-After` or an exponential backoff; a requested wait above `max_retry_after_s` ends
    the retries instead. An item that still fails gets a no-answer record: `model_served: null` and its
    `error_code`. Each record is passed to `on_record` as soon as it exists, so a run that stops early
    keeps what it already paid for. A measured cost above its reservation raises `LedgerUnusable` after
    that record is passed on.

    Every record is one `read_records` takes: a `label` is checked before anything is reserved
    (`check_labels`), and a response that would not read back is settled as the `invalid_response` a
    shipped backend raises, not retried, and recorded with no answer. A `cost_usd` that is neither valid
    nor `None` still raises from `Ledger.settle`.

    Never writes raw `state` into returned `s1-record/1` dicts or ledger metadata.
    """
    if not isinstance(model_requested, str) or not model_requested:
        raise ValueError(f"model_requested must be a non-empty string, got {model_requested!r}")
    problem = surrogate_problem("model_requested", model_requested)
    if problem is not None:
        raise ValueError(problem)
    work_questions = probe_tasks(questions, probe_mode=probe_mode)

    check_states(states)
    check_labels(states, work_questions)
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

        for base_name, role, q_model, inverse_source in work_questions:
            qid = question_id(q_model)
            q_wire = json.dumps(q_model.wire_dict())
            est_tokens = token_bound(state_text) + token_bound(q_wire) + reserve_overhead_tokens
            worst_usd = max(1e-6, round(est_tokens * worst_usd_per_token, 8))
            envelope: dict[str, Any] = {
                "schema": "s1-record/1",
                "dataset": dataset,
                "item_id": item_id,
                "state_sha256": state_sha,
                "question_id": qid,
                "primitive": q_model.type,
                "model_requested": model_requested,
            }
            if item.get("split") is not None:
                envelope["split"] = item["split"]
            tail: dict[str, Any] = {}
            if "label" in item and role in LABELED_ROLES:
                tail["label"] = item["label"]
            if probe_mode and role in ("noul", "twin", "inverse"):
                tail["probe_role"] = role
                tail["probe_question"] = base_name
                if inverse_source is not None:
                    tail["probe_inverse"] = inverse_source

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
                        # Before settling: `Ledger.settle` itself refuses a `model_served` JSON cannot hold.
                        answered = _answered_record(envelope, resp, q_model.type, tail)
                        if answered is None:
                            error = BackendError("invalid_response", sent=True, usage_usd=resp.cost_usd)
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

            if error is None:
                rec = answered
            else:
                rec = {**envelope, "model_served": None, "error_code": error.error_code, **tail}
                if not _readable(rec):
                    rec["error_code"] = "invalid_response"
            records_out.append(rec)
            if on_record is not None:
                on_record(rec)
            if ledger.overrun:
                raise LedgerUnusable(ledger.overrun)
            if refused is not None:
                raise refused

    return records_out
