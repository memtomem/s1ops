"""Live System 1 backends implementing the `Backend` protocol.

`HttpSystemOneBackend` calls `POST /v1/systemone` through `typesafe-sdk` (the `typesafe` extra), and
`OpenAIDecisionsBackend` calls `POST /v1/decisions` through `openai` (the `openai` extra). SDK
retries are off: `run_batch` retries, so that every attempt gets its own ledger reservation.

safe-v1: errors carry only an error code (`BackendError`), never server text, because a server may
echo the request and the request carries the raw `state`. The SDKs' and their transport's logs carry
it too (request bodies at DEBUG, request IDs and response headers, an unknown answer type at
WARNING, the reason phrase at INFO) and reach any handler the caller configured, so records those
loggers emit during `infer` are dropped. Their logs outside that call pass through.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
import re
import time
from collections.abc import Mapping
from typing import Any

from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, QuestionModel, ScoreQuestion, to_openai_decision
from s1ops.contracts.records import RecordError, parse_record
from s1ops.monitor.runner import BackendError, BackendResponse

RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
_NAME = "q"
# `model_served` is server text that reaches records, ledgers and exports, so it must look like a model name.
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}")

_IN_INFER: contextvars.ContextVar[bool] = contextvars.ContextVar("s1ops_backend_infer", default=False)

# A logger's filter does not see records from its child loggers, so every logger that can carry server
# text is named. Some exist only after a lazy import (`httpcore2.http2`, `httpcore2.socks`); creating
# them here hands the later import the same, filtered, logger. tests/test_openai_backend.py fails if a
# library adds a logger under these prefixes that is not listed.
QUIET_LOGGERS = (
    "typesafe_sdk",
    "openai",
    "openai._base_client",
    "openai._response",
    "openai._legacy_response",
    "httpx2",
    "httpcore2.connection",
    "httpcore2.http11",
    "httpcore2.http2",
    "httpcore2.proxy",
    "httpcore2.socks",
)


class _DropDuringInfer(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _IN_INFER.get()


_DROP_DURING_INFER = _DropDuringInfer()


def _quiet_sdk_logs() -> None:
    for name in QUIET_LOGGERS:
        # addFilter ignores a filter already added.
        logging.getLogger(name).addFilter(_DROP_DURING_INFER)


def _usd(tokens: int | None, usd_per_input_token: float) -> float | None:
    """Tokens times the price, or `None` when the product is not a finite number of dollars.

    A server-reported count can be any JSON integer; `10**400 * price` raises OverflowError, and this
    can run while an SDK error is being handled, where an escaping exception chains to its server text.
    """
    if tokens is None:
        return None
    try:
        usd = tokens * usd_per_input_token
    except OverflowError:
        return None
    return usd if math.isfinite(usd) else None


def _check_price(usd_per_input_token: object) -> float:
    if isinstance(usd_per_input_token, bool) or not (
        isinstance(usd_per_input_token, (int, float)) and math.isfinite(usd_per_input_token) and usd_per_input_token >= 0
    ):
        raise ValueError(f"usd_per_input_token must be a finite number >= 0, got {usd_per_input_token!r}")
    return float(usd_per_input_token)


def retry_after_s(headers: Mapping[str, str]) -> float | None:
    """The server's requested wait in seconds from `retry-after-ms` or `Retry-After`, if valid."""
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            value = float(ms)
        except ValueError:
            value = math.nan
        if math.isfinite(value) and value >= 0:
            return value / 1000
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        # This runs while the SDK error is being handled, so an escaping exception would chain to it
        # and a traceback would print the server's body. An absurd offset raises OverflowError.
        try:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
    return value if math.isfinite(value) and value >= 0 else None


class _Invalid(Exception):
    pass


def _tokens(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _body_tokens(body: Any) -> int | None:
    """Billable input tokens from an error body, when it reports valid usage."""
    usage = body.get("usage") if isinstance(body, dict) else None
    return _tokens(usage.get("input_tokens")) if isinstance(usage, dict) else None


def _validated(question: QuestionModel, out: dict[str, Any]) -> dict[str, Any]:
    """Check mapped answer keys the way `s1-record/1` will, plus the score range it does not check."""
    if isinstance(question, ScoreQuestion):
        score = out["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= len(question.criteria) - 1:
            raise _Invalid
    envelope = {
        "schema": "s1-record/1",
        "item_id": "-",
        "question_id": "-",
        "primitive": question.type,
        "model_requested": "-",
        "model_served": "-",
    }
    try:
        parse_record({**envelope, **out})
    except RecordError:
        raise _Invalid from None
    return out


def _typesafe_answer(question: QuestionModel, answer: Any, sdk: Any) -> dict[str, Any]:
    """Map the SDK answer to `s1-record/1` answer keys, checking it against the question asked."""
    if isinstance(question, NoulQuestion):
        if not isinstance(answer, sdk.NoulAnswer):
            raise _Invalid
        out: dict[str, Any] = {"p": answer.noul}
    elif isinstance(question, ChoiceQuestion):
        if not isinstance(answer, sdk.ChoiceAnswer) or set(answer.probabilities) != set(question.criteria):
            raise _Invalid
        out = {
            "choice": answer.choice,
            "probabilities": {k: answer.probabilities[k] for k in question.criteria},
            "confidence": answer.confidence,
        }
    elif isinstance(question, ScoreQuestion):
        levels = range(len(question.criteria))
        if not isinstance(answer, sdk.ScoreAnswer) or set(answer.probabilities) != set(levels):
            raise _Invalid
        out = {
            "score": answer.score,
            "probabilities": {str(k): answer.probabilities[k] for k in levels},
            "confidence": answer.confidence,
        }
    else:
        raise _Invalid
    return _validated(question, out)


class HttpSystemOneBackend:
    """`POST {base_url}/v1/systemone`, one question per request, priced by billable input tokens.

    `api_key`, `base_url` and `timeout` default to the SDK's (`TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`).
    The response carries token counts, not dollars: cost is `usage.input_tokens * usd_per_input_token`
    (output tokens are free). `transport` is an `httpx2` transport, for tests.
    """

    # Measured 2026-10-25 on jev-1.13.0 / jev-1.14.0: each request bills about 235 input tokens more
    # than `token_bound` estimates for its state and question; `run_batch` adds this to every reservation.
    reserve_overhead_tokens = 300.0

    def __init__(
        self,
        *,
        usd_per_input_token: float,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: Any = None,
    ) -> None:
        try:
            import typesafe_sdk
        except ImportError:
            raise ImportError("HttpSystemOneBackend needs the typesafe extra: pip install 's1ops[typesafe]'") from None
        self.usd_per_input_token = _check_price(usd_per_input_token)
        self._sdk = typesafe_sdk
        _quiet_sdk_logs()
        try:
            self._client = typesafe_sdk.TypeSafeClient(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                transport=transport,
                retry=typesafe_sdk.RetryPolicy(max_retries=0),
            )
        except typesafe_sdk.TypeSafeError as e:
            # Configuration messages name the setting, never the key itself.
            raise ValueError(f"TypeSafe client: {e}") from None

    def _usd(self, tokens: int | None) -> float | None:
        return _usd(tokens, self.usd_per_input_token)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpSystemOneBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def infer(
        self,
        state: str,
        question: QuestionModel,
        *,
        model_requested: str,
    ) -> BackendResponse:
        sdk = self._sdk
        started = time.perf_counter()
        quiet = _IN_INFER.set(True)
        try:
            resp = self._client.system_one(
                state=state,
                questions={_NAME: question.wire_dict()},
                model=model_requested,
            )
        # Subclasses first: a validation error is an API error, and a timeout is a connection error.
        # A body that failed the SDK's schema, or an HTTP error body, may still report valid usage:
        # keep it as the measured cost, so that an overrun there stops the run too.
        except sdk.TypeSafeAPIResponseValidationError as e:
            raise BackendError("invalid_response", sent=True, usage_usd=self._usd(_body_tokens(e.body))) from None
        except sdk.TypeSafeAPIError as e:
            raise BackendError(
                f"http_{e.status}",
                sent=True,
                retryable=e.status in RETRY_STATUSES,
                retry_after_s=retry_after_s(e.headers),
                usage_usd=self._usd(_body_tokens(e.body)),
            ) from None
        except sdk.TypeSafeAPITimeoutError:
            raise BackendError("timeout", sent=None, retryable=True) from None
        except sdk.TypeSafeAPIConnectionError:
            # The SDK does not say whether the request left before the connection failed.
            raise BackendError("connect", sent=None, retryable=True) from None
        except sdk.TypeSafeError:
            raise BackendError("invalid_request", sent=False) from None
        finally:
            _IN_INFER.reset(quiet)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        usage_usd = self._usd(_tokens(resp.usage.input_tokens))
        try:
            if not isinstance(resp.model, str) or not _MODEL_NAME.fullmatch(resp.model):
                raise _Invalid
            answer = _typesafe_answer(question, resp.answers.get(_NAME), sdk)
        except _Invalid:
            raise BackendError("invalid_response", sent=True, usage_usd=usage_usd) from None
        if usage_usd is None:
            raise BackendError("invalid_usage", sent=True)
        return BackendResponse(
            model_served=resp.model,
            answer=answer,
            cost_usd=usage_usd,
            latency_ms=latency_ms,
        )


class _Refused(Exception):
    pass


def _json_body(response: Any) -> Any:
    """The decoded JSON body, or `None` when there is none to read."""
    try:
        return json.loads(response.content)
    except Exception:
        # Not JSON, not UTF-8, or never read (httpx2.ResponseNotRead). This runs while an SDK error is
        # being handled, and an escaping exception would chain to that error and its server text.
        return None


def _openai_tokens(body: Any) -> int | None:
    """Billable input tokens, or `None` when the usage is missing or its price is not known.

    Requests disable prompt caching, so a cache write (charged at a rate other than the input
    price) means the measured cost is unknown. Cached reads are discounted, so the input price is
    an upper bound for them.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    details = usage.get("input_tokens_details", {})
    if not isinstance(details, dict) or _tokens(details.get("cache_write_tokens", 0)) != 0:
        return None
    return _tokens(usage.get("input_tokens"))


def _distribution(entries: Any, keys: list[Any], key_ok: Any) -> dict[Any, Any]:
    """`[{value, probability}]` as a dict over exactly `keys`.

    One entry per key, checked before the dict exists: as many entries as keys whose values cover the
    keys leaves no room for a duplicate, which a dict would otherwise collapse into a valid-looking
    distribution.
    """
    if not isinstance(entries, list) or len(entries) != len(keys):
        raise _Invalid
    values = [e.get("value") if isinstance(e, dict) else None for e in entries]
    if not all(key_ok(v) for v in values) or set(values) != set(keys):
        raise _Invalid
    return {e["value"]: e.get("probability") for e in entries}


def _openai_answer(question: QuestionModel, body: Any) -> dict[str, Any]:
    """Map the one answer in a `/v1/decisions` body to `s1-record/1` answer keys."""
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], dict):
        raise _Invalid
    answer = answers[0]
    kind = answer.get("type")
    if answer.get("name") != _NAME:
        raise _Invalid
    if kind == "refusal":
        raise _Refused
    if isinstance(question, NoulQuestion) and kind == "predicate":
        out: dict[str, Any] = {"p": answer.get("probability")}
    elif isinstance(question, ChoiceQuestion) and kind == "choice":
        keys = list(question.criteria)
        probs = _distribution(answer.get("probabilities"), keys, lambda v: isinstance(v, str))
        out = {
            "choice": answer.get("choice"),
            "probabilities": {k: probs[k] for k in keys},
            "confidence": answer.get("confidence"),
        }
    elif isinstance(question, ScoreQuestion) and kind == "score":
        levels = list(range(len(question.criteria)))
        probs = _distribution(
            answer.get("probabilities"), levels, lambda v: isinstance(v, int) and not isinstance(v, bool)
        )
        out = {
            "score": answer.get("score"),
            "probabilities": {str(k): probs[k] for k in levels},
            "confidence": answer.get("confidence"),
        }
    else:
        raise _Invalid
    return _validated(question, out)


class OpenAIDecisionsBackend:
    """`POST {base_url}/v1/decisions`, one question per request, priced by billable input tokens.

    `api_key`, `base_url` and `timeout` default to the SDK's (`OPENAI_API_KEY`, `OPENAI_BASE_URL`).
    Prompt caching is disabled (`prompt_cache_options={"mode": "explicit"}` with no breakpoints), so
    cost is `usage.input_tokens * usd_per_input_token` and the state is not kept in a server cache.
    The SDK does not validate responses by default, so the raw JSON is read and checked here.
    `http_client` is an `httpx2.Client`, for tests.
    """

    # Measured 2026-10-25 on gpt-6-luna: each request bills about 120 input tokens more than
    # `token_bound` estimates for its state and question; `run_batch` adds this to every reservation.
    reserve_overhead_tokens = 200.0

    def __init__(
        self,
        *,
        usd_per_input_token: float,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: Any = None,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAIDecisionsBackend needs the openai extra: pip install 's1ops[openai]'") from None
        self.usd_per_input_token = _check_price(usd_per_input_token)
        self._sdk = openai
        _quiet_sdk_logs()
        options: dict[str, Any] = {"api_key": api_key, "base_url": base_url, "max_retries": 0, "http_client": http_client}
        # The SDK applies its default timeout only when the argument is omitted; None disables it.
        if timeout is not None:
            options["timeout"] = timeout
        try:
            self._client = openai.OpenAI(**options)
        except openai.OpenAIError as e:
            # Configuration messages name the setting, never the key itself.
            raise ValueError(f"OpenAI client: {e}") from None

    def _usd(self, tokens: int | None) -> float | None:
        return _usd(tokens, self.usd_per_input_token)

    def check_question(self, question: QuestionModel) -> None:
        """Raise `ValueError` for a question `/v1/decisions` cannot take, before anything is reserved."""
        to_openai_decision(_NAME, question)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIDecisionsBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def infer(
        self,
        state: str,
        question: QuestionModel,
        *,
        model_requested: str,
    ) -> BackendResponse:
        sdk = self._sdk
        wire = to_openai_decision(_NAME, question)
        started = time.perf_counter()
        quiet = _IN_INFER.set(True)
        try:
            raw = self._client.decisions.with_raw_response.create(
                model=model_requested,
                input=state,
                questions=[wire],
                prompt_cache_options={"mode": "explicit"},
            )
            body = _json_body(raw.http_response)
        # Subclasses first: a timeout is a connection error. `e.body` holds only the inner `error`
        # object, so usage is read from the whole response.
        except sdk.APIResponseValidationError as e:
            raise BackendError("invalid_response", sent=True, usage_usd=self._usd(_openai_tokens(_json_body(e.response)))) from None
        except sdk.APIStatusError as e:
            raise BackendError(
                f"http_{e.status_code}",
                sent=True,
                retryable=e.status_code in RETRY_STATUSES,
                retry_after_s=retry_after_s(e.response.headers),
                usage_usd=self._usd(_openai_tokens(_json_body(e.response))),
            ) from None
        except sdk.APITimeoutError:
            raise BackendError("timeout", sent=None, retryable=True) from None
        except sdk.APIConnectionError:
            raise BackendError("connect", sent=None, retryable=True) from None
        except sdk.OpenAIError:
            raise BackendError("invalid_request", sent=False) from None
        finally:
            _IN_INFER.reset(quiet)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        usage_usd = self._usd(_openai_tokens(body))
        model = body.get("model") if isinstance(body, dict) else None
        try:
            if not isinstance(model, str) or not _MODEL_NAME.fullmatch(model):
                raise _Invalid
            answer = _openai_answer(question, body)
        except _Refused:
            raise BackendError("refusal", sent=True, usage_usd=usage_usd) from None
        except _Invalid:
            raise BackendError("invalid_response", sent=True, usage_usd=usage_usd) from None
        if usage_usd is None:
            raise BackendError("invalid_usage", sent=True)
        return BackendResponse(
            model_served=model,
            answer=answer,
            cost_usd=usage_usd,
            latency_ms=latency_ms,
        )
