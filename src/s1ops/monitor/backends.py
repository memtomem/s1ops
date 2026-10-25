"""Live System 1 backends implementing the `Backend` protocol.

`HttpSystemOneBackend` calls `POST /v1/systemone` through `typesafe-sdk` (the `typesafe` extra).
Retries are left to `run_batch`, so that every attempt gets its own ledger reservation.

safe-v1: errors carry only an error code (`BackendError`), never server text, because a server may
echo the request and the request carries the raw `state`. The SDK's own logs carry it too (request
bodies at DEBUG, the request ID at INFO, an unknown answer type at WARNING) and reach any handler
the caller configured, so records the `typesafe_sdk` logger emits during `infer` are dropped. Its
logs outside that call, from other code in the process, pass through.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import math
import re
import time
from collections.abc import Mapping
from typing import Any

from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, QuestionModel, ScoreQuestion
from s1ops.contracts.records import RecordError, parse_record
from s1ops.monitor.runner import BackendError, BackendResponse

RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
_NAME = "q"
# `model_served` is server text that reaches records, ledgers and exports, so it must look like a model name.
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}")

_IN_INFER: contextvars.ContextVar[bool] = contextvars.ContextVar("s1ops_systemone_infer", default=False)


class _DropDuringInfer(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _IN_INFER.get()


_DROP_DURING_INFER = _DropDuringInfer()


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


def _answer(question: QuestionModel, answer: Any, sdk: Any) -> dict[str, Any]:
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
        # parse_record checks only that a score is finite.
        if not 0 <= answer.score <= levels[-1]:
            raise _Invalid
        out = {
            "score": answer.score,
            "probabilities": {str(k): answer.probabilities[k] for k in levels},
            "confidence": answer.confidence,
        }
    else:
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


class HttpSystemOneBackend:
    """`POST {base_url}/v1/systemone`, one question per request, priced by billable input tokens.

    `api_key`, `base_url` and `timeout` default to the SDK's (`TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`).
    The response carries token counts, not dollars: cost is `usage.input_tokens * usd_per_input_token`
    (output tokens are free). `transport` is an `httpx2` transport, for tests.
    """

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
        if isinstance(usd_per_input_token, bool) or not (
            isinstance(usd_per_input_token, (int, float)) and math.isfinite(usd_per_input_token) and usd_per_input_token >= 0
        ):
            raise ValueError(f"usd_per_input_token must be a finite number >= 0, got {usd_per_input_token!r}")
        self._sdk = typesafe_sdk
        # The SDK logs through this one logger; addFilter ignores a filter already added.
        logging.getLogger("typesafe_sdk").addFilter(_DROP_DURING_INFER)
        self.usd_per_input_token = float(usd_per_input_token)
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
        return None if tokens is None else tokens * self.usd_per_input_token

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
            answer = _answer(question, resp.answers.get(_NAME), sdk)
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
