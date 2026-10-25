"""Live System 1 backends implementing the `Backend` protocol.

`HttpSystemOneBackend` calls `POST /v1/systemone` through `typesafe-sdk` (the `typesafe` extra).
Retries are left to `run_batch`, so that every attempt gets its own ledger reservation.

safe-v1: errors carry only an error code (`BackendError`), never server text, because a server may
echo the request and the request carries the raw `state`. The SDK logs request bodies at DEBUG level
on the `typesafe_sdk` logger, which has only a `NullHandler`; `s1` never configures logging, so they
go nowhere. A library caller that attaches a handler must keep that logger above DEBUG.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time
from collections.abc import Mapping
from typing import Any

from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, QuestionModel, ScoreQuestion
from s1ops.contracts.records import RecordError, parse_record
from s1ops.monitor.runner import BackendError, BackendResponse

RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
_NAME = "q"


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
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    return value if math.isfinite(value) and value >= 0 else None


class _Invalid(Exception):
    pass


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
        try:
            resp = self._client.system_one(
                state=state,
                questions={_NAME: question.wire_dict()},
                model=model_requested,
            )
        # Subclasses first: a validation error is an API error, and a timeout is a connection error.
        except sdk.TypeSafeAPIResponseValidationError:
            raise BackendError("invalid_response", sent=True) from None
        except sdk.TypeSafeAPIError as e:
            raise BackendError(
                f"http_{e.status}",
                sent=True,
                retryable=e.status in RETRY_STATUSES,
                retry_after_s=retry_after_s(e.headers),
            ) from None
        except sdk.TypeSafeAPITimeoutError:
            raise BackendError("timeout", sent=None, retryable=True) from None
        except sdk.TypeSafeAPIConnectionError:
            # The SDK does not say whether the request left before the connection failed.
            raise BackendError("connect", sent=None, retryable=True) from None
        except sdk.TypeSafeError:
            raise BackendError("invalid_request", sent=False) from None
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        tokens = resp.usage.input_tokens
        usage_usd = None
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
            usage_usd = tokens * self.usd_per_input_token
        try:
            if not isinstance(resp.model, str) or not resp.model:
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
