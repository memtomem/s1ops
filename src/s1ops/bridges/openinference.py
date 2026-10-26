"""OpenInference / Arize Phoenix span exporter for System 1 records.

Compatible with `tracegraph` contracts:
- `safe-v1` redaction profile by default (never emits raw item `state`)
- Status normalized to `"OK" | "ERROR" | "UNSET"`
- Single-currency cost guard (`MultipleCurrenciesError`)
- OpenInference span kinds:
  - `noul` -> `GUARDRAIL`
  - `choice` -> `EVALUATOR`
  - `score` -> `RERANKER`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from s1ops.bridges.langfuse import _predict_info
from s1ops.contracts.records import Record
from s1ops.contracts.schemas import PolicyEntry


class MultipleCurrenciesError(ValueError):
    """Raised when spans in a single export mix multiple cost currencies."""


_SPAN_KIND_BY_PRIMITIVE = {
    "noul": "GUARDRAIL",
    "choice": "EVALUATOR",
    "score": "RERANKER",
}

_VALID_STATUSES = {"OK", "ERROR", "UNSET"}


def normalize_status(status: str | None) -> str:
    """Normalize span status string to OpenInference 'OK' | 'ERROR' | 'UNSET'."""
    if not status:
        return "OK"
    upper = status.strip().upper()
    if upper in _VALID_STATUSES:
        return upper
    if upper in ("SUCCESS", "COMPLETED", "PASS"):
        return "OK"
    if upper in ("FAIL", "FAILED", "EXCEPTION"):
        return "ERROR"
    return "UNSET"


def build_openinference_spans(
    records: list[Record],
    *,
    run_name: str = "s1ops-eval",
    policies: list[PolicyEntry] | None = None,
    redaction: str = "safe-v1",
    currency: str = "USD",
    record_currencies: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenInference / Arize Phoenix spans (safe-v1 body-free by default)."""
    if redaction != "safe-v1":
        raise ValueError(
            f"Unsupported redaction profile {redaction!r}; s1ops enforces 'safe-v1' (body-free)"
        )

    if record_currencies:
        unique_currencies = set(record_currencies.values())
        if len(unique_currencies) > 1:
            raise MultipleCurrenciesError(
                f"Mixed currencies in export batch: {sorted(unique_currencies)}"
            )
        if unique_currencies:
            currency = next(iter(unique_currencies))

    policy_map: dict[tuple[str, str, str], PolicyEntry] = {}
    if policies:
        for pol in policies:
            policy_map[(pol.question_id, pol.model_served, pol.primitive)] = pol

    spans: list[dict[str, Any]] = []
    for r in records:
        if r.model_served is None:
            continue
        pol = policy_map.get((r.question_id, r.model_served, r.primitive))
        info = _predict_info(r, pol)
        span_kind = _SPAN_KIND_BY_PRIMITIVE.get(r.primitive, "GUARDRAIL")

        state_sha = r.raw.get("state_sha256", "")
        trace_seed = f"{run_name}:{r.dataset}:{r.item_id}:{state_sha}"
        span_seed = f"{trace_seed}:{r.question_id}:{r.model_served}:{r.primitive}"
        trace_id = hashlib.sha256(trace_seed.encode("utf-8")).hexdigest()[:32]
        span_id = hashlib.sha256(span_seed.encode("utf-8")).hexdigest()[:16]

        attributes: dict[str, Any] = {
            "openinference.span.kind": span_kind,
            "llm.model_name": r.model_served,
            "s1.redaction": "safe-v1",
            "s1.dataset": r.dataset,
            "s1.item_id": r.item_id,
            "s1.state_sha256": state_sha,
            "s1.question_id": r.question_id,
            "s1.primitive": r.primitive,
            "s1.model_served": r.model_served,
            "s1.model_requested": r.model_requested,
            "s1.p_raw": info["p_raw"],
            "s1.p_calib": info["p_calib"],
            "s1.p_effective": info["p_effective"],
            "s1.predicted": info["predicted"],
            "s1.brier": info["brier"],
            "s1.policy_action": info["policy_action"],
            "s1.cost.currency": currency,
        }
        if r.label is not None:
            attributes["s1.label"] = r.label
        if pol is not None:
            attributes["s1.policy_sha256"] = pol.policy_sha256
            attributes["s1.threshold"] = pol.threshold
            attributes["s1.band_lo"] = pol.band_lo
            attributes["s1.band_hi"] = pol.band_hi
            attributes["s1.routing_rule"] = info["routing_rule"]

        spans.append(
            {
                "name": f"s1.{r.primitive}.{r.question_id}",
                "context": {
                    "trace_id": trace_id,
                    "span_id": span_id,
                },
                "span_kind": span_kind,
                "status_code": normalize_status("OK"),
                "latency_ms": r.raw.get("latency_ms"),
                "timestamp": r.raw.get("timestamp"),
                "attributes": attributes,
            }
        )

    return spans


def export_openinference_jsonl(
    path: str | Path,
    records: list[Record],
    *,
    run_name: str = "s1ops-eval",
    policies: list[PolicyEntry] | None = None,
) -> list[dict[str, Any]]:
    """Write OpenInference spans as JSONL to `path` and return the list of spans."""
    spans = build_openinference_spans(
        records,
        run_name=run_name,
        policies=policies,
        redaction="safe-v1",
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for span in spans:
            fh.write(json.dumps(span, sort_keys=True) + "\n")
    return spans
