"""Langfuse exporter for System 1 records, scores, dataset runs, and annotation queues.

Enforces `safe-v1` (body-free by default): item `state` text is never emitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from s1ops.contracts.records import Record, parse_record
from s1ops.contracts.schemas import PolicyEntry
from s1ops.runtime.routing import ROUTING_RULE, route


def _trace_id(record: Record, run_name: str = "s1ops-run") -> str:
    state_hash = record.raw.get("state_sha256", "")
    raw = f"{run_name}:{record.dataset}:{record.item_id}:{state_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _observation_id(record: Record, run_name: str = "s1ops-run") -> str:
    raw = (
        f"{run_name}:{record.dataset}:{record.item_id}:"
        f"{record.question_id}:{record.model_served}:{record.primitive}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _predict_info(
    record: Record,
    policy: PolicyEntry | None = None,
) -> dict[str, Any]:
    """Compute scalar p_raw, p_calib, brier, and policy_action for a Record."""
    raw_row = record.raw
    calib = raw_row.get("calib") if isinstance(raw_row.get("calib"), dict) else None

    if record.primitive == "noul":
        p_raw = float(raw_row["p"])
        # The parser checks calib.p, as drift sees it.
        p_cal = parse_record(raw_row, record.line, use_calib=True).probs[1] if calib and "p" in calib else None
        p_eff = p_cal if p_cal is not None else p_raw
        brier_val: float | None = None
        if isinstance(record.label, bool):
            y = 1.0 if record.label else 0.0
            brier_val = (p_eff - y) ** 2

        if policy is not None:
            routed = route(policy, (False, True), (1.0 - p_eff, p_eff))
            action, predicted = routed.action, routed.predicted
        else:
            predicted = p_eff >= 0.5
            action = "ALLOW" if predicted else "BLOCK"

        return {
            "p_raw": round(p_raw, 6),
            "p_calib": round(p_cal, 6) if p_cal is not None else None,
            "p_effective": round(p_eff, 6),
            "predicted": predicted,
            "brier": round(brier_val, 6) if brier_val is not None else None,
            "policy_action": action,
            "routing_rule": ROUTING_RULE if policy is not None else None,
        }

    # choice or score
    probs_raw: dict[str, float] = dict(raw_row["probabilities"])
    probs_cal: dict[str, float] | None = (
        dict(calib["probabilities"]) if calib and "probabilities" in calib else None
    )
    probs_eff = probs_cal if probs_cal is not None else probs_raw
    # The parser checks the calibrated keys against the raw ones and renormalizes, as drift sees it.
    parsed = parse_record(raw_row, record.line, use_calib=True) if probs_cal is not None else record
    predicted = max(probs_eff.items(), key=lambda kv: (kv[1], kv[0]))[0]
    action = f"SELECT:{predicted}"
    if policy is not None:
        # Export labels as the record wrote them ("01" is not reported as "1").
        routed = route(policy, parsed.classes, parsed.probs)
        label_of = {int(k): k for k in probs_eff} if record.primitive == "score" else {k: k for k in probs_eff}
        predicted = label_of[routed.predicted]
        action = "ESCALATE" if routed.escalated else f"SELECT:{predicted}"
    p_raw = float(probs_raw[predicted])
    p_cal = float(probs_cal[predicted]) if probs_cal is not None else None
    p_eff = float(probs_eff[predicted])

    brier_val = None
    if record.label is not None and str(record.label) in probs_eff:
        lbl_str = str(record.label)
        brier_val = sum(
            (float(p) - (1.0 if k == lbl_str else 0.0)) ** 2
            for k, p in probs_eff.items()
        )

    return {
        "p_raw": round(p_raw, 6),
        "p_calib": round(p_cal, 6) if p_cal is not None else None,
        "p_effective": round(p_eff, 6),
        "predicted": predicted,
        "brier": round(brier_val, 6) if brier_val is not None else None,
        "policy_action": action,
        "routing_rule": ROUTING_RULE if policy is not None else None,
    }


def build_langfuse_events(
    records: list[Record],
    *,
    run_name: str = "s1ops-eval",
    policies: list[PolicyEntry] | None = None,
    annotation_candidates: list[dict[str, Any]] | None = None,
    redaction: str = "safe-v1",
) -> dict[str, Any]:
    """Build Langfuse-compatible batch ingestion events (safe-v1 body-free by default)."""
    if redaction != "safe-v1":
        raise ValueError(
            f"Unsupported redaction profile {redaction!r}; s1ops enforces 'safe-v1' (body-free)"
        )

    policy_map: dict[tuple[str, str, str], PolicyEntry] = {}
    if policies:
        for pol in policies:
            policy_map[(pol.question_id, pol.model_served, pol.primitive)] = pol

    dataset_run_items: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []

    for r in records:
        if r.model_served is None:
            continue
        tid = _trace_id(r, run_name=run_name)
        oid = _observation_id(r, run_name=run_name)
        pol = policy_map.get((r.question_id, r.model_served, r.primitive))
        info = _predict_info(r, pol)

        obs = {
            "id": oid,
            "traceId": tid,
            "type": "GENERATION",
            "name": f"s1.{r.primitive}.{r.question_id}",
            "model": r.model_served,
            "input": {
                "dataset": r.dataset,
                "item_id": r.item_id,
                "state_sha256": r.raw.get("state_sha256"),
                "question_id": r.question_id,
                "primitive": r.primitive,
            },
            "output": {
                "p_raw": info["p_raw"],
                "p_calib": info["p_calib"],
                "predicted": info["predicted"],
                "policy_action": info["policy_action"],
            },
            "metadata": {
                "redaction": "safe-v1",
                "model_requested": r.model_requested,
                "latency_ms": r.raw.get("latency_ms"),
            },
        }
        if info["routing_rule"] is not None:
            obs["metadata"]["routing_rule"] = info["routing_rule"]
        observations.append(obs)

        dataset_run_items.append(
            {
                "runName": run_name,
                "datasetName": r.dataset or "default",
                "datasetItemId": r.item_id,
                "traceId": tid,
                "observationId": oid,
            }
        )

        prefix = f"s1.{r.question_id}"
        scores.append(
            {
                "traceId": tid,
                "observationId": oid,
                "name": f"{prefix}.p_raw",
                "dataType": "NUMERIC",
                "value": info["p_raw"],
            }
        )
        if info["p_calib"] is not None:
            scores.append(
                {
                    "traceId": tid,
                    "observationId": oid,
                    "name": f"{prefix}.p_calib",
                    "dataType": "NUMERIC",
                    "value": info["p_calib"],
                }
            )
        if info["brier"] is not None:
            scores.append(
                {
                    "traceId": tid,
                    "observationId": oid,
                    "name": f"{prefix}.brier",
                    "dataType": "NUMERIC",
                    "value": info["brier"],
                }
            )
        scores.append(
            {
                "traceId": tid,
                "observationId": oid,
                "name": f"{prefix}.policy_action",
                "dataType": "CATEGORICAL",
                "stringValue": info["policy_action"],
            }
        )

    queue_items: list[dict[str, Any]] = []
    if annotation_candidates:
        for item in annotation_candidates:
            safe_item = {k: v for k, v in item.items() if k != "state"}
            queue_items.append(
                {
                    "queueName": f"{run_name}-labelaudit",
                    "objectType": "OBSERVATION",
                    "dataset": safe_item.get("dataset"),
                    "item_id": safe_item.get("item_id"),
                    "question_id": safe_item.get("question_id"),
                    "metadata": safe_item,
                }
            )

    return {
        "schema": "s1-langfuse-export/1",
        "redaction": "safe-v1",
        "run_name": run_name,
        "summary": {
            "n_records": len(records),
            "n_observations": len(observations),
            "n_scores": len(scores),
            "n_annotation_queue_items": len(queue_items),
        },
        "dataset_run_items": dataset_run_items,
        "observations": observations,
        "scores": scores,
        "annotation_queue_items": queue_items,
    }


def export_langfuse_file(
    path: str | Path,
    records: list[Record],
    *,
    run_name: str = "s1ops-eval",
    policies: list[PolicyEntry] | None = None,
    annotation_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write Langfuse export bundle to disk and return the payload."""
    payload = build_langfuse_events(
        records,
        run_name=run_name,
        policies=policies,
        annotation_candidates=annotation_candidates,
        redaction="safe-v1",
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
