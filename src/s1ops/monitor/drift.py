"""Distribution, latency, abstain, and silent model-upgrade drift detector for System 1 models.

Compares a reference window (`records_ref`) against a current production/canary window (`records_cur`)
per `question_id` without ever requiring ground-truth labels on the current window:
1. **Population Stability Index (PSI)** & **Jensen-Shannon Divergence (JSD)** on predicted probabilities.
2. **Quantile & Entropy Shift**: `p10`, `p50`, `p90`, and mean Shannon entropy delta.
3. **Escalation & Abstain Drift**: shift in `model_served is None` failure rate and `[band_lo, band_hi)`
   escalation rate under an active `PolicyEntry`.
4. **Silent `model_served` Upgrade / Mismatch Detector**: immediately alerts if `model_served` changed
   between reference and current windows or if `model_requested != model_served`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np

from s1ops.contracts.records import GroupKey, Record, has_calib, parse_record, read_records, short_repr
from s1ops.contracts.schemas import PolicyEntry
from s1ops.runtime.routing import RoutingInputError, route_record

PSI_EPS = 1e-4
PSI_WARN = 0.10
PSI_CRITICAL = 0.25


class DriftInputError(ValueError):
    """`audit_drift(use_calib=True)` was given an answered record without calibrated
    probabilities, or records of one group calibrated through different fits. Carries the 1-based
    line and the side (`"reference"` or `"current"`)."""

    def __init__(self, line: int, message: str, side: str) -> None:
        self.line, self.message, self.side = line, message, side
        super().__init__(f"{side} records: line {line}: {message}")


def _one_primitive_per_question(refs: list[Record], curs: list[Record]) -> None:
    """Drift pools a question's models on purpose, so a model change shows, but never its
    primitives: a `question_id` names one wire question, and its metrics read one kind of answer."""
    first: dict[str, Record] = {}
    for side, records in (("reference", refs), ("current", curs)):
        for r in records:
            seen = first.setdefault(r.question_id, r)
            if r.primitive != seen.primitive:
                raise DriftInputError(
                    r.line,
                    f"question {r.question_id} is answered as {r.primitive} here and as {seen.primitive} elsewhere; "
                    "a question_id names one wire question, so one primitive: the file is hand-edited or mislabeled",
                    side,
                )


def _answered(rec: Record) -> bool:
    return rec.model_served is not None and bool(rec.probs)


def _compared(records: Sequence[Record], use_calib: bool, side: str) -> list[Record]:
    """`records` with each answered one re-parsed from its `raw` on the distribution the run
    compares, so the result does not depend on how the caller parsed them.

    A stored `calib` is validated in both modes. With `use_calib`, every answered record must carry
    one: falling back to raw per record would compare calibrated against raw probabilities.
    """
    out: list[Record] = []
    for r in records:
        if not _answered(r):
            out.append(r)
            continue
        checked = parse_record(r.raw, r.line, use_calib=True)
        if not use_calib:
            out.append(parse_record(r.raw, r.line))
        elif has_calib(r.raw, r.primitive):
            out.append(checked)
        else:
            raise DriftInputError(
                r.line,
                "a calibrated comparison needs calibrated probabilities and this record has none; "
                "run `s1 apply` on both files, or compare raw probabilities",
                side,
            )
    return out


def _fit_name(fit_sha256: str | None) -> str:
    return f"fit {fit_sha256!r}" if fit_sha256 is not None else "no fit"


def _one_fit(refs: Sequence[Record], curs: Sequence[Record]) -> None:
    """With `use_calib`, every answered record of a `(question_id, model_served, primitive)` group,
    on both sides, must name one `calib.fit_sha256`, as `s1 policy tune --use-calib` requires:
    otherwise a refit shows as drift. A group whose records all name none (applied before the
    hash) passes. Groups, not questions: a question's report pools models, and another model's
    records were calibrated by another fit by design."""
    first: dict[GroupKey, tuple[str | None, str, int]] = {}
    for side, records in (("reference", refs), ("current", curs)):
        for r in records:
            if not _answered(r):
                continue
            h = r.raw["calib"].get("fit_sha256")
            if h is not None and not isinstance(h, str):
                raise DriftInputError(r.line, f"calib.fit_sha256 must be a string, got {short_repr(h)}", side)
            seen, seen_side, seen_line = first.setdefault(r.key, (h, side, r.line))
            if h != seen:
                raise DriftInputError(
                    r.line,
                    f"calibrated through {_fit_name(h)}, but {seen_side} records line {seen_line} of the same "
                    f"group ({r.question_id!r}, {r.model_served!r}) through {_fit_name(seen)}; a calibrated "
                    "comparison needs one fit per group: run `s1 apply` on both files with one "
                    "fits.json, or compare raw probabilities",
                    side,
                )


def _shannon_entropy(probs: Sequence[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p, 2)
    return h


def psi(
    ref_scores: Sequence[float],
    cur_scores: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between reference and current score distributions."""
    r = np.asarray(ref_scores, dtype=float)
    c = np.asarray(cur_scores, dtype=float)
    if len(r) == 0 or len(c) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    r_counts, _ = np.histogram(r, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)
    r_prop = np.clip(r_counts / len(r), PSI_EPS, 1.0)
    c_prop = np.clip(c_counts / len(c), PSI_EPS, 1.0)
    return float(np.sum((c_prop - r_prop) * np.log(c_prop / r_prop)))


def jensen_shannon(
    ref_scores: Sequence[float],
    cur_scores: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """Compute Jensen-Shannon divergence (base 2, in [0, 1]) between histograms of scores."""
    r = np.asarray(ref_scores, dtype=float)
    c = np.asarray(cur_scores, dtype=float)
    if len(r) == 0 or len(c) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    r_counts, _ = np.histogram(r, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)
    p = r_counts / len(r)
    q = c_counts / len(c)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


def _primary_score(rec: Record) -> float:
    """Return `p` for `noul`, or max class probability for `choice`/`score`."""
    if rec.primitive == "noul":
        return float(rec.probs[1])
    return float(max(rec.probs)) if rec.probs else 0.0


def _quantiles(scores: Sequence[float]) -> dict[str, float]:
    if not scores:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}
    arr = np.asarray(scores, dtype=float)
    q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90])
    return {
        "p10": round(float(q10), 6),
        "p50": round(float(q50), 6),
        "p90": round(float(q90), 6),
        "mean": round(float(arr.mean()), 6),
    }


def _latency_summary(records: Sequence[Record]) -> dict[str, float | None]:
    vals = [
        float(r.raw["latency_ms"])
        for r in records
        if isinstance(r.raw.get("latency_ms"), (int, float))
        and not isinstance(r.raw.get("latency_ms"), bool)
    ]
    if not vals:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
    arr = np.asarray(vals, dtype=float)
    p50, p95 = np.quantile(arr, [0.50, 0.95])
    return {
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "mean_ms": round(float(arr.mean()), 2),
    }


def _escalation_rate(
    records: Sequence[Record], policies: Mapping[GroupKey, PolicyEntry], side: str | None = None
) -> float | None:
    """The share of answered records escalated by the policy of their own group, over the records
    that have one; `None` when none has."""
    escalated: list[bool] = []
    for r in records:
        pol = policies.get((r.question_id, r.model_served, r.primitive)) if r.model_served is not None and r.probs else None
        if pol is None:
            continue
        try:
            escalated.append(route_record(pol, r).escalated)
        except RoutingInputError as e:
            raise (e.with_side(side) if side else e) from None
    if not escalated:
        return None
    return round(sum(escalated) / len(escalated), 6)


@dataclass(frozen=True)
class DriftReport:
    question_id: str
    primitive: str
    ref_models: list[str]
    cur_models: list[str]
    model_drift_detected: bool
    model_mismatch_count_cur: int
    n_ref: int
    n_cur: int
    failure_rate_ref: float
    failure_rate_cur: float
    psi: float
    js_divergence: float
    quantiles_ref: dict[str, float]
    quantiles_cur: dict[str, float]
    mean_entropy_ref: float
    mean_entropy_cur: float
    entropy_delta: float
    escalation_rate_ref: float | None
    escalation_rate_cur: float | None
    latency_ref: dict[str, float | None]
    latency_cur: dict[str, float | None]
    status: str
    alerts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_drift(
    ref_records: str | Path | Iterable[Record],
    cur_records: str | Path | Iterable[Record],
    *,
    policies: Sequence[PolicyEntry] | None = None,
    bins: int = 10,
    use_calib: bool = False,
) -> dict[str, Any]:
    """Compare reference records against current records per `question_id`.

    The distributions compared (PSI, JS divergence, quantiles, entropy) are the raw ones, or with
    `use_calib` the calibrated ones, which every answered record on both sides must then carry,
    made by one fit per `(question_id, model_served, primitive)` group (`DriftInputError`
    otherwise). The escalation rate routes what each policy was tuned on either
    way.

    Records passed in are re-parsed from their `raw` with the default `sum_tol`, validating any
    stored `calib` in both modes, so one the caller parsed without `use_calib` or with a looser
    tolerance can raise `RecordError` here.
    """
    # Path reads validate a stored `calib` while reading, so its error names the file; the result
    # is the same, since `_compared` re-parses every answered record from `raw`.
    if isinstance(ref_records, (str, Path)):
        ref_records = read_records(ref_records, require_answer=True, use_calib=True)
    if isinstance(cur_records, (str, Path)):
        cur_records = read_records(cur_records, require_answer=True, use_calib=True)
    refs = _compared(list(ref_records), use_calib, "reference")
    curs = _compared(list(cur_records), use_calib, "current")
    if use_calib:
        _one_fit(refs, curs)

    _one_primitive_per_question(refs, curs)
    pol_by_key: dict[GroupKey, PolicyEntry] = {(p.question_id, p.model_served, p.primitive): p for p in policies or ()}

    ref_by_q: dict[str, list[Record]] = defaultdict(list)
    cur_by_q: dict[str, list[Record]] = defaultdict(list)
    for r in refs:
        ref_by_q[r.question_id].append(r)
    for r in curs:
        cur_by_q[r.question_id].append(r)

    shared_qids = sorted(ref_by_q.keys() & cur_by_q.keys())
    reports: list[DriftReport] = []

    for qid in shared_qids:
        r_all = ref_by_q[qid]
        c_all = cur_by_q[qid]
        prim = r_all[0].primitive

        r_ok = [r for r in r_all if _answered(r)]
        c_ok = [r for r in c_all if _answered(r)]

        fail_ref = (len(r_all) - len(r_ok)) / len(r_all) if r_all else 0.0
        fail_cur = (len(c_all) - len(c_ok)) / len(c_all) if c_all else 0.0

        ref_models = sorted({str(r.model_served) for r in r_ok})
        cur_models = sorted({str(r.model_served) for r in c_ok})
        model_drift = ref_models != cur_models
        mismatch_cur = sum(1 for r in c_ok if r.model_mismatch)

        r_scores = [_primary_score(r) for r in r_ok]
        c_scores = [_primary_score(r) for r in c_ok]

        psi_val = round(psi(r_scores, c_scores, bins=bins), 6)
        jsd_val = round(jensen_shannon(r_scores, c_scores, bins=bins), 6)

        ent_ref = (
            round(sum(_shannon_entropy(r.probs) for r in r_ok) / len(r_ok), 6)
            if r_ok
            else 0.0
        )
        ent_cur = (
            round(sum(_shannon_entropy(r.probs) for r in c_ok) / len(c_ok), 6)
            if c_ok
            else 0.0
        )

        esc_ref = _escalation_rate(r_ok, pol_by_key, "reference")
        esc_cur = _escalation_rate(c_ok, pol_by_key, "current")

        alerts: list[str] = []
        status = "OK"

        if model_drift:
            alerts.append(
                f"model_served changed from {ref_models} to {cur_models}; "
                "existing fits.json and policy.json are invalidated"
            )
            status = "CRITICAL"
        if mismatch_cur > 0:
            alerts.append(
                f"{mismatch_cur}/{len(c_ok)} current records have model_requested != model_served"
            )
            if status == "OK":
                status = "WARN"
        if psi_val >= PSI_CRITICAL:
            alerts.append(f"severe probability distribution shift: PSI={psi_val:.3f} >= {PSI_CRITICAL}")
            status = "CRITICAL"
        elif psi_val >= PSI_WARN:
            alerts.append(f"moderate probability distribution shift: PSI={psi_val:.3f} >= {PSI_WARN}")
            if status == "OK":
                status = "WARN"
        if fail_cur - fail_ref >= 0.05:
            alerts.append(
                f"failure/abstain rate jumped from {fail_ref:.1%} to {fail_cur:.1%}"
            )
            status = "CRITICAL"

        reports.append(
            DriftReport(
                question_id=qid,
                primitive=prim,
                ref_models=ref_models,
                cur_models=cur_models,
                model_drift_detected=model_drift,
                model_mismatch_count_cur=mismatch_cur,
                n_ref=len(r_all),
                n_cur=len(c_all),
                failure_rate_ref=round(fail_ref, 6),
                failure_rate_cur=round(fail_cur, 6),
                psi=psi_val,
                js_divergence=jsd_val,
                quantiles_ref=_quantiles(r_scores),
                quantiles_cur=_quantiles(c_scores),
                mean_entropy_ref=ent_ref,
                mean_entropy_cur=ent_cur,
                entropy_delta=round(ent_cur - ent_ref, 6),
                escalation_rate_ref=esc_ref,
                escalation_rate_cur=esc_cur,
                latency_ref=_latency_summary(r_ok),
                latency_cur=_latency_summary(c_ok),
                status=status,
                alerts=alerts,
            )
        )

    counts = Counter(r.status for r in reports)
    return {
        "schema": "s1-drift-report/1",
        "distribution": "calibrated" if use_calib else "raw",
        "summary": {
            "n_questions": len(reports),
            "ok": counts["OK"],
            "warn": counts["WARN"],
            "critical": counts["CRITICAL"],
        },
        "questions": [r.as_dict() for r in reports],
    }
