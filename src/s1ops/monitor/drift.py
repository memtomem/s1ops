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
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from s1ops.contracts.records import GroupKey, Record, has_calib, read_records, reparse, short_repr
from s1ops.contracts.schemas import PolicyEntry
from s1ops.runtime.routing import RoutingInputError, route_record

PSI_EPS = 1e-4
PSI_WARN = 0.10
PSI_CRITICAL = 0.25
# A PSI over a threshold raises the status only when a permutation test of PSI puts it outside
# sampling noise: PSI of two samples from one distribution grows like (bins - 1)(1/n_ref + 1/n_cur),
# about 0.33 at 54 records per side.
PSI_ALPHA_WARN = 0.05
PSI_ALPHA_CRITICAL = 0.01
PSI_RESAMPLES = 1000
PSI_MIN_RESAMPLES = 1000
PSI_SEED = 0


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
        checked = reparse(r, use_calib=True)
        if not use_calib:
            out.append(reparse(r))
        elif has_calib(r.raw, r.primitive):
            out.append(checked)
        else:
            raise DriftInputError(
                r.line,
                "a calibrated comparison needs calibrated probabilities and this record has none; "
                "run `s1 apply` on both files, pass a `--fit` with a fit for this record's group, "
                "or compare raw probabilities",
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


def _bin_counts(scores: Sequence[float], bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(np.asarray(scores, dtype=float), bins=edges)
    return counts


def _psi_terms(r_counts: np.ndarray, n_ref: int, c_counts: np.ndarray, n_cur: int) -> np.ndarray:
    """Each bin's term of PSI (`score_histograms` serves them)."""
    r_prop = np.clip(r_counts / n_ref, PSI_EPS, 1.0)
    c_prop = np.clip(c_counts / n_cur, PSI_EPS, 1.0)
    return (c_prop - r_prop) * np.log(c_prop / r_prop)


def _psi_from_counts(r_counts: np.ndarray, n_ref: int, c_counts: np.ndarray, n_cur: int) -> np.ndarray:
    """PSI of bin counts; rows of a 2-D array are separate samples. The one formula `psi` and the
    permutation test share, so the observed value and every resample are computed alike."""
    return np.sum(_psi_terms(r_counts, n_ref, c_counts, n_cur), axis=-1)


def psi(
    ref_scores: Sequence[float],
    cur_scores: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between reference and current score distributions."""
    if len(ref_scores) == 0 or len(cur_scores) == 0:
        return 0.0
    return float(
        _psi_from_counts(_bin_counts(ref_scores, bins), len(ref_scores), _bin_counts(cur_scores, bins), len(cur_scores))
    )


def _psi_seed(question_id: str) -> list[int]:
    """One generator per question, so adding or removing a question never changes another's p."""
    return [PSI_SEED, int(hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16], 16)]


def psi_test(
    ref_scores: Sequence[float],
    cur_scores: Sequence[float],
    *,
    bins: int = 10,
    resamples: int = PSI_RESAMPLES,
    seed: Sequence[int] | int = PSI_SEED,
) -> tuple[float, float | None]:
    """`(psi, p)`: PSI at full precision, and the permutation p-value of a PSI this large or larger
    when both samples come from one distribution (`None` when a side is empty or `resamples` is 0).

    A uniform random split of the pooled scores into the two sample sizes depends on the scores only
    through their bins, so the reference side's bin counts are multivariate hypergeometric over the
    pooled counts: each resample draws those counts directly, in O(bins). The p-value is a Monte
    Carlo estimate, `(1 + exceedances) / (resamples + 1)`, a resample counting as reaching the
    observed PSI within a relative tolerance of 1e-12."""
    n_ref, n_cur = len(ref_scores), len(cur_scores)
    if n_ref == 0 or n_cur == 0:
        return 0.0, None
    r_counts, c_counts = _bin_counts(ref_scores, bins), _bin_counts(cur_scores, bins)
    observed = float(_psi_from_counts(r_counts, n_ref, c_counts, n_cur))
    if resamples == 0:
        return observed, None
    pooled = r_counts + c_counts
    draws = np.random.default_rng(seed).multivariate_hypergeometric(pooled, n_ref, size=resamples)
    resampled = _psi_from_counts(draws, n_ref, pooled - draws, n_cur)
    exceed = int(np.count_nonzero(resampled >= observed - 1e-12 * max(1.0, observed)))
    return observed, (1 + exceed) / (resamples + 1)


def _check_resamples(resamples: object) -> int:
    if isinstance(resamples, bool) or not isinstance(resamples, int):
        raise ValueError(f"psi_resamples must be an integer, got {short_repr(resamples)}")
    if resamples != 0 and resamples < PSI_MIN_RESAMPLES:
        raise ValueError(
            f"psi_resamples must be 0 (no test) or at least {PSI_MIN_RESAMPLES}, got {resamples}: "
            "fewer resamples make the p-value too coarse for its alpha"
        )
    return resamples


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


def _latency_ms(value: int | float) -> float:
    try:
        return float(value)
    except OverflowError:  # a JSON integer past the float range: counted, not dropped, so its statistics go null
        return math.inf if value > 0 else -math.inf


def _rounded_ms(value: float) -> float | None:
    value = float(value)
    return round(value, 2) if math.isfinite(value) else None


def _latency_summary(records: Sequence[Record]) -> dict[str, float | None]:
    """p50, p95 and mean of the records' numeric `latency_ms`. Each is null when there is no number to
    summarize or the statistic is not finite, which finite values can reach: `latency_ms` is not validated,
    so a sum or an interpolation can overflow."""
    vals = [
        _latency_ms(r.raw["latency_ms"])
        for r in records
        if isinstance(r.raw.get("latency_ms"), (int, float))
        and not isinstance(r.raw.get("latency_ms"), bool)
    ]
    if not vals:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
    arr = np.asarray(vals, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        p50, p95 = np.quantile(arr, [0.50, 0.95])
        mean = arr.mean()
    return {"p50_ms": _rounded_ms(p50), "p95_ms": _rounded_ms(p95), "mean_ms": _rounded_ms(mean)}


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
    n_ref_answered: int
    n_cur_answered: int
    failure_rate_ref: float
    failure_rate_cur: float
    psi: float
    psi_p_value: float | None
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
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def psi_test_info(resamples: int = PSI_RESAMPLES) -> dict[str, Any]:
    """The report's `psi_test` block: how a PSI over a threshold is tested before it alerts."""
    if resamples == 0:
        return {"method": "none", "resamples": 0}
    return {
        "method": "permutation-by-bin-counts",
        "resamples": resamples,
        "seed": PSI_SEED,
        "alpha_warn": PSI_ALPHA_WARN,
        "alpha_critical": PSI_ALPHA_CRITICAL,
    }


def _prepared(
    ref_records: str | Path | Iterable[Record],
    cur_records: str | Path | Iterable[Record],
    use_calib: bool,
) -> tuple[dict[str, list[Record]], dict[str, list[Record]], list[str]]:
    """Both sides read, re-parsed and checked as every drift statistic needs them, by question,
    and the questions both sides answer, sorted: the one preparation `audit_drift` and
    `score_histograms` share, so they refuse the same inputs and see the same records."""
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

    ref_by_q: dict[str, list[Record]] = defaultdict(list)
    cur_by_q: dict[str, list[Record]] = defaultdict(list)
    for r in refs:
        ref_by_q[r.question_id].append(r)
    for r in curs:
        cur_by_q[r.question_id].append(r)
    return ref_by_q, cur_by_q, sorted(ref_by_q.keys() & cur_by_q.keys())


def audit_drift(
    ref_records: str | Path | Iterable[Record],
    cur_records: str | Path | Iterable[Record],
    *,
    policies: Sequence[PolicyEntry] | None = None,
    bins: int = 10,
    use_calib: bool = False,
    psi_resamples: int = PSI_RESAMPLES,
) -> dict[str, Any]:
    """Compare reference records against current records per `question_id`.

    The distributions compared (PSI, JS divergence, quantiles, entropy) are the raw ones, or with
    `use_calib` the calibrated ones, which every answered record on both sides must then carry,
    made by one fit per `(question_id, model_served, primitive)` group (`DriftInputError`
    otherwise). The escalation rate routes what each policy was tuned on either
    way.

    PSI over `PSI_WARN` / `PSI_CRITICAL` raises the status only when its permutation p-value
    (`psi_test`, `psi_resamples` draws) is under `PSI_ALPHA_WARN` / `PSI_ALPHA_CRITICAL`; a PSI over
    a threshold that reaches neither is reported in `notes`. `psi_resamples=0` turns the test off
    (PSI thresholds alone); otherwise it must be at least `PSI_MIN_RESAMPLES`.

    Records passed in are re-parsed from their `raw` with the default `sum_tol`, validating any
    stored `calib` in both modes, so one the caller parsed without `use_calib` or with a looser
    tolerance can raise `RecordError` here.
    """
    psi_resamples = _check_resamples(psi_resamples)
    ref_by_q, cur_by_q, shared_qids = _prepared(ref_records, cur_records, use_calib)
    pol_by_key: dict[GroupKey, PolicyEntry] = {(p.question_id, p.model_served, p.primitive): p for p in policies or ()}
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

        psi_full, psi_p = psi_test(r_scores, c_scores, bins=bins, resamples=psi_resamples, seed=_psi_seed(qid))
        psi_val = round(psi_full, 6)
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
        notes: list[str] = []
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
        tested = "" if psi_p is None else f" (p={psi_p:.3f}, n_ref={len(r_ok)}, n_cur={len(c_ok)})"
        critical_ok = psi_p is None or psi_p < PSI_ALPHA_CRITICAL
        warn_ok = psi_p is None or psi_p < PSI_ALPHA_WARN
        if psi_val >= PSI_CRITICAL and critical_ok:
            alerts.append(f"severe probability distribution shift: PSI={psi_val:.3f} >= {PSI_CRITICAL}{tested}")
            status = "CRITICAL"
        elif psi_val >= PSI_WARN and warn_ok:
            if psi_val >= PSI_CRITICAL:
                alerts.append(
                    f"moderate probability distribution shift: PSI={psi_val:.3f} >= {PSI_CRITICAL}, "
                    f"significant at {PSI_ALPHA_WARN} but not at {PSI_ALPHA_CRITICAL}{tested}"
                )
            else:
                alerts.append(f"moderate probability distribution shift: PSI={psi_val:.3f} >= {PSI_WARN}{tested}")
            if status == "OK":
                status = "WARN"
        elif psi_val >= PSI_WARN:
            reached = PSI_CRITICAL if psi_val >= PSI_CRITICAL else PSI_WARN
            notes.append(
                f"PSI={psi_val:.3f} >= {reached}, but the PSI test did not reach significance at "
                f"n_ref={len(r_ok)}, n_cur={len(c_ok)} (p={psi_p:.3f}); not alerted"
            )
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
                n_ref_answered=len(r_ok),
                n_cur_answered=len(c_ok),
                failure_rate_ref=round(fail_ref, 6),
                failure_rate_cur=round(fail_cur, 6),
                psi=psi_val,
                psi_p_value=None if psi_p is None else round(psi_p, 6),
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
                notes=notes,
            )
        )

    counts = Counter(r.status for r in reports)
    return {
        "schema": "s1-drift-report/1",
        "distribution": "calibrated" if use_calib else "raw",
        "psi_test": psi_test_info(psi_resamples),
        "summary": {
            "n_questions": len(reports),
            "ok": counts["OK"],
            "warn": counts["WARN"],
            "critical": counts["CRITICAL"],
        },
        "questions": [r.as_dict() for r in reports],
    }


def score_histograms(
    ref_records: str | Path | Iterable[Record],
    cur_records: str | Path | Iterable[Record],
    *,
    bins: int = 10,
    use_calib: bool = False,
) -> dict[str, Any]:
    """The binned scores `audit_drift`'s PSI is computed from, per question: the same inputs,
    refused alike, the same answered records with their models pooled, the same score (`p` for
    noul, the top-class probability otherwise) and the same `bins` equal-width bins over [0, 1]
    (half-open, the last one closed, as `np.histogram` counts).

    Each question's `psi` is the report's `psi` for the same arguments, and `psi_terms` are the
    per-bin terms of that sum, rounded one by one (so they need not add up to the rounded `psi`);
    both are 0.0 / null when a side has no answered record.
    """
    ref_by_q, cur_by_q, shared_qids = _prepared(ref_records, cur_records, use_calib)
    edges = [float(e) for e in np.linspace(0.0, 1.0, bins + 1)]
    questions: list[dict[str, Any]] = []
    for qid in shared_qids:
        prim = ref_by_q[qid][0].primitive
        r_scores = [_primary_score(r) for r in ref_by_q[qid] if _answered(r)]
        c_scores = [_primary_score(r) for r in cur_by_q[qid] if _answered(r)]
        r_counts, c_counts = _bin_counts(r_scores, bins), _bin_counts(c_scores, bins)
        if r_scores and c_scores:
            full = _psi_terms(r_counts, len(r_scores), c_counts, len(c_scores))
            terms: list[float] | None = [round(float(t), 6) for t in full]
            value = round(float(np.sum(full, axis=-1)), 6)
        else:
            terms, value = None, 0.0
        questions.append({
            "question_id": qid,
            "primitive": prim,
            "score": "p" if prim == "noul" else "top_probability",
            "edges": edges,
            "ref": {"n_answered": len(r_scores), "counts": [int(n) for n in r_counts]},
            "cur": {"n_answered": len(c_scores), "counts": [int(n) for n in c_counts]},
            "psi": value,
            "psi_terms": terms,
        })
    return {
        "schema": "s1-drift-histograms/1",
        "distribution": "calibrated" if use_calib else "raw",
        "bins": bins,
        "questions": questions,
    }
