"""Audit one group: point metrics, bootstrap intervals, tie-mass, reliability tables, and before/after a fit."""

import numpy as np

from s1ops.contracts.records import Group
from s1ops.eval.metrics import (
    MIN_PER_BIN,
    bootstrap,
    direction,
    ece_floor,
    point_metrics,
    reliability,
    reliability_points,
    tie_mass,
)


def _tables(primitive: str, P: np.ndarray, y: np.ndarray, bins: int) -> dict[str, list[dict]]:
    score, target = reliability_points(primitive, P, y)
    return {
        strategy: [{**b.__dict__, "gap": b.gap} for b in reliability(score, target, bins, strategy)]
        for strategy in ("width", "mass")
    }


def _floor(primitive: str, P: np.ndarray, y: np.ndarray, bins: int, reps: int, seed: int) -> dict:
    score, _ = reliability_points(primitive, P, y)
    return ece_floor(score, bins, reps, seed) if reps else {}


def audit_group(
    group: Group,
    *,
    bins: int = 10,
    reps: int = 1000,
    seed: int = 0,
    after: np.ndarray | None = None,
) -> dict:
    levels = list(group.classes) if group.primitive == "score" else None
    P, y, prim = group.P, group.y, group.primitive
    score_before, _ = reliability_points(prim, P, y)

    def stat(idx: np.ndarray) -> dict[str, float]:
        before = point_metrics(prim, P[idx], y[idx], bins, levels)
        out = {f"before.{k}": v for k, v in before.items()}
        if after is not None:
            post = point_metrics(prim, after[idx], y[idx], bins, levels)
            out.update({f"after.{k}": v for k, v in post.items()})
            out.update({f"delta.{k}": post[k] - before[k] for k in post})
        return out

    ci = bootstrap(stat, group.n, reps, seed) if reps else {}

    def section(prefix: str) -> dict[str, list[float]]:
        return {k.split(".", 1)[1]: list(v) for k, v in ci.items() if k.startswith(prefix + ".")}

    warnings = []
    if group.n < MIN_PER_BIN * bins:
        warnings.append(
            f"n={group.n} is below {MIN_PER_BIN}x{bins} bins: ECE is biased upward and the reliability table is noisy; "
            f"compare against ece_floor"
        )
    ties = tie_mass(score_before)
    if ties["max_tie_count"] >= max(10, int(0.15 * group.n)):
        warnings.append(
            f"tie cluster: {ties['max_tie_count']}/{group.n} ({ties['max_tie_share']:.1%}) answers tied at {ties['max_tie_value']:.4g}"
        )

    before = point_metrics(prim, P, y, bins, levels)
    out = {
        "question_id": group.question_id,
        "model_served": group.model_served,
        "primitive": prim,
        "n": group.n,
        "splits": sorted({s for s in group.splits if s is not None}),
        "bins": bins,
        "bootstrap": {"reps": reps, "seed": seed, "level": 0.95},
        "before": before,
        "before_ci": section("before"),
        "direction": direction(tuple(ci["before.confidence_gap"])) if ci else None,
        "ece_floor": _floor(prim, P, y, bins, reps, seed),
        "tie_mass": ties,
        "reliability": _tables(prim, P, y, bins),
        "warnings": warnings,
    }
    if after is not None:
        score_after, _ = reliability_points(prim, after, y)
        out.update(
            after=point_metrics(prim, after, y, bins, levels),
            after_ci=section("after"),
            delta_ci=section("delta"),
            direction_after=direction(tuple(ci["after.confidence_gap"])) if ci else None,
            ece_floor_after=_floor(prim, after, y, bins, reps, seed),
            tie_mass_after=tie_mass(score_after),
            reliability_after=_tables(prim, after, y, bins),
        )
    return out
