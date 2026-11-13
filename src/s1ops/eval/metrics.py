"""Calibration and coverage metrics for one group of System 1 answers.

Includes:
- Log loss and Brier score (proper scoring rules)
- Equal-width and Equal-mass ECE
- Monte Carlo calibrated noise floor (`ece_floor`: expected ECE under true calibration at the same `n`)
- Tie-mass diagnostics (`tie_mass`: detects probability rounding ties such as 2-decimal Jev clusters at 1.0)
- Seeded percentile bootstrap intervals
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

EPS = 1e-6
MIN_PER_BIN = 10


def top_label(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probability of the argmax class and whether it equals the label (ties go to the first class)."""
    pred = P.argmax(axis=1)
    return P[np.arange(len(y)), pred], (pred == y).astype(float)


def noul_routed_label(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a noul group (`P` rows `(1 - p, p)`): `max(1 - p, p)`, the confidence an `error_budget`
    policy compares with its threshold, and whether the routing rule's answer equals the label.

    The rule answers `p >= 0.5`, so at `p == 0.5` it answers true where `top_label` breaks the tie
    to the first class, false; a threshold tuned on `top_label` can then exceed its budget."""
    return P.max(axis=1), (P[:, 1] >= 0.5) == (y == 1)


def top_tied(P: np.ndarray) -> np.ndarray:
    """Rows whose largest probability occurs more than once, compared exactly, as the routing rule
    compares a choice or score answer (`probs.count(top) > 1`): it escalates such a row at every
    threshold. Not for noul, whose rule answers true at `p == 0.5` rather than escalating."""
    return (P == P.max(axis=1, keepdims=True)).sum(axis=1) > 1


def routed_confidence(primitive: str, P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """What an `error_budget` policy routes, as `select_threshold` and `threshold_curve` take it:
    the confidence and correctness of every row the policy can accept, and how many rows it
    escalates at every threshold.

    Noul: every row, counted as the rule answers (`noul_routed_label`). Choice and score: the rows
    with a unique top probability (`top_label`); a tie at the top escalates, so it counts in
    neither an accepted error's numerator nor its denominator, only in `n`."""
    if primitive == "noul":
        conf, correct = noul_routed_label(P, y)
        return conf, correct, 0
    tied = top_tied(P)
    conf, correct = top_label(P[~tied], y[~tied])
    return conf, correct, int(tied.sum())


def reliability_points(primitive: str, P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(score, target)` pairs that ECE bins: `(p, label)` for Noul, top label otherwise."""
    if primitive == "noul":
        return P[:, 1], y.astype(float)
    return top_label(P, y)


def _bins(score: np.ndarray, bins: int, strategy: str) -> list[np.ndarray]:
    if strategy == "width":
        idx = np.minimum(np.floor(score * bins).astype(int), bins - 1)
        return [np.flatnonzero(idx == b) for b in range(bins)]
    if strategy == "mass":
        return list(np.array_split(np.argsort(score, kind="stable"), bins))
    raise ValueError(f"strategy must be 'width' or 'mass', got {strategy!r}")


def ece(score: np.ndarray, target: np.ndarray, bins: int = 10, strategy: str = "width") -> float:
    """Expected calibration error: `sum_b (n_b / n) * |mean(score_b) - mean(target_b)|`."""
    n = len(score)
    if n == 0:
        return 0.0
    return float(
        sum(len(b) / n * abs(score[b].mean() - target[b].mean()) for b in _bins(score, bins, strategy) if len(b))
    )


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_score: float
    observed: float

    @property
    def gap(self) -> float:
        return self.mean_score - self.observed


def reliability(score: np.ndarray, target: np.ndarray, bins: int = 10, strategy: str = "width") -> list[Bin]:
    rows = []
    for b, members in enumerate(_bins(score, bins, strategy)):
        if not len(members):
            continue
        if strategy == "width":
            lo, hi = b / bins, (b + 1) / bins
        else:
            lo, hi = float(score[members].min()), float(score[members].max())
        rows.append(Bin(lo, hi, len(members), float(score[members].mean()), float(target[members].mean())))
    return rows


def brier(primitive: str, P: np.ndarray, y: np.ndarray) -> float:
    if primitive == "noul":
        return float(np.mean((P[:, 1] - y) ** 2))
    onehot = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def log_loss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))))


def tie_mass(score: np.ndarray) -> dict[str, float | int]:
    """Detect tied probability masses (e.g. 2-decimal rounding ties at 1.00 that break fine quantile splits)."""
    n = len(score)
    if n == 0:
        return {"max_tie_value": 0.0, "max_tie_count": 0, "max_tie_share": 0.0, "unique_values": 0}
    vals, counts = np.unique(np.round(score, 8), return_counts=True)
    top_idx = int(np.argmax(counts))
    return {
        "max_tie_value": float(vals[top_idx]),
        "max_tie_count": int(counts[top_idx]),
        "max_tie_share": float(counts[top_idx] / n),
        "unique_values": int(len(vals)),
    }


def point_metrics(
    primitive: str,
    P: np.ndarray,
    y: np.ndarray,
    bins: int,
    levels: Sequence[int] | None = None,
) -> dict[str, float]:
    conf, correct = top_label(P, y)
    score, target = reliability_points(primitive, P, y)
    out = {
        "accuracy": float(correct.mean()),
        "brier": brier(primitive, P, y),
        "log_loss": log_loss(P, y),
        "ece_width": ece(score, target, bins, "width"),
        "ece_mass": ece(score, target, bins, "mass"),
        "confidence_gap": float(conf.mean() - correct.mean()),
    }
    if levels is not None:
        lv = np.asarray(levels, dtype=float)
        out["mae"] = float(np.mean(np.abs((P * lv).sum(axis=1) - lv[y])))
    return out


def bootstrap(
    stat: Callable[[np.ndarray], dict[str, float]],
    n: int,
    reps: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(reps):
        for name, value in stat(rng.integers(0, n, n)).items():
            draws.setdefault(name, []).append(value)
    alpha = (1 - level) / 2
    return {
        name: (float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha)))
        for name, v in draws.items()
    }


def ece_floor(score: np.ndarray, bins: int, reps: int, seed: int) -> dict[str, dict[str, float]]:
    """ECE of the same scores with labels drawn *from* them (perfectly calibrated noise floor)."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"width": [], "mass": []}
    for _ in range(reps):
        target = (rng.random(len(score)) < score).astype(float)
        for strategy, values in draws.items():
            values.append(ece(score, target, bins, strategy))
    return {k: {"mean": float(np.mean(v)), "p95": float(np.quantile(v, 0.95))} for k, v in draws.items()}


def direction(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if lo > 0:
        return "over-confident"
    if hi < 0:
        return "under-confident"
    return "no clear direction"
