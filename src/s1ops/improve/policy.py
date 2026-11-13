"""Threshold and uncertainty band compilers for System 1 decision models.

Supports two complementary policy modes:
1. Error-Budget Threshold (`select_threshold`, `clopper_pearson_upper`, `expected_cost`, `threshold_curve`):
   Sort by max probability descending (tie-safe: equal confidences accepted or rejected together).
   Pick the lowest threshold where empirical error rate (or Clopper-Pearson upper bound) <= `budget`.
2. Hybrid Escalation Band (`Band`, `Route`, `BandResult`, `sweep`, `pareto`, `choose`):
   For binary (`noul`) cascades: `p < lo` -> FALSE, `p >= hi` -> TRUE, `lo <= p < hi` -> ESCALATE to LLM.
3. False-Pass Cap (`select_false_pass_cutoff`, `FalsePassCut`):
   For a `noul` consumer that acts only on FALSE: the smallest cutoff at which at most a given share of
   label-false items reach `p >= cutoff`, written as a band with `lo == hi`.

Tuning on calibrated probabilities (`tuned_rows`, `tuned_fit_sha256`, `CalibInputError`): the checks
`s1 policy tune --use-calib`, `s1 policy curve --use-calib` and the web simulator run on a group.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
from scipy.stats import beta as sp_beta

from s1ops.contracts.records import Group, GroupKey, Record, has_calib, short_repr


def _check_cost(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")


def _check_rate(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")


def clopper_pearson_upper(errors: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) Clopper-Pearson exact upper confidence bound on binomial error rate."""
    if n <= 0 or not (0 <= errors <= n):
        raise ValueError(f"need 0 <= errors <= n and n > 0, got errors={errors}, n={n}")
    if errors == n:
        return 1.0
    return float(sp_beta.ppf(1.0 - alpha, errors + 1, n - errors))


def select_threshold(
    confidence,
    correct,
    budget: float,
    min_accepted: int = 1,
    *,
    use_cp_upper: bool = False,
    alpha: float = 0.05,
) -> float | None:
    """Tie-safe threshold selection at error budget `budget`.

    Sort by confidence descending (stable); accept equal confidences as one group; return the
    confidence of the last group at which `errors <= budget * accepted` (or Clopper-Pearson upper
    bound <= `budget` when `use_cp_upper=True`) and `accepted >= min_accepted`.
    `None` means no threshold meets the budget: accept nothing (delegate all).
    """
    if not math.isfinite(budget) or not 0 <= budget <= 1 or min_accepted < 1:
        raise ValueError("invalid error budget or minimum accepted count")
    if use_cp_upper and not (math.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct)
    if confidence.ndim != 1 or correct.shape != confidence.shape:
        raise ValueError("confidence and correctness must be equal-length 1-D vectors")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("confidence must be finite and in [0, 1]")
    if not np.isin(correct, [False, True]).all():
        raise ValueError("correctness must be boolean")
    correct = correct.astype(bool)
    if not len(confidence):
        return None
    order = np.argsort(-confidence, kind="stable")
    ordered, errors = confidence[order], np.cumsum(~correct[order])
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(order) - 1]
    accepted = ends + 1
    if use_cp_upper:
        cp_ok = np.array(
            [clopper_pearson_upper(int(errors[e]), int(accepted[i]), alpha) <= budget for i, e in enumerate(ends)],
            dtype=bool,
        )
        ok = np.flatnonzero(cp_ok & (accepted >= min_accepted))
    else:
        ok = np.flatnonzero((errors[ends] <= budget * accepted) & (accepted >= min_accepted))
    return float(ordered[ends[ok[-1]]]) if len(ok) else None


@dataclass(frozen=True)
class FalsePassCut:
    """A noul cutoff `c` where an item passes (is not blocked) iff `p >= c`, with its counts."""

    cutoff: float
    n_true: int
    n_false: int
    true_pass: int
    false_pass: int

    @property
    def true_pass_rate(self) -> float | None:
        return self.true_pass / self.n_true if self.n_true else None

    @property
    def false_pass_rate(self) -> float:
        return self.false_pass / self.n_false


def select_false_pass_cutoff(
    p,
    label,
    max_false_pass: float,
    *,
    use_cp_upper: bool = False,
    alpha: float = 0.05,
) -> FalsePassCut | None:
    """Smallest cutoff at which at most `max_false_pass` of the label-false items pass.

    For a consumer that acts only on BLOCK (`p < band_lo`) and treats ALLOW and ESCALATE alike.
    Candidates are the observed `p` values and `1.0`; an item at exactly the cutoff passes, as in
    routing. A candidate is feasible when `false_pass <= max_false_pass * n_false` (or its
    Clopper-Pearson upper bound is `<= max_false_pass` when `use_cp_upper=True`); the smallest
    feasible one passes the most label-true items. `None` means no candidate is feasible.
    """
    if not math.isfinite(max_false_pass) or not 0 <= max_false_pass < 1:
        raise ValueError(f"max_false_pass must be in [0, 1), got {max_false_pass!r}")
    if use_cp_upper and not (math.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    p, label = np.asarray(p, dtype=float), np.asarray(label)
    if p.ndim != 1 or label.shape != p.shape:
        raise ValueError("p and label must be equal-length 1-D vectors")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p must be finite and in [0, 1]")
    if label.dtype != bool:
        raise ValueError("label must be boolean")
    false_p, true_p = np.sort(p[~label]), np.sort(p[label])
    if not len(false_p):
        raise ValueError("no label-false items: nothing bounds the false-pass rate")
    candidates = np.unique(np.r_[p, 1.0])
    false_pass = len(false_p) - np.searchsorted(false_p, candidates, side="left")
    if use_cp_upper:
        ok = np.array([clopper_pearson_upper(int(k), len(false_p), alpha) <= max_false_pass for k in false_pass], dtype=bool)
    else:
        ok = false_pass <= max_false_pass * len(false_p)
    feasible = np.flatnonzero(ok)
    if not len(feasible):
        return None
    i = int(feasible[0])
    c = float(candidates[i])
    return FalsePassCut(
        cutoff=c,
        n_true=len(true_p),
        n_false=len(false_p),
        true_pass=int(len(true_p) - np.searchsorted(true_p, c, side="left")),
        false_pass=int(false_pass[i]),
    )


def expected_cost(
    tickets: Sequence[dict],
    threshold: float | None,
    c_s1: float,
    c_delegate: float,
    c_error: float,
) -> dict:
    """USD per 1,000 items for 'System 1 first, delegate when unsure'.

    `tickets`: dicts with `conf` (System 1 max probability, `None` if failed), `ok` (System 1 right)
    and `delegate_ok` (delegate right, default True).
    """
    n = len(tickets)
    if n == 0:
        raise ValueError("no tickets")
    auto = [t["conf"] is not None and threshold is not None and t["conf"] >= threshold for t in tickets]
    delegated = n - sum(auto)
    errors_auto = sum(a and not t["ok"] for a, t in zip(auto, tickets))
    errors_delegate = sum(not a and not t.get("delegate_ok", True) for a, t in zip(auto, tickets))
    parts = {
        "s1": n * c_s1,
        "delegate": delegated * c_delegate,
        "error": (errors_auto + errors_delegate) * c_error,
    }
    per_1k = {k: v / n * 1000 for k, v in parts.items()}
    return {
        "threshold": threshold,
        "total": sum(per_1k.values()),
        **per_1k,
        "n": n,
        "automated": n - delegated,
        "coverage": (n - delegated) / n,
        "delegated": delegated,
        "errors_auto": errors_auto,
        "error_rate_auto": errors_auto / (n - delegated) if (n - delegated) else 0.0,
        "errors_delegate": errors_delegate,
    }


def threshold_curve(
    confidence,
    correct,
    *,
    c_s1: float = 0.00004,
    c_delegate: float = 0.004,
    c_error: float = 0.10,
    delegate_accuracy: float = 0.98,
    always_escalated: int = 0,
) -> list[dict]:
    """Compute coverage, empirical error rate, Clopper-Pearson upper bound, and $/1k cost across tie groups.

    `always_escalated` counts items outside `confidence` that are delegated at every threshold (a
    choice or score tie at the top): they add to `n`, so to coverage's denominator and to the
    delegated items, never to `accepted` or `errors`."""
    for name, value in (("c_s1", c_s1), ("c_delegate", c_delegate), ("c_error", c_error)):
        _check_cost(name, value)
    _check_rate("delegate_accuracy", delegate_accuracy)
    if isinstance(always_escalated, bool) or not isinstance(always_escalated, int) or always_escalated < 0:
        raise ValueError(f"always_escalated must be a non-negative integer, got {always_escalated!r}")
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(confidence) == 0:
        return []
    n = len(confidence) + always_escalated
    order = np.argsort(-confidence, kind="stable")
    ordered, errors = confidence[order], np.cumsum(~correct[order])
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(order) - 1]
    points: list[dict] = []
    for e in ends:
        acc_n = int(e + 1)
        err_n = int(errors[e])
        tau = float(ordered[e])
        del_n = n - acc_n
        err_del = del_n * (1.0 - delegate_accuracy)
        s1_1k = c_s1 * 1000.0
        del_1k = (del_n / n) * c_delegate * 1000.0
        err_1k = ((err_n + err_del) / n) * c_error * 1000.0
        points.append(
            {
                "threshold": tau,
                "accepted": acc_n,
                "coverage": acc_n / n,
                "errors": err_n,
                "error_rate": err_n / acc_n,
                "cp_upper_95": clopper_pearson_upper(err_n, acc_n, 0.05),
                "cost_per_1k_usd": s1_1k + del_1k + err_1k,
                "s1_per_1k_usd": s1_1k,
                "delegate_per_1k_usd": del_1k,
                "error_per_1k_usd": err_1k,
            }
        )
    return points


class Route(str, Enum):
    TRUE = "true"
    FALSE = "false"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Band:
    """Escalate when `lo <= p < hi`; otherwise trust the System 1 Noul answer."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lo <= self.hi and self.lo <= 1.0 and (self.hi <= 1.0 or self.hi == float("inf"))):
            raise ValueError(f"band must satisfy 0 <= lo <= hi <= 1 (or hi = inf), got lo={self.lo}, hi={self.hi}")

    def route(self, p: float) -> Route:
        if p >= self.hi:
            return Route.TRUE
        if p < self.lo:
            return Route.FALSE
        return Route.ESCALATE

    @classmethod
    def parse(cls, text: str) -> "Band":
        lo, hi = (float(part) for part in text.split(","))
        return cls(lo, hi)

    def __str__(self) -> str:
        return f"[{self.lo:g}, {self.hi:g})"


@dataclass(frozen=True)
class DevRecord:
    id: str
    p_jev: float
    label: bool
    llm: bool | None = None
    llm_unresolved: bool = False
    jev_cost_usd: float | None = None
    llm_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.p_jev) and 0.0 <= self.p_jev <= 1.0):
            raise ValueError(f"{self.id}: p_jev must be a finite probability, got {self.p_jev!r}")
        for name in ("jev_cost_usd", "llm_cost_usd"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{self.id}: {name} must be a finite non-negative number, got {value!r}")


@dataclass(frozen=True)
class Prices:
    jev_usd: float = 0.00004
    llm_usd: float = 0.004

    def __post_init__(self) -> None:
        _check_cost("jev_usd", self.jev_usd)
        _check_cost("llm_usd", self.llm_usd)


@dataclass(frozen=True)
class BandResult:
    band: Band
    n: int
    escalated: int
    tp: int
    fp: int
    fn: int
    tn: int
    unresolved: int
    cost_usd: float
    cost_estimated: int

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    def metric(self, name: str) -> float:
        return {"f1": self.f1, "accuracy": self.accuracy}[name]

    def as_row(self) -> dict:
        row = asdict(self)
        row["band"] = str(self.band)
        row["lo"] = self.band.lo
        row["hi"] = self.band.hi
        row.update(
            escalation_rate=self.escalation_rate,
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            accuracy=self.accuracy,
        )
        return row


def evaluate(records: Sequence[DevRecord], band: Band, prices: Prices = Prices()) -> BandResult | None:
    tp = fp = fn = tn = escalated = unresolved = estimated = 0
    cost = 0.0
    for r in records:
        cost += prices.jev_usd if r.jev_cost_usd is None else r.jev_cost_usd
        estimated += r.jev_cost_usd is None
        route = band.route(r.p_jev)
        if route is Route.ESCALATE:
            if r.llm is None and not r.llm_unresolved:
                return None
            escalated += 1
            cost += prices.llm_usd if r.llm_cost_usd is None else r.llm_cost_usd
            estimated += r.llm_cost_usd is None
            if r.llm_unresolved:
                unresolved += 1
                predicted = not r.label
            else:
                predicted = r.llm
        else:
            predicted = route is Route.TRUE
        if predicted and r.label:
            tp += 1
        elif predicted:
            fp += 1
        elif r.label:
            fn += 1
        else:
            tn += 1
    return BandResult(band, len(records), escalated, tp, fp, fn, tn, unresolved, cost, estimated)


def grid(step: float = 0.05) -> list[Band]:
    k = round(1 / step)
    if abs(k * step - 1) > 1e-9:
        raise ValueError(f"step must divide 1 evenly, got {step}")
    points = [round(i * step, 6) for i in range(k + 1)]
    return [Band(lo, hi) for i, lo in enumerate(points) for hi in points[i:]]


def sweep(records: Sequence[DevRecord], bands: Iterable[Band], prices: Prices = Prices()) -> list[BandResult]:
    results = (evaluate(records, band, prices) for band in bands)
    return [r for r in results if r is not None]


def pareto(results: Iterable[BandResult], metric: str = "f1") -> list[BandResult]:
    ordered = sorted(results, key=lambda r: (r.cost_usd, -r.metric(metric)))
    frontier: list[BandResult] = []
    best = float("-inf")
    for r in ordered:
        if r.metric(metric) > best:
            frontier.append(r)
            best = r.metric(metric)
    return frontier


def choose(
    results: Sequence[BandResult],
    metric: str = "f1",
    *,
    max_cost_usd: float | None = None,
    max_escalation_rate: float | None = None,
    min_metric: float | None = None,
) -> BandResult | None:
    if max_escalation_rate is not None:
        _check_rate("max_escalation_rate", max_escalation_rate)
    feasible = [
        r
        for r in results
        if (max_cost_usd is None or r.cost_usd <= max_cost_usd)
        and (max_escalation_rate is None or r.escalation_rate <= max_escalation_rate)
    ]
    if min_metric is not None:
        feasible = [r for r in feasible if r.metric(metric) >= min_metric]
        return min(feasible, key=lambda r: (r.cost_usd, -r.metric(metric)), default=None)
    return max(feasible, key=lambda r: (r.metric(metric), -r.escalated, -r.cost_usd), default=None)


class CalibInputError(ValueError):
    """Records a calibrated threshold cannot be tuned or simulated on: a tuned record without the
    calibrated distribution, or a group calibrated through more than one fit."""


def tuned_rows(records: Iterable[Record]) -> dict[GroupKey, list[Record]]:
    """The records `select(records, allow_model_mismatch=True)` tunes on, by group: answered and
    labeled (`select` refuses a repeated item within a group)."""
    rows: dict[GroupKey, list[Record]] = defaultdict(list)
    for r in records:
        if r.model_served is not None and r.label is not None:
            rows[r.key].append(r)
    return rows


def tuned_fit_sha256(grp: Group, rows: Sequence[Record]) -> str | None:
    """The `fit_sha256` that every record tuned into `grp` was calibrated through, or `None` for
    records from before `s1 apply` wrote it. A record without the calibrated distribution would be
    tuned on its raw one (`parse_record` falls back), so it is refused rather than counted."""
    name = f"{grp.question_id} ({grp.model_served})"
    for r in rows:
        if not has_calib(r.raw, grp.primitive):
            raise CalibInputError(
                f"{name}: line {r.line} has no calibrated probabilities; a calibrated policy is tuned on "
                "the `calib` that `s1 apply` writes, so apply the fit to every record first"
            )
    for r in rows:
        h = r.raw["calib"].get("fit_sha256")
        if h is not None and not isinstance(h, str):
            raise CalibInputError(f"{name}: line {r.line}: calib.fit_sha256 must be a string, got {short_repr(h)}")
    hashes = {r.raw["calib"].get("fit_sha256") for r in rows}
    if len(hashes) > 1:
        shown = ", ".join(sorted(repr(h) for h in hashes))
        raise CalibInputError(f"{name}: records were calibrated through different fits ({shown}); apply one fit file")
    return hashes.pop() if hashes else None
