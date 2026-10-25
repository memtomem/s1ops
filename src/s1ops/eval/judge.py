"""Rubric grader agreement audit against a reference judge (exact, within-1, quadratic κ, bias)."""

import math
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class Agreement:
    n: int
    exact: float
    within_one: float
    weighted_kappa: float
    """NaN when kappa is undefined (e.g. both graders give every item the same grade)."""
    bias: float
    pass_agreement: float | None = None
    passes_bar: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        if math.isnan(self.weighted_kappa):
            d["weighted_kappa"] = None  # NaN is not valid JSON
        return d


def agreement(
    jev: Sequence[int],
    reference: Sequence[int],
    levels: int,
    *,
    pass_level: int | None = None,
    min_kappa: float = 0.70,
    min_pass_agreement: float = 0.95,
) -> Agreement:
    if len(jev) != len(reference) or not jev:
        raise ValueError("need two non-empty sequences of the same length")
    if levels < 2 or any(not 0 <= x < levels for x in (*jev, *reference)):
        raise ValueError(f"levels must be >= 2 and every grade in [0, {levels})")
    n = len(jev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        kappa = float(
            cohen_kappa_score(
                jev,
                reference,
                labels=list(range(levels)),
                weights="quadratic",
                replace_undefined_by=float("nan"),
            )
        )
    exact = sum(a == b for a, b in zip(jev, reference)) / n
    within_one = sum(abs(a - b) <= 1 for a, b in zip(jev, reference)) / n
    bias = sum(a - b for a, b in zip(jev, reference)) / n
    pass_agree = None
    if pass_level is not None:
        pass_agree = sum((a >= pass_level) == (b >= pass_level) for a, b in zip(jev, reference)) / n
    passes = (kappa >= min_kappa) and (pass_agree is None or pass_agree >= min_pass_agreement)
    return Agreement(
        n=n,
        exact=exact,
        within_one=within_one,
        weighted_kappa=kappa,
        bias=bias,
        pass_agreement=pass_agree,
        passes_bar=passes,
    )


def by_criterion(
    rows: Iterable[tuple[str, int, int]],
    levels: int,
    *,
    pass_level: int | None = None,
    min_kappa: float = 0.70,
) -> dict[str, Agreement]:
    groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for criterion, j, r in rows:
        groups[criterion][0].append(j)
        groups[criterion][1].append(r)
    return {
        c: agreement(j, r, levels, pass_level=pass_level, min_kappa=min_kappa)
        for c, (j, r) in sorted(groups.items())
    }
