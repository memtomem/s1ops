"""Both budget selectors compare the rate they report: `errors / accepted <= budget` (`select_threshold`)
and `false_pass / n_false <= cap` (`select_false_pass_cutoff`), never a count against a rounded product.

A correctly rounded quotient is the double nearest the exact rate, so 29 errors of 100 is exactly the
0.29 budget, as `threshold_curve` reports it; `29 <= 0.29 * 100` is not (`28.999999999999996`). The two
rules part only at such float boundaries, and in both directions: `5/6` rounds above the double
`0.8333333333333333`, whose product with 6 is `5.0`; `1/49` rounds to `0.02040816326530612`, although
the exact `1/49` lies above that decimal. Every expected value below is exact."""

from __future__ import annotations

import numpy as np
import pytest

from s1ops.improve.policy import FalsePassCut, select_false_pass_cutoff, select_threshold, threshold_curve
from tests.test_max_false_pass import oracle as false_pass_oracle


def threshold_oracle(conf, correct, budget, min_accepted=1):
    """The tuned threshold by a plain loop over tie groups, highest first: the last group whose
    cumulative error rate is within the budget with at least `min_accepted` accepted, else None."""
    pairs = sorted(zip((float(c) for c in conf), (bool(k) for k in correct)), key=lambda t: -t[0])
    accepted = errors = 0
    tuned = None
    for i, (c, ok) in enumerate(pairs):
        accepted += 1
        errors += not ok
        if i + 1 == len(pairs) or pairs[i + 1][0] != c:
            if accepted >= min_accepted and errors / accepted <= budget:
                tuned = c
    return tuned


def _groups(*groups: tuple[float, int, int]) -> tuple[list[float], list[bool]]:
    """`(confidence, right, wrong)` per tie group."""
    conf: list[float] = []
    correct: list[bool] = []
    for c, right, wrong in groups:
        conf += [c] * (right + wrong)
        correct += [True] * right + [False] * wrong
    return conf, correct


def _false_pass(*groups: tuple[float, int, int]) -> tuple[list[float], np.ndarray]:
    """`(p, label-false items, label-true items)` per value of p."""
    p: list[float] = []
    label: list[bool] = []
    for value, n_false, n_true in groups:
        p += [value] * (n_false + n_true)
        label += [False] * n_false + [True] * n_true
    return p, np.array(label)


# --- the issue's case and the two directions -------------------------------------------------------

# 50 right at 0.9; 29 wrong and 21 right at 0.8: the 0.8 point is 29 errors of 100.
ISSUE = _groups((0.9, 50, 0), (0.8, 21, 29))


def test_a_group_exactly_on_the_budget_is_accepted() -> None:
    conf, correct = ISSUE
    assert 0.29 * 100 == 28.999999999999996
    assert [(pt["threshold"], pt["error_rate"]) for pt in threshold_curve(conf, correct)] == [(0.9, 0.0), (0.8, 0.29)]
    assert select_threshold(conf, correct, budget=0.29) == 0.8


def test_a_rate_that_rounds_above_the_budget_is_refused() -> None:
    budget = 0.8333333333333333
    conf, correct = _groups((0.9, 1, 0), (0.8, 0, 5))  # the 0.8 point: 5 errors of 6
    assert budget * 6 == 5.0 and 5 / 6 == 0.8333333333333334
    assert select_threshold(conf, correct, budget=budget) == 0.9


def test_a_rate_that_rounds_onto_the_budget_is_accepted() -> None:
    budget = 0.02040816326530612
    conf, correct = _groups((0.9, 48, 0), (0.8, 0, 1))  # the 0.8 point: 1 error of 49
    assert 1 / 49 == budget and budget * 49 < 1
    assert select_threshold(conf, correct, budget=budget) == 0.8


def test_false_pass_exactly_on_the_cap_is_accepted() -> None:
    # 29 of 100 label-false items at 0.9, the rest at 0.1; ten label-true items at 0.95.
    p, label = _false_pass((0.9, 29, 0), (0.1, 71, 0), (0.95, 0, 10))
    assert select_false_pass_cutoff(p, label, 0.29) == FalsePassCut(cutoff=0.9, n_true=10, n_false=100, true_pass=10, false_pass=29)


def test_false_pass_that_rounds_above_the_cap_is_refused() -> None:
    p, label = _false_pass((0.9, 5, 0), (0.1, 1, 0), (0.95, 0, 1))  # cutoff 0.9 passes 5 of 6
    assert select_false_pass_cutoff(p, label, 0.8333333333333333) == FalsePassCut(cutoff=0.95, n_true=1, n_false=6, true_pass=1, false_pass=0)


def test_false_pass_that_rounds_onto_the_cap_is_accepted() -> None:
    p, label = _false_pass((0.9, 1, 0), (0.1, 48, 0), (0.95, 0, 1))  # cutoff 0.9 passes 1 of 49
    assert select_false_pass_cutoff(p, label, 0.02040816326530612) == FalsePassCut(cutoff=0.9, n_true=1, n_false=49, true_pass=1, false_pass=1)


# --- sweeps against the plain-loop oracles -----------------------------------------------------------


def _threshold_inputs() -> list[tuple[list[float], list[bool]]]:
    rng = np.random.default_rng(161)
    drawn = []
    for n in (7, 40, 300):
        conf = np.round(rng.uniform(0.5, 1.0, n), 2)
        correct = rng.uniform(0, 1, n) < conf
        drawn.append((conf.tolist(), correct.tolist()))
    return [
        ISSUE,
        # 0 of 1, then 1 of 2, then 1 of 4: at 0.25 the tuned point comes back after an infeasible one.
        _groups((0.9, 1, 0), (0.8, 0, 1), (0.7, 2, 0)),
        _groups((0.9, 48, 0), (0.8, 0, 1)),
        _groups((0.9, 1, 0), (0.8, 0, 5)),
        *drawn,
    ]


@pytest.mark.parametrize("min_accepted", [1, 2, 3, 50])
def test_select_threshold_matches_the_oracle_on_every_cent_of_budget(min_accepted) -> None:
    for conf, correct in _threshold_inputs():
        for k in range(101):
            budget = k / 100
            assert select_threshold(conf, correct, budget, min_accepted) == threshold_oracle(conf, correct, budget, min_accepted), (budget, min_accepted, len(conf))


def test_the_sweep_sees_the_recovery_and_the_minimum() -> None:
    conf, correct = _groups((0.9, 1, 0), (0.8, 0, 1), (0.7, 2, 0))
    assert select_threshold(conf, correct, 0.25) == 0.7  # 0.8 is infeasible (1 of 2), 0.7 is not (1 of 4)
    assert select_threshold(conf, correct, 0.25, min_accepted=2) == 0.7
    assert select_threshold(conf, correct, 0.0, min_accepted=2) is None  # only 0.9 (1 accepted) has no error


def _false_pass_inputs() -> list[tuple[list[float], np.ndarray]]:
    rng = np.random.default_rng(1610)
    drawn = []
    for n in (9, 60, 400):
        p = np.round(rng.uniform(0, 1, n), 2)
        label = rng.uniform(0, 1, n) < p
        label[0] = False  # at least one label-false item
        drawn.append((p.tolist(), label))
    return [
        _false_pass((0.9, 29, 0), (0.1, 71, 0), (0.95, 0, 10)),
        _false_pass((0.9, 5, 0), (0.1, 1, 0), (0.95, 0, 1)),
        _false_pass((0.9, 1, 0), (0.1, 48, 0), (0.95, 0, 1)),
        *drawn,
    ]


def test_select_false_pass_cutoff_matches_the_oracle_on_every_cent_of_cap() -> None:
    for p, label in _false_pass_inputs():
        for k in range(100):  # a cap of 1.0 is refused
            cap = k / 100
            assert select_false_pass_cutoff(p, label, cap) == false_pass_oracle(p, label, cap), (cap, len(p))
