"""Tests for evaluation metrics, calibration fitting, shadow comparison, and judge audit."""

from __future__ import annotations

import math
import random

import numpy as np

from s1ops.contracts.records import parse_record, select
from s1ops.eval.audit import audit_group
from s1ops.eval.compare import compare_records, holm, mcnemar_exact
from s1ops.eval.judge import agreement, by_criterion
from s1ops.eval.metrics import brier, ece, ece_floor, log_loss, tie_mass
from s1ops.improve.calibrate import apply_to_record, fit_group, transform


def _make_overconfident_records(n: int = 80, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        z = rng.gauss(0.0, 1.2)
        p_true = 1.0 / (1.0 + math.exp(-z))
        y = rng.random() < p_true
        p_raw = 1.0 / (1.0 + math.exp(-z * 2.2))
        out.append(
            {
                "schema": "s1-record/1",
                "dataset": "synth",
                "item_id": f"it-{i}",
                "question_id": "q-abcdef0123456789",
                "primitive": "noul",
                "model_requested": "jev-1.13.0",
                "model_served": "jev-1.13.0",
                "p": round(p_raw, 6),
                "label": bool(y),
            }
        )
    return out


def test_metrics_ece_floor_and_tie_mass() -> None:
    scores = np.array([0.5] * 40 + [0.9] * 40, dtype=float)
    y = np.array([1, 0] * 20 + [1] * 36 + [0] * 4, dtype=int)
    P = np.column_stack([1.0 - scores, scores])

    assert brier("noul", P, y) < 0.25
    assert log_loss(P, y) < 0.70
    assert ece(scores, y.astype(float), bins=10, strategy="mass") >= 0.0

    tm = tie_mass(scores)
    assert tm["unique_values"] == 2
    assert tm["max_tie_count"] == 40

    fl = ece_floor(scores, bins=10, reps=100, seed=0)
    assert fl["mass"]["p95"] >= fl["mass"]["mean"] >= 0.0


def test_calibrate_temperature_reduces_nll_and_audits() -> None:
    raw_objs = _make_overconfident_records(100, seed=11)
    train_recs = [parse_record(o) for o in raw_objs[:60]]
    test_recs = [parse_record(o) for o in raw_objs[60:]]

    grp_train = select(train_recs).groups[0]
    fit = fit_group(grp_train, method="temperature", split="cal")
    assert fit.params["temperature"] > 1.1
    assert fit.nll_after <= fit.nll_before

    grp_test = select(test_recs).groups[0]
    after_P = transform(fit.method, fit.primitive, fit.params, grp_test.P)
    rep = audit_group(grp_test, bins=10, reps=50, seed=0, after=after_P)
    assert "after" in rep
    assert rep["after"]["log_loss"] <= rep["before"]["log_loss"]

    applied = apply_to_record(fit, test_recs[0])
    assert "calib" in applied and "p" in applied["calib"]


def test_shadow_compare_mcnemar_holm_and_judge() -> None:
    mc = mcnemar_exact([True] * 12 + [False] * 2, [False] * 12 + [True] * 2)
    assert mc["p"] < 0.02
    adj = holm({"q1": 0.01, "q2": 0.04, "q3": 0.20})
    assert adj["q1"] <= adj["q2"] <= adj["q3"]

    objs_a = _make_overconfident_records(30, seed=3)
    objs_b = [
        {
            **o,
            "model_requested": "kev-1.0",
            "model_served": "kev-1.0",
            "p": 0.55,
        }
        for o in objs_a
    ]
    cmp = compare_records(objs_a, objs_b)
    assert cmp["shared_pairs"] == 30
    assert len(cmp["questions"]) == 1

    jev_grades = [3, 1, 4, 2, 0]
    ref_grades = [3, 1, 3, 2, 1]
    ag = agreement(jev_grades, ref_grades, levels=5, pass_level=3)
    assert ag.n == 5
    assert ag.weighted_kappa > 0.5

    rows = [
        ("faithfulness", 3, 3),
        ("faithfulness", 1, 1),
        ("faithfulness", 4, 3),
        ("tone", 2, 2),
        ("tone", 0, 1),
    ]
    by_c = by_criterion(rows, levels=5, pass_level=2)
    assert "faithfulness" in by_c and "tone" in by_c
