"""Tests for policy threshold & escalation tuning, slice discovery, and blind label audit."""

from __future__ import annotations

import re

from s1ops.contracts.records import parse_record
from s1ops.improve.labelaudit import (
    build_sheet,
    estimate,
    rank,
    score_sheet,
)
from s1ops.improve.policy import (
    Band,
    DevRecord,
    Prices,
    choose,
    evaluate,
    pareto,
    select_threshold,
    sweep,
)
from s1ops.improve.slices import mine_error_slices, slice_report


def test_policy_select_threshold_tie_safety_and_escalation() -> None:
    # Tie at conf=0.80 with mixed correctness (1 right, 3 wrong):
    # Tie-safe group selection must not split the 0.80 tie group.
    conf = [0.95, 0.92, 0.80, 0.80, 0.80, 0.80, 0.60]
    correct = [True, True, True, False, False, False, False]
    tau = select_threshold(conf, correct, budget=0.10)
    assert tau == 0.92

    records = [
        DevRecord(id="1", p_jev=0.92, label=True, llm=True),
        DevRecord(id="2", p_jev=0.52, label=True, llm=True),
        DevRecord(id="3", p_jev=0.45, label=False, llm=False),
        DevRecord(id="4", p_jev=0.08, label=False, llm=False),
    ]
    res = evaluate(records, Band(0.3, 0.7), Prices(jev_usd=0.00004, llm_usd=0.004))
    assert res is not None
    assert res.escalated == 2
    assert res.accuracy == 1.0

    swept = sweep(records, [Band(0.2, 0.8), Band(0.3, 0.7), Band(0.5, 0.5)])
    front = pareto(swept, metric="f1")
    assert len(front) >= 1
    chosen = choose(swept, metric="f1", max_escalation_rate=0.60)
    assert chosen is not None


def test_slices_answerability_and_bh_fdr_mining() -> None:
    slice_items = []
    mine_rows = []
    for i in range(60):
        is_bad = i < 20
        # Bad slice has 16/20 errors; clean slice has 4/40 errors
        is_err = (i < 16) if is_bad else (i in (25, 30, 35, 40))
        slice_items.append(
            {
                "slice": "trap" if is_bad else "clean",
                "predicted": not is_err,
                "label": True,
            }
        )
        mine_rows.append(
            {
                "error": is_err,
                "features": {
                    "domain": "slang" if is_bad else "formal",
                    "noise": "a" if (i % 2 == 0) else "b",
                },
            }
        )

    rep = {r.slice: r for r in slice_report(slice_items, min_n=15)}
    assert "trap" in rep and "clean" in rep
    assert rep["clean"].accuracy > rep["trap"].accuracy

    mined = mine_error_slices(mine_rows, min_support=10, fdr_q=0.10)
    assert len(mined) >= 1
    assert mined[0].feature == "domain=slang"
    assert mined[0].significant is True


def test_labelaudit_ranking_and_blind_sheet_estimation() -> None:
    rows = []
    states = {}
    flipped_ids = {f"it-{i:02d}" for i in range(6)}
    for i in range(30):
        item_id = f"it-{i:02d}"
        states[("ds", item_id)] = f"State text #{i}"
        given = False if i < 6 else True
        rows.append(
            {
                "schema": "s1-record/1",
                "dataset": "ds",
                "item_id": item_id,
                "question_id": "q-1234567890abcdef",
                "primitive": "noul",
                "model_requested": "jev-1.13.0",
                "model_served": "jev-1.13.0",
                "p": 0.96,
                "label": given,
            }
        )

    flags = rank(rows, balance=False)
    assert {f.item_id for f in flags[:6]} == flipped_ids

    records = [parse_record(row, n) for n, row in enumerate(rows, 1)]
    sheet = build_sheet(records, states, 6, 12, options=["True", "False"], seed=0, balance=False)
    completed_form = re.sub(r"^답: $", "답: True", sheet.form, flags=re.M)
    scored = score_sheet(completed_form, sheet.key)
    est = estimate(
        scored["flag"]["reviewer_disagrees_with_label"],
        scored["flag"]["n"],
        scored["control"]["reviewer_disagrees_with_label"],
        scored["control"]["n"],
    )
    assert est.flag_precision == 1.0
    assert est.control_rate == 0.0
    assert est.passes() is True

