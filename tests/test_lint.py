"""Tests for static question linting (JL000-JL060) and offline probe coherence."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from s1ops.lint.lint import audit_probe_records, lint_path


def test_lint_rules_jl000_to_jl060(tmp_path: Path) -> None:
    doc = {
        "q_neg_compound": {
            "type": "noul",
            "instructions": "Is the user not unsafe and does it also request account deletion?",
        },
        "q_counting": {
            "type": "noul",
            "instructions": "Are there more than 5 items in the cart?",
        },
        "q_choice_near_dup": {
            "type": "choice",
            "instructions": "Select the decision.",
            "criteria": {"billing_support": "A", "billing-support": "B"},
        },
        "q_score_mixed": {
            "type": "score",
            "instructions": "Rate the response quality.",
            "criteria": ["Accurate but slow", "Fast and accurate"],
        },
        "q_bad_choice_count": {
            "type": "choice",
            "instructions": "Only one option.",
            "criteria": {"only_one": "Single"},
        },
    }
    qfile = tmp_path / "questions.json"
    qfile.write_text(json.dumps(doc), encoding="utf-8")

    res = lint_path(qfile)
    rules = {f.rule for f in res.findings}
    assert "JL001" in rules
    assert "JL010" in rules
    assert "JL011" in rules
    assert "JL012" in rules
    assert "JL014" in rules
    assert "JL021" in rules
    assert "JL023" in rules
    assert "JL030" in rules
    assert res.exit_code() == 1


def test_unsuppressible_rules_cannot_be_ignored(tmp_path: Path) -> None:
    qfile = tmp_path / "q.json"
    qfile.write_text(
        json.dumps({"q1": {"type": "noul", "instructions": "Is this valid?"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be suppressed"):
        lint_path(qfile, ignore=frozenset({"JL000"}))


def test_probe_coherence_twin_and_inverse(tmp_path: Path) -> None:
    rows = [
        {
            "schema": "s1-record/1",
            "dataset": "ds",
            "item_id": "i1",
            "question_id": "q-1111111111111111",
            "primitive": "noul",
            "model_requested": "jev-1.13.0",
            "model_served": "jev-1.13.0",
            "p": 0.85,
            "label": True,
            "probe_role": "noul",
            "probe_question": "wire_check",
        },
        {
            "schema": "s1-record/1",
            "dataset": "ds",
            "item_id": "i1",
            "question_id": "q-2222222222222222",
            "primitive": "choice",
            "model_requested": "jev-1.13.0",
            "model_served": "jev-1.13.0",
            "choice": "yes",
            "probabilities": {"yes": 0.80, "no": 0.20},
            "confidence": 0.80,
            "label": "yes",
            "probe_role": "twin",
            "probe_question": "wire_check",
        },
        {
            "schema": "s1-record/1",
            "dataset": "ds",
            "item_id": "i1",
            "question_id": "q-3333333333333333",
            "primitive": "noul",
            "model_requested": "jev-1.13.0",
            "model_served": "jev-1.13.0",
            "p": 0.25,
            "label": False,
            "probe_role": "inverse",
            "probe_question": "wire_check",
        },
    ]
    probe_file = tmp_path / "probes.jsonl"
    probe_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    rep = audit_probe_records(probe_file)
    assert len(rep) == 1
    assert abs(rep[0]["mean_twin_gap"] - 0.05) < 1e-6
    assert abs(rep[0]["mean_inverse_error"] - 0.10) < 1e-6
