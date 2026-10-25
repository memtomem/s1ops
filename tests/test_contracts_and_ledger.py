"""Tests for s1ops contracts (questions, records, schemas) and reserve-then-settle ledger."""

from __future__ import annotations

from pathlib import Path
import pytest

import s1
import s1ops
from s1ops.contracts.questions import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    parse_wire_question,
    question_id,
    to_openai_decision,
)
from s1ops.contracts.records import (
    RecordError,
    parse_record,
    select,
)
from s1ops.contracts.schemas import PolicyEntry, load_policies, save_policies
from s1ops.ledger.journal import (
    BudgetExceeded,
    Ledger,
    LedgerCorrupt,
    LedgerUnusable,
    lock_path,
    migrate,
    read,
    recover,
    token_bound,
)


def test_package_aliases_and_question_wire_conversions() -> None:
    assert s1.__version__ == s1ops.__version__

    nq = NoulQuestion(instructions="Is this message safe?")
    qid = question_id(nq)
    assert qid.startswith("q-")
    assert len(qid) == 18

    s1_payload = nq.wire_dict()
    assert s1_payload == {"type": "noul", "instructions": "Is this message safe?"}
    assert question_id(parse_wire_question(s1_payload)) == qid

    dec_payload = to_openai_decision("safety_check", nq)
    assert dec_payload["type"] == "predicate"
    assert question_id(parse_wire_question(dec_payload)) == qid

    cq = ChoiceQuestion(
        instructions="Pick intent",
        criteria={"billing": "Billing questions", "support": "Support questions"},
    )
    cq_qid = question_id(cq)
    assert question_id(parse_wire_question(cq.wire_dict())) == cq_qid
    assert question_id(parse_wire_question(to_openai_decision("intent", cq))) == cq_qid

    sq = ScoreQuestion(instructions="Relevance score", criteria=["low", "high"])
    assert sq.type == "score"
    assert question_id(sq).startswith("q-")


def test_record_parsing_and_strict_grouping() -> None:
    valid_noul = {
        "schema": "s1-record/1",
        "dataset": "ds1",
        "item_id": "item-1",
        "question_id": "q-1234567890abcdef",
        "primitive": "noul",
        "model_requested": "jev-1.13.0",
        "model_served": "jev-1.13.0",
        "p": 0.8,
        "label": True,
    }
    rec = parse_record(valid_noul)
    assert rec.key == ("q-1234567890abcdef", "jev-1.13.0", "noul")
    assert rec.item == ("ds1", "item-1")

    # Reject bool in probability
    with pytest.raises(RecordError, match="must be a number"):
        parse_record({**valid_noul, "p": True})

    # Reject choice sum out of tolerance (0.02)
    bad_choice = {
        "schema": "s1-record/1",
        "dataset": "ds1",
        "item_id": "item-2",
        "question_id": "q-1234567890abcdef",
        "primitive": "choice",
        "model_requested": "jev-1.13.0",
        "model_served": "jev-1.13.0",
        "choice": "a",
        "probabilities": {"a": 0.5, "b": 0.2},
        "confidence": 0.5,
        "label": "a",
    }
    with pytest.raises(RecordError, match="sum to"):
        parse_record(bad_choice)

    # Check select() never pools across model_served
    rec2 = parse_record({**valid_noul, "model_requested": "kev-1.0", "model_served": "kev-1.0"})
    sel = select([rec, rec2])
    assert len(sel.groups) == 2


def test_policy_schema_roundtrip(tmp_path: Path) -> None:
    pol = PolicyEntry(
        question_id="q-1234567890abcdef",
        model_served="jev-1.13.0",
        primitive="noul",
        mode="escalation_band",
        split="eval",
        n=100,
        threshold=0.72,
        error_budget=0.10,
        band_lo=0.25,
        band_hi=0.75,
        calibrated=True,
    )
    assert len(pol.policy_sha256) == 16
    path = tmp_path / "policy.json"
    save_policies([pol], path)
    loaded = load_policies(path)
    assert len(loaded) == 1
    assert loaded[0].policy_sha256 == pol.policy_sha256


def test_ledger_reserve_settle_migrate_recover(tmp_path: Path) -> None:
    lpath = tmp_path / "journal.jsonl"
    assert lock_path(lpath).name == "journal.jsonl.lock"
    assert token_bound("안녕하세요 hello") > 10.0

    with Ledger(lpath, cap_usd=1.00) as ledger:
        res1 = ledger.reserve(0.40, model="jev-1.13.0")
        assert len(ledger.outstanding) == 1
        ledger.settle(res1, 0.25, outcome="measured")
        assert len(ledger.outstanding) == 0

        # Exceeding cap raises BudgetExceeded
        with pytest.raises(BudgetExceeded):
            ledger.reserve(0.80)

        # Overrun (measured > reserved) locks out further reservations
        res2 = ledger.reserve(0.10)
        ledger.settle(res2, 0.15, outcome="measured")
        with pytest.raises(LedgerUnusable):
            ledger.reserve(0.01)

    st = read(lpath)
    assert abs(st.spent_usd - 0.40) < 1e-6

    # Legacy migration
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text('{"usd": 0.12}\n{"usd": 0.08}\n', encoding="utf-8")
    migrated_path = tmp_path / "migrated.jsonl"
    st_mig = migrate(legacy_path, migrated_path, acknowledge_lost_reservations=True)
    assert abs(st_mig.opening_usd - 0.20) < 1e-6

    # Partial trailing write recovery
    broken_path = tmp_path / "broken.jsonl"
    broken_path.write_bytes(migrated_path.read_bytes() + b'{"v": 1, "event": "res')
    with pytest.raises(LedgerCorrupt):
        read(broken_path)
    rec_out = tmp_path / "recovered.jsonl"
    recover(broken_path, rec_out)
    assert abs(read(rec_out).opening_usd - 0.20) < 1e-6
