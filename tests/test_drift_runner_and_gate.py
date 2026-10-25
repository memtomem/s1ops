"""Tests for Phase 2/3 modules: drift monitor, budget-guarded probe runner, and runtime DecisionGate."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import NoulQuestion
from s1ops.demo import generate_workspace
from s1ops.ledger.journal import BudgetExceeded, Ledger, read as ledger_read
from s1ops.lint.lint import audit_probe_records
from s1ops.monitor.drift import audit_drift, jensen_shannon, psi
from s1ops.monitor.runner import DeterministicMockBackend, expand_probes, run_batch
from s1ops.runtime.gate import (
    DecisionGate,
    ModelServedMismatchError,
    PolicyIntegrityError,
)
from s1ops.web.app import create_app


def test_drift_psi_jsd_and_silent_model_upgrade(tmp_path: Path) -> None:
    ws_dir = tmp_path / "ws"
    ws = generate_workspace(ws_dir)

    # Identical distributions have 0 PSI and 0 JSD
    assert psi([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == 0.0
    assert jensen_shannon([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == 0.0

    rep = audit_drift(ws["records"], ws["records_canary"])
    assert rep["summary"]["n_questions"] == 1
    assert rep["summary"]["critical"] == 1
    q_rep = rep["questions"][0]
    assert q_rep["model_drift_detected"] is True
    assert "jev-1.14.0-rc1" in q_rep["cur_models"]
    assert q_rep["psi"] > 0.10

    # CLI drift returns 0 by default, 4 with --fail-on-critical
    assert cli_main(["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"]]) == 0
    assert (
        cli_main(
            [
                "drift",
                ws["records"],
                ws["records_canary"],
                "--policy",
                ws["policy"],
                "--fail-on-critical",
            ]
        )
        == 4
    )


def test_budget_guarded_probe_runner_and_zero_state_privacy(tmp_path: Path) -> None:
    q = NoulQuestion(instructions="Does the message mention a wire transfer?")
    expanded = expand_probes({"wire": q})
    roles = [r for _, r, _ in expanded]
    assert roles == ["noul", "twin", "inverse"]

    states = [
        {"dataset": "d1", "item_id": "s1", "state": "TOP_SECRET_BODY_1", "label": True},
        {"dataset": "d1", "item_id": "s2", "state": "TOP_SECRET_BODY_2", "label": False},
    ]
    journal_path = tmp_path / "probe_journal.jsonl"
    backend = DeterministicMockBackend(served_model="jev-1.13.0")

    with Ledger(journal_path, cap_usd=0.50) as ledger:
        recs = run_batch(
            states,
            {"wire": q},
            backend=backend,
            ledger=ledger,
            model_requested="jev-1.13.0",
            probe_mode=True,
        )

    assert len(recs) == 6
    # Verify safe-v1 zero-state privacy on both records and ledger journal
    serialized_recs = json.dumps(recs)
    assert "TOP_SECRET_BODY" not in serialized_recs
    assert "TOP_SECRET_BODY" not in journal_path.read_text(encoding="utf-8")

    st = ledger_read(journal_path)
    assert st.spent_usd > 0.0
    assert len(st.outstanding) == 0

    # Verify audit_probe_records works directly on runner output
    probe_out = tmp_path / "probes.jsonl"
    probe_out.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    coherence = audit_probe_records(probe_out)
    assert len(coherence) == 1
    assert coherence[0]["n_states"] == 2

    # Tiny cap_usd triggers BudgetExceeded before backend is called
    tiny_journal = tmp_path / "tiny.jsonl"
    with Ledger(tiny_journal, cap_usd=1e-9) as tiny_ledger:
        with pytest.raises(BudgetExceeded):
            run_batch(states, {"wire": q}, backend=backend, ledger=tiny_ledger)


def test_runtime_decision_gate_and_cli_web(tmp_path: Path) -> None:
    ws_dir = tmp_path / "ws"
    ws = generate_workspace(ws_dir)

    gate = DecisionGate.from_files(ws["policy"], ws["fits"])
    pol_doc = json.loads(Path(ws["policy"]).read_text(encoding="utf-8"))
    qid = pol_doc["policies"][0]["question_id"]
    pol_sha = pol_doc["policies"][0]["policy_sha256"]

    # High probability -> ALLOW
    d_allow = gate.evaluate_noul(
        question_id=qid,
        model_served="jev-1.13.0",
        p_raw=0.98,
        expected_policy_sha256=pol_sha,
    )
    assert d_allow.action == "ALLOW"
    assert d_allow.escalated is False
    assert d_allow.p_calibrated is not None

    # Mid probability -> ESCALATE
    d_esc = gate.evaluate_noul(
        question_id=qid,
        model_served="jev-1.13.0",
        p_raw=0.50,
    )
    assert d_esc.action == "ESCALATE"
    assert d_esc.escalated is True

    # Low probability -> BLOCK
    d_blk = gate.evaluate_noul(
        question_id=qid,
        model_served="jev-1.13.0",
        p_raw=0.02,
    )
    assert d_blk.action == "BLOCK"
    assert d_blk.escalated is False

    # Model mismatch raises ModelServedMismatchError
    with pytest.raises(ModelServedMismatchError):
        gate.evaluate_noul(question_id=qid, model_served="jev-1.14.0-rc1", p_raw=0.90)

    # Policy SHA-256 mismatch raises PolicyIntegrityError
    with pytest.raises(PolicyIntegrityError):
        gate.evaluate_noul(
            question_id=qid,
            model_served="jev-1.13.0",
            p_raw=0.90,
            expected_policy_sha256="deadbeef00000000",
        )

    # Test CLI `s1 gate` and `s1 probe`
    assert (
        cli_main(
            [
                "gate",
                ws["policy"],
                "--fit",
                ws["fits"],
                "--question",
                qid,
                "--model",
                "jev-1.13.0",
                "--p",
                "0.95",
            ]
        )
        == 0
    )
    probe_cli_out = tmp_path / "cli_probes.jsonl"
    probe_cli_ledger = tmp_path / "cli_ledger.jsonl"
    assert (
        cli_main(
            [
                "probe",
                ws["questions"],
                "--states",
                ws["states_local"],
                "-o",
                str(probe_cli_out),
                "--ledger",
                str(probe_cli_ledger),
                "--cap-usd",
                "5.0",
            ]
        )
        == 0
    )

    # Test Web Workbench /api/drift
    app = create_app(workspace_dir=ws_dir)
    client = TestClient(app)
    r_drift = client.get("/api/drift")
    assert r_drift.status_code == 200
    assert r_drift.json()["summary"]["critical"] == 1
