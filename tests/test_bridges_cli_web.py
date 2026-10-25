"""Tests for Langfuse/OpenInference safe-v1 exporters, unified CLI, and FastAPI Web Workbench."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from s1ops.bridges.langfuse import build_langfuse_events
from s1ops.bridges.openinference import (
    MultipleCurrenciesError,
    build_openinference_spans,
    normalize_status,
)
from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records
from s1ops.demo import generate_workspace
from s1ops.web.app import create_app


def test_bridges_safe_v1_and_currency_guard(tmp_path: Path) -> None:
    ws = generate_workspace(tmp_path / "ws")
    records = read_records(ws["records"])

    lf = build_langfuse_events(
        records[:10],
        run_name="test-run",
        annotation_candidates=[
            {"dataset": "d", "item_id": "i1", "question_id": "q1", "state": "SECRET_TEXT"}
        ],
    )
    assert lf["redaction"] == "safe-v1"
    serialized_lf = json.dumps(lf)
    assert "SECRET_TEXT" not in serialized_lf
    assert len(lf["observations"]) == 10
    assert len(lf["scores"]) >= 30

    spans = build_openinference_spans(records[:10], run_name="test-run")
    assert len(spans) == 10
    assert spans[0]["span_kind"] == "GUARDRAIL"
    assert spans[0]["status_code"] == "OK"
    assert normalize_status("completed") == "OK"
    assert normalize_status("failed") == "ERROR"

    with pytest.raises(MultipleCurrenciesError):
        build_openinference_spans(
            records[:2],
            record_currencies={"a": "USD", "b": "KRW"},
        )


def test_cli_end_to_end_and_refusal_exit_codes(tmp_path: Path) -> None:
    ws_dir = tmp_path / "ws"
    ws = generate_workspace(ws_dir)

    # 1. lint rules & probe-audit return 0; lint check with --strict on warnings returns 1
    assert cli_main(["lint", "rules"]) == 0
    assert cli_main(["lint", "probe-audit", ws["probe_records"]]) == 0
    assert cli_main(["lint", "check", ws["questions"]]) == 0
    assert cli_main(["lint", "check", "--strict", ws["questions"]]) == 1

    # 2. fit on train records
    fit_out = tmp_path / "custom_fits.json"
    assert cli_main(["fit", ws["records_train"], "-o", str(fit_out)]) == 0

    # 3. apply on in-sample train records must refuse with exit code 2
    in_sample_out = tmp_path / "in_sample_applied.jsonl"
    assert cli_main(["apply", ws["records_train"], "--fit", str(fit_out), "-o", str(in_sample_out)]) == 2

    # 4. apply on mismatched model (kev-1.0 vs jev-1.13.0 fit) must refuse with exit code 3
    mismatch_out = tmp_path / "mismatch_applied.jsonl"
    assert cli_main(["apply", ws["records_kev"], "--fit", str(fit_out), "-o", str(mismatch_out)]) == 3

    # 5. apply on held-out records succeeds with exit code 0
    heldout_out = tmp_path / "heldout_applied.jsonl"
    assert cli_main(["apply", ws["records"], "--fit", str(fit_out), "-o", str(heldout_out)]) == 0

    # 6. audit, policy tune, slice, labelaudit, compare, judge, export
    assert cli_main(["audit", ws["records"], "--fit", ws["fits"], "--boot", "30"]) == 0
    pol_out = tmp_path / "pol.json"
    assert cli_main(["policy", "tune", ws["records"], "-o", str(pol_out), "--use-calib"]) == 0
    assert cli_main(["slice", "answerable", ws["records"], "--features", ws["features"]]) == 0
    assert cli_main(["slice", "mine", ws["records"], "--features", ws["features"]]) == 0
    assert cli_main(["labelaudit", "rank", ws["records"], "--fit", ws["fits"], "--top", "5"]) == 0
    assert cli_main(["compare", ws["records"], ws["records_kev"]]) == 0
    assert cli_main(["judge", "audit", ws["judge_grades"]]) == 0

    lf_out = tmp_path / "lf.json"
    px_out = tmp_path / "px.jsonl"
    assert cli_main(["export", "langfuse", ws["records"], "-o", str(lf_out), "--policy", str(pol_out)]) == 0
    assert cli_main(["export", "phoenix", ws["records"], "-o", str(px_out), "--policy", str(pol_out)]) == 0


def test_web_workbench_endpoints(tmp_path: Path) -> None:
    ws_dir = tmp_path / "ws"
    generate_workspace(ws_dir)
    app = create_app(workspace_dir=ws_dir)
    client = TestClient(app)

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "s1ops" in r_root.text

    r_ov = client.get("/api/overview")
    assert r_ov.status_code == 200
    assert r_ov.json()["n_records"] == 160

    r_aud = client.get("/api/audit?n_boot=30")
    assert r_aud.status_code == 200
    assert len(r_aud.json()["groups"]) == 1

    r_pol = client.get("/api/simulate-policy?tau_low=0.25&tau_high=0.75")
    assert r_pol.status_code == 200
    assert "active_band" in r_pol.json()

    r_lint = client.get("/api/lint")
    assert r_lint.status_code == 200
    assert r_lint.json()["n_findings"] > 0

    r_sl = client.get("/api/slices")
    assert r_sl.status_code == 200
    assert len(r_sl.json()["error_slices"]) >= 1

    r_la = client.get("/api/labelaudit")
    assert r_la.status_code == 200
    assert len(r_la.json()["top_flags"]) > 0

    r_cmp = client.get("/api/compare")
    assert r_cmp.status_code == 200
    assert r_cmp.json()["judge"]["overall"]["n"] == 90
