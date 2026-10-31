"""`policy.json` rows: a contradictory row and a row edited after `s1 policy tune` are refused at load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops import PolicyFingerprintError
from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import load_policies
from s1ops.demo import generate_workspace
from s1ops.web.app import create_app


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _doc(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _write(path: str, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc))


def _edit(path: str, index: int = 0, **fields) -> None:
    doc = _doc(path)
    doc["policies"][index].update(fields)
    _write(path, doc)


def test_demo_policies_load_with_their_stored_fingerprints(ws) -> None:
    for name in ("policy", "policy_categorical"):
        stored = [row["policy_sha256"] for row in _doc(ws[name])["policies"]]
        assert [e.policy_sha256 for e in load_policies(ws[name])] == stored


def test_an_edited_threshold_is_refused(ws) -> None:
    stored = _doc(ws["policy"])["policies"][0]["policy_sha256"]
    _edit(ws["policy"], threshold=0.01)
    with pytest.raises(PolicyFingerprintError) as e:
        load_policies(ws["policy"])
    message = str(e.value)
    assert f"policies.0: policy_sha256 {stored} does not match its routing fields" in message
    assert "remove policy_sha256 to accept the edit" in message


@pytest.mark.parametrize("field", ["band_lo", "band_hi", "calibrated", "fit_sha256", "mode"])
def test_any_routing_field_edit_is_refused(ws, field: str) -> None:
    row = _doc(ws["policy"])["policies"][0]
    edits = {
        "band_lo": {"band_lo": 0.2},
        "band_hi": {"band_hi": 0.8},
        "calibrated": {"calibrated": False, "fit_sha256": None},
        "fit_sha256": {"fit_sha256": "0" * 16},
        "mode": {"mode": "error_budget", "band_lo": None, "band_hi": None},
    }[field]
    assert all(row[k] != v for k, v in edits.items() if k != "fit_sha256" or row[k] is not None)
    _edit(ws["policy"], **edits)
    with pytest.raises(PolicyFingerprintError):
        load_policies(ws["policy"])


def test_a_non_routing_edit_still_loads(ws) -> None:
    _edit(ws["policy"], n=7, created_at="2000-01-01T00:00:00+00:00", metrics_on_split={"note": "hand"})
    assert load_policies(ws["policy"])[0].n == 7


@pytest.mark.parametrize("stored", [None, "absent"])
def test_an_edit_without_a_stored_fingerprint_is_accepted(ws, stored) -> None:
    doc = _doc(ws["policy"])
    row = doc["policies"][0]
    row["threshold"] = 0.01
    if stored == "absent":
        del row["policy_sha256"]
    else:
        row["policy_sha256"] = None
    _write(ws["policy"], doc)
    (entry,) = load_policies(ws["policy"])
    assert entry.threshold == 0.01


def test_cli_and_web_report_an_edited_policy(ws, capsys) -> None:
    _edit(ws["policy"], threshold=0.01)
    assert cli_main(["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"]]) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("s1: ") and err.count("\n") == 1
    assert "policy.json: policies.0: policy_sha256" in err
    client = TestClient(create_app(workspace_dir=Path(ws["records"]).parent))
    r = client.get("/api/drift")
    assert r.status_code == 400
    assert "policy.json" in r.json()["detail"] and "does not match its routing fields" in r.json()["detail"]


def _rebaseline(path: str, index: int = 0, **fields) -> None:
    # A contradictory row with its fingerprint removed, so only the schema can refuse it.
    doc = _doc(path)
    doc["policies"][index].update(fields)
    doc["policies"][index].pop("policy_sha256", None)
    _write(path, doc)


@pytest.mark.parametrize(
    ("name", "fields", "message"),
    [
        ("policy", {"primitive": "binary"}, "policies.0.primitive: Input should be 'noul', 'choice' or 'score'"),
        ("policy", {"n": 0}, "policies.0.n: Input should be greater than or equal to 1"),
        ("policy", {"error_budget": 7.5}, "policies.0.error_budget: Input should be less than or equal to 1"),
        ("policy", {"error_budget": -0.1}, "policies.0.error_budget: Input should be greater than or equal to 0"),
        (
            "policy",
            {"calibrated": False},
            "policies.0: an uncalibrated policy has no fit; fit_sha256 must be null when calibrated is false",
        ),
        (
            "policy_categorical",
            {"mode": "escalation_band", "band_lo": 0.2, "band_hi": 0.8},
            "policies.0: an escalation_band policy is noul-only; score policies use error_budget",
        ),
    ],
)
def test_contradictory_rows_are_refused_at_load(ws, name: str, fields: dict, message: str) -> None:
    _rebaseline(ws[name], **fields)
    with pytest.raises(ValueError) as e:
        load_policies(ws[name])
    assert message in str(e.value)
    assert not isinstance(e.value, PolicyFingerprintError)


def test_an_uncalibrated_row_without_calibrated_key_and_with_a_fit_is_refused(ws) -> None:
    doc = _doc(ws["policy"])
    row = doc["policies"][0]
    del row["calibrated"]
    row.pop("policy_sha256")
    _write(ws["policy"], doc)
    with pytest.raises(ValueError, match="fit_sha256 must be null when calibrated is false"):
        load_policies(ws["policy"])
