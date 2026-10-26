"""Reports route the distribution a policy was tuned on, under the policy of the record's own group.

`DecisionGate` routes calibrated probabilities exactly when `policy.calibrated`; `s1 drift`'s
escalation rate and both exporters' `policy_action` must decide the same way (#14).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from s1ops.bridges.langfuse import _predict_info
from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, parse_record, read_records, select
from s1ops.contracts.schemas import PolicyEntry, load_policies
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import apply_to_record, fit_group, load_fits, transform
from s1ops.monitor.drift import audit_drift
from s1ops.runtime.gate import DecisionGate
from s1ops.runtime.routing import ROUTING_RULE, RoutingInputError, policy_distribution, route_record
from s1ops.web.app import create_app

MODEL = "jev-1.13.0"


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> dict:
    ws = generate_workspace(tmp_path_factory.mktemp("demo") / "ws")
    (band,) = load_policies(ws["policy"])
    return {"ws": ws, "band": band, "fits": load_fits(ws["fits"]), "rows": _rows(ws["records"])}


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _policies(band: PolicyEntry) -> list[PolicyEntry]:
    budget = dataclasses.replace(band, mode="error_budget")
    return [band, budget, dataclasses.replace(band, calibrated=False), dataclasses.replace(budget, calibrated=False)]


def _assert_reports_route_like_the_gate(rows: list[dict], fit, pol: PolicyEntry) -> None:
    gate = DecisionGate([pol], [fit])
    records = [parse_record(row, n) for n, row in enumerate(rows, 1)]
    escalated = []
    for row, rec in zip(rows, records):
        d = gate.evaluate_noul(question_id=pol.question_id, model_served=MODEL, p_raw=row["p"])
        info = _predict_info(rec, pol)
        assert (info["policy_action"], info["predicted"], info["p_effective"]) == (d.action, d.predicted, d.p_effective), rec.line
        # Unrounded: the stored calib is the gate's own transform of the same raw p (same fit).
        p = row["p"]
        expected = transform(fit.method, "noul", fit.params, np.array([[1.0 - p, p]]))[0, 1] if pol.calibrated else p
        assert policy_distribution(pol, rec)[1][1] == expected, rec.line
        escalated.append(d.escalated)
    rep = audit_drift(records, records, policies=[pol])
    (q,) = rep["questions"]
    assert q["escalation_rate_ref"] == q["escalation_rate_cur"] == round(sum(escalated) / len(escalated), 6)


@pytest.mark.parametrize("which", range(4), ids=["band-cal", "budget-cal", "band-raw", "budget-raw"])
def test_reports_route_the_demo_like_the_gate(demo, which: int) -> None:
    (fit,) = demo["fits"]
    _assert_reports_route_like_the_gate(demo["rows"], fit, _policies(demo["band"])[which])


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_reports_route_like_the_gate_under_other_fits(demo, method: str) -> None:
    (grp,) = select(read_records(demo["ws"]["records_train"])).groups
    fit = fit_group(grp, method=method, split="cal", min_n=30)
    raw = [{k: v for k, v in row.items() if k != "calib"} for row in demo["rows"]]
    applied = [apply_to_record(fit, parse_record(row)) for row in raw]
    for pol in _policies(demo["band"]):
        _assert_reports_route_like_the_gate(applied, fit, pol)


def _row(primitive: str, **extra) -> dict:
    answer = {
        "noul": {"p": 0.9},
        "choice": {"choice": "A", "confidence": 0.9, "probabilities": {"A": 0.9, "B": 0.1}},
        "score": {"score": 0, "confidence": 0.9, "probabilities": {"0": 0.9, "1": 0.1}},
    }[primitive]
    return {
        "schema": "s1-record/1", "item_id": "i", "question_id": "q", "primitive": primitive,
        "model_requested": "m", "model_served": "m", **answer, **extra,
    }


def _policy(primitive: str, **kw) -> PolicyEntry:
    base = {"question_id": "q", "model_served": "m", "primitive": primitive, "mode": "error_budget", "split": "eval", "n": 1, "threshold": 0.5}
    return PolicyEntry(**{**base, **kw})


@pytest.mark.parametrize("primitive", ["noul", "choice", "score"])
@pytest.mark.parametrize("calib", [None, {}, {"method": "temperature"}], ids=["absent", "empty", "no-values"])
def test_a_calibrated_policy_refuses_a_record_without_calib(primitive: str, calib) -> None:
    rec = parse_record(_row(primitive, **({} if calib is None else {"calib": calib})), 5)
    pol = _policy(primitive, calibrated=True)
    for call in (lambda: route_record(pol, rec), lambda: _predict_info(rec, pol)):
        with pytest.raises(RoutingInputError, match=r"^line 5: policy for q \(m\) was tuned on calibrated probabilities .*`s1 apply`") as exc:
            call()
        assert exc.value.line == 5 and exc.value.side is None
    # A raw policy routes the raw answer.
    assert route_record(_policy(primitive), rec).escalated is False
    with pytest.raises(RoutingInputError, match=r"^current records: line 5: policy for q") as exc:
        audit_drift([parse_record(_row(primitive, calib=_calib(primitive)), 1)], [rec], policies=[pol])
    assert str(exc.value).count("line") == 1 and exc.value.side == "current"


def _calib(primitive: str) -> dict:
    return {"p": 0.4} if primitive == "noul" else {"probabilities": {"A" if primitive == "choice" else "0": 0.4, "B" if primitive == "choice" else "1": 0.6}}


@pytest.mark.parametrize("primitive", ["noul", "choice", "score"])
def test_the_policy_not_the_record_picks_the_distribution(primitive: str) -> None:
    # Raw 0.9 on the first class; calib 0.4 / 0.6. A 0.8 budget accepts raw and escalates calib.
    rec = parse_record(_row(primitive, calib=_calib(primitive)))
    raw, cal = _policy(primitive, threshold=0.8), _policy(primitive, threshold=0.8, calibrated=True)
    assert route_record(raw, rec).escalated is False
    assert route_record(cal, rec).escalated is True
    assert _predict_info(rec, raw)["policy_action"] != "ESCALATE"
    assert _predict_info(rec, cal)["policy_action"] == "ESCALATE"


def test_the_issue_repro_now_matches_the_gate() -> None:
    rec = parse_record(_row("noul", calib={"p": 0.6}))
    pol = _policy("noul", threshold=0.8)
    gate = DecisionGate([pol]).evaluate_noul(question_id="q", model_served="m", p_raw=0.9)
    assert _predict_info(rec, pol)["policy_action"] == gate.action == "ALLOW"


def test_invalid_stored_calib_is_still_refused_under_a_raw_policy(tmp_path: Path, capsys) -> None:
    # The router does not read calib for a raw policy, but the reports still validate it.
    rec = parse_record(_row("noul", calib={"p": "nan"}), 2)
    pol = _policy("noul")
    assert route_record(pol, rec).escalated is False
    with pytest.raises(RecordError, match="calib.p"):
        _predict_info(rec, pol)
    records = tmp_path / "r.jsonl"
    records.write_text(json.dumps(_row("noul", calib={"p": "nan"})) + "\n")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "s1-policy/1", "policies": [pol.to_dict()]}))
    for args in (["drift", str(records), str(records), "--policy", str(policy)],
                 ["export", "phoenix", str(records), "-o", str(tmp_path / "o"), "--policy", str(policy)]):
        assert cli_main(args) == 1
        assert "calib.p" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["drift", "export langfuse", "export phoenix"])
def test_cli_refuses_a_calibrated_policy_without_calib_in_one_line(tmp_path: Path, command: str, capsys) -> None:
    records = tmp_path / "r.jsonl"
    records.write_text(json.dumps(_row("noul")) + "\n")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "s1-policy/1", "policies": [_policy("noul", calibrated=True).to_dict()]}))
    r = str(records)
    args = {
        "drift": ["drift", r, r, "--policy", str(policy)],
        "export langfuse": ["export", "langfuse", r, "-o", str(tmp_path / "o"), "--policy", str(policy)],
        "export phoenix": ["export", "phoenix", r, "-o", str(tmp_path / "o"), "--policy", str(policy)],
    }[command]
    assert cli_main(args) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err and "`s1 apply`" in err
    prefix = "s1: reference records: line 1: " if command == "drift" else "s1: line 1: "
    assert err.startswith(prefix)


def test_drift_routes_each_record_under_its_own_groups_policy() -> None:
    def rec(model: str, p: float, n: int):
        return parse_record({**_row("noul", p=p), "model_served": model, "item_id": f"i{n}"}, n)

    strict = _policy("noul", model_served="m1", threshold=0.95)
    lenient = _policy("noul", model_served="m2", threshold=0.6)
    ref = [rec("m1", 0.9, 1), rec("m2", 0.9, 2)]  # m1 escalates under 0.95, m2 does not under 0.6
    cur = [rec("m1", 0.9, 1), rec("m3", 0.9, 2), rec("m3", 0.99, 3)]  # m3 has no policy
    (q,) = audit_drift(ref, cur, policies=[strict, lenient])["questions"]
    assert q["escalation_rate_ref"] == 0.5
    assert q["escalation_rate_cur"] == 1.0  # over the one m1 record only
    (q,) = audit_drift(ref, [rec("m3", 0.9, 1)], policies=[strict, lenient])["questions"]
    assert q["escalation_rate_cur"] is None
    (q,) = audit_drift(ref, cur)["questions"]
    assert q["escalation_rate_ref"] is None and q["escalation_rate_cur"] is None


def test_demo_drift_routes_only_the_policys_model(demo, capsys) -> None:
    ws, (fit,) = demo["ws"], demo["fits"]
    canary = _rows(ws["records_canary"])
    gate = DecisionGate([demo["band"]], [fit])

    def gate_rate(rows: list[dict]) -> float:
        esc = [gate.evaluate_noul(question_id=demo["band"].question_id, model_served=MODEL, p_raw=r["p"]).escalated for r in rows]
        return round(sum(esc) / len(esc), 6)

    own = [r for r in canary if r["model_served"] == MODEL]
    assert len(own) == 80
    assert cli_main(["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"]]) == 0
    (q,) = json.loads(capsys.readouterr().out)["questions"]
    assert q["escalation_rate_ref"] == gate_rate(demo["rows"])
    assert q["escalation_rate_cur"] == gate_rate(own)
    # --raw-only changes what PSI compares, not what the policy routes.
    assert cli_main(["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"], "--raw-only"]) == 0
    (raw_only,) = json.loads(capsys.readouterr().out)["questions"]
    assert (raw_only["escalation_rate_ref"], raw_only["escalation_rate_cur"]) == (q["escalation_rate_ref"], q["escalation_rate_cur"])
    assert raw_only["psi"] != q["psi"]


def test_demo_canary_calibrates_only_the_policys_model(demo) -> None:
    rows = _rows(demo["ws"]["records_canary"])
    assert Counter((r["model_served"], "calib" in r) for r in rows) == {(MODEL, True): 80, ("jev-1.14.0-rc1", False): 40}
    for r in rows:
        r.pop("calib", None)
    # The raw canary, byte for byte, as main @ 0635b06 generated it before any calib was added.
    digest = hashlib.sha256("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()).hexdigest()
    assert digest == "d388974b8068f399e68d889c7ea8220ed8779d45e5d1244597f8d7e9abb9804b"


def _strip_calib(path: str, model: str) -> int:
    rows = _rows(path)
    n = next(i for i, r in enumerate(rows) if r["model_served"] == model)
    rows[n].pop("calib")
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return n + 1


@pytest.mark.parametrize("which", ["records", "records_canary"])
def test_web_drift_names_the_file_without_calib(tmp_path: Path, which: str) -> None:
    ws = generate_workspace(tmp_path / "ws")
    line = _strip_calib(ws[which], MODEL)
    r = TestClient(create_app(workspace_dir=Path(ws["records"]).parent)).get("/api/drift")
    assert r.status_code == 400
    assert r.json()["detail"].startswith(f"{which}.jsonl: line {line}: policy for ")


def test_web_drift_keeps_a_computation_bug_a_500(tmp_path: Path, monkeypatch) -> None:
    ws = generate_workspace(tmp_path / "ws")

    def broken(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("s1ops.monitor.drift.route_record", broken)
    r = TestClient(create_app(workspace_dir=Path(ws["records"]).parent), raise_server_exceptions=False).get("/api/drift")
    assert r.status_code == 500


def test_the_routing_rule_is_3() -> None:
    assert ROUTING_RULE == "3"
