"""The noul gate for a per-call consumer (decision-plane's hook): a light import path, and a fail-closed
load and evaluation, so a policy is routed only through the fit it was tuned through."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import s1ops
from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import PolicyEntry, load_policies, save_policies
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import Fit, apply_to_record, load_fits, save_fits
from s1ops.runtime.gate import DecisionGate, PolicyIntegrityError

QID, MODEL = "q-0000000000000001", "kev-latest"
PARAMS = {"temperature": {"temperature": 2.5}, "platt": {"a": 1.3, "b": -0.2}, "isotonic": {"x": [0.0, 0.4, 1.0], "y": [0.0, 0.3, 1.0]}}


def _fit(method: str = "temperature", params: dict | None = None, **kw) -> Fit:
    fields = dict(
        question_id=QID, model_served=MODEL, primitive="noul", method=method, split="cal", n=2,
        classes=[False, True], params=PARAMS[method] if params is None else params, nll_before=0.7,
        nll_after=0.6, fitted_at="2026-10-26T00:00:00+00:00", source_sha256="", warnings=[],
        items=[["cal", "i1"], ["cal", "i2"]],
    )
    return Fit(**{**fields, **kw})


def _policy(fit: Fit | None = None, **kw) -> PolicyEntry:
    fields = dict(
        question_id=QID, model_served=MODEL, primitive="noul", mode="error_budget", split="eval", n=40,
        threshold=0.8, error_budget=0.1, calibrated=True, fit_sha256=fit.fit_sha256 if fit else None,
    )
    return PolicyEntry(**{**fields, **kw})


# --- import path -----------------------------------------------------------------------------------

_PROBE = """
import json, sys
from s1ops.contracts.schemas import load_policies
from s1ops.improve.calibrate import load_fits
from s1ops.runtime.gate import DecisionGate
gate = DecisionGate(load_policies(sys.argv[1]), load_fits(sys.argv[2]))
d = gate.evaluate_noul(question_id=sys.argv[3], model_served=sys.argv[4], p_raw=0.7)
heavy = ("scipy.optimize", "sklearn", "fastapi", "uvicorn", "s1ops.cli", "s1ops.web")
print(json.dumps({"action": d.action, "loaded": sorted(m for m in heavy if m in sys.modules)}))
"""


@pytest.mark.parametrize("method", sorted(PARAMS))
def test_loading_and_evaluating_keeps_the_fitting_stack_and_web_app_unloaded(tmp_path: Path, method: str) -> None:
    fit = _fit(method)
    save_fits([fit], tmp_path / "fits.json")
    save_policies([_policy(fit)], tmp_path / "policy.json")
    out = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE, str(tmp_path / "policy.json"), str(tmp_path / "fits.json"), QID, MODEL],
        capture_output=True, text=True, check=True, cwd=tmp_path,
    )
    result = json.loads(out.stdout)
    assert result["loaded"] == [] and result["action"] in ("ALLOW", "BLOCK", "ESCALATE")


def test_every_export_resolves_lazily_to_its_home_object() -> None:
    assert set(s1ops._LAZY) == set(s1ops.__all__) - {"__version__"}
    for name, module in s1ops._LAZY.items():
        assert getattr(s1ops, name) is getattr(sys.modules[module], name), name
    assert set(s1ops.__all__) <= set(dir(s1ops))
    with pytest.raises(AttributeError):
        s1ops.no_such_name  # noqa: B018


def test_the_s1_alias_still_exports_everything() -> None:
    import s1

    for name in s1ops.__all__:
        assert getattr(s1, name) is getattr(s1ops, name), name


# --- calibrated policies need their own fit ----------------------------------------------------------


def test_a_calibrated_noul_policy_without_its_fit_is_refused() -> None:
    # Before: routed the raw 0.1 as BLOCK. The fit would have made it about 0.33, ESCALATE.
    for fits in ([], [_fit(question_id="q-other")], [_fit(model_served="other")]):
        gate = DecisionGate([_policy(_fit())], fits)
        with pytest.raises(ValueError, match="tuned on calibrated probabilities"):
            gate.evaluate_noul(question_id=QID, model_served=MODEL, p_raw=0.1)


def test_a_raw_policy_needs_no_fit() -> None:
    gate = DecisionGate([_policy(calibrated=False)], [])
    assert gate.evaluate_noul(question_id=QID, model_served=MODEL, p_raw=0.1).action == "BLOCK"


def test_a_substituted_fit_for_the_same_group_is_refused() -> None:
    tuned, swapped = _fit(params={"temperature": 4.0}), _fit(params={"temperature": 1.0})
    good = DecisionGate([_policy(tuned)], [tuned]).evaluate_noul(question_id=QID, model_served=MODEL, p_raw=0.1)
    assert good.action == "ESCALATE"  # calibrated about 0.366
    with pytest.raises(PolicyIntegrityError, match="tuned through fit"):
        DecisionGate([_policy(tuned)], [swapped]).evaluate_noul(question_id=QID, model_served=MODEL, p_raw=0.1)


def test_a_policy_without_a_recorded_fit_hash_takes_any_fit() -> None:
    gate = DecisionGate([_policy(None)], [_fit(params={"temperature": 1.0})])
    assert gate.evaluate_noul(question_id=QID, model_served=MODEL, p_raw=0.1).action == "BLOCK"


def test_the_gate_cli_refuses_a_calibrated_policy_without_fits(tmp_path: Path, capsys) -> None:
    save_policies([_policy(_fit())], tmp_path / "policy.json")
    rc = cli_main(["gate", str(tmp_path / "policy.json"), "--question", QID, "--model", MODEL, "--p", "0.1"])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 1 and len(err) == 1 and err[0].startswith("s1: ") and "fits.json" in err[0]


# --- fit_sha256 ------------------------------------------------------------------------------------


def test_the_fit_hash_is_pinned_on_fixed_inputs() -> None:
    canonical = (
        '{"classes":[false,true],"method":"temperature","model_served":"kev-latest",'
        '"params":{"temperature":2.5},"primitive":"noul","question_id":"q-0000000000000001"}'
    )
    fit = _fit()
    assert fit.fit_sha256 == hashlib.sha256(canonical.encode()).hexdigest()[:16] == "7dc60e8aae145b20"
    assert _policy(fit).policy_sha256 == "01fdf075db167061"


def test_the_fit_hash_follows_the_map_not_the_bookkeeping() -> None:
    fit = _fit()
    for change in (dict(fitted_at="later"), dict(items=[["cal", "i3"], ["cal", "i4"]]), dict(n=3), dict(nll_after=0.5), dict(split="x")):
        assert dataclasses.replace(fit, **change).fit_sha256 == fit.fit_sha256, change
    for change in (dict(params={"temperature": 2.6}), dict(method="platt", params=PARAMS["platt"]), dict(model_served="m2")):
        assert dataclasses.replace(fit, **change).fit_sha256 != fit.fit_sha256, change


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("light") / "ws")


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_apply_and_tune_carry_the_fit_hash_into_the_policy(demo, tmp_path: Path) -> None:
    (fit,) = load_fits(demo["fits"])
    applied = tmp_path / "applied.jsonl"
    assert cli_main(["apply", demo["records"], "--fit", demo["fits"], "-o", str(applied)]) == 0
    assert {r["calib"]["fit_sha256"] for r in _rows(str(applied)) if "calib" in r} == {fit.fit_sha256}
    assert cli_main(["policy", "tune", str(applied), "-o", str(tmp_path / "p.json"), "--use-calib"]) == 0
    (pol,) = load_policies(tmp_path / "p.json")
    assert pol.fit_sha256 == fit.fit_sha256
    assert cli_main(["policy", "tune", str(applied), "-o", str(tmp_path / "raw.json")]) == 0
    assert load_policies(tmp_path / "raw.json")[0].fit_sha256 is None


def _tune(tmp_path: Path, rows: list[dict], capsys) -> tuple[int, str]:
    rc = cli_main(["policy", "tune", _write(tmp_path / "r.jsonl", rows), "-o", str(tmp_path / "p.json"), "--use-calib"])
    return rc, capsys.readouterr().err


def _labeled(demo) -> list[dict]:
    return [r for r in _rows(demo["records"]) if r.get("label") is not None and "calib" in r]


def test_tune_refuses_records_calibrated_through_different_fits(demo, tmp_path: Path, capsys) -> None:
    rows = _labeled(demo)
    rows[3]["calib"]["fit_sha256"] = "0000000000000000"
    rc, err = _tune(tmp_path, rows, capsys)
    assert rc == 1 and "different fits" in err


def test_tune_refuses_a_record_tuned_on_its_raw_probability(demo, tmp_path: Path, capsys) -> None:
    for drop in ("p", None):  # calib with a hash but no calibrated p; no calib at all
        rows = _labeled(demo)
        if drop:
            del rows[5]["calib"][drop]
        else:
            del rows[5]["calib"]
        rc, err = _tune(tmp_path, rows, capsys)
        assert rc == 1 and "has no calibrated probabilities" in err, drop


def test_tune_refuses_a_hash_that_is_not_a_string(demo, tmp_path: Path, capsys) -> None:
    rows = _labeled(demo)
    rows[0]["calib"]["fit_sha256"] = 7
    rc, err = _tune(tmp_path, rows, capsys)
    assert rc == 1 and "must be a string" in err


def test_tune_keeps_a_null_hash_for_records_from_before_it(demo, tmp_path: Path, capsys) -> None:
    rows = _labeled(demo)
    for r in rows:
        del r["calib"]["fit_sha256"]
    rc, _ = _tune(tmp_path, rows, capsys)
    assert rc == 0 and load_policies(tmp_path / "p.json")[0].fit_sha256 is None


def test_the_demo_policies_record_their_fits(demo) -> None:
    for policy, fits in (("policy", "fits"), ("policy_categorical", "fits_categorical")):
        by_key = {f.key: f for f in load_fits(demo[fits])}
        for pol in load_policies(demo[policy]):
            assert pol.fit_sha256 == by_key[(pol.question_id, pol.model_served, pol.primitive)].fit_sha256


def test_apply_writes_the_hash_of_the_fit_it_applied() -> None:
    from s1ops.contracts.records import parse_record

    row = {"schema": "s1-record/1", "item_id": "i", "question_id": QID, "primitive": "noul",
           "model_requested": MODEL, "model_served": MODEL, "p": 0.3}
    fit = _fit("platt")
    assert apply_to_record(fit, parse_record(row))["calib"]["fit_sha256"] == fit.fit_sha256


# --- policy files ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(mode="escalation_band", band_lo=2.0, band_hi=3.0), "0 <= band_lo <= band_hi <= 1"),
        (dict(mode="escalation_band", band_lo=0.8, band_hi=0.2), "0 <= band_lo <= band_hi <= 1"),
        (dict(mode="escalation_band", band_lo=-0.1, band_hi=0.5), "0 <= band_lo <= band_hi <= 1"),
        (dict(mode="escalation_band", band_lo=0.2), "needs both band_lo and band_hi"),
        (dict(mode="escalation_band", band_hi=0.8), "needs both band_lo and band_hi"),
        (dict(mode="escalation_band"), "needs both band_lo and band_hi"),
        (dict(band_lo=0.2, band_hi=0.8), "has no band"),
        (dict(band_hi=0.8), "has no band"),
    ],
)
def test_a_band_that_routing_would_misread_is_refused(tmp_path: Path, change: dict, message: str) -> None:
    save_policies([_policy(None, **change)], tmp_path / "policy.json")
    with pytest.raises(ValueError, match=rf"policy\.json: policies\.0: .*{message}"):
        load_policies(tmp_path / "policy.json")


@pytest.mark.parametrize(("lo", "hi"), [(0.0, 1.0), (0.3, 0.3), (0.25, 0.75)])
def test_a_band_inside_the_unit_interval_loads(tmp_path: Path, lo: float, hi: float) -> None:
    save_policies([_policy(None, mode="escalation_band", band_lo=lo, band_hi=hi)], tmp_path / "policy.json")
    assert load_policies(tmp_path / "policy.json")[0].band_lo == lo


def test_two_policies_for_one_group_are_refused_in_either_order(tmp_path: Path) -> None:
    strict, loose = _policy(None, threshold=0.95), _policy(None, threshold=0.5)
    for pair in ([strict, loose], [loose, strict]):
        save_policies(pair, tmp_path / "policy.json")
        with pytest.raises(ValueError, match="more than one policy"):
            load_policies(tmp_path / "policy.json")
        with pytest.raises(ValueError, match="more than one policy"):
            DecisionGate(pair, [])


def test_two_fits_for_one_group_are_refused_by_the_gate() -> None:
    with pytest.raises(ValueError, match="more than one fit"):
        DecisionGate([_policy(None)], [_fit(), _fit(params={"temperature": 1.0})])
