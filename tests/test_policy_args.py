"""`policy tune` / `policy curve` numbers: out-of-range arguments are refused, not silently mapped to another policy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from s1ops.improve.policy import Band, Prices, choose, evaluate, select_threshold, threshold_curve


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _error(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("s1: ") and err.count("\n") == 1
    assert "Traceback" not in err
    return err.rstrip("\n")


_CONF = [0.95, 0.9, 0.8, 0.7, 0.6]
_CORRECT = [True, True, False, True, False]


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf])
def test_select_threshold_refuses_alpha_outside_the_open_unit_interval(alpha: float) -> None:
    # #36: alpha 1 accepted everything, NaN accepted nothing.
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        select_threshold(_CONF, _CORRECT, budget=0.2, use_cp_upper=True, alpha=alpha)


def test_select_threshold_ignores_alpha_without_the_upper_bound() -> None:
    assert select_threshold(_CONF, _CORRECT, budget=0.2, alpha=1.0) == select_threshold(_CONF, _CORRECT, budget=0.2)


@pytest.mark.parametrize("name", ["c_s1", "c_delegate", "c_error"])
@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf, True])
def test_threshold_curve_refuses_a_bad_cost(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a finite non-negative number"):
        threshold_curve(_CONF, _CORRECT, **{name: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan])
def test_threshold_curve_refuses_a_bad_delegate_accuracy(value: float) -> None:
    with pytest.raises(ValueError, match=r"delegate_accuracy must be a finite number in \[0, 1\]"):
        threshold_curve(_CONF, _CORRECT, delegate_accuracy=value)


def test_threshold_curve_accepts_zero_costs() -> None:
    points = threshold_curve(_CONF, _CORRECT, c_s1=0.0, c_delegate=0.0, c_error=0.0)
    assert [p["cost_per_1k_usd"] for p in points] == [0.0] * len(points)


@pytest.mark.parametrize(("field", "value"), [("jev_usd", -1.0), ("llm_usd", math.nan)])
def test_prices_refuse_a_bad_cost(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a finite non-negative number"):
        Prices(**{field: value})


@pytest.mark.parametrize("cap", [-1.0, 1.5, math.nan])
def test_choose_refuses_an_escalation_cap_outside_the_unit_interval(cap: float) -> None:
    from s1ops.improve.policy import DevRecord

    recs = [DevRecord(id=str(i), p_jev=p, label=p > 0.5, llm=p > 0.5) for i, p in enumerate([0.1, 0.4, 0.6, 0.9])]
    results = [evaluate(recs, Band(0.3, 0.7), Prices())]
    with pytest.raises(ValueError, match=r"max_escalation_rate must be a finite number in \[0, 1\]"):
        choose(results, max_escalation_rate=cap)
    assert choose(results, max_escalation_rate=None) is not None


@pytest.mark.parametrize("value", ["-1", "2", "nan"])
@pytest.mark.parametrize("records", ["records", "records_categorical"])
def test_tune_refuses_max_escalation_outside_the_unit_interval(ws, tmp_path: Path, capsys, value: str, records: str) -> None:
    # A categorical-only selection never reaches choose(), so the CLI checks before the group loop.
    args = ["policy", "tune", ws[records], "-o", str(tmp_path / "p.json"), "--use-calib", "--max-escalation", value]
    assert cli_main(args) == 1
    assert _error(capsys) == f"s1: --max-escalation must be a number in [0, 1], got {float(value)!r}"
    assert not (tmp_path / "p.json").exists()


@pytest.mark.parametrize("value", ["0", "1", "nan"])
def test_tune_refuses_cp_alpha_only_with_the_upper_bound(ws, tmp_path: Path, capsys, value: str) -> None:
    base = ["policy", "tune", ws["records"], "-o", str(tmp_path / "p.json"), "--use-calib", "--cp-alpha", value]
    assert cli_main([*base, "--use-cp-upper"]) == 1
    assert _error(capsys) == f"s1: --cp-alpha must be in (0, 1), got {float(value)!r}"
    assert cli_main(base) == 0
    capsys.readouterr()


@pytest.mark.parametrize("sub", ["tune", "curve"])
@pytest.mark.parametrize("records", ["records", "records_categorical"])
@pytest.mark.parametrize(("flag", "value"), [("--cost-s1", "-5"), ("--cost-sys2", "nan")])
def test_costs_are_checked_for_both_subcommands_and_every_primitive(
    ws, tmp_path: Path, capsys, sub: str, records: str, flag: str, value: str
) -> None:
    args = ["policy", sub, ws[records], "--use-calib", flag, value]
    if sub == "tune":
        args += ["-o", str(tmp_path / "p.json")]
    assert cli_main(args) == 1
    assert _error(capsys) == f"s1: {flag} must be a finite non-negative number, got {float(value)!r}"


def test_curve_with_no_matching_group_exits_1_like_tune(ws, tmp_path: Path, capsys) -> None:
    assert cli_main(["policy", "curve", ws["records"], "--use-calib", "--question", "q-nope"]) == 1
    curve_err = _error(capsys)
    assert cli_main(["policy", "tune", ws["records"], "-o", str(tmp_path / "p.json"), "--use-calib", "--question", "q-nope"]) == 1
    assert _error(capsys) == curve_err
    assert curve_err.startswith("s1: no labeled groups matching --question q-nope in ")
