"""Zero-overhead runtime policy & calibration gate (`DecisionGate`).

Loads compiled `fits.json` (`jev-calib-fit/2` / `s1-calib-fit/1`) and `policy.json`
(`s1-policy/1`) artifacts, enforces `model_served` at runtime, checks `policy_sha256`
against a caller-supplied expected fingerprint when one is given, applies post-hoc calibration (`temperature` / `platt` / `isotonic`), and executes
`error_budget` or `escalation_band` routing in sub-millisecond time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from s1ops.contracts.records import GroupKey
from s1ops.contracts.schemas import PolicyEntry, load_policies
from s1ops.improve.calibrate import Fit, load_fits, transform


class ModelServedMismatchError(RuntimeError):
    """Raised when the live `model_served` does not match the compiled policy/fit."""


class PolicyIntegrityError(RuntimeError):
    """Raised when `expected_policy_sha256` does not match the loaded policy."""


@dataclass(frozen=True)
class GateDecision:
    question_id: str
    model_served: str
    primitive: str
    action: str
    """`'ALLOW'`, `'BLOCK'`, `'ESCALATE'`, or `'SELECT:<label>'`."""
    escalated: bool
    predicted: bool | str | int
    p_raw: float
    p_calibrated: float | None
    p_effective: float
    policy_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionGate:
    """In-memory runtime gate that applies `fits.json` and `policy.json` to live System 1 outputs."""

    def __init__(
        self,
        policies: list[PolicyEntry],
        fits: list[Fit] | None = None,
    ) -> None:
        self._policies: dict[GroupKey, PolicyEntry] = {
            (p.question_id, p.model_served, p.primitive): p for p in policies
        }
        self._policies_by_qid: dict[str, list[PolicyEntry]] = {}
        for p in policies:
            self._policies_by_qid.setdefault(p.question_id, []).append(p)

        self._fits: dict[GroupKey, Fit] = {
            f.key: f for f in (fits or [])
        }

    @classmethod
    def from_files(
        cls,
        policy_path: str | Path,
        fits_path: str | Path | None = None,
    ) -> "DecisionGate":
        policies = load_policies(policy_path)
        fits = load_fits(fits_path) if fits_path is not None else []
        return cls(policies=policies, fits=fits)

    def evaluate_noul(
        self,
        *,
        question_id: str,
        model_served: str,
        p_raw: float,
        expected_policy_sha256: str | None = None,
    ) -> GateDecision:
        key: GroupKey = (question_id, model_served, "noul")
        pol = self._policies.get(key)
        if pol is None:
            same_q = self._policies_by_qid.get(question_id, [])
            if same_q:
                expected_models = sorted({x.model_served for x in same_q})
                raise ModelServedMismatchError(
                    f"Policy for {question_id} was compiled for model_served={expected_models}, "
                    f"but runtime received model_served={model_served!r}"
                )
            raise KeyError(f"No policy found for group {key}")

        if expected_policy_sha256 is not None and pol.policy_sha256 != expected_policy_sha256:
            raise PolicyIntegrityError(
                f"Policy SHA-256 mismatch for {question_id}: "
                f"expected {expected_policy_sha256!r}, got {pol.policy_sha256!r}"
            )

        p_cal: float | None = None
        fit = self._fits.get(key)
        if fit is not None:
            P = np.array([[1.0 - float(p_raw), float(p_raw)]], dtype=float)
            Q = transform(fit.method, "noul", fit.params, P)
            p_cal = float(Q[0, 1])

        p_eff = p_cal if (pol.calibrated and p_cal is not None) else float(p_raw)

        if pol.mode == "escalation_band" and pol.band_lo is not None and pol.band_hi is not None:
            if p_eff >= pol.band_hi:
                action = "ALLOW"
                escalated = False
                pred = True
            elif p_eff < pol.band_lo:
                action = "BLOCK"
                escalated = False
                pred = False
            else:
                action = "ESCALATE"
                escalated = True
                pred = p_eff >= 0.5
        else:
            tau = pol.threshold
            conf = max(p_eff, 1.0 - p_eff)
            pred = p_eff >= 0.5
            if tau is None or conf < tau:
                action = "ESCALATE"
                escalated = True
            else:
                action = "ALLOW" if pred else "BLOCK"
                escalated = False

        return GateDecision(
            question_id=question_id,
            model_served=model_served,
            primitive="noul",
            action=action,
            escalated=escalated,
            predicted=pred,
            p_raw=round(float(p_raw), 6),
            p_calibrated=round(p_cal, 6) if p_cal is not None else None,
            p_effective=round(p_eff, 6),
            policy_sha256=pol.policy_sha256,
        )
