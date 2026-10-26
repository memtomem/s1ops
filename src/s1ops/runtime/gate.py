"""Zero-overhead runtime policy & calibration gate (`DecisionGate`).

Loads compiled `fits.json` (`jev-calib-fit/2` / `s1-calib-fit/1`) and `policy.json`
(`s1-policy/1`) artifacts, enforces `model_served` at runtime, checks `policy_sha256`
against a caller-supplied expected fingerprint when one is given, applies post-hoc calibration (`temperature` / `platt` / `isotonic`), and routes
through the shared `route` rule (`error_budget` or `escalation_band`) in sub-millisecond time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from s1ops.contracts.records import DEFAULT_SUM_TOL, GroupKey, probability_total
from s1ops.contracts.schemas import PolicyEntry, load_policies
from s1ops.improve.calibrate import Fit, load_fits, transform
from s1ops.runtime.routing import route


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
    """Noul: `p >= 0.5` unless the band decided. Choice / score: the top label (on a tie, which
    always escalates, the first tied label: arbitrary)."""
    p_raw: float
    """Noul: raw `p`. Choice / score: the raw probability of `predicted`."""
    p_calibrated: float | None
    p_effective: float
    """The value routed on: noul `p`, or the top probability of the routed distribution."""
    policy_sha256: str | None
    probabilities: dict[str, float] | None = None
    """Choice / score: the routed distribution, keyed by label. `None` for noul."""
    expected_score: float | None = None
    """Score only: `sum(level * p)` over the routed distribution. Reported, not routed on."""

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

    def primitive_of(self, question_id: str) -> str | None:
        """The primitive the policies for `question_id` were compiled for, or `None` if none were."""
        primitives = sorted({p.primitive for p in self._policies_by_qid.get(question_id, [])})
        return primitives[0] if len(primitives) == 1 else None

    def _policy(self, key: GroupKey, expected_policy_sha256: str | None) -> PolicyEntry:
        pol = self._policies.get(key)
        if pol is None:
            question_id, model_served, primitive = key
            same_q = self._policies_by_qid.get(question_id, [])
            same_primitive = [x for x in same_q if x.primitive == primitive]
            if same_primitive:
                expected_models = sorted({x.model_served for x in same_primitive})
                raise ModelServedMismatchError(
                    f"Policy for {question_id} was compiled for model_served={expected_models}, "
                    f"but runtime received model_served={model_served!r}"
                )
            if same_q:
                compiled = sorted({x.primitive for x in same_q})
                raise ValueError(f"policy for {question_id} is for primitive {', '.join(compiled)}, not {primitive}")
            raise KeyError(f"No policy found for group {key}")

        if expected_policy_sha256 is not None and pol.policy_sha256 != expected_policy_sha256:
            raise PolicyIntegrityError(
                f"Policy SHA-256 mismatch for {key[0]}: "
                f"expected {expected_policy_sha256!r}, got {pol.policy_sha256!r}"
            )
        return pol

    def evaluate_noul(
        self,
        *,
        question_id: str,
        model_served: str,
        p_raw: float,
        expected_policy_sha256: str | None = None,
    ) -> GateDecision:
        key: GroupKey = (question_id, model_served, "noul")
        pol = self._policy(key, expected_policy_sha256)

        p_cal: float | None = None
        fit = self._fits.get(key)
        if fit is not None:
            P = np.array([[1.0 - float(p_raw), float(p_raw)]], dtype=float)
            Q = transform(fit.method, "noul", fit.params, P)
            p_cal = float(Q[0, 1])

        p_eff = p_cal if (pol.calibrated and p_cal is not None) else float(p_raw)
        routed = route(pol, (False, True), (1.0 - p_eff, p_eff))

        return GateDecision(
            question_id=question_id,
            model_served=model_served,
            primitive="noul",
            action=routed.action,
            escalated=routed.escalated,
            predicted=routed.predicted,
            p_raw=round(float(p_raw), 6),
            p_calibrated=round(p_cal, 6) if p_cal is not None else None,
            p_effective=round(p_eff, 6),
            policy_sha256=pol.policy_sha256,
        )

    def evaluate_choice(
        self,
        *,
        question_id: str,
        model_served: str,
        probs_raw: Mapping[str, float],
        expected_policy_sha256: str | None = None,
    ) -> GateDecision:
        """Route a choice answer. `probs_raw` maps each option to its probability, in option order."""
        if not isinstance(probs_raw, Mapping):
            raise ValueError(f"choice probabilities must be a mapping of option to probability, got {probs_raw!r}")
        bad = [c for c in probs_raw if not (isinstance(c, str) and c)]
        if bad:
            raise ValueError(f"choice options must be non-empty strings, got {bad!r}")
        classes = list(probs_raw)
        return self._evaluate_categorical(
            (question_id, model_served, "choice"),
            classes,
            [probs_raw[c] for c in classes],
            expected_policy_sha256,
        )

    def evaluate_score(
        self,
        *,
        question_id: str,
        model_served: str,
        probs_raw: Mapping[str | int, float],
        expected_policy_sha256: str | None = None,
    ) -> GateDecision:
        """Route a score answer. Keys are levels: non-negative integers or their decimal strings."""
        if not isinstance(probs_raw, Mapping):
            raise ValueError(f"score probabilities must be a mapping of level to probability, got {probs_raw!r}")
        levels: dict[int, float] = {}
        for k, v in probs_raw.items():
            level = _level(k)
            if level in levels:
                raise ValueError(f"score level {k!r} repeats level {level}")
            levels[level] = v
        classes = sorted(levels)
        return self._evaluate_categorical(
            (question_id, model_served, "score"),
            classes,
            [levels[c] for c in classes],
            expected_policy_sha256,
        )

    def _evaluate_categorical(
        self,
        key: GroupKey,
        classes: list,
        values: Sequence[object],
        expected_policy_sha256: str | None,
    ) -> GateDecision:
        question_id, model_served, primitive = key
        pol = self._policy(key, expected_policy_sha256)
        raw = _normalized(classes, values)

        fit = self._fits.get(key)
        if pol.calibrated and fit is None:
            raise ValueError(
                f"policy for {question_id} ({model_served}) was tuned on calibrated probabilities; "
                "pass the fits.json it was compiled with"
            )
        cal = _calibrated(fit, classes, raw) if fit is not None else None

        eff = cal if (pol.calibrated and cal is not None) else raw
        routed = route(pol, classes, eff)
        i = classes.index(routed.predicted)

        return GateDecision(
            question_id=question_id,
            model_served=model_served,
            primitive=primitive,
            action=routed.action,
            escalated=routed.escalated,
            predicted=routed.predicted,
            p_raw=round(raw[i], 6),
            p_calibrated=round(cal[i], 6) if cal is not None else None,
            p_effective=round(routed.confidence, 6),
            policy_sha256=pol.policy_sha256,
            probabilities={str(c): round(q, 6) for c, q in zip(classes, eff)},
            expected_score=(
                round(math.fsum(c * q for c, q in zip(classes, eff)), 6) if primitive == "score" else None
            ),
        )


def _calibrated(fit: Fit, classes: list, raw: list[float]) -> list[float]:
    """Calibrate a normalized distribution by the steps the tuner's input went through: `s1 apply`
    transforms it in fit order, and `parse_record(use_calib=True)` renormalizes the result."""
    if set(classes) != set(fit.classes):
        raise ValueError(
            f"{fit.question_id} ({fit.model_served}): answer classes {sorted(map(str, classes))} "
            f"differ from the fit's {sorted(map(str, fit.classes))}"
        )
    by_class = dict(zip(classes, raw))
    P = np.array([[by_class[c] for c in fit.classes]], dtype=float)
    Q = dict(zip(fit.classes, (float(q) for q in transform(fit.method, fit.primitive, fit.params, P)[0])))
    total = probability_total(Q.values())
    return [Q[c] / total for c in classes]


def _level(key: object) -> int:
    if isinstance(key, int) and not isinstance(key, bool) and key >= 0:
        return key
    if isinstance(key, str) and key.isascii() and key.isdigit():
        return int(key)
    raise ValueError(f"score levels must be non-negative integers or their decimal strings, got {key!r}")


def _normalized(classes: Sequence, values: Sequence[object]) -> list[float]:
    """Validate a distribution and renormalize it exactly as `parse_record` does."""
    if len(classes) < 2:
        raise ValueError(f"need at least two classes, got {list(classes)!r}")
    probs: list[float] = []
    for c, v in zip(classes, values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"probability for {c!r} must be a number, got {v!r}")
        try:
            finite = math.isfinite(v)
        except OverflowError:
            finite = False
        if not finite or not 0.0 <= v <= 1.0:
            raise ValueError(f"probability for {c!r} must be a finite number in [0, 1], got {v!r:.40}")
        probs.append(float(v))
    total = probability_total(probs)
    if abs(total - 1.0) > DEFAULT_SUM_TOL:
        raise ValueError(f"probabilities sum to {total:.6g}, not 1 within {DEFAULT_SUM_TOL}")
    return [q / total for q in probs]
