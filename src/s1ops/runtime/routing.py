"""One routing rule for every consumer of a compiled `s1-policy/1` entry.

`DecisionGate`, the escalation rate in `s1 drift` and the Langfuse / Phoenix exporters all decide
through `route`, so a policy means the same thing at runtime and in every report. Callers choose
which distribution to route (raw or calibrated); only the decision is shared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from s1ops.contracts.schemas import PolicyEntry

ROUTING_RULE = "2"
"""Exported with every policy-routed record. Exports made before the rule was shared (when the
exporters and drift each applied a policy their own way) carry no rule, which tells them apart:
their `policy_sha256` is the same, but `policy_action` and `predicted` can differ."""


@dataclass(frozen=True)
class Routed:
    action: str
    """`'ALLOW'`, `'BLOCK'`, `'ESCALATE'`, or `'SELECT:<label>'`."""
    escalated: bool
    predicted: bool | str | int
    """System 1's answer. On a tie at the top it is the first tied class, which is arbitrary."""
    confidence: float
    """Noul: `p`. Choice / score: the top probability, unrounded, as compared with the threshold."""


def route(policy: PolicyEntry, classes: Sequence, probs: Sequence[float]) -> Routed:
    """Route one answer under `policy`; `probs` is a normalized distribution aligned with `classes`.

    The primitive comes from the policy, never from `classes`: `(0, 1) == (False, True)` in
    Python, so a two-level score would otherwise be routed as a Noul.
    """
    probs = [float(q) for q in probs]
    if len(classes) != len(probs):
        raise ValueError(f"{len(classes)} classes but {len(probs)} probabilities")
    if policy.primitive == "noul":
        if len(classes) != 2 or not all(isinstance(c, bool) for c in classes) or list(classes) != [False, True]:
            raise ValueError(f"noul classes must be [false, true], got {list(classes)!r}")
        return _route_noul(policy, probs[1])
    if policy.primitive in ("choice", "score"):
        return _route_categorical(policy, list(classes), probs)
    raise ValueError(f"policy for {policy.question_id} has unknown primitive {policy.primitive!r}")


def _route_noul(pol: PolicyEntry, p: float) -> Routed:
    if pol.mode == "escalation_band" and pol.band_lo is not None and pol.band_hi is not None:
        if p >= pol.band_hi:
            return Routed("ALLOW", False, True, p)
        if p < pol.band_lo:
            return Routed("BLOCK", False, False, p)
        return Routed("ESCALATE", True, p >= 0.5, p)
    pred = p >= 0.5
    if pol.threshold is None or max(p, 1.0 - p) < pol.threshold:
        return Routed("ESCALATE", True, pred, p)
    return Routed("ALLOW" if pred else "BLOCK", False, pred, p)


def _route_categorical(pol: PolicyEntry, classes: list, probs: list[float]) -> Routed:
    if pol.mode != "error_budget":
        raise ValueError(
            f"policy for {pol.question_id} ({pol.model_served}) has mode {pol.mode!r}; "
            f"{pol.primitive} policies support error_budget only"
        )
    if len(classes) < 2 or len(set(map(repr, classes))) != len(classes):
        raise ValueError(f"{pol.primitive} needs at least two distinct classes, got {classes!r}")
    if pol.primitive == "choice":
        ok = all(isinstance(c, str) and c for c in classes)
    else:
        ok = all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in classes)
    if not ok:
        kind = "option names" if pol.primitive == "choice" else "non-negative integer levels"
        raise ValueError(f"{pol.primitive} classes must be {kind}, got {classes!r}")
    top = max(probs)
    label = classes[probs.index(top)]
    # The tuner breaks a tie by the first record's option order, which file order and
    # `s1 apply` (sort_keys) change, so a tie cannot be routed the way it was tuned.
    tied = probs.count(top) > 1
    if tied or pol.threshold is None or top < pol.threshold:
        return Routed("ESCALATE", True, label, top)
    return Routed(f"SELECT:{label}", False, label, top)
