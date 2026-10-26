"""One routing rule for every consumer of a compiled `s1-policy/1` entry.

`DecisionGate`, the escalation rate in `s1 drift` and the Langfuse / Phoenix exporters all decide
through `route`, so a policy means the same thing at runtime and in every report. The policy also
chooses what is routed: the calibrated distribution exactly when it was tuned on one
(`policy.calibrated`). The gate calibrates a live answer with its fit; the reports route a record
through `route_record`, which takes the record's stored `calib`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from s1ops.contracts.records import Record, has_calib, parse_record
from s1ops.contracts.schemas import PolicyEntry

ROUTING_RULE = "3"
"""Exported with every policy-routed record, so exports made under different rules can be told
apart: their `policy_sha256` is the same, but `policy_action` and `predicted` can differ.

- none: before the rule was shared, the exporters and drift each applied a policy their own way.
- "2": one rule, but reports routed `calib` whenever a record carried it, and drift looked the
  policy up by `question_id` alone.
- "3": reports route the distribution the policy was tuned on (`policy.calibrated`), refuse a
  calibrated policy for a record without `calib`, and drift routes each record under the policy
  of its own `(question_id, model_served, primitive)` group. A policy-routed choice / score export
  reports `p_effective` and `brier` of the renormalized distribution it routed, as the gate does.
"""


class RoutingInputError(ValueError):
    """A record cannot be routed the way its policy was tuned. Carries the 1-based line and, from
    `s1 drift`, the side (`"reference"` or `"current"`) the record came from."""

    def __init__(self, line: int, message: str, side: str | None = None) -> None:
        self.line, self.message, self.side = line, message, side
        super().__init__(f"{side + ' records: ' if side else ''}line {line}: {message}")

    def with_side(self, side: str) -> RoutingInputError:
        return RoutingInputError(self.line, self.message, side)


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


def policy_distribution(policy: PolicyEntry, record: Record) -> tuple[tuple, tuple[float, ...]]:
    """The classes and distribution `policy` was tuned on: the record's `calib` exactly when
    `policy.calibrated`, as the gate routes.

    Re-parsed from `record.raw` with the default `sum_tol`, so the result does not depend on how the
    caller parsed the record (one parsed with a looser tolerance can be refused here).
    """
    if policy.calibrated and not has_calib(record.raw, policy.primitive):
        raise RoutingInputError(
            record.line,
            f"policy for {policy.question_id} ({policy.model_served}) was tuned on calibrated "
            "probabilities and this record has none; run `s1 apply` first "
            "(`--pass-through-unfitted` keeps records of models it has no fit for)",
        )
    parsed = parse_record(record.raw, record.line, use_calib=policy.calibrated)
    return parsed.classes, parsed.probs


def route_record(policy: PolicyEntry, record: Record) -> Routed:
    """Route a decision record under `policy`, on the distribution the policy was tuned on."""
    return route(policy, *policy_distribution(policy, record))


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
