"""Policy and receipt contracts (`s1-policy/1` and `s1-receipt/1`)."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s1ops.contracts._validation import pydantic_errors

POLICY_SCHEMA = "s1-policy/1"
RECEIPT_SCHEMA = "s1-receipt/1"


def _reject_constant(name: str) -> float:
    raise ValueError(f"non-finite number {name} is not valid JSON")


@dataclass(frozen=True)
class PolicyEntry:
    question_id: str
    model_served: str
    primitive: str
    mode: str
    """`'error_budget'` (single threshold `>=` on max probability) or `'escalation_band'` (`[lo, hi)` on noul `p`)."""
    split: str
    n: int
    threshold: float | None = None
    """For `'error_budget'`: `None` means accept nothing (delegate all)."""
    error_budget: float | None = None
    band_lo: float | None = None
    band_hi: float | None = None
    calibrated: bool = False
    fit_sha256: str | None = None
    source_sha256: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    metrics_on_split: dict = field(default_factory=dict)

    @property
    def policy_sha256(self) -> str:
        core = {
            "question_id": self.question_id,
            "model_served": self.model_served,
            "primitive": self.primitive,
            "mode": self.mode,
            "threshold": self.threshold,
            "error_budget": self.error_budget,
            "band_lo": self.band_lo,
            "band_hi": self.band_hi,
            "calibrated": self.calibrated,
            "fit_sha256": self.fit_sha256,
        }
        canonical = json.dumps(core, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["policy_sha256"] = self.policy_sha256
        return d


def save_policies(entries: list[PolicyEntry], path: str | Path) -> None:
    doc = {"schema": POLICY_SCHEMA, "policies": [e.to_dict() for e in entries]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


class _PolicyRow(BaseModel):
    """On-disk shape of one `PolicyEntry`; strict so a JSON string or bool never passes as a number."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    question_id: str
    model_served: str
    primitive: str
    mode: Literal["error_budget", "escalation_band"]
    split: str
    n: int
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    error_budget: float | None = None
    band_lo: float | None = None
    band_hi: float | None = None
    calibrated: bool = False
    fit_sha256: str | None = None
    source_sha256: str = ""
    created_at: str | None = None
    metrics_on_split: dict = Field(default_factory=dict)
    policy_sha256: str | None = None
    """Written by `save_policies`; recomputed from the fields, not compared."""


class _PolicyDocument(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    policies: list[_PolicyRow] = Field(default_factory=list)


def load_policies(path: str | Path) -> list[PolicyEntry]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f, parse_constant=_reject_constant)
    if not isinstance(doc, dict) or doc.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"{path}: not a {POLICY_SCHEMA} file")
    try:
        _PolicyDocument.model_validate(doc)
    except ValidationError as e:
        raise ValueError(f"{path}: " + "; ".join(pydantic_errors(e))) from None
    rows = doc.get("policies", [])
    for i, row in enumerate(rows):
        problem = _band_problem(row)
        if problem:
            raise ValueError(f"{path}: policies.{i}: {problem}")
    keys = [(row["question_id"], row["model_served"], row["primitive"]) for row in rows]
    repeated = sorted(k for k, n in Counter(keys).items() if n > 1)
    if repeated:
        raise ValueError(f"{path}: more than one policy for {repeated}")
    # Build from the file's own values: the validated model turns 1 into 1.0, which changes policy_sha256.
    return [PolicyEntry(**{k: v for k, v in row.items() if k != "policy_sha256"}) for row in rows]


def _band_problem(row: dict) -> str | None:
    """Routing reads the band only when both bounds are set (`routing._route_noul`), so a band
    missing a bound would silently route by threshold; one outside [0, 1] or reversed routes
    probabilities in ways no tuned band produces (`[2, 3)` blocks `p = 1`)."""
    lo, hi = row.get("band_lo"), row.get("band_hi")
    if row["mode"] == "error_budget":
        if lo is not None or hi is not None:
            return "an error_budget policy has no band; band_lo and band_hi must be null"
        return None
    if lo is None or hi is None:
        return "an escalation_band policy needs both band_lo and band_hi"
    if not 0.0 <= lo <= hi <= 1.0:
        return f"band_lo and band_hi must satisfy 0 <= band_lo <= band_hi <= 1, got [{lo!r}, {hi!r})"
    return None
