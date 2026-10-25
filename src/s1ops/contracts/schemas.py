"""Policy and receipt contracts (`s1-policy/1` and `s1-receipt/1`)."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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


def load_policies(path: str | Path) -> list[PolicyEntry]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f, parse_constant=_reject_constant)
    if not isinstance(doc, dict) or doc.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"{path}: not a {POLICY_SCHEMA} file")
    out: list[PolicyEntry] = []
    for row in doc.get("policies", []):
        clean = {k: v for k, v in row.items() if k != "policy_sha256"}
        entry = PolicyEntry(**clean)
        if entry.mode not in ("error_budget", "escalation_band"):
            raise ValueError(f"{path}: unsupported policy mode {entry.mode!r}")
        if entry.threshold is not None and not (math.isfinite(entry.threshold) and 0.0 <= entry.threshold <= 1.0):
            raise ValueError(f"{path}: threshold must be in [0, 1] or null")
        out.append(entry)
    return out
