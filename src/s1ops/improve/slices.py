"""Slice answerability pre-flight (Wilson CI vs null/blind) and BH-FDR confident-error slice mining."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from scipy.stats import fisher_exact


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n > 0")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class SliceReport:
    slice: str
    n: int
    accuracy: float
    ci: tuple[float, float]
    null_accuracy: float
    null_ci: tuple[float, float]
    null_kind: str
    answerable: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _accuracy(items: list[Mapping], key: str) -> tuple[int, int]:
    hits = sum(1 for it in items if str(it[key]) == str(it["label"]))
    return hits, len(items)


def slice_report(items: Iterable[Mapping], min_n: int = 20) -> list[SliceReport]:
    """Per-slice accuracy with Wilson intervals against majority class or blind masked-state probe."""
    groups: dict[str, list[Mapping]] = {}
    for it in items:
        groups.setdefault(str(it["slice"]), []).append(it)
    out = []
    for name in sorted(groups):
        rows = groups[name]
        hits, n = _accuracy(rows, "predicted")
        acc, ci = hits / n, wilson(hits, n)
        if all(r.get("blind_predicted") is not None for r in rows):
            bh, _ = _accuracy(rows, "blind_predicted")
            null_acc, null_ci, kind = bh / n, wilson(bh, n), "blind"
        else:
            counts: dict[str, int] = {}
            for r in rows:
                counts[str(r["label"])] = counts.get(str(r["label"]), 0) + 1
            top = max(counts.values())
            null_acc, null_ci, kind = top / n, wilson(top, n), "majority"
        answerable = n >= min_n and ci[0] > null_ci[1]
        out.append(SliceReport(name, n, acc, ci, null_acc, null_ci, kind, answerable))
    return out


def overall(items: Iterable[Mapping]) -> tuple[float, tuple[float, float]]:
    rows = list(items)
    hits, n = _accuracy(rows, "predicted")
    return hits / n, wilson(hits, n)


@dataclass(frozen=True)
class ErrorSliceFinding:
    feature: str
    slice_n: int
    slice_errors: int
    slice_error_rate: float
    rest_n: int
    rest_errors: int
    rest_error_rate: float
    odds_ratio: float
    p_value: float
    q_value: float
    significant: bool

    def as_dict(self) -> dict:
        return asdict(self)


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR q-values (monotone, capped at 1.0)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [1.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, pvalues[idx] * m / (rank + 1))
        q[idx] = running
    return q


def mine_error_slices(
    rows: Iterable[Mapping],
    *,
    min_support: int = 20,
    fdr_q: float = 0.10,
) -> list[ErrorSliceFinding]:
    """Mine pre-declared boolean/categorical code features where errors concentrate (Fisher + BH-FDR).

    Each row in `rows` must have `error: bool` (or `label` != `predicted`) and `features: dict[str, bool | str]`.
    """
    items = list(rows)
    total_n = len(items)
    if total_n == 0:
        return []
    errors = [
        bool(r["error"]) if "error" in r else (str(r.get("predicted")) != str(r.get("label")))
        for r in items
    ]
    total_err = sum(errors)

    feature_names: set[str] = set()
    normalized_feats: list[set[str]] = []
    for r in items:
        feats = r.get("features") or r.get("slice_features") or {}
        active: set[str] = set()
        for k, v in feats.items():
            if isinstance(v, bool):
                if v:
                    active.add(str(k))
            elif v is not None:
                active.add(f"{k}={v}")
        feature_names.update(active)
        normalized_feats.append(active)

    candidates: list[tuple[str, int, int, int, int, float, float]] = []
    for fname in sorted(feature_names):
        in_slice = [i for i, fset in enumerate(normalized_feats) if fname in fset]
        s_n = len(in_slice)
        if s_n < min_support or (total_n - s_n) < 1:
            continue
        s_err = sum(errors[i] for i in in_slice)
        s_ok = s_n - s_err
        r_n = total_n - s_n
        r_err = total_err - s_err
        r_ok = r_n - r_err
        odds, p_val = fisher_exact([[s_err, s_ok], [r_err, r_ok]], alternative="greater")
        candidates.append((fname, s_n, s_err, r_n, r_err, float(odds), float(p_val)))

    q_vals = benjamini_hochberg([c[6] for c in candidates])
    findings = [
        ErrorSliceFinding(
            feature=c[0],
            slice_n=c[1],
            slice_errors=c[2],
            slice_error_rate=c[2] / c[1],
            rest_n=c[3],
            rest_errors=c[4],
            rest_error_rate=c[4] / c[3] if c[3] else 0.0,
            odds_ratio=c[5],
            p_value=c[6],
            q_value=q,
            significant=q <= fdr_q,
        )
        for c, q in zip(candidates, q_vals)
    ]
    findings.sort(key=lambda f: (not f.significant, f.q_value, -f.slice_error_rate))
    return findings
