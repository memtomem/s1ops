"""Slice answerability pre-flight (Wilson CI vs null/blind) and BH-FDR confident-error slice mining."""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from scipy.stats import fisher_exact

from s1ops.contracts.records import GroupKey, Record, reparse


FeatureKey = tuple[str | None, str]
"""`(dataset, item_id)` as records key an item: no `dataset` is `None`, `item_id` is a string."""

_FEATURE_VALUE = (str, int, float, bool)


def parse_feature_rows(rows: Sequence[object]) -> tuple[dict[FeatureKey, dict], dict[FeatureKey, str]]:
    """Check `features.jsonl` rows and index them by item: `(features_by_item, slice_by_item)`.

    A row is `{"dataset"?, "item_id", "slice"?, "features"?}`. `item_id` is a non-empty string or
    an integer (not a boolean); `dataset` is absent, `null` or a non-empty string; `slice` and
    `features.slice` are strings; `features` is an object of strings, numbers, booleans or nulls
    (`mine_error_slices` turns each value into a slice name, so a list or object has no stable
    one); no item appears twice. A row's own `slice` wins over `features.slice`. Errors name the
    1-based row and the field, never a value.
    """
    features_by_item: dict[FeatureKey, dict] = {}
    slice_by_item: dict[FeatureKey, str] = {}
    seen: dict[FeatureKey, int] = {}
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {i} is not an object")
        if "item_id" not in row:
            raise ValueError(f"row {i} is missing 'item_id'")
        item_id = row["item_id"]
        if isinstance(item_id, bool) or not isinstance(item_id, (str, int)) or item_id == "":
            raise ValueError(f"row {i}: 'item_id' must be a non-empty string or an integer, got {type(item_id).__name__}")
        dataset = row.get("dataset")
        if dataset is not None and (not isinstance(dataset, str) or not dataset):
            raise ValueError(f"row {i}: 'dataset' must be a non-empty string or null, got {type(dataset).__name__}")
        key = (dataset, str(item_id))
        if key in seen:
            raise ValueError(f"row {i} is the same item as row {seen[key]}")
        seen[key] = i
        feats = row.get("features", {})
        if not isinstance(feats, Mapping):
            raise ValueError(f"row {i}: 'features' must be an object, got {type(feats).__name__}")
        for name, value in feats.items():
            if value is not None and not isinstance(value, _FEATURE_VALUE):
                raise ValueError(
                    f"row {i}: each 'features' value must be a string, number, boolean or null, got {type(value).__name__}"
                )
        for where, holder in (("features.slice", feats), ("slice", row)):
            if "slice" in holder:
                if not isinstance(holder["slice"], str):
                    raise ValueError(f"row {i}: '{where}' must be a string, got {type(holder['slice']).__name__}")
                slice_by_item[key] = holder["slice"]
        features_by_item[key] = dict(feats)
    return features_by_item, slice_by_item


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
        """`odds_ratio` is `None` when it is not finite: Fisher's ratio is `inf` when its denominator
        is zero (e.g. no error outside the feature) and `nan` when both products are (e.g. a group
        with no errors). Neither is valid JSON."""
        d = asdict(self)
        if not math.isfinite(d["odds_ratio"]):
            d["odds_ratio"] = None
        return d


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


@dataclass(frozen=True)
class GroupSliceInputs:
    """One `(question_id, model_served, primitive)` group's inputs to `slice_report` (`items`) and
    `mine_error_slices` (`rows`), one entry per record, in input order."""

    key: GroupKey
    items: list[dict]
    rows: list[dict]

    def header(self) -> dict:
        question_id, model_served, primitive = self.key
        return {"question_id": question_id, "model_served": model_served, "primitive": primitive, "n": len(self.items)}


def slice_inputs_by_group(
    records: Iterable[Record],
    features_by_item: Mapping[FeatureKey, Mapping],
    slice_by_item: Mapping[FeatureKey, str],
    *,
    question_id: str | None = None,
    model_served: str | None = None,
    raw: bool = False,
) -> list[GroupSliceInputs]:
    """Split the labeled, answered records into their groups, in sorted key order, so slice
    statistics are never pooled across groups. Call `select(records, allow_model_mismatch=True)`
    first: it refuses a repeated item or differing classes within a group, which this does not.

    Each record keeps its own prediction: the first of its own classes with the top probability
    (a group may list a choice's keys in different orders). A record's slice is its item's
    (`(dataset, item_id)`, then `(None, item_id)`), else `"default"`; its features likewise, else none.

    With `raw`, each kept record predicts from its raw distribution (`reparse`), whatever it was
    read with, so records read with `use_calib=True` (which validates a stored `calib`) can still be
    sliced raw. Without it, each predicts from the distribution it holds.
    """
    by_key: dict[GroupKey, tuple[list[dict], list[dict]]] = defaultdict(lambda: ([], []))
    for r in records:
        if r.model_served is None or r.label is None:
            continue
        if (question_id is not None and r.question_id != question_id) or (
            model_served is not None and r.model_served != model_served
        ):
            continue
        if raw:
            r = reparse(r)
        predicted = r.classes[max(range(len(r.probs)), key=lambda i: r.probs[i])]
        items, rows = by_key[r.key]
        sl = slice_by_item.get(r.item) or slice_by_item.get((None, r.item_id)) or "default"
        feats = features_by_item.get(r.item) or features_by_item.get((None, r.item_id)) or {}
        items.append({"slice": sl, "predicted": predicted, "label": r.label})
        rows.append({"error": predicted != r.label, "features": feats})
    return [GroupSliceInputs(key, *by_key[key]) for key in sorted(by_key)]
