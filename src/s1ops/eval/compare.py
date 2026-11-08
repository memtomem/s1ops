"""Multi-backend shadow and matrix comparison (`|Δp|`, TVD, flip rate, McNemar-Holm)."""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from s1ops.contracts.records import SCHEMAS, read_records, score_level, score_levels
from s1ops.improve.trainset import check_not_trained, load_trained

SUM_TOLERANCE = 0.02


class CompareInputError(ValueError):
    """The primary and shadow records cannot be paired or are not valid records."""


@dataclass(frozen=True)
class Disagreement:
    dataset: str | None
    item_id: str
    question_id: str
    primitive: str
    distance: float
    """Noul: `|p_a - p_b|`. Choice and Score: total variation distance between distributions."""
    flipped: bool
    """Decision flip: Noul across 0.5, Choice returned `choice`, Score argmax level."""
    primary_pred: object = None
    shadow_pred: object = None
    label: object = None


def _tvd(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in a.keys() | b.keys())


def _top(dist: dict[str, float]) -> str:
    return max(sorted(dist), key=lambda k: dist[k])


def _finite_number(x: object) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return isinstance(x, int) or math.isfinite(x)


def validate(record: dict) -> None:
    if record.get("schema") not in SCHEMAS:
        raise CompareInputError(f"not a valid record: schema={record.get('schema')!r}")
    kind = record.get("primitive")
    if kind == "noul":
        p = record.get("p")
        if not _finite_number(p) or not 0.0 <= p <= 1.0:
            raise CompareInputError(f"noul p must be a number in [0, 1], got {p!r}")
    elif kind in ("choice", "score"):
        dist = record.get("probabilities")
        if not isinstance(dist, dict) or not dist or any(not _finite_number(v) or v < 0 for v in dist.values()):
            raise CompareInputError(f"{kind} probabilities must be a non-empty map of finite non-negative numbers")
        if abs(sum(dist.values()) - 1.0) > SUM_TOLERANCE:
            raise CompareInputError(f"{kind} probabilities sum to {sum(dist.values()):.4f}, outside 1 ± {SUM_TOLERANCE}")
        if kind == "choice" and record.get("choice") not in dist:
            raise CompareInputError(f"choice {record.get('choice')!r} is not one of its probability labels")
        if kind == "score":
            try:
                score_levels(dist)
            except ValueError as e:
                raise CompareInputError(str(e)) from None
    else:
        raise CompareInputError(f"unknown primitive {kind!r}")


def compare(primary: dict, shadow: dict) -> Disagreement:
    validate(primary)
    validate(shadow)
    for key in ("dataset", "item_id", "question_id", "primitive"):
        if primary.get(key) != shadow.get(key):
            raise CompareInputError(f"{key} differs: {primary.get(key)!r} vs {shadow.get(key)!r}")
    kind = primary["primitive"]
    label = primary.get("label") if primary.get("label") is not None else shadow.get("label")
    if kind == "noul":
        a, b = float(primary["p"]), float(shadow["p"])
        pa, pb = a >= 0.5, b >= 0.5
        return Disagreement(
            primary.get("dataset"), primary["item_id"], primary["question_id"], kind, abs(a - b), pa != pb, pa, pb, label
        )
    if kind == "choice":
        a, b = primary["probabilities"], shadow["probabilities"]
        pa, pb = primary["choice"], shadow["choice"]
        return Disagreement(
            primary.get("dataset"), primary["item_id"], primary["question_id"], kind, _tvd(a, b), pa != pb, pa, pb, label
        )
    if kind == "score":
        a, b = primary["probabilities"], shadow["probabilities"]
        pa, pb = score_level(_top(a)), score_level(_top(b))
        return Disagreement(
            primary.get("dataset"), primary["item_id"], primary["question_id"], kind, _tvd(a, b), pa != pb, pa, pb, label
        )
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Summary:
    n: int
    flip_rate: float
    mean_distance: float


def summarize(disagreements: Iterable[Disagreement]) -> Summary:
    rows = list(disagreements)
    if not rows:
        return Summary(0, 0.0, 0.0)
    return Summary(len(rows), sum(d.flipped for d in rows) / len(rows), sum(d.distance for d in rows) / len(rows))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on discordant pairs."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired outcomes must have the same length")
    a_only = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    b_only = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = a_only + b_only
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / (2**n) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p": min(1.0, 2 * tail)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    out: dict[str, float] = {}
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        out[k] = running
    return out


def _one_group_per_question(pairs: list[tuple[dict, dict]]) -> None:
    """A question's report pools its paired rows, so on each side they must be one group: one
    `model_served` and one `primitive`. Pooling two models would report one model's name over both."""
    seen: dict[tuple[str, str], dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for a, b in pairs:
        for side, row in (("primary", a), ("shadow", b)):
            values = seen[(side, row["question_id"])]
            values["primitive"].add(str(row.get("primitive")))
            values["model_served"].add(str(row.get("model_served")))
    for (side, qid), values in sorted(seen.items()):
        if len(values["primitive"]) > 1:
            raise CompareInputError(
                f"{side} records answer question {qid} as more than one primitive {sorted(values['primitive'])}; "
                "a question_id names one wire question, so one primitive: the file is hand-edited or mislabeled"
            )
        if len(values["model_served"]) > 1:
            raise CompareInputError(
                f"{side} records hold more than one model_served for question {qid}: {sorted(values['model_served'])}; "
                "compare one model per file (split it by model_served)"
            )


def compare_records(
    primary_records: str | Path | Iterable[dict],
    shadow_records: str | Path | Iterable[dict],
    *,
    max_flip_rate: float = 0.03,
    max_accuracy_drop: float = 0.02,
    training_manifests: Sequence[str | Path] = (),
) -> dict:
    """Compare two sets of records paired on `(dataset, item_id, question_id)`.

    `training_manifests` refuses a compared pair in which a side's model trained on the item
    (`TrainedItemError`). Only pairs are checked: a candidate's training item that the other file
    lacks is not compared."""
    if isinstance(primary_records, (str, Path)):
        prim_rows = [r.raw for r in read_records(primary_records) if r.model_served is not None]
    else:
        prim_rows = [r for r in primary_records if r.get("model_served") is not None]
    if isinstance(shadow_records, (str, Path)):
        shad_rows = [r.raw for r in read_records(shadow_records) if r.model_served is not None]
    else:
        shad_rows = [r for r in shadow_records if r.get("model_served") is not None]

    def index(rows: list[dict]) -> dict[tuple, dict]:
        idx: dict[tuple, dict] = {}
        for r in rows:
            k = (r.get("dataset"), r["item_id"], r["question_id"])
            if k in idx:
                raise CompareInputError(f"duplicate record key {k}")
            idx[k] = r
        return idx

    idx_a, idx_b = index(prim_rows), index(shad_rows)
    shared = sorted(idx_a.keys() & idx_b.keys(), key=lambda x: (x[0] or "", x[1], x[2]))
    _one_group_per_question([(idx_a[k], idx_b[k]) for k in shared])
    if training_manifests:
        name = {
            side: str(source) if isinstance(source, (str, Path)) else f"{side} records"
            for side, source in (("primary", primary_records), ("shadow", shadow_records))
        }
        trained = load_trained(
            training_manifests,
            [r["model_served"] for r in (*prim_rows, *shad_rows)],
            f"{name['primary']} or {name['shadow']}",
        )
        for k in shared:
            for side, row in (("primary", idx_a[k]), ("shadow", idx_b[k])):
                group = (row["question_id"], str(row["model_served"]), str(row.get("primitive")))
                check_not_trained(group, [(row.get("dataset"), row["item_id"])], trained, name[side])
    by_q: dict[str, list[Disagreement]] = defaultdict(list)
    models_a: dict[str, str] = {}
    models_b: dict[str, str] = {}
    for k in shared:
        a, b = idx_a[k], idx_b[k]
        d = compare(a, b)
        by_q[d.question_id].append(d)
        models_a[d.question_id] = str(a.get("model_served"))
        models_b[d.question_id] = str(b.get("model_served"))

    raw_p: dict[str, float] = {}
    q_reports: list[dict] = []
    for qid, ds in sorted(by_q.items()):
        s = summarize(ds)
        labeled = [d for d in ds if d.label is not None]
        acc_a = acc_b = None
        mc = None
        replaceable = s.flip_rate <= max_flip_rate
        if labeled:
            ca = [d.primary_pred == d.label for d in labeled]
            cb = [d.shadow_pred == d.label for d in labeled]
            acc_a = sum(ca) / len(ca)
            acc_b = sum(cb) / len(cb)
            mc = mcnemar_exact(ca, cb)
            raw_p[qid] = mc["p"]
            replaceable = replaceable and ((acc_a - acc_b) <= max_accuracy_drop)
        q_reports.append(
            {
                "question_id": qid,
                "primitive": ds[0].primitive,
                "primary_model": models_a[qid],
                "shadow_model": models_b[qid],
                "n": s.n,
                "flip_rate": s.flip_rate,
                "mean_distance": s.mean_distance,
                "max_distance": max(d.distance for d in ds),
                "n_labeled": len(labeled),
                "primary_accuracy": acc_a,
                "shadow_accuracy": acc_b,
                "accuracy_delta": (acc_b - acc_a) if (acc_a is not None and acc_b is not None) else None,
                "mcnemar": mc,
                "replaceable": replaceable,
                "flipped_items": [asdict(d) for d in ds if d.flipped][:50],
            }
        )
    adjusted = holm(raw_p) if raw_p else {}
    for rep in q_reports:
        if rep["mcnemar"] is not None:
            rep["mcnemar"]["p_holm"] = adjusted[rep["question_id"]]
            rep["mcnemar"]["significant_05"] = adjusted[rep["question_id"]] < 0.05

    return {
        "schema": "s1-shadow-compare/1",
        "shared_pairs": len(shared),
        "primary_only": len(idx_a) - len(shared),
        "shadow_only": len(idx_b) - len(shared),
        "questions": q_reports,
    }
