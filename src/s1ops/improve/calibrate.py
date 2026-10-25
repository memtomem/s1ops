"""Fit, save, load and apply recalibration maps keyed to `(question_id, model_served, primitive)`."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_expit, log_softmax, logit, softmax
from sklearn.isotonic import IsotonicRegression

from s1ops.contracts.records import Group, GroupKey, ItemKey, Record
from s1ops.eval.metrics import EPS, log_loss

FIT_SCHEMAS = ("jev-calib-fit/2", "s1-calib-fit/1")
DEFAULT_FIT_SCHEMA = "jev-calib-fit/2"
METHODS = {"temperature": ("noul", "choice", "score"), "platt": ("noul",), "isotonic": ("noul",)}
LOG_T_BOUNDS = (math.log(0.01), math.log(100.0))


class FitError(ValueError):
    pass


@dataclass(frozen=True)
class Fit:
    question_id: str
    model_served: str
    primitive: str
    method: str
    split: str
    n: int
    classes: list
    params: dict
    nll_before: float
    nll_after: float
    fitted_at: str
    source_sha256: str
    warnings: list[str]
    items: list[list]

    @property
    def key(self) -> GroupKey:
        return (self.question_id, self.model_served, self.primitive)


def _z(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, EPS, 1 - EPS))


def _binary_nll(s: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.where(y == 1, log_expit(s), log_expit(-s))))


def transform(method: str, primitive: str, params: dict, P: np.ndarray) -> np.ndarray:
    if primitive == "noul":
        p = P[:, 1]
        if method == "temperature":
            q = expit(_z(p) / params["temperature"])
        elif method == "platt":
            q = expit(params["a"] * _z(p) + params["b"])
        elif method == "isotonic":
            q = np.interp(p, params["x"], params["y"])
        else:
            raise FitError(f"unknown method {method!r}")
        return _checked(np.column_stack([1 - q, q]))
    if method != "temperature":
        raise FitError(f"{method} is only defined for noul")
    return _checked(softmax(np.log(np.clip(P, EPS, 1.0)) / params["temperature"], axis=1))


def _checked(Q: np.ndarray) -> np.ndarray:
    if not (np.isfinite(Q).all() and (Q >= 0).all() and (Q <= 1).all() and np.allclose(Q.sum(axis=1), 1.0)):
        raise FitError("the fitted map produced values that are not probabilities")
    return Q


def _fit_params(method: str, primitive: str, P: np.ndarray, y: np.ndarray) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(P[:, 1], y)
        return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}, warnings
    if method == "platt":
        z = _z(P[:, 1])
        res = minimize(lambda ab: _binary_nll(ab[0] * z + ab[1], y), x0=[1.0, 0.0], method="L-BFGS-B")
        if not res.success:
            raise FitError(f"platt fit did not converge: {res.message}")
        return {"a": float(res.x[0]), "b": float(res.x[1])}, warnings
    if primitive == "noul":
        z = _z(P[:, 1])

        def objective(t: float) -> float:
            return _binary_nll(z / math.exp(t), y)
    else:
        L = np.log(np.clip(P, EPS, 1.0))
        rows = np.arange(len(y))

        def objective(t: float) -> float:
            return float(-np.mean(log_softmax(L / math.exp(t), axis=1)[rows, y]))

    res = minimize_scalar(objective, bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-6})
    if not res.success:
        raise FitError(f"temperature fit did not converge: {res.message}")
    if min(abs(res.x - b) for b in LOG_T_BOUNDS) < 1e-3:
        warnings.append(
            f"temperature hit the search bound ({math.exp(res.x):.3g}); the answers may carry no usable signal"
        )
    return {"temperature": float(math.exp(res.x))}, warnings


def fit_group(group: Group, method: str, split: str, *, source_sha256: str = "", min_n: int = 0) -> Fit:
    if method not in METHODS:
        raise FitError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    if group.primitive not in METHODS[method]:
        raise FitError(
            f"{method} is not defined for {group.primitive}; use one of {[m for m, p in METHODS.items() if group.primitive in p]}"
        )
    if len(np.unique(group.y)) < 2:
        raise FitError(f"{group.question_id} ({group.model_served}): every label is the same class; nothing to fit")
    params, warnings = _fit_params(method, group.primitive, group.P, group.y)
    problem = _params_problem(method, params) or _classes_problem(group.primitive, list(group.classes))
    if problem:
        raise FitError(f"{group.question_id} ({group.model_served}): {problem}")
    if group.n < min_n:
        warnings.insert(0, f"n={group.n} is small for a recalibration fit (warning threshold {min_n})")
    after = transform(method, group.primitive, params, group.P)
    return Fit(
        question_id=group.question_id,
        model_served=group.model_served,
        primitive=group.primitive,
        method=method,
        split=split,
        n=group.n,
        classes=list(group.classes),
        params=params,
        nll_before=log_loss(group.P, group.y),
        nll_after=log_loss(after, group.y),
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_sha256=source_sha256,
        warnings=warnings,
        items=[list(i) for i in sorted(group.items, key=_item_order)],
    )


def _item_order(item: ItemKey) -> tuple:
    dataset, item_id = item
    return (dataset is not None, dataset or "", item_id)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_fits(fits: list[Fit], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": DEFAULT_FIT_SCHEMA, "fits": [asdict(x) for x in fits]}, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _reject_constant(name: str) -> float:
    raise FitError(f"non-finite number {name} is not valid JSON")


def load_fits(path: str | Path) -> list[Fit]:
    with open(path, encoding="utf-8") as f:
        try:
            doc = json.load(f, parse_constant=_reject_constant)
        except FitError as e:
            raise FitError(f"{path}: {e}") from None
    if isinstance(doc, dict) and doc.get("schema") == "jev-calib-fit/1":
        raise FitError(f"{path}: jev-calib-fit/1 stores item ids without their dataset; refit to get {DEFAULT_FIT_SCHEMA}")
    if not isinstance(doc, dict) or doc.get("schema") not in FIT_SCHEMAS:
        raise FitError(f"{path}: not a {DEFAULT_FIT_SCHEMA} file")
    fits = []
    for row in doc.get("fits", []):
        try:
            fit = Fit(**row)
        except TypeError as e:
            raise FitError(f"{path}: malformed fit entry: {e}") from None
        if fit.method not in METHODS or fit.primitive not in METHODS[fit.method]:
            raise FitError(f"{path}: {fit.method} is not a method for {fit.primitive}")
        if not (isinstance(fit.items, list) and len(fit.items) == fit.n and all(_is_item(i) for i in fit.items)):
            raise FitError(
                f"{path}: fit for {fit.question_id} needs items, a list of its n={fit.n} [dataset, item_id] pairs"
            )
        problem = _classes_problem(fit.primitive, fit.classes) or _params_problem(fit.method, fit.params)
        if problem:
            raise FitError(f"{path}: fit for {fit.question_id} ({fit.method}, {fit.primitive}): {problem}")
        fits.append(fit)
    keys = [x.key for x in fits]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise FitError(f"{path}: more than one fit for {sorted(duplicates)}; keep one method per file")
    return fits


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _params_problem(method: str, params: object) -> str | None:
    expected = {"temperature": {"temperature"}, "platt": {"a", "b"}, "isotonic": {"x", "y"}}[method]
    if not isinstance(params, dict) or set(params) != expected:
        return f"params must have exactly the keys {sorted(expected)}, got {params!r}"
    if method == "temperature":
        t = params["temperature"]
        return None if _finite(t) and t > 0 else f"params temperature must be a finite number > 0, got {t!r}"
    if method == "platt":
        bad = [k for k in ("a", "b") if not _finite(params[k])]
        return f"params {bad} must be finite numbers, got {params!r}" if bad else None
    x, y = params["x"], params["y"]
    if not (isinstance(x, list) and isinstance(y, list) and x and len(x) == len(y)):
        return "params x and y must be non-empty lists of the same length"
    if not all(_finite(v) for v in (*x, *y)):
        return "params x and y must hold finite numbers"
    if any(b <= a for a, b in zip(x, x[1:])):
        return "params x must be strictly increasing"
    if any(not 0.0 <= v <= 1.0 for v in y) or any(b < a for a, b in zip(y, y[1:])):
        return "params y must be non-decreasing and within [0, 1]"
    return None


def _classes_problem(primitive: str, classes: object) -> str | None:
    if not isinstance(classes, list) or len(classes) < 2 or len(classes) != len(set(map(repr, classes))):
        return f"classes must be a list of at least two distinct values, got {classes!r}"
    if primitive == "noul":
        ok = len(classes) == 2 and all(isinstance(c, bool) for c in classes) and classes == [False, True]
        return None if ok else f"noul classes must be [false, true], got {classes!r}"
    if primitive == "choice":
        return None if all(isinstance(c, str) and c for c in classes) else f"choice classes must be option names, got {classes!r}"
    ok = all(isinstance(c, int) and not isinstance(c, bool) for c in classes) and classes == sorted(classes)
    return None if ok else f"score classes must be integer levels in ascending order, got {classes!r}"


def _is_item(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and (value[0] is None or isinstance(value[0], str))
        and isinstance(value[1], str)
    )


def refusal(fits: list[Fit], key: GroupKey, classes: tuple) -> str | None:
    qid, served, primitive = key
    for x in fits:
        if x.key == key:
            if set(x.classes) != set(classes):
                return f"answer classes {sorted(map(str, classes))} differ from the fit's {sorted(map(str, x.classes))}"
            return None
    same_question = [x for x in fits if x.question_id == qid]
    if not same_question:
        return f"no fit for question_id {qid}"
    models = sorted({x.model_served for x in same_question if x.primitive == primitive})
    if models:
        return f"fit is for model_served {', '.join(models)}; this answer was served by {served}"
    return f"fit for {qid} is for primitive {same_question[0].primitive}, not {primitive}"


def fit_for(fits: list[Fit], key: GroupKey) -> Fit | None:
    return next((x for x in fits if x.key == key), None)


def fitted_items(fits: list[Fit]) -> dict[GroupKey, frozenset[ItemKey]]:
    return {x.key: frozenset((d, i) for d, i in x.items) for x in fits}


def apply_to_record(fit: Fit, record: Record, *, in_fit_sample: bool = False) -> dict:
    order = {c: i for i, c in enumerate(record.classes)}
    P = np.array([[record.probs[order[c]] for c in fit.classes]])
    Q = transform(fit.method, fit.primitive, fit.params, P)[0]
    calib: dict = {"method": fit.method, "fit_split": fit.split, "fit_n": fit.n, "fitted_at": fit.fitted_at}
    if in_fit_sample:
        calib["in_fit_sample"] = True
    if fit.primitive == "noul":
        calib["p"] = float(Q[1])
    elif fit.primitive == "choice":
        calib["probabilities"] = {c: float(q) for c, q in zip(fit.classes, Q)}
    else:
        calib["probabilities"] = {str(c): float(q) for c, q in zip(fit.classes, Q)}
        calib["score"] = float((np.asarray(fit.classes, dtype=float) * Q).sum())
    return {**record.raw, "calib": calib}
