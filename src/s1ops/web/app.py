"""FastAPI Local Web Workbench for s1ops.

Serves a 6-tab interactive System 1 LLMOps console on `127.0.0.1`:
1. Calibration & Noise Floor Explorer (Reliability SVG + ECE Floor + Tie-Mass)
2. Threshold & Escalation Band Simulator (Interactive band_lo / band_hi sliders + Pareto)
3. Question Lint & Probe Coherence Inspector (JL000-JL060 + Noul/Choice & Inverse gap)
4. Slice & Blind Label Audit Studio (BH-FDR error slices + Disagreement ranker)
5. Multi-Backend Shadow Compare & Judge Audit (McNemar-Holm + Weighted Kappa)
6. Distribution & Model Drift Monitor (PSI + JSD + silent `model_served` upgrade)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from s1ops import __version__
from s1ops.contracts.records import GroupKey, Record, read_records, select
from s1ops.contracts.schemas import load_policies
from s1ops.eval.audit import audit_group
from s1ops.eval.compare import CompareInputError, compare_records
from s1ops.eval.judge import agreement as judge_agreement
from s1ops.eval.judge import by_criterion as judge_by_criterion
from s1ops.eval.metrics import top_label
from s1ops.improve.calibrate import Fit, fit_for, load_fits, transform
from s1ops.improve.labelaudit import LabelAuditInputError, rank_with_skips
from s1ops.improve.policy import (
    Band,
    DevRecord,
    Prices,
    choose,
    evaluate as evaluate_band,
    grid as band_grid,
    pareto,
    select_threshold,
    sweep,
    threshold_curve,
)
from s1ops.improve.slices import mine_error_slices, parse_feature_rows, slice_report
from s1ops.lint.lint import ProbeInputError, audit_probe_records, lint_path
from s1ops.lint.rules import RULES
from s1ops.monitor.drift import audit_drift
from s1ops.runtime.routing import RoutingInputError


STATIC_DIR = Path(__file__).parent / "static"

def _load_or_400(what: str | Path, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run one input-reading or input-validation call; its `ValueError` becomes HTTP 400 naming `what`.

    Every input error in s1ops is a ValueError (RecordError, FitError, JSONDecodeError, load_policies).
    Wrap only the read of one file, never a computation, so a server bug stays a 500.
    """
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{what}: {e}") from None


_ONE_LEVEL = "every grade is 0; agreement needs grades on at least two levels"


def _is_grade(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file or JSON array; a missing or empty file is no rows, a non-object row is an error."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        data = json.loads(text)
        rows = data if isinstance(data, list) else []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"element {i}: expected a JSON object")
        return rows
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {n}: not valid JSON: {e}") from None
        if not isinstance(row, dict):
            raise ValueError(f"line {n}: expected a JSON object")
        out.append(row)
    return out


def create_app(
    workspace_dir: str | Path | None = None,
    *,
    records_path: str | Path | None = None,
    compare_records_path: str | Path | None = None,
    fits_path: str | Path | None = None,
    questions_path: str | Path | None = None,
    probe_records_path: str | Path | None = None,
    features_path: str | Path | None = None,
    judge_path: str | Path | None = None,
) -> FastAPI:
    """Create the s1ops FastAPI application bound to a local workspace."""
    base = Path(workspace_dir) if workspace_dir else Path.cwd()
    if not base.is_dir():
        raise FileNotFoundError(f"workspace directory not found: {base}")

    def _resolve(explicit: str | Path | None, default_rel: str) -> Path:
        if explicit is not None:
            return Path(explicit)
        candidate = base / default_rel
        if candidate.exists():
            return candidate
        alt = base / "examples" / "data" / default_rel
        if alt.exists():
            return alt
        return candidate

    rec_file = _resolve(records_path, "records.jsonl")
    cmp_file = _resolve(compare_records_path, "records_kev.jsonl")
    fit_file = _resolve(fits_path, "fits.json")
    q_file = _resolve(questions_path, "questions.json")
    probe_file = _resolve(probe_records_path, "probe_records.jsonl")
    feat_file = _resolve(features_path, "features.jsonl")
    judge_file = _resolve(judge_path, "judge_grades.jsonl")

    app = FastAPI(
        title="s1ops Workbench",
        description="System 1 Typed Decision Model LLMOps Workbench",
        version=__version__,
    )

    def _get_records(use_calib: bool = False) -> list[Record]:
        if not rec_file.exists():
            return []
        return _load_or_400(rec_file, read_records, rec_file, use_calib=use_calib)

    def _get_fits() -> list[Fit]:
        if not fit_file.exists():
            return []
        return _load_or_400(fit_file, load_fits, fit_file)

    def _select(records: list[Record]) -> Any:
        # Duplicate (dataset, item_id) rows pass the per-line parser and fail here.
        return _load_or_400(rec_file, select, records, allow_model_mismatch=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/overview")
    def api_overview() -> dict[str, Any]:
        records = _get_records()
        fits = _get_fits()
        fit_keys: set[GroupKey] = {f.key for f in fits}
        optional_files = {
            "records": rec_file,
            "compare_records": cmp_file,
            "fits": fit_file,
            "questions": q_file,
            "probe_records": probe_file,
            "features": feat_file,
            "judge_grades": judge_file,
            "records_canary": _resolve(None, "records_canary.jsonl"),
            "policy": _resolve(None, "policy.json"),
        }
        groups_map: dict[GroupKey, int] = {}
        calibrated_count = 0
        for r in records:
            if r.model_served is None:
                continue
            groups_map[r.key] = groups_map.get(r.key, 0) + 1
            if isinstance(r.raw.get("calib"), dict):
                calibrated_count += 1

        groups = [
            {
                "question_id": qid,
                "model_served": model,
                "primitive": prim,
                "n_records": count,
                "has_fit": (qid, model, prim) in fit_keys,
            }
            for (qid, model, prim), count in sorted(groups_map.items())
        ]
        return {
            "workspace": str(base),
            "files": {
                name: str(path) if path.exists() else None
                for name, path in optional_files.items()
            },
            "missing_files": [path.name for path in optional_files.values() if not path.exists()],
            "n_records": len(records),
            "n_calibrated_records": calibrated_count,
            "n_groups": len(groups),
            "groups": groups,
        }

    @app.get("/api/audit")
    def api_audit(
        question_id: str | None = Query(default=None),
        model_served: str | None = Query(default=None),
        n_boot: int = Query(default=200, ge=20, le=2000),
    ) -> dict[str, Any]:
        records = _get_records(use_calib=False)
        if not records:
            return {"groups": []}
        fits = _get_fits()
        sel = _select(records)
        results: list[dict[str, Any]] = []
        for grp in sel.groups:
            if question_id and grp.question_id != question_id:
                continue
            if model_served and grp.model_served != model_served:
                continue
            fit = fit_for(fits, grp.key)
            after_P = (
                transform(fit.method, fit.primitive, fit.params, grp.P)
                if fit is not None
                else None
            )
            report = audit_group(
                grp,
                bins=10,
                reps=n_boot,
                seed=0,
                after=after_P,
            )
            if fit is not None:
                report["fit"] = {
                    "method": fit.method,
                    "params": fit.params,
                    "nll_before": fit.nll_before,
                    "nll_after": fit.nll_after,
                }
            results.append(report)
        return {"groups": results}

    @app.get("/api/simulate-policy")
    def api_simulate_policy(
        question_id: str | None = Query(default=None),
        model_served: str | None = Query(default=None),
        tau_low: float = Query(default=0.25, ge=0.0, le=1.0),
        tau_high: float = Query(default=0.75, ge=0.0, le=1.0),
        cost_s1: float = Query(default=0.00004, ge=0.0),
        cost_sys2: float = Query(default=0.004, ge=0.0),
        error_budget: float = Query(default=0.10, ge=0.0, le=1.0),
        use_calib: bool = Query(default=True),
    ) -> dict[str, Any]:
        records = _get_records(use_calib=use_calib)
        if not records:
            raise HTTPException(status_code=404, detail="No records loaded")

        sel = _select(records)
        chosen_grp = None
        for grp in sel.groups:
            if question_id and grp.question_id != question_id:
                continue
            if model_served and grp.model_served != model_served:
                continue
            if grp.primitive == "noul":
                chosen_grp = grp
                break
        if chosen_grp is None and sel.groups:
            chosen_grp = sel.groups[0]
        if chosen_grp is None:
            raise HTTPException(status_code=404, detail="No matching labeled group found")

        conf, correct = top_label(chosen_grp.P, chosen_grp.y)
        tau = select_threshold(conf, correct, budget=error_budget)
        curve = threshold_curve(
            conf,
            correct,
            c_s1=cost_s1,
            c_delegate=cost_sys2,
        )

        # Build DevRecords for hybrid escalation band evaluation
        low = min(tau_low, tau_high)
        high = max(tau_low, tau_high)
        dev_records: list[DevRecord] = []
        for idx, (_, item_id) in enumerate(chosen_grp.items):
            p_yes = float(chosen_grp.P[idx, 1]) if chosen_grp.primitive == "noul" else float(conf[idx])
            lbl = bool(chosen_grp.y[idx] == 1)
            dev_records.append(
                DevRecord(
                    id=item_id,
                    p_jev=p_yes,
                    label=lbl,
                    llm=lbl,
                    jev_cost_usd=cost_s1,
                    llm_cost_usd=cost_sys2,
                )
            )

        prices = Prices(jev_usd=cost_s1, llm_usd=cost_sys2)
        active_res = evaluate_band(dev_records, Band(low, high), prices)
        swept = sweep(dev_records, band_grid(0.10), prices)
        frontier = [r.as_row() for r in pareto(swept, metric="f1")]
        recommended = choose(swept, metric="f1", max_escalation_rate=0.35)

        return {
            "group": {
                "question_id": chosen_grp.question_id,
                "model_served": chosen_grp.model_served,
                "primitive": chosen_grp.primitive,
                "n": chosen_grp.n,
            },
            "active_band": active_res.as_row() if active_res else None,
            "tuned_threshold": {
                "threshold": tau,
                "error_budget": error_budget,
            },
            "recommended_band_under_35pct_esc": recommended.as_row() if recommended else None,
            "pareto_frontier": frontier,
            "threshold_curve": curve,
        }

    @app.get("/api/lint")
    def api_lint() -> dict[str, Any]:
        lint_dict: dict[str, Any] = {"findings": [], "counts": {}}
        if q_file.exists():
            # lint_path reports an unreadable or malformed questions file as findings, not exceptions.
            res = lint_path(q_file)
            lint_dict = res.as_dict()

        probe_out: list[dict[str, Any]] = []
        if probe_file.exists():
            # audit_probe_records only takes a path, so validate the file first and run the audit unwrapped.
            _load_or_400(probe_file, read_records, probe_file)
            try:
                probe_out = audit_probe_records(probe_file)
            except ProbeInputError as e:
                raise HTTPException(status_code=400, detail=f"{probe_file.name}: {e}") from None

        rules_catalog = [
            {
                "id": r.id,
                "title": r.title,
                "severity": r.severity,
                "fix": r.fix,
            }
            for r in RULES.values()
        ]
        return {
            "questions_file": str(q_file) if q_file.exists() else None,
            "n_findings": len(lint_dict.get("findings", [])),
            "counts": lint_dict.get("counts", {}),
            "findings": lint_dict.get("findings", []),
            "probe_audit": probe_out,
            "rules": rules_catalog,
        }

    @app.get("/api/slices")
    def api_slices(
        question_id: str | None = Query(default=None),
        model_served: str | None = Query(default=None),
        fdr_q: float = Query(default=0.10, ge=0.001, le=0.50),
    ) -> dict[str, Any]:
        records = _get_records(use_calib=True)
        features_rows = _load_or_400(feat_file, _load_json_or_jsonl, feat_file)
        features_by_item, slice_by_item = _load_or_400(feat_file, parse_feature_rows, features_rows)

        filtered = [
            r
            for r in records
            if r.model_served is not None
            and r.label is not None
            and (question_id is None or r.question_id == question_id)
            and (model_served is None or r.model_served == model_served)
        ]
        if not filtered:
            return {"answerability": [], "error_slices": []}

        slice_items: list[dict[str, Any]] = []
        mine_rows: list[dict[str, Any]] = []
        for r in filtered:
            pred = r.classes[int(max(range(len(r.probs)), key=lambda i: r.probs[i]))]
            sl = slice_by_item.get(r.item) or slice_by_item.get((None, r.item_id)) or "default"
            feats = features_by_item.get(r.item) or features_by_item.get((None, r.item_id)) or {}
            slice_items.append(
                {
                    "slice": sl,
                    "predicted": pred,
                    "label": r.label,
                }
            )
            mine_rows.append(
                {
                    "error": pred != r.label,
                    "features": feats,
                }
            )

        ans = [s.as_dict() for s in slice_report(slice_items, min_n=10)]
        err_slices = [
            e.as_dict()
            for e in mine_error_slices(mine_rows, min_support=5, fdr_q=fdr_q)
        ]
        return {
            "answerability": ans,
            "error_slices": err_slices,
        }

    @app.get("/api/labelaudit")
    def api_labelaudit(
        top_k: int = Query(default=15, ge=1, le=100),
    ) -> dict[str, Any]:
        records = _get_records(use_calib=False)
        if not records:
            return {"top_flags": [], "skipped": {}}
        # A fits.json that exists means calibrate, even with no fits in it; only a missing one ranks raw.
        fits = _get_fits() if fit_file.exists() else None
        raw_rows = [r.raw for r in records]
        try:
            flags, skipped = rank_with_skips(raw_rows, temperature=fits)
        except LabelAuditInputError as e:
            raise HTTPException(status_code=400, detail=f"{rec_file.name} vs {fit_file.name}: {e}") from None
        return {
            "n_total_candidates": len(flags),
            "skipped": {"no_response": skipped.no_response, "unlabeled": skipped.unlabeled},
            "top_flags": [
                {
                    "rank": i + 1,
                    **f.as_dict(),
                }
                for i, f in enumerate(flags[:top_k])
            ],
        }

    @app.get("/api/compare")
    def api_compare() -> dict[str, Any]:
        shadow_out: dict[str, Any] = {"questions": []}
        if rec_file.exists() and cmp_file.exists():
            primary = [r.raw for r in _get_records()]
            shadow = [r.raw for r in _load_or_400(cmp_file, read_records, cmp_file)]
            try:
                shadow_out = compare_records(primary, shadow)
            except CompareInputError as e:
                raise HTTPException(status_code=400, detail=f"{rec_file.name} vs {cmp_file.name}: {e}") from None

        # The judge file is optional: missing or empty means "judge": null, not an error.
        judge_rows = _load_or_400(judge_file, _load_json_or_jsonl, judge_file)
        judge_out: dict[str, Any] | None = None
        if judge_rows:
            for i, r in enumerate(judge_rows, start=1):
                if not all(_is_grade(r.get(k)) for k in ("judge", "human")):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{judge_file}: row {i}: 'judge' and 'human' must be non-negative integers",
                    )
            j_vals = [r["judge"] for r in judge_rows]
            h_vals = [r["human"] for r in judge_rows]
            levels = max(max(j_vals), max(h_vals)) + 1
            if levels < 2:
                raise HTTPException(status_code=400, detail=f"{judge_file}: {_ONE_LEVEL}")
            crit_tuples = [
                (str(r.get("criterion", "default")), r["judge"], r["human"])
                for r in judge_rows
            ]
            overall = judge_agreement(j_vals, h_vals, levels=levels, pass_level=3)
            by_crit = judge_by_criterion(crit_tuples, levels=levels, pass_level=3)
            judge_out = {
                "overall": overall.as_dict(),
                "by_criterion": {k: v.as_dict() for k, v in by_crit.items()},
            }

        return {
            "shadow": shadow_out,
            "judge": judge_out,
        }

    @app.get("/api/drift")
    def api_drift() -> dict[str, Any]:
        canary_file = _resolve(None, "records_canary.jsonl")
        pol_file = _resolve(None, "policy.json")
        if not rec_file.exists() or not canary_file.exists():
            return {"schema": "s1-drift-report/1", "distribution": "raw", "summary": {"n_questions": 0, "ok": 0, "warn": 0, "critical": 0}, "questions": []}
        policies = _load_or_400(pol_file, load_policies, pol_file) if pol_file.exists() else None
        # Read with use_calib so a malformed stored calib is a 400 naming its file; the report
        # still compares raw probabilities, re-parsed by audit_drift.
        refs = _load_or_400(rec_file, read_records, rec_file, require_answer=True, use_calib=True)
        curs = _load_or_400(canary_file, read_records, canary_file, require_answer=True, use_calib=True)
        try:
            return audit_drift(refs, curs, policies=policies)
        except RoutingInputError as e:
            path = rec_file if e.side == "reference" else canary_file
            raise HTTPException(status_code=400, detail=f"{path.name}: line {e.line}: {e.message}") from None

    return app

