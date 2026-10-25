"""Unified CLI for s1ops (`s1` and `s1ops` commands).

Provides offline-first System 1 evaluation, calibration, policy compilation,
slice & blind label audit, shadow comparison, judge agreement, Langfuse/Phoenix
safe-v1 exporters, ledger management, and the local Web Workbench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from s1ops.bridges.langfuse import export_langfuse_file
from s1ops.bridges.openinference import export_openinference_jsonl
from s1ops.contracts.records import read_records, select
from s1ops.contracts.schemas import PolicyEntry, load_policies, save_policies
from s1ops.eval.audit import audit_group
from s1ops.eval.compare import compare_records
from s1ops.eval.judge import agreement as judge_agreement
from s1ops.eval.judge import by_criterion as judge_by_criterion
from s1ops.eval.metrics import top_label
from s1ops.improve.calibrate import (
    Fit,
    apply_to_record,
    fit_for,
    fit_group,
    fitted_items,
    load_fits,
    refusal,
    save_fits,
    sha256_file,
    transform,
)
from s1ops.improve.labelaudit import (
    blind_sheet,
    estimate,
    rank_with_skips,
    score_sheet,
    temperatures_from_fits,
)
from s1ops.improve.policy import (
    Band,
    DevRecord,
    Prices,
    choose,
    grid as band_grid,
    select_threshold,
    sweep,
    threshold_curve,
)
from s1ops.improve.slices import mine_error_slices, slice_report
from s1ops.ledger.journal import (
    Ledger,
    migrate as ledger_migrate,
    read as ledger_read,
    recover as ledger_recover,
)
from s1ops.lint.lint import audit_probe_records, format_text, lint_path
from s1ops.lint.rules import RULES
from s1ops.monitor.drift import audit_drift
from s1ops.monitor.runner import DeterministicMockBackend, run_batch
from s1ops.runtime.gate import DecisionGate



def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _cmd_lint(args: argparse.Namespace) -> int:
    if args.lint_cmd == "rules":
        rows = [
            {
                "id": r.id,
                "title": r.title,
                "severity": r.severity,
                "fix": r.fix,
                "evidence": r.evidence,
                "evidence_url": r.evidence_url,
            }
            for r in RULES.values()
        ]
        print(json.dumps(rows, indent=2))
        return 0

    if args.lint_cmd == "probe-audit":
        report = audit_probe_records(args.records)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    ignore = frozenset(args.ignore or [])
    result = lint_path(args.path, ignore=ignore)
    if args.format == "json":
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(format_text(result))
    return result.exit_code(strict=args.strict)


def _cmd_audit(args: argparse.Namespace) -> int:
    records = read_records(args.records)
    fits = load_fits(args.fit) if args.fit else []
    sel = select(records, split=args.split, allow_model_mismatch=True)
    reports = []
    for grp in sel.groups:
        if args.question and grp.question_id != args.question:
            continue
        if args.model and grp.model_served != args.model:
            continue
        fit = fit_for(fits, grp.key)
        after_P = (
            transform(fit.method, fit.primitive, fit.params, grp.P)
            if fit is not None
            else None
        )
        rep = audit_group(
            grp,
            bins=args.bins,
            reps=args.boot,
            seed=args.seed,
            after=after_P,
        )
        reports.append(rep)
    print(json.dumps({"groups": reports}, indent=2, sort_keys=True))
    return 0 if reports else 1


def _cmd_fit(args: argparse.Namespace) -> int:
    records = read_records(args.records)
    sel = select(records, split=args.split, allow_model_mismatch=True)
    src_sha = sha256_file(args.records)
    fits: list[Fit] = []
    for grp in sel.groups:
        if args.question and grp.question_id != args.question:
            continue
        if args.model and grp.model_served != args.model:
            continue
        f = fit_group(
            grp,
            method=args.method,
            split=args.split or "cal",
            source_sha256=src_sha,
            min_n=args.min_n,
        )
        fits.append(f)

    if not fits:
        return 1
    save_fits(fits, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_fits": len(fits),
                "groups": [
                    {
                        "question_id": f.question_id,
                        "model_served": f.model_served,
                        "primitive": f.primitive,
                        "method": f.method,
                        "n": f.n,
                        "nll_before": f.nll_before,
                        "nll_after": f.nll_after,
                    }
                    for f in fits
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    records = read_records(args.records, require_answer=True)
    fits = load_fits(args.fit)
    in_sample_map = fitted_items(fits)

    out_objs: list[dict[str, Any]] = []
    for rec in records:
        if rec.model_served is None:
            out_objs.append(rec.raw)
            continue
        fit = fit_for(fits, rec.key)
        if fit is None:
            msg = refusal(fits, rec.key, rec.classes) or f"no fit for {rec.key}"
            print(f"Refusing to apply fit: {msg}", file=sys.stderr)
            if "model_served" in msg:
                return 3
            return 1

        err = refusal(fits, rec.key, rec.classes)
        if err is not None:
            print(f"Refusing to apply fit: {err}", file=sys.stderr)
            return 1

        in_sample = rec.item in in_sample_map.get(fit.key, frozenset())
        if in_sample and not args.allow_in_sample:
            print(
                f"Refusing in-sample calibration apply on {rec.item}; "
                "pass --allow-in-sample to override",
                file=sys.stderr,
            )
            return 2

        out_objs.append(apply_to_record(fit, rec, in_fit_sample=in_sample))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for o in out_objs:
            fh.write(json.dumps(o, sort_keys=True) + "\n")
    return 0


def _cmd_policy(args: argparse.Namespace) -> int:
    records = read_records(args.records, use_calib=args.use_calib)
    sel = select(records, allow_model_mismatch=True)
    if args.policy_cmd == "curve":
        curves: list[dict[str, Any]] = []
        for grp in sel.groups:
            if args.question and grp.question_id != args.question:
                continue
            if args.model and grp.model_served != args.model:
                continue
            conf, correct = top_label(grp.P, grp.y)
            curves.append(
                {
                    "question_id": grp.question_id,
                    "model_served": grp.model_served,
                    "primitive": grp.primitive,
                    "curve": threshold_curve(
                        conf,
                        correct,
                        c_s1=args.cost_s1,
                        c_delegate=args.cost_sys2,
                    ),
                }
            )
        print(json.dumps({"groups": curves}, indent=2, sort_keys=True))
        return 0

    src_sha = sha256_file(args.records)
    policies: list[PolicyEntry] = []
    for grp in sel.groups:
        if args.question and grp.question_id != args.question:
            continue
        if args.model and grp.model_served != args.model:
            continue
        conf, correct = top_label(grp.P, grp.y)
        tau = select_threshold(
            conf,
            correct,
            budget=args.error_budget,
            use_cp_upper=args.use_cp_upper,
            alpha=args.cp_alpha,
        )

        band_lo = None
        band_hi = None
        if grp.primitive == "noul":
            dev_recs = [
                DevRecord(
                    id=item_id,
                    p_jev=float(grp.P[i, 1]),
                    label=bool(grp.y[i] == 1),
                    llm=bool(grp.y[i] == 1),
                    jev_cost_usd=args.cost_s1,
                    llm_cost_usd=args.cost_sys2,
                )
                for i, (_, item_id) in enumerate(grp.items)
            ]
            swept = sweep(
                dev_recs,
                band_grid(0.10),
                Prices(jev_usd=args.cost_s1, llm_usd=args.cost_sys2),
            )
            best = choose(swept, metric="f1", max_escalation_rate=args.max_escalation)
            if best is not None:
                band_lo = best.band.lo
                band_hi = best.band.hi

        mode = "escalation_band" if (band_lo is not None and band_hi is not None) else "error_budget"
        policies.append(
            PolicyEntry(
                question_id=grp.question_id,
                model_served=grp.model_served,
                primitive=grp.primitive,
                mode=mode,
                split="eval",
                n=grp.n,
                threshold=tau,
                error_budget=args.error_budget,
                band_lo=band_lo,
                band_hi=band_hi,
                calibrated=args.use_calib,
                source_sha256=src_sha,
            )
        )

    if not policies:
        return 1
    save_policies(policies, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_policies": len(policies),
                "policies": [p.to_dict() for p in policies],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_slice(args: argparse.Namespace) -> int:
    records = read_records(args.records, use_calib=True)
    features_by_item: dict[tuple[str | None, str], dict[str, Any]] = {}
    slice_by_item: dict[tuple[str | None, str], str] = {}
    if getattr(args, "features", None):
        for row in _load_jsonl_dicts(Path(args.features)):
            key = (row.get("dataset"), str(row.get("item_id", "")))
            feats = row.get("features", {})
            if isinstance(feats, dict):
                features_by_item[key] = feats
                if "slice" in feats:
                    slice_by_item[key] = str(feats["slice"])
            if "slice" in row:
                slice_by_item[key] = str(row["slice"])

    filtered = [
        r
        for r in records
        if r.model_served is not None
        and r.label is not None
        and (args.question is None or r.question_id == args.question)
        and (args.model is None or r.model_served == args.model)
    ]

    if args.slice_cmd == "answerable":
        items = []
        for r in filtered:
            pred = r.classes[int(max(range(len(r.probs)), key=lambda i: r.probs[i]))]
            sl = slice_by_item.get(r.item) or slice_by_item.get((None, r.item_id)) or "default"
            items.append({"slice": sl, "predicted": pred, "label": r.label})
        rep = [s.as_dict() for s in slice_report(items, min_n=args.min_n)]
        print(json.dumps({"slices": rep}, indent=2, sort_keys=True))
        return 0

    rows = []
    for r in filtered:
        pred = r.classes[int(max(range(len(r.probs)), key=lambda i: r.probs[i]))]
        feats = features_by_item.get(r.item) or features_by_item.get((None, r.item_id)) or {}
        rows.append({"error": pred != r.label, "features": feats})
    mined = [
        e.as_dict()
        for e in mine_error_slices(rows, min_support=args.min_support, fdr_q=args.fdr_q)
    ]
    print(json.dumps({"error_slices": mined}, indent=2, sort_keys=True))
    return 0


def _cmd_labelaudit(args: argparse.Namespace) -> int:
    if args.la_cmd == "estimate":
        form_text = Path(args.sheet).read_text(encoding="utf-8")
        key_doc = json.loads(Path(args.key).read_text(encoding="utf-8"))
        scored = score_sheet(form_text, key_doc)
        est = estimate(
            scored["flag"]["reviewer_disagrees_with_label"],
            scored["flag"]["n"],
            scored["control"]["reviewer_disagrees_with_label"],
            scored["control"]["n"],
        )
        print(
            json.dumps(
                {"scored": scored, "estimate": est.as_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    records = read_records(args.records)
    fits = load_fits(args.fit) if getattr(args, "fit", None) else []
    temps = temperatures_from_fits(fits) if fits else None
    raw_rows = [r.raw for r in records]
    flags, skipped = rank_with_skips(raw_rows, temperature=temps)

    if args.la_cmd == "rank":
        payload = {
            "n_candidates": len(flags),
            "skipped": {"no_response": skipped.no_response, "unlabeled": skipped.unlabeled},
            "top": [
                {
                    "rank": i + 1,
                    **f.as_dict(),
                }
                for i, f in enumerate(flags[: args.top])
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    states_rows = _load_jsonl_dicts(Path(args.states))
    states_map: dict[tuple[str | None, str], str] = {}
    for row in states_rows:
        states_map[(row.get("dataset"), str(row["item_id"]))] = str(row["state"])

    options = args.options.split(",") if args.options else ["True", "False"]
    sheet = blind_sheet(
        flags,
        text_of=states_map,
        k=args.top,
        control_k=args.controls,
        options=options,
        seed=args.seed,
    )
    Path(args.output).write_text(sheet.form, encoding="utf-8")
    Path(args.key).write_text(json.dumps(sheet.key, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sheet": str(args.output),
                "key": str(args.key),
                "n_items": len(sheet.key["items"]),
            },
            indent=2,
        )
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    rep = compare_records(args.primary, args.shadow)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    rows = _load_jsonl_dicts(Path(args.grades))
    j_vals = [int(r["judge"]) for r in rows]
    h_vals = [int(r["human"]) for r in rows]
    levels = max(max(j_vals), max(h_vals)) + 1
    crit_tuples = [
        (str(r.get("criterion", "default")), int(r["judge"]), int(r["human"]))
        for r in rows
    ]
    overall = judge_agreement(j_vals, h_vals, levels=levels, pass_level=args.pass_level)
    by_crit = judge_by_criterion(crit_tuples, levels=levels, pass_level=args.pass_level)
    out = {
        "overall": overall.as_dict(),
        "by_criterion": {k: v.as_dict() for k, v in by_crit.items()},
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    records = read_records(args.records)
    policies: list[PolicyEntry] | None = None
    if args.policy:
        policies = load_policies(args.policy)

    if args.export_cmd == "langfuse":
        payload = export_langfuse_file(
            args.output,
            records,
            run_name=args.run_name,
            policies=policies,
        )
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        return 0

    spans = export_openinference_jsonl(
        args.output,
        records,
        run_name=args.run_name,
        policies=policies,
    )
    print(
        json.dumps(
            {"output": str(args.output), "n_spans": len(spans), "redaction": "safe-v1"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    if args.ledger_cmd == "status":
        st = ledger_read(args.path)
        print(
            json.dumps(
                {
                    "format": st.format,
                    "opening_usd": st.opening_usd,
                    "spent_usd": st.spent_usd,
                    "outstanding_usd": st.outstanding_usd,
                    "n_settlements": len(st.settlements),
                    "n_outstanding": len(st.outstanding),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.ledger_cmd == "migrate":
        st = ledger_migrate(
            args.path,
            args.output,
            acknowledge_lost_reservations=args.acknowledge_lost_reservations,
        )
        print(
            json.dumps(
                {"format": st.format, "opening_usd": st.opening_usd, "spent_usd": st.spent_usd},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    out = ledger_recover(args.path, args.output)
    print(json.dumps({"recovered": str(out)}, indent=2, sort_keys=True))
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    policies = load_policies(args.policy) if args.policy else None
    rep = audit_drift(
        args.reference,
        args.current,
        policies=policies,
        bins=args.bins,
        use_calib=not args.raw_only,
    )
    print(json.dumps(rep, indent=2, sort_keys=True))
    if args.fail_on_critical and rep["summary"]["critical"] > 0:
        return 1
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    q_doc = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if isinstance(q_doc, dict) and "questions" in q_doc and isinstance(q_doc["questions"], dict):
        q_map = q_doc["questions"]
    elif isinstance(q_doc, dict):
        q_map = {k: v for k, v in q_doc.items() if isinstance(v, dict) and "type" in v}
    else:
        raise ValueError("questions file must be a JSON object mapping names to questions")

    states = _load_jsonl_dicts(Path(args.states))
    backend = DeterministicMockBackend(served_model=args.served_model or args.model)
    with Ledger(args.ledger, cap_usd=args.cap_usd) as ledger:
        recs = run_batch(
            states,
            q_map,
            backend=backend,
            ledger=ledger,
            model_requested=args.model,
            probe_mode=not args.no_expand_probes,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    st = ledger_read(args.ledger)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "n_records": len(recs),
                "spent_usd": round(st.spent_usd, 8),
                "cap_usd": args.cap_usd,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    gate = DecisionGate.from_files(args.policy, args.fit)
    dec = gate.evaluate_noul(
        question_id=args.question,
        model_served=args.model,
        p_raw=args.p,
        expected_policy_sha256=args.expected_sha256,
    )
    print(json.dumps(dec.as_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    from s1ops.web.app import create_app

    app = create_app(workspace_dir=args.workspace)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s1",
        description="s1ops: Dedicated System 1 LLMOps CLI & Workbench",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # lint
    p_lint = sub.add_parser("lint", help="Static question lint & probe coherence")
    sub_lint = p_lint.add_subparsers(dest="lint_cmd", required=True)
    p_lc = sub_lint.add_parser("check", help="Lint a question specification file")
    p_lc.add_argument("path")
    p_lc.add_argument("--format", choices=["text", "json"], default="text")
    p_lc.add_argument("--strict", action="store_true")
    p_lc.add_argument("--ignore", action="append")
    sub_lint.add_parser("rules", help="List JL000-JL060 rules")
    p_lp = sub_lint.add_parser("probe-audit", help="Audit Noul/Choice twin & inverse coherence")
    p_lp.add_argument("records")
    p_lint.set_defaults(func=_cmd_lint)

    # audit
    p_aud = sub.add_parser("audit", help="Calibration & noise floor audit per group")
    p_aud.add_argument("records")
    p_aud.add_argument("--fit", default=None)
    p_aud.add_argument("--question", default=None)
    p_aud.add_argument("--model", default=None)
    p_aud.add_argument("--split", default=None)
    p_aud.add_argument("--bins", type=int, default=10)
    p_aud.add_argument("--boot", type=int, default=200)
    p_aud.add_argument("--seed", type=int, default=0)
    p_aud.set_defaults(func=_cmd_audit)

    # fit
    p_fit = sub.add_parser("fit", help="Fit per-group post-hoc calibration (temp/platt/isotonic)")
    p_fit.add_argument("records")
    p_fit.add_argument("-o", "--output", required=True)
    p_fit.add_argument(
        "--method",
        choices=["temperature", "platt", "isotonic"],
        default="temperature",
    )
    p_fit.add_argument("--question", default=None)
    p_fit.add_argument("--model", default=None)
    p_fit.add_argument("--split", default=None)
    p_fit.add_argument("--min-n", type=int, default=30)
    p_fit.set_defaults(func=_cmd_fit)

    # apply
    p_app = sub.add_parser("apply", help="Apply calibration fit to held-out records")
    p_app.add_argument("records")
    p_app.add_argument("--fit", required=True)
    p_app.add_argument("-o", "--output", required=True)
    p_app.add_argument("--allow-in-sample", action="store_true")
    p_app.set_defaults(func=_cmd_apply)

    # policy
    p_pol = sub.add_parser("policy", help="Threshold tuning & System 1->2 escalation compiler")
    sub_pol = p_pol.add_subparsers(dest="policy_cmd", required=True)
    p_pt = sub_pol.add_parser("tune", help="Compile error-budget & escalation policy.json")
    p_pt.add_argument("records")
    p_pt.add_argument("-o", "--output", required=True)
    p_pt.add_argument("--question", default=None)
    p_pt.add_argument("--model", default=None)
    p_pt.add_argument("--error-budget", type=float, default=0.10)
    p_pt.add_argument("--use-cp-upper", action="store_true")
    p_pt.add_argument("--cp-alpha", type=float, default=0.05)
    p_pt.add_argument("--cost-s1", type=float, default=0.00004)
    p_pt.add_argument("--cost-sys2", type=float, default=0.004)
    p_pt.add_argument("--max-escalation", type=float, default=0.35)
    p_pt.add_argument("--use-calib", action="store_true")
    p_pc = sub_pol.add_parser("curve", help="Generate threshold operating curve")
    p_pc.add_argument("records")
    p_pc.add_argument("--question", default=None)
    p_pc.add_argument("--model", default=None)
    p_pc.add_argument("--cost-s1", type=float, default=0.00004)
    p_pc.add_argument("--cost-sys2", type=float, default=0.004)
    p_pc.add_argument("--use-calib", action="store_true")
    p_pol.set_defaults(func=_cmd_policy)

    # slice
    p_sl = sub.add_parser("slice", help="Answerability decomposition & BH-FDR error slice mining")
    sub_sl = p_sl.add_subparsers(dest="slice_cmd", required=True)
    p_sa = sub_sl.add_parser("answerable", help="Slice metrics vs majority & blind null")
    p_sa.add_argument("records")
    p_sa.add_argument("--features", default=None)
    p_sa.add_argument("--question", default=None)
    p_sa.add_argument("--model", default=None)
    p_sa.add_argument("--min-n", type=int, default=10)
    p_sm = sub_sl.add_parser("mine", help="Mine high-error feature slices with BH-FDR control")
    p_sm.add_argument("records")
    p_sm.add_argument("--features", required=True)
    p_sm.add_argument("--question", default=None)
    p_sm.add_argument("--model", default=None)
    p_sm.add_argument("--fdr-q", type=float, default=0.10)
    p_sm.add_argument("--min-support", type=int, default=5)
    p_sl.set_defaults(func=_cmd_slice)

    # labelaudit
    p_la = sub.add_parser("labelaudit", help="Out-of-fold label disagreement & blind review sheets")
    sub_la = p_la.add_subparsers(dest="la_cmd", required=True)
    p_lar = sub_la.add_parser("rank", help="Rank candidate label errors by calibrated disagreement")
    p_lar.add_argument("records")
    p_lar.add_argument("--fit", default=None)
    p_lar.add_argument("--top", type=int, default=20)
    p_las = sub_la.add_parser("sheet", help="Generate blinded markdown review sheet + secret key")
    p_las.add_argument("records")
    p_las.add_argument("--states", required=True)
    p_las.add_argument("--fit", default=None)
    p_las.add_argument("-o", "--output", required=True)
    p_las.add_argument("-k", "--key", required=True)
    p_las.add_argument("--top", type=int, default=10)
    p_las.add_argument("--controls", type=int, default=5)
    p_las.add_argument("--options", default="True,False")
    p_las.add_argument("--seed", type=int, default=0)
    p_lae = sub_la.add_parser("estimate", help="Score completed blind sheet against secret key")
    p_lae.add_argument("sheet")
    p_lae.add_argument("--key", required=True)
    p_la.set_defaults(func=_cmd_labelaudit)

    # compare
    p_cmp = sub.add_parser("compare", help="Multi-backend shadow comparison (McNemar-Holm)")
    p_cmp.add_argument("primary")
    p_cmp.add_argument("shadow")
    p_cmp.set_defaults(func=_cmd_compare)

    # judge
    p_jdg = sub.add_parser("judge", help="Rubric judge agreement & weighted kappa audit")
    sub_jdg = p_jdg.add_subparsers(dest="judge_cmd", required=True)
    p_ja = sub_jdg.add_parser("audit", help="Compute exact, within-1, weighted kappa, and bias")
    p_ja.add_argument("grades")
    p_ja.add_argument("--pass-level", type=int, default=3)
    p_jdg.set_defaults(func=_cmd_judge)

    # drift
    p_drf = sub.add_parser("drift", help="Distribution PSI, quantile, and model_served drift detector")
    p_drf.add_argument("reference")
    p_drf.add_argument("current")
    p_drf.add_argument("--policy", default=None)
    p_drf.add_argument("--bins", type=int, default=10)
    p_drf.add_argument("--raw-only", action="store_true")
    p_drf.add_argument("--fail-on-critical", action="store_true")
    p_drf.set_defaults(func=_cmd_drift)

    # probe
    p_prb = sub.add_parser("probe", help="Budget-guarded System 1 probe & batch runner")
    p_prb.add_argument("questions")
    p_prb.add_argument("--states", required=True)
    p_prb.add_argument("-o", "--output", required=True)
    p_prb.add_argument("--ledger", required=True)
    p_prb.add_argument("--cap-usd", type=float, default=1.00)
    p_prb.add_argument("--model", default="jev-1.13.0")
    p_prb.add_argument("--served-model", default=None)
    p_prb.add_argument("--no-expand-probes", action="store_true")
    p_prb.set_defaults(func=_cmd_probe)

    # gate
    p_gte = sub.add_parser("gate", help="Evaluate runtime DecisionGate on a live System 1 output")
    p_gte.add_argument("policy")
    p_gte.add_argument("--fit", default=None)
    p_gte.add_argument("--question", required=True)
    p_gte.add_argument("--model", required=True)
    p_gte.add_argument("--p", type=float, required=True)
    p_gte.add_argument("--expected-sha256", default=None)
    p_gte.set_defaults(func=_cmd_gate)

    # export
    p_exp = sub.add_parser("export", help="Safe-v1 exporters for Langfuse and Arize Phoenix")
    sub_exp = p_exp.add_subparsers(dest="export_cmd", required=True)
    p_el = sub_exp.add_parser("langfuse", help="Export safe-v1 Langfuse Dataset Run & Scores")
    p_el.add_argument("records")
    p_el.add_argument("-o", "--output", required=True)
    p_el.add_argument("--policy", default=None)
    p_el.add_argument("--run-name", default="s1ops-eval")
    p_ep = sub_exp.add_parser("phoenix", help="Export safe-v1 OpenInference spans JSONL")
    p_ep.add_argument("records")
    p_ep.add_argument("-o", "--output", required=True)
    p_ep.add_argument("--policy", default=None)
    p_ep.add_argument("--run-name", default="s1ops-eval")
    p_exp.set_defaults(func=_cmd_export)

    # ledger
    p_led = sub.add_parser("ledger", help="Reserve-then-settle budget ledger operations")
    sub_led = p_led.add_subparsers(dest="ledger_cmd", required=True)
    p_ls = sub_led.add_parser("status", help="Inspect ledger USD balances")
    p_ls.add_argument("path")
    p_lm = sub_led.add_parser("migrate", help="Migrate legacy ledger to reserve/settle journal")
    p_lm.add_argument("path")
    p_lm.add_argument("-o", "--output", required=True)
    p_lm.add_argument("--acknowledge-lost-reservations", action="store_true")
    p_lr = sub_led.add_parser("recover", help="Recover journal from trailing partial write")
    p_lr.add_argument("path")
    p_lr.add_argument("-o", "--output", required=True)
    p_led.set_defaults(func=_cmd_ledger)

    # web
    p_web = sub.add_parser("web", help="Launch local System 1 LLMOps Web Workbench (127.0.0.1)")
    p_web.add_argument("--workspace", default=".")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.set_defaults(func=_cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

