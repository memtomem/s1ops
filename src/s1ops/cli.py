"""Unified CLI for s1ops (`s1` and `s1ops` commands).

Provides offline-first System 1 evaluation, calibration, policy compilation,
slice & blind label audit, shadow comparison, judge agreement, Langfuse/Phoenix
safe-v1 exporters, ledger management, and the local Web Workbench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from s1ops import __version__
from s1ops.bridges.langfuse import export_langfuse_file
from s1ops.bridges.openinference import export_openinference_jsonl
from s1ops.contracts.records import Selection, read_records, select
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
    BudgetExceeded,
    Ledger,
    migrate as ledger_migrate,
    read as ledger_read,
    recover as ledger_recover,
)
from s1ops.lint.lint import audit_probe_records, format_text, lint_path
from s1ops.lint.rules import RULES
from s1ops.monitor.drift import audit_drift
from s1ops.monitor.runner import DeterministicMockBackend, run_batch
from s1ops.runtime.gate import DecisionGate, ModelServedMismatchError, PolicyIntegrityError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_IN_SAMPLE = 2
EXIT_MODEL_MISMATCH = 3
EXIT_DRIFT_CRITICAL = 4


class CliError(Exception):
    """A user-facing failure: printed as one `s1: <message>` line, no traceback."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


def _fail(message: str, code: int = EXIT_ERROR) -> int:
    print(f"s1: {message}", file=sys.stderr)
    return code


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 1, so exit 2 stays reserved for the in-sample refusal."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"s1: {message}\n")


# Input arguments by dest name: (what it is, the command that produces it).
# Output paths (`-o`, `labelaudit sheet --key`, `probe --ledger`) are deliberately absent.
_INPUTS: dict[str, tuple[str, str | None]] = {
    "records": ("records file", None),
    "reference": ("reference records file", None),
    "current": ("current records file", None),
    "primary": ("primary records file", None),
    "shadow": ("shadow records file", None),
    "questions": ("questions file", None),
    "grades": ("grades file", None),
    "features": ("features file", None),
    "states": ("states file", None),
    "path": ("input file", None),
    "fit": ("fits file", "s1 fit"),
    "policy": ("policy file", "s1 policy tune"),
    "sheet": ("review sheet", "s1 labelaudit sheet"),
}


def _require_file(value: str, what: str, producer: str | None) -> None:
    p = Path(value)
    if p.is_dir():
        raise CliError(f"{what} is a directory: {p}")
    if not p.exists():
        hint = f" (run `{producer}` first)" if producer else ""
        raise CliError(f"{what} not found: {p}{hint}")


def _check_inputs(args: argparse.Namespace) -> None:
    if getattr(args, "lint_cmd", None) == "check":
        # lint reports an unreadable questions file as a JL000 finding in its own (--format json) report.
        return
    for dest, (what, producer) in _INPUTS.items():
        value = getattr(args, dest, None)
        if value is not None:
            _require_file(value, what, producer)


def _no_groups(args: argparse.Namespace, sel: Selection, records_path: str) -> CliError:
    filters = [
        f"--{name} {value}"
        for name, value in (
            ("question", getattr(args, "question", None)),
            ("model", getattr(args, "model", None)),
            ("split", getattr(args, "split", None)),
        )
        if value
    ]
    where = f" matching {' '.join(filters)}" if filters else ""
    skipped = (
        f"{len(sel.groups)} group(s) before filters; skipped {sel.no_response} without a response, "
        f"{sel.unlabeled} unlabeled, {sel.other_split} in other splits"
    )
    return CliError(f"no labeled groups{where} in {records_path} ({skipped})")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CliError(f"{path}: not valid JSON: {e}") from None


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, or a JSON array of objects; every row must be an object."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        rows = _read_json(path)
        if not isinstance(rows, list):
            raise CliError(f"{path}: expected a JSON array of objects")
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CliError(f"{path}: element {i}: expected a JSON object")
        return rows
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise CliError(f"{path}: line {n}: not valid JSON: {e}") from None
        if not isinstance(row, dict):
            raise CliError(f"{path}: line {n}: expected a JSON object")
        out.append(row)
    return out


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
    if not reports:
        raise _no_groups(args, sel, args.records)
    print(json.dumps({"groups": reports}, indent=2, sort_keys=True))
    return EXIT_OK


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
        raise _no_groups(args, sel, args.records)
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
            code = EXIT_MODEL_MISMATCH if "model_served" in msg else EXIT_ERROR
            return _fail(f"refusing to apply fit: {msg}", code)

        err = refusal(fits, rec.key, rec.classes)
        if err is not None:
            return _fail(f"refusing to apply fit: {err}")

        in_sample = rec.item in in_sample_map.get(fit.key, frozenset())
        if in_sample and not args.allow_in_sample:
            return _fail(
                f"refusing in-sample calibration apply on {rec.item}; "
                "pass --allow-in-sample to override",
                EXIT_IN_SAMPLE,
            )

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
        raise _no_groups(args, sel, args.records)
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
        _require_file(args.key, "secret key", "s1 labelaudit sheet")
        form_text = Path(args.sheet).read_text(encoding="utf-8")
        key_doc = _read_json(Path(args.key))
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
    for i, row in enumerate(states_rows, start=1):
        if "item_id" not in row or "state" not in row:
            raise CliError(f"{args.states}: row {i} is missing 'item_id' or 'state'")
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
    if not rows:
        raise CliError(f"no grade rows in {args.grades}")
    for i, r in enumerate(rows, start=1):
        if "judge" not in r or "human" not in r:
            raise CliError(f"{args.grades}: row {i} is missing 'judge' or 'human'")
        if not all(isinstance(r[k], int) and not isinstance(r[k], bool) and r[k] >= 0 for k in ("judge", "human")):
            raise CliError(f"{args.grades}: row {i}: 'judge' and 'human' must be non-negative integers")
    j_vals = [r["judge"] for r in rows]
    h_vals = [r["human"] for r in rows]
    levels = max(max(j_vals), max(h_vals)) + 1
    if levels < 2:
        raise CliError(f"{args.grades}: every grade is 0; agreement needs grades on at least two levels")
    crit_tuples = [
        (str(r.get("criterion", "default")), r["judge"], r["human"])
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
        return EXIT_DRIFT_CRITICAL
    return EXIT_OK


def _cmd_probe(args: argparse.Namespace) -> int:
    q_doc = _read_json(Path(args.questions))
    if isinstance(q_doc, dict) and "questions" in q_doc and isinstance(q_doc["questions"], dict):
        q_map = q_doc["questions"]
    elif isinstance(q_doc, dict):
        q_map = {k: v for k, v in q_doc.items() if isinstance(v, dict) and "type" in v}
    else:
        raise CliError(f"{args.questions}: must be a JSON object mapping names to questions")

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
    try:
        dec = gate.evaluate_noul(
            question_id=args.question,
            model_served=args.model,
            p_raw=args.p,
            expected_policy_sha256=args.expected_sha256,
        )
    except KeyError as e:
        raise CliError(f"{e.args[0]} in {args.policy}") from None
    print(json.dumps(dec.as_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    # Only a new or empty directory: the generated names include fits.json and policy.json,
    # which a real workspace also holds, so nothing is ever overwritten.
    out = Path(args.out_dir)
    if out.exists() and not out.is_dir():
        raise CliError(f"demo target is not a directory: {out}")
    if out.is_dir() and any(out.iterdir()):
        raise CliError(f"demo target is not empty: {out} (remove it or choose a new directory)")

    from s1ops.demo import generate_workspace

    paths = generate_workspace(out, seed=args.seed)
    print(json.dumps(paths, indent=2, sort_keys=True))
    print(f"next: s1 web --workspace {shlex.quote(str(out))}", file=sys.stderr)
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    if not Path(args.workspace).is_dir():
        raise CliError(f"workspace directory not found: {args.workspace}")

    import uvicorn

    from s1ops.web.app import create_app

    app = create_app(workspace_dir=args.workspace)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show `(default: ...)` only when the default says something; argparse passes this per subparser."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default in (None, False, argparse.SUPPRESS):
            return action.help
        return super()._get_help_string(action)


_FMT = {"formatter_class": _HelpFormatter}

_H_QUESTION = "only the group with this question_id"
_H_MODEL = "only the group with this model_served"
_H_RECORDS = "decision records (jev-record/1 or s1-record/1 JSONL)"
_H_OUTPUT = "output path"
_H_FIT_IN = "fits.json from `s1 fit`"
_H_POLICY_IN = "policy.json from `s1 policy tune`"
_H_USE_CALIB = "use the calibrated probabilities written by `s1 apply` instead of the raw ones"
_H_COST_S1 = "USD cost of one System 1 call"
_H_COST_SYS2 = "USD cost of escalating one item to System 2 (LLM or human)"


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="s1",
        description="s1ops: Dedicated System 1 LLMOps CLI & Workbench",
        **_FMT,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # lint
    p_lint = sub.add_parser("lint", help="Static question lint & probe coherence", **_FMT)
    sub_lint = p_lint.add_subparsers(dest="lint_cmd", required=True)
    p_lc = sub_lint.add_parser("check", help="Lint a question specification file", **_FMT)
    p_lc.add_argument("path", help="questions JSON file")
    p_lc.add_argument("--format", choices=["text", "json"], default="text", help="report format")
    p_lc.add_argument("--strict", action="store_true", help="exit 1 on warnings as well as errors")
    p_lc.add_argument("--ignore", action="append", help="rule id to suppress (repeatable); JL000/JL001/JL003/JL020 cannot be suppressed")
    sub_lint.add_parser("rules", help="List JL000-JL060 rules", **_FMT)
    p_lp = sub_lint.add_parser("probe-audit", help="Audit Noul/Choice twin & inverse coherence", **_FMT)
    p_lp.add_argument("records", help="probe records JSONL from `s1 probe`")
    p_lint.set_defaults(func=_cmd_lint)

    # audit
    p_aud = sub.add_parser("audit", help="Calibration & noise floor audit per group", **_FMT)
    p_aud.add_argument("records", help=_H_RECORDS)
    p_aud.add_argument("--fit", default=None, help=f"{_H_FIT_IN}; adds an after-calibration report")
    p_aud.add_argument("--question", default=None, help=_H_QUESTION)
    p_aud.add_argument("--model", default=None, help=_H_MODEL)
    p_aud.add_argument("--split", default=None, help="only records whose `split` field equals this")
    p_aud.add_argument("--bins", type=int, default=10, help="reliability/ECE bins")
    p_aud.add_argument("--boot", type=int, default=200, help="Monte Carlo replicates for the ECE noise floor and bootstrap CIs")
    p_aud.add_argument("--seed", type=int, default=0, help="random seed")
    p_aud.set_defaults(func=_cmd_audit)

    # fit
    p_fit = sub.add_parser("fit", help="Fit per-group post-hoc calibration (temp/platt/isotonic)", **_FMT)
    p_fit.add_argument("records", help=f"calibration-fold {_H_RECORDS}")
    p_fit.add_argument("-o", "--output", required=True, help="fits.json to write")
    p_fit.add_argument(
        "--method",
        choices=["temperature", "platt", "isotonic"],
        default="temperature",
        help="calibration map; platt and isotonic are noul-only",
    )
    p_fit.add_argument("--question", default=None, help=_H_QUESTION)
    p_fit.add_argument("--model", default=None, help=_H_MODEL)
    p_fit.add_argument("--split", default=None, help="only records whose `split` field equals this; the fit is tagged `cal` when unset")
    p_fit.add_argument("--min-n", type=int, default=30, help="groups with fewer labeled records are still fitted but get a small-sample warning")
    p_fit.set_defaults(func=_cmd_fit)

    # apply
    p_app = sub.add_parser("apply", help="Apply calibration fit to held-out records", **_FMT)
    p_app.add_argument("records", help=f"held-out {_H_RECORDS}")
    p_app.add_argument("--fit", required=True, help=_H_FIT_IN)
    p_app.add_argument("-o", "--output", required=True, help="calibrated records JSONL to write")
    p_app.add_argument("--allow-in-sample", action="store_true", help="apply even to items the fit was trained on (otherwise exit 2); they are marked in_fit_sample")
    p_app.set_defaults(func=_cmd_apply)

    # policy
    p_pol = sub.add_parser("policy", help="Threshold tuning & System 1->2 escalation compiler", **_FMT)
    sub_pol = p_pol.add_subparsers(dest="policy_cmd", required=True)
    p_pt = sub_pol.add_parser("tune", help="Compile error-budget & escalation policy.json", **_FMT)
    p_pt.add_argument("records", help=_H_RECORDS)
    p_pt.add_argument("-o", "--output", required=True, help="policy.json to write")
    p_pt.add_argument("--question", default=None, help=_H_QUESTION)
    p_pt.add_argument("--model", default=None, help=_H_MODEL)
    p_pt.add_argument("--error-budget", type=float, default=0.10, help="maximum error rate among auto-accepted items; the threshold is the lowest tie-safe one that meets it")
    p_pt.add_argument("--use-cp-upper", action="store_true", help="judge the budget against the Clopper-Pearson upper bound instead of the point estimate")
    p_pt.add_argument("--cp-alpha", type=float, default=0.05, help="significance level of the Clopper-Pearson bound")
    p_pt.add_argument("--cost-s1", type=float, default=0.00004, help=_H_COST_S1)
    p_pt.add_argument("--cost-sys2", type=float, default=0.004, help=_H_COST_SYS2)
    p_pt.add_argument("--max-escalation", type=float, default=0.35, help="largest fraction of items the noul escalation band may send to System 2")
    p_pt.add_argument("--use-calib", action="store_true", help=_H_USE_CALIB)
    p_pc = sub_pol.add_parser("curve", help="Generate threshold operating curve", **_FMT)
    p_pc.add_argument("records", help=_H_RECORDS)
    p_pc.add_argument("--question", default=None, help=_H_QUESTION)
    p_pc.add_argument("--model", default=None, help=_H_MODEL)
    p_pc.add_argument("--cost-s1", type=float, default=0.00004, help=_H_COST_S1)
    p_pc.add_argument("--cost-sys2", type=float, default=0.004, help=_H_COST_SYS2)
    p_pc.add_argument("--use-calib", action="store_true", help=_H_USE_CALIB)
    p_pol.set_defaults(func=_cmd_policy)

    # slice
    p_sl = sub.add_parser("slice", help="Answerability decomposition & BH-FDR error slice mining", **_FMT)
    sub_sl = p_sl.add_subparsers(dest="slice_cmd", required=True)
    p_sa = sub_sl.add_parser("answerable", help="Slice metrics vs majority & blind null", **_FMT)
    p_sa.add_argument("records", help=_H_RECORDS)
    p_sa.add_argument("--features", default=None, help="features JSONL tagging each item with a `slice`; untagged items fall in `default`")
    p_sa.add_argument("--question", default=None, help=_H_QUESTION)
    p_sa.add_argument("--model", default=None, help=_H_MODEL)
    p_sa.add_argument("--min-n", type=int, default=10, help="slices with fewer items are reported but never marked answerable")
    p_sm = sub_sl.add_parser("mine", help="Mine high-error feature slices with BH-FDR control", **_FMT)
    p_sm.add_argument("records", help=_H_RECORDS)
    p_sm.add_argument("--features", required=True, help="features JSONL with pre-declared per-item feature tags")
    p_sm.add_argument("--question", default=None, help=_H_QUESTION)
    p_sm.add_argument("--model", default=None, help=_H_MODEL)
    p_sm.add_argument("--fdr-q", type=float, default=0.10, help="Benjamini-Hochberg false discovery rate")
    p_sm.add_argument("--min-support", type=int, default=5, help="minimum items in a feature slice to test it")
    p_sl.set_defaults(func=_cmd_slice)

    # labelaudit
    p_la = sub.add_parser("labelaudit", help="Out-of-fold label disagreement & blind review sheets", **_FMT)
    sub_la = p_la.add_subparsers(dest="la_cmd", required=True)
    p_lar = sub_la.add_parser("rank", help="Rank candidate label errors by calibrated disagreement", **_FMT)
    p_lar.add_argument("records", help=_H_RECORDS)
    p_lar.add_argument("--fit", default=None, help=f"{_H_FIT_IN}; its temperatures rescale the disagreement score")
    p_lar.add_argument("--top", type=int, default=20, help="candidates to print")
    p_las = sub_la.add_parser("sheet", help="Generate blinded markdown review sheet + secret key", **_FMT)
    p_las.add_argument("records", help=_H_RECORDS)
    p_las.add_argument("--states", required=True, help="local-only states JSONL (`item_id`, `state`) to show reviewers; never written to records")
    p_las.add_argument("--fit", default=None, help=f"{_H_FIT_IN}; its temperatures rescale the disagreement score")
    p_las.add_argument("-o", "--output", required=True, help="markdown review sheet to write")
    p_las.add_argument("-k", "--key", required=True, help="secret answer key JSON to write; keep it away from reviewers")
    p_las.add_argument("--top", type=int, default=10, help="flagged candidates on the sheet")
    p_las.add_argument("--controls", type=int, default=5, help="random control items mixed in to measure baseline disagreement")
    p_las.add_argument("--options", default="True,False", help="comma-separated answer options shown to reviewers")
    p_las.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p_lae = sub_la.add_parser("estimate", help="Score completed blind sheet against secret key", **_FMT)
    p_lae.add_argument("sheet", help="review sheet completed by the reviewer")
    p_lae.add_argument("--key", required=True, help="secret key JSON from `s1 labelaudit sheet`")
    p_la.set_defaults(func=_cmd_labelaudit)

    # compare
    p_cmp = sub.add_parser("compare", help="Multi-backend shadow comparison (McNemar-Holm)", **_FMT)
    p_cmp.add_argument("primary", help=f"primary backend {_H_RECORDS}")
    p_cmp.add_argument("shadow", help=f"shadow backend {_H_RECORDS} on the same items")
    p_cmp.set_defaults(func=_cmd_compare)

    # judge
    p_jdg = sub.add_parser("judge", help="Rubric judge agreement & weighted kappa audit", **_FMT)
    sub_jdg = p_jdg.add_subparsers(dest="judge_cmd", required=True)
    p_ja = sub_jdg.add_parser("audit", help="Compute exact, within-1, weighted kappa, and bias", **_FMT)
    p_ja.add_argument("grades", help="JSONL rows with integer `judge` and `human` grades and an optional `criterion`")
    p_ja.add_argument("--pass-level", type=int, default=3, help="grade at or above which an item counts as passing, for pass/fail agreement")
    p_jdg.set_defaults(func=_cmd_judge)

    # drift
    p_drf = sub.add_parser("drift", help="Distribution PSI, quantile, and model_served drift detector", **_FMT)
    p_drf.add_argument("reference", help=f"baseline {_H_RECORDS}")
    p_drf.add_argument("current", help=f"current/canary {_H_RECORDS}")
    p_drf.add_argument("--policy", default=None, help=f"{_H_POLICY_IN}; enables escalation-rate drift")
    p_drf.add_argument("--bins", type=int, default=10, help="histogram bins for PSI and Jensen-Shannon")
    p_drf.add_argument("--raw-only", action="store_true", help="compare raw probabilities even when records carry calibrated ones")
    p_drf.add_argument("--fail-on-critical", action="store_true", help="exit 4 if any question is CRITICAL")
    p_drf.set_defaults(func=_cmd_drift)

    # probe
    p_prb = sub.add_parser("probe", help="Budget-guarded System 1 probe & batch runner", **_FMT)
    p_prb.add_argument("questions", help="questions JSON file")
    p_prb.add_argument("--states", required=True, help="local-only states JSONL; only state_sha256 reaches the output")
    p_prb.add_argument("-o", "--output", required=True, help="probe records JSONL to write")
    p_prb.add_argument("--ledger", required=True, help="budget journal; created if absent")
    p_prb.add_argument("--cap-usd", type=float, default=1.00, help="hard spend cap; the run stops before a call whose worst-case cost would exceed it")
    p_prb.add_argument("--model", default="jev-1.13.0", help="model_requested sent to the backend")
    p_prb.add_argument("--served-model", default=None, help="model_served the mock backend reports (defaults to --model); set it differently to simulate a silent upgrade")
    p_prb.add_argument("--no-expand-probes", action="store_true", help="send only the questions as written, without twin and inverse probes")
    p_prb.set_defaults(func=_cmd_probe)

    # gate
    p_gte = sub.add_parser("gate", help="Evaluate runtime DecisionGate on a live System 1 output", **_FMT)
    p_gte.add_argument("policy", help=_H_POLICY_IN)
    p_gte.add_argument("--fit", default=None, help=f"{_H_FIT_IN}; calibrates --p when the policy was compiled with --use-calib")
    p_gte.add_argument("--question", required=True, help="question_id of the live call")
    p_gte.add_argument("--model", required=True, help="model_served of the live call; a mismatch with the policy exits 3")
    p_gte.add_argument("--p", type=float, required=True, help="raw noul probability returned by the model")
    p_gte.add_argument("--expected-sha256", default=None, help="expected policy_sha256; a mismatch exits 1")
    p_gte.set_defaults(func=_cmd_gate)

    # export
    p_exp = sub.add_parser("export", help="Safe-v1 exporters for Langfuse and Arize Phoenix", **_FMT)
    sub_exp = p_exp.add_subparsers(dest="export_cmd", required=True)
    p_el = sub_exp.add_parser("langfuse", help="Export safe-v1 Langfuse Dataset Run & Scores", **_FMT)
    p_el.add_argument("records", help=_H_RECORDS)
    p_el.add_argument("-o", "--output", required=True, help="Langfuse export JSON to write")
    p_el.add_argument("--policy", default=None, help=f"{_H_POLICY_IN}; routes each record with the compiled threshold or band instead of a fixed 0.5")
    p_el.add_argument("--run-name", default="s1ops-eval", help="dataset run name")
    p_ep = sub_exp.add_parser("phoenix", help="Export safe-v1 OpenInference spans JSONL", **_FMT)
    p_ep.add_argument("records", help=_H_RECORDS)
    p_ep.add_argument("-o", "--output", required=True, help="OpenInference spans JSONL to write")
    p_ep.add_argument("--policy", default=None, help=f"{_H_POLICY_IN}; routes with the compiled threshold or band and adds s1.policy_sha256, s1.threshold, s1.band_lo, s1.band_hi")
    p_ep.add_argument("--run-name", default="s1ops-eval", help="run name attribute")
    p_exp.set_defaults(func=_cmd_export)

    # ledger
    p_led = sub.add_parser("ledger", help="Reserve-then-settle budget ledger operations", **_FMT)
    sub_led = p_led.add_subparsers(dest="ledger_cmd", required=True)
    p_ls = sub_led.add_parser("status", help="Inspect ledger USD balances", **_FMT)
    p_ls.add_argument("path", help="ledger journal")
    p_lm = sub_led.add_parser("migrate", help="Migrate legacy ledger to reserve/settle journal", **_FMT)
    p_lm.add_argument("path", help="legacy ledger")
    p_lm.add_argument("-o", "--output", required=True, help="new journal to create; must not exist")
    p_lm.add_argument("--acknowledge-lost-reservations", action="store_true", help="required: a legacy ledger cannot show reservations lost before migration")
    p_lr = sub_led.add_parser("recover", help="Recover journal from trailing partial write", **_FMT)
    p_lr.add_argument("path", help="journal with a partial trailing write")
    p_lr.add_argument("-o", "--output", required=True, help="recovered journal to write")
    p_led.set_defaults(func=_cmd_ledger)

    # demo
    p_demo = sub.add_parser("demo", help="Write a seeded synthetic demo workspace", **_FMT)
    p_demo.add_argument("out_dir", help="directory to create, or an existing empty one")
    p_demo.add_argument("--seed", type=int, default=42, help="random seed for the generated data")
    p_demo.set_defaults(func=_cmd_demo)

    # web
    p_web = sub.add_parser("web", help="Launch local System 1 LLMOps Web Workbench (127.0.0.1)", **_FMT)
    p_web.add_argument("--workspace", default=".", help="directory holding records.jsonl, fits.json, policy.json, ...")
    p_web.add_argument("--host", default="127.0.0.1", help="bind address")
    p_web.add_argument("--port", type=int, default=8765, help="bind port")
    p_web.set_defaults(func=_cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    debug = bool(os.environ.get("S1_DEBUG"))
    try:
        _check_inputs(args)
        return int(args.func(args))
    except CliError as e:
        return _fail(str(e), e.code)
    except ModelServedMismatchError as e:
        if debug:
            raise
        return _fail(str(e), EXIT_MODEL_MISMATCH)
    except OSError as e:
        if debug:
            raise
        return _fail(f"{e.filename}: {e.strerror}" if e.filename else str(e))
    except (ValueError, PolicyIntegrityError, BudgetExceeded) as e:
        # ValueError covers RecordError, FitError, LedgerCorrupt and json.JSONDecodeError.
        if debug:
            raise
        return _fail(str(e))


if __name__ == "__main__":
    sys.exit(main())

