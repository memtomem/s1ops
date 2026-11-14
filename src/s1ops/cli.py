"""Unified CLI for s1ops (`s1` and `s1ops` commands).

Provides offline-first System 1 evaluation, calibration, policy compilation,
slice & blind label audit, shadow comparison, judge agreement, Langfuse/Phoenix
safe-v1 exporters, ledger management, and the local Web Workbench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from s1ops import __version__
from s1ops.bridges.langfuse import export_langfuse_file
from s1ops.bridges.openinference import export_openinference_jsonl
from s1ops.contracts._validation import UNDECODABLE
from s1ops.contracts.records import Group, GroupKey, Record, Selection, read_records, read_records_hashed, select, surrogate_problem
from s1ops.contracts.schemas import PolicyEntry, load_policies, save_policies
from s1ops.eval.audit import audit_group
from s1ops.eval.compare import compare_records
from s1ops.eval.judge import agreement as judge_agreement
from s1ops.eval.judge import by_criterion as judge_by_criterion
from s1ops.eval.metrics import routed_confidence, top_label
from s1ops.improve.calibrate import (
    AppliedFits,
    Fit,
    FitApplyError,
    apply_fits,
    fit_for,
    fit_group,
    load_fits,
    save_fits,
    transform,
)
from s1ops.improve.labelaudit import (
    LabelAuditInputError,
    SheetInputError,
    build_sheet,
    check_key,
    estimate,
    rank_with_skips,
    score_sheet,
)
from s1ops.improve.policy import (
    Band,
    CalibInputError,
    DevRecord,
    Prices,
    choose,
    grid as band_grid,
    select_false_pass_cutoff,
    select_threshold,
    sweep,
    threshold_curve,
    tuned_fit_sha256,
    tuned_rows,
)
from s1ops.improve.slices import FeatureKey, mine_error_slices, parse_feature_rows, slice_inputs_by_group, slice_report
from s1ops.improve.trainset import Trained, TrainedItemError, build_manifest, check_not_trained, load_manifest, load_trained, save_manifest
from s1ops.ledger.journal import (
    BudgetExceeded,
    Ledger,
    lock_path,
    migrate as ledger_migrate,
    read as ledger_read,
    recover as ledger_recover,
)
from s1ops.lint.lint import KNOWN_RULES, ProbeInputError, audit_probe_records, format_text, lint_path
from s1ops.lint.loader import parse_question, read_question_entries
from s1ops.lint.rules import RULES
from s1ops.monitor.drift import audit_drift
from s1ops.monitor.backends import HttpSystemOneBackend, OpenAIDecisionsBackend
from s1ops.monitor.runner import (
    DeterministicMockBackend,
    check_labels,
    check_questions,
    check_reservation,
    check_states,
    probe_tasks,
    run_batch,
)
from s1ops.runtime.gate import DecisionGate, ModelServedMismatchError, PolicyIntegrityError
from s1ops.runtime.routing import routed_metrics

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_IN_SAMPLE = 2
EXIT_MODEL_MISMATCH = 3
EXIT_DRIFT_CRITICAL = 4
_FIT_APPLY_EXIT = {"in_sample": EXIT_IN_SAMPLE, "model_mismatch": EXIT_MODEL_MISMATCH, "other": EXIT_ERROR}


class CliError(Exception):
    """A user-facing failure: printed as one `s1: <message>` line (escaped by `_one_line`), no
    traceback."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


def _one_line(message: str) -> str:
    """`message` with every non-printable character escaped (`\\n`, `\\x1b`, `\\u202e`, ...), so an
    error line stays one line, and cannot recolor or reorder the terminal, whatever ids or paths
    it interpolates. Printable text, non-ASCII included, and backslashes are kept as they are."""
    return "".join(c if c.isprintable() else c.encode("unicode_escape").decode("ascii") for c in message)


def _fail(message: str, code: int = EXIT_ERROR) -> int:
    print(f"s1: {_one_line(message)}", file=sys.stderr)
    return code


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 1, so exit 2 stays reserved for the in-sample refusal."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.exit(EXIT_ERROR, f"s1: {_one_line(f'{message}; see {self.prog} -h')}\n")


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
    "training_manifest": ("training manifest", "s1 trainset export"),
    "parent": ("parent training manifest", "s1 trainset export"),
}


def _require_file(value: str, what: str, producer: str | None) -> None:
    """Refuse a missing input, a directory, or one that cannot be inspected, naming why. Other
    non-regular files (a FIFO from `<(...)` process substitution) are accepted."""
    p = Path(value)
    try:
        st = os.stat(p)
    except (FileNotFoundError, NotADirectoryError):
        try:
            os.lstat(p)
        except OSError:
            hint = f" (run `{producer}` first)" if producer else ""
            raise CliError(f"{what} not found: {p}{hint}") from None
        raise CliError(f"{what} is a dangling symbolic link: {p}") from None
    except OSError as e:  # a symlink loop, no search permission on a parent
        raise CliError(f"{what} cannot be read: {p}: {e.strerror or e}") from None
    if stat.S_ISDIR(st.st_mode):
        raise CliError(f"{what} is a directory: {p}")


def _check_inputs(args: argparse.Namespace) -> None:
    if getattr(args, "lint_cmd", None) == "check":
        # lint reports an unreadable questions file as a JL000 finding in its own (--format json) report.
        return
    for dest, (what, producer) in _INPUTS.items():
        value = getattr(args, dest, None)
        for one in value if isinstance(value, list) else [value] if value is not None else []:
            _require_file(one, what, producer)


def _selected(args: argparse.Namespace, sel: Selection) -> list[Group]:
    """The groups `--question` and `--model` leave."""
    return [
        grp
        for grp in sel.groups
        if not (args.question and grp.question_id != args.question) and not (args.model and grp.model_served != args.model)
    ]


def _trained(args: argparse.Namespace, records: list[Record]) -> Trained:
    """What `--training-manifest` says each checkpoint trained on, bound to the whole input file."""
    paths = args.training_manifest or []
    return load_trained(paths, [r.model_served for r in records], args.records) if paths else {}


def _refuse_trained(args: argparse.Namespace, groups: list[Group], trained: Trained) -> None:
    for grp in groups:
        check_not_trained(grp.key, grp.items, trained, args.records)


def _tuned_fit_sha256(grp: Group, rows: list[Record]) -> str | None:
    try:
        return tuned_fit_sha256(grp, rows)
    except CalibInputError as e:
        raise CliError(str(e)) from None


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


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise CliError(f"{path}: not valid UTF-8: {e}") from None


def _decode_json(text: str, where: str) -> Any:
    """The JSON value in `text`; a failure is one error line starting with `where` (a path, or a path
    and line). Besides a syntax error, the decoder refuses an integer past Python's digit limit with a
    plain `ValueError` and overflows the stack on deep nesting, which would otherwise escape as a
    message without the file or as a traceback (`read_json_file`'s wording for both)."""
    try:
        return json.loads(text)
    except ValueError as e:  # JSONDecodeError, or the integer-digit limit
        raise CliError(f"{where}: not valid JSON: {e}") from None
    except RecursionError:
        raise CliError(f"{where}: not valid JSON: nested too deeply") from None


def _read_json(path: Path) -> Any:
    return _decode_json(_read_text(path), str(path))


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, or a JSON array of objects; every row must be an object."""
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        # Decode the text already read: a FIFO (`<(cat rows.json)`) has only the one read.
        bad = UNDECODABLE.search(text)
        if bad:
            line = text.count("\n", 0, bad.start()) + 1
            raise CliError(f"{path}: line {line}: not valid UTF-8")
        rows = _decode_json(text, str(path))
        if not isinstance(rows, list):
            raise CliError(f"{path}: expected a JSON array of objects")
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CliError(f"{path}: element {i}: expected a JSON object")
        return rows
    out: list[dict[str, Any]] = []
    # read_text already turned \r\n and \r into \n; str.splitlines would also split on \x0b, \x1c
    # or U+2028, which a JSON string may hold raw, and count lines the file does not have.
    for n, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if UNDECODABLE.search(line):
            raise CliError(f"{path}: line {n}: not valid UTF-8")
        row = _decode_json(line, f"{path}: line {n}")
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
        try:
            report = audit_probe_records(args.records)
        except ProbeInputError as e:
            raise CliError(f"{args.records}: {e}") from None
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
    trained = _trained(args, records)
    sel = select(records, split=args.split, allow_model_mismatch=True)
    selected = _selected(args, sel)
    _refuse_trained(args, selected, trained)
    reports = []
    for grp in selected:
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
    records, src_sha = read_records_hashed(args.records)
    trained = _trained(args, records)
    sel = select(records, split=args.split, allow_model_mismatch=True)
    selected = _selected(args, sel)
    _refuse_trained(args, selected, trained)
    fits: list[Fit] = []
    for grp in selected:
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
    # `safe-v1`: no row is written with its item `state`.
    state_dropped = sum(1 for rec in records if "state" in rec.raw)
    applied = apply_fits(records, fits, allow_in_sample=args.allow_in_sample, pass_through_unfitted=args.pass_through_unfitted)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for line in applied.lines:
            fh.write(line + "\n")
    summary: dict[str, Any] = {"calibrated": applied.calibrated}
    if args.pass_through_unfitted:
        summary["passed_through"] = applied.passed_through
    if state_dropped:
        summary["state_dropped"] = state_dropped
    if len(summary) > 1:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _check_policy_args(args: argparse.Namespace) -> None:
    """Refuse an out-of-range number before any group is read, whichever primitive the groups turn out to be."""
    for flag, value in (("--cost-s1", args.cost_s1), ("--cost-sys2", args.cost_sys2)):
        if not (math.isfinite(value) and value >= 0.0):
            raise CliError(f"{flag} must be a finite non-negative number, got {value!r}")
    if args.policy_cmd != "tune":
        return
    if args.max_escalation is not None and not (math.isfinite(args.max_escalation) and 0.0 <= args.max_escalation <= 1.0):
        raise CliError(f"--max-escalation must be a number in [0, 1], got {args.max_escalation!r}")
    if args.use_cp_upper and not (math.isfinite(args.cp_alpha) and 0.0 < args.cp_alpha < 1.0):
        raise CliError(f"--cp-alpha must be in (0, 1), got {args.cp_alpha!r}")


def _cmd_policy(args: argparse.Namespace) -> int:
    _check_policy_args(args)
    records, src_sha = read_records_hashed(args.records, use_calib=args.use_calib)
    trained = _trained(args, records)
    sel = select(records, allow_model_mismatch=True)
    selected = _selected(args, sel)
    _refuse_trained(args, selected, trained)
    tuned = tuned_rows(records) if args.use_calib else {}
    if args.policy_cmd == "curve":
        curves: list[dict[str, Any]] = []
        for grp in selected:
            if args.use_calib:
                _tuned_fit_sha256(grp, tuned[grp.key])
            # A point counts what an `error_budget` policy at its threshold routes.
            conf, correct, escalated = routed_confidence(grp.primitive, grp.P, grp.y)
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
                        always_escalated=escalated,
                    ),
                }
            )
        if not curves:
            raise _no_groups(args, sel, args.records)
        print(json.dumps({"groups": curves}, indent=2, sort_keys=True))
        return 0

    false_pass = args.max_false_pass
    if false_pass is not None and args.max_escalation is not None:
        raise CliError("--max-escalation sizes the F1 band, which --max-false-pass replaces; pass one of them")
    by_threshold = args.noul_routing == "threshold"
    if by_threshold and args.max_escalation is not None:
        raise CliError("--max-escalation sizes the F1 band, which --noul-routing threshold replaces; pass one of them")
    if by_threshold and false_pass is not None:
        raise CliError("--noul-routing threshold and --max-false-pass each replace the F1 band; pass one of them")
    max_escalation = 0.35 if args.max_escalation is None else args.max_escalation
    if false_pass is not None and selected and not any(grp.primitive == "noul" for grp in selected):
        raise CliError("--max-false-pass applies to noul groups, and none is selected")
    if by_threshold and selected and not any(grp.primitive == "noul" for grp in selected):
        raise CliError("--noul-routing applies to noul groups, and none is selected")
    policies: list[PolicyEntry] = []
    for grp in selected:
        if false_pass is not None and grp.primitive == "noul":
            policies.append(_false_pass_policy(args, grp, src_sha, tuned))
            continue
        # A threshold that routes is tuned on what it routes (a choice or score tie at the top is
        # escalated, so left out). Under the noul band it does not route, and keeps `top_label`'s,
        # so a band policy's file stays as it was.
        if grp.primitive == "noul" and not by_threshold:
            conf, correct = top_label(grp.P, grp.y)
        else:
            conf, correct, _ = routed_confidence(grp.primitive, grp.P, grp.y)
        tau = select_threshold(
            conf,
            correct,
            budget=args.error_budget,
            use_cp_upper=args.use_cp_upper,
            alpha=args.cp_alpha,
        )

        band_lo = None
        band_hi = None
        if grp.primitive == "noul" and not by_threshold:
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
            best = choose(swept, metric="f1", max_escalation_rate=max_escalation)
            if best is not None:
                band_lo = best.band.lo
                band_hi = best.band.hi

        mode = "escalation_band" if (band_lo is not None and band_hi is not None) else "error_budget"
        policy = PolicyEntry(
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
            fit_sha256=_tuned_fit_sha256(grp, tuned[grp.key]) if args.use_calib else None,
            source_sha256=src_sha,
        )
        metrics = routed_metrics(policy, grp, use_cp_upper=args.use_cp_upper, cp_alpha=args.cp_alpha)
        policies.append(replace(policy, metrics_on_split=metrics))

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


def _false_pass_policy(args: argparse.Namespace, grp: Group, src_sha: str, tuned: dict[GroupKey, list[Record]]) -> PolicyEntry:
    """A noul policy that passes (does not block) `p >= cutoff`, the smallest cutoff at which at
    most `--max-false-pass` of the group's label-false items pass. Written as a band with
    `band_lo == band_hi`, so it never escalates and the gate routes it unchanged."""
    name = f"{grp.question_id} ({grp.model_served})"
    p, label = grp.P[:, 1], grp.y == 1
    try:
        cut = select_false_pass_cutoff(p, label, args.max_false_pass, use_cp_upper=args.use_cp_upper, alpha=args.cp_alpha)
    except ValueError as e:
        raise CliError(f"{name}: {e}") from None
    if cut is None:
        bound = f"the Clopper-Pearson upper bound (alpha {args.cp_alpha:g})" if args.use_cp_upper else "the false-pass rate"
        raise CliError(
            f"{name}: no cutoff keeps {bound} at or under {args.max_false_pass:g} with {int((~label).sum())} "
            "label-false items; more label-false items, a larger --max-false-pass or a larger --cp-alpha may help"
        )
    return PolicyEntry(
        question_id=grp.question_id,
        model_served=grp.model_served,
        primitive=grp.primitive,
        mode="escalation_band",
        split="eval",
        n=grp.n,
        threshold=None,
        error_budget=None,
        band_lo=cut.cutoff,
        band_hi=cut.cutoff,
        calibrated=args.use_calib,
        fit_sha256=_tuned_fit_sha256(grp, tuned[grp.key]) if args.use_calib else None,
        source_sha256=src_sha,
        metrics_on_split={
            "selection": "max_false_pass",
            "max_false_pass": args.max_false_pass,
            "use_cp_upper": args.use_cp_upper,
            "cp_alpha": args.cp_alpha if args.use_cp_upper else None,  # unchecked and unused otherwise
            "n_true": cut.n_true,
            "n_false": cut.n_false,
            "true_pass": cut.true_pass,
            "false_pass": cut.false_pass,
            "true_pass_rate": cut.true_pass_rate,
            "false_pass_rate": cut.false_pass_rate,
        },
    )


def _cmd_slice(args: argparse.Namespace) -> int:
    records = read_records(args.records, use_calib=True)
    features_by_item: dict[FeatureKey, dict] = {}
    slice_by_item: dict[FeatureKey, str] = {}
    if getattr(args, "features", None):
        try:
            features_by_item, slice_by_item = parse_feature_rows(_load_jsonl_dicts(Path(args.features)))
        except ValueError as e:
            raise CliError(f"{args.features}: {e}") from None

    # Refuses a repeated item or differing classes in any group, before the filters, as `s1 audit`.
    sel = select(records, allow_model_mismatch=True)
    groups = slice_inputs_by_group(
        records,
        features_by_item,
        slice_by_item,
        question_id=args.question,
        model_served=args.model,
        raw=not args.use_calib,
    )
    if not groups:
        raise _no_groups(args, sel, args.records)
    if args.use_calib:
        # What `s1 policy tune --use-calib` refuses, in the groups reported: a record that would be
        # sliced on its raw distribution, or a group calibrated through more than one fit.
        by_key = {g.key: g for g in sel.groups}
        tuned = tuned_rows(records)
        for g in groups:
            try:
                tuned_fit_sha256(by_key[g.key], tuned[g.key])
            except CalibInputError as e:
                raise CliError(f"{args.records}: {e}") from None

    if args.slice_cmd == "answerable":
        out = [{**g.header(), "slices": [s.as_dict() for s in slice_report(g.items, min_n=args.min_n)]} for g in groups]
    else:
        out = [
            {
                **g.header(),
                "error_slices": [
                    e.as_dict() for e in mine_error_slices(g.rows, min_support=args.min_support, fdr_q=args.fdr_q)
                ],
            }
            for g in groups
        ]
    distribution = "calibrated" if args.use_calib else "raw"
    print(json.dumps({"distribution": distribution, "groups": out}, indent=2, sort_keys=True))
    return 0


def _filtered_records(args: argparse.Namespace) -> list:
    """`args.records` narrowed by `--question` / `--model`. A filter that keeps nothing, or keeps
    only records that cannot be ranked, is an error naming what the file holds."""
    records = read_records(args.records)
    for value, attr, name in ((args.question, "question_id", "question_id"), (args.model, "model_served", "model_served")):
        if value is None:
            continue
        present = sorted({str(getattr(r, attr)) for r in records})
        records = [r for r in records if getattr(r, attr) == value]
        if not records:
            shown = ", ".join(present[:5]) + (", ..." if len(present) > 5 else "")
            raise CliError(f"{args.records}: no record has {name} {value!r}; the records hold {shown or 'none'}")
    if args.question is not None or args.model is not None:
        if not any(r.model_served is not None and r.label is not None for r in records):
            unanswered = sum(r.model_served is None for r in records)
            raise CliError(
                f"{args.records}: nothing to rank for the selection: {unanswered} unanswered and "
                f"{len(records) - unanswered} unlabeled record(s)"
            )
    return records


def _question_entries(path: str) -> dict[str, Any]:
    """A questions file's raw entries, as `s1 lint check` reads them. A read failure is `main()`'s to report."""
    entries, errors = read_question_entries(path, Path(path).read_bytes())
    if errors:
        raise CliError(f"{path}: {'; '.join(errors)}")
    return entries


def _question_from_file(path: str, question_id: str):
    """The question with `question_id` in a questions file."""
    found = None
    for name, raw in _question_entries(path).items():  # every entry is checked, wherever the match is
        spec = parse_question(name, raw, KNOWN_RULES)
        if spec.errors:
            raise CliError(f"{path}: question {name!r}: {'; '.join(spec.errors)}")
        if found is None and spec.qid == question_id:
            found = spec.question
    if found is None:
        raise CliError(f"{path}: no question has question_id {question_id}")
    return found


def _fit_refused(args: argparse.Namespace, error: LabelAuditInputError) -> CliError:
    """A records file and a `--fit` file that do not go together: the error is about the pair."""
    return CliError(f"{args.records} with --fit {args.fit}: {error}")


def _cmd_labelaudit(args: argparse.Namespace) -> int:
    if args.la_cmd == "estimate":
        _require_file(args.key, "secret key", "s1 labelaudit sheet")
        form_text = _read_text(Path(args.sheet))
        key_doc = _read_json(Path(args.key))
        try:
            check_key(key_doc)
        except ValueError as e:
            raise CliError(f"{args.key}: {e}") from None
        try:
            scored = score_sheet(form_text, key_doc)
        except ValueError as e:
            if os.environ.get("S1_DEBUG"):
                raise
            raise CliError(f"{args.sheet}: {e}") from None
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

    records = _filtered_records(args)
    # `--fit` given means calibrate, even when the file holds no fits: every record is then refused.
    fits = load_fits(args.fit) if getattr(args, "fit", None) else None
    if args.la_cmd == "rank":
        try:
            flags, skipped = rank_with_skips([r.raw for r in records], temperature=fits)
        except LabelAuditInputError as e:
            if os.environ.get("S1_DEBUG"):
                raise
            raise _fit_refused(args, e) from None
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
    try:
        check_states(states_rows)
    except ValueError as e:
        raise CliError(f"{args.states}: {e}") from None
    # Keyed as `s1 probe` keys the records it writes from these rows: no `dataset` is "default".
    states_map = {(row.get("dataset", "default"), str(row["item_id"])): row["state"] for row in states_rows}

    # The questions file is read only to find the records' one question; build_sheet checks the
    # rest (one question, options, the question's primitive and classes).
    question_ids = {r.question_id for r in records if r.model_served is not None and r.label is not None}
    question = _question_from_file(args.questions, next(iter(question_ids))) if args.questions and len(question_ids) == 1 else None
    try:
        sheet = build_sheet(
            records,
            states_map,
            args.top,
            args.controls,
            options=args.options.split(",") if args.options is not None else None,
            question=question,
            fit=fits,
            seed=args.seed,
        )
    except LabelAuditInputError as e:
        if os.environ.get("S1_DEBUG"):
            raise
        raise _fit_refused(args, e) from None
    except SheetInputError as e:
        if os.environ.get("S1_DEBUG"):
            raise
        where = {"question": args.questions, "states": args.states}.get(e.source, args.records)
        raise CliError(f"{where}: {e}" + (": pass --question" if e.many_questions else "")) from None
    question_id, options = sheet.key["question_id"], sheet.key["options"]
    Path(args.output).write_text(sheet.form, encoding="utf-8")
    Path(args.key).write_text(json.dumps(sheet.key, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sheet": str(args.output),
                "key": str(args.key),
                "n_items": len(sheet.key["items"]),
                "question_id": question_id,
                "options": options,
                # The wording is shown only from a questions file; say so rather than print a note
                # on stderr, which carries error lines.
                "question_shown": question is not None,
            },
            indent=2,
        )
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    rep = compare_records(args.primary, args.shadow, training_manifests=args.training_manifest or ())
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


def _cmd_trainset(args: argparse.Namespace) -> int:
    records, src_sha = read_records_hashed(args.records)
    parent = load_manifest(args.parent) if args.parent else None
    built = build_manifest(records, split=args.split, checkpoint=args.checkpoint, source_sha256=src_sha, parent=parent, source=args.records)
    output = Path(args.output)
    if output.exists():
        for other in (args.records, args.parent):
            if other and output.samefile(other):
                raise CliError(f"--output {output} is the same file as {other}")
    save_manifest(built.manifest, output)
    summary: dict[str, Any] = {
        "output": str(output),
        "checkpoint": built.manifest.checkpoint,
        "split": built.manifest.split,
        "source_sha256": built.manifest.source_sha256,
        "items": len(built.manifest.items),
        "questions": len({i.question_id for i in built.manifest.items}),
        "inherited_items": len(built.manifest.inherited_items),
        "skipped": {"no_response": built.no_response, "unlabeled": built.unlabeled, "other_split": built.other_split},
    }
    if built.also_in_other_splits:
        summary["also_in_other_splits"] = {"n": len(built.also_in_other_splits), "first": list(built.also_in_other_splits[0])}
    if built.answered_as_checkpoint:
        summary["answered_as_checkpoint"] = built.answered_as_checkpoint
    print(json.dumps(summary, indent=2, sort_keys=True))
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


def _check_fit_args(args: argparse.Namespace) -> None:
    if args.allow_in_sample and not args.fit:
        raise CliError("--allow-in-sample applies only with --fit")


def _applied_in_memory(path: str, fits: list[Fit], args: argparse.Namespace) -> AppliedFits:
    """`path` as `s1 apply <path> --fit ... --pass-through-unfitted` would write it."""
    return apply_fits(read_records(path), fits, allow_in_sample=args.allow_in_sample, pass_through_unfitted=True)


def _cmd_export(args: argparse.Namespace) -> int:
    _check_fit_args(args)
    policies: list[PolicyEntry] | None = None
    if args.policy:
        policies = load_policies(args.policy)
    applied = _applied_in_memory(args.records, load_fits(args.fit), args) if args.fit else None
    records = applied.records if applied else read_records(args.records)
    fit_info = {"fit": applied.summary()} if applied else {}

    if args.export_cmd == "langfuse":
        payload = export_langfuse_file(
            args.output,
            records,
            run_name=args.run_name,
            policies=policies,
        )
        print(json.dumps({**payload["summary"], **fit_info}, indent=2, sort_keys=True))
        return 0

    spans = export_openinference_jsonl(
        args.output,
        records,
        run_name=args.run_name,
        policies=policies,
    )
    print(
        json.dumps(
            {"output": str(args.output), "n_spans": len(spans), "redaction": "safe-v1", **fit_info},
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
    _check_fit_args(args)
    if args.fit and not (args.policy or args.calibrated):
        raise CliError(
            "--fit does not change a raw comparison without a policy: "
            "pass --calibrated to compare calibrated probabilities, or --policy to route them"
        )
    policies = load_policies(args.policy) if args.policy else None
    reference, current, fit_info = args.reference, args.current, {}
    if args.fit:
        fits = load_fits(args.fit)
        applied_ref, applied_cur = _applied_in_memory(reference, fits, args), _applied_in_memory(current, fits, args)
        reference, current = applied_ref.records, applied_cur.records
        fit_info = {"fit": {"reference": applied_ref.summary(), "current": applied_cur.summary()}}
    rep = audit_drift(
        reference,
        current,
        policies=policies,
        bins=args.bins,
        use_calib=args.calibrated,
    )
    print(json.dumps({**rep, **fit_info}, indent=2, sort_keys=True))
    if args.fail_on_critical and rep["summary"]["critical"] > 0:
        return EXIT_DRIFT_CRITICAL
    return EXIT_OK


_MOCK_MODEL = "jev-1.13.0"


def _probe_backend(args: argparse.Namespace):
    # The name is written into every record. Argument bytes that are not UTF-8 arrive as lone surrogates.
    problem = surrogate_problem("--model", args.model) if args.model else None
    if problem is not None:
        raise CliError(problem)
    if args.backend == "mock":
        if not args.model:
            raise CliError("--model must not be empty")
        return DeterministicMockBackend(served_model=args.served_model or args.model)
    if args.served_model is not None:
        raise CliError("--served-model applies only to --backend mock")
    if not args.model:
        raise CliError(f"--backend {args.backend} needs --model: there is no default, since a server that echoes the request would report it as the model that answered")
    if args.usd_per_input_token is None:
        raise CliError(f"--backend {args.backend} needs --usd-per-input-token: the API reports tokens, not dollars")
    try:
        live = HttpSystemOneBackend if args.backend == "systemone" else OpenAIDecisionsBackend
        return live(
            usd_per_input_token=args.usd_per_input_token,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    except ImportError as e:
        raise CliError(str(e)) from None


def _close_backend(backend: Any) -> Exception | None:
    """Close `backend` and return what `close()` raised, so that it cannot replace the command's own result."""
    close = getattr(backend, "close", None)
    if close is None:
        return None
    try:
        close()
    except Exception as e:
        return e
    return None


def _open_output_apart(path: Path, others: list[Path]):
    """Open `path` for writing, refusing if it is the same file as any of `others`.

    Compares file identity, not names: a symlink, a hard link or a case-insensitive filesystem can
    make two names one file. Nothing is truncated: the caller truncates when the first record
    arrives, so a run that fails before any request keeps the previous output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
    except FileExistsError:
        fd = os.open(path, os.O_WRONLY)
        created = False
    try:
        st = os.fstat(fd)
        for other in others:
            try:
                ost = os.stat(other)
            except FileNotFoundError:
                continue
            if (ost.st_dev, ost.st_ino) == (st.st_dev, st.st_ino):
                if created:
                    os.unlink(path)
                raise CliError(f"--output {path} is the same file as {other}")
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "w", encoding="utf-8")


def _os_error_message(e: OSError) -> str:
    return f"{e.filename}: {e.strerror}" if e.filename else str(e)


def _cmd_probe(args: argparse.Namespace) -> int:
    q_map = _question_entries(args.questions)
    # Every question, its extension and the backend's refusals are checked before the ledger or
    # the output is opened; run_batch repeats the same deterministic expansion.
    try:
        tasks = probe_tasks(q_map, probe_mode=not args.no_expand_probes)
    except ValueError as e:
        raise CliError(f"{args.questions}: {e}") from None
    if args.max_retries < 0:
        raise CliError("--max-retries must be >= 0")
    # None takes the backend's measured default inside run_batch; check an explicit value up front.
    check_reservation(args.worst_usd_per_token, args.reserve_overhead_tokens or 0.0)

    states = _load_jsonl_dicts(Path(args.states))
    if not states:
        # Before the backend, the ledger or the output is touched: a run that sends nothing would
        # leave an empty output that the next command refuses, one step away from the cause.
        raise CliError(f"{args.states}: holds no state rows, so the probe would send nothing")
    try:
        check_states(states)
        check_labels(states, tasks)
    except ValueError as e:
        raise CliError(f"{args.states}: {e}") from None
    if args.model is None and args.backend == "mock":
        args.model = _MOCK_MODEL
    backend = _probe_backend(args)
    out_path = Path(args.output)
    ledger_path = Path(args.ledger)
    written = 0
    failed = 0
    lost = False
    try:
        try:  # inside the `finally` that closes the backend, before the ledger or output is opened
            check_questions(tasks, backend)
        except ValueError as e:
            raise CliError(f"{args.questions}: {e}") from None
        with Ledger(ledger_path, cap_usd=args.cap_usd) as ledger:
            others = [ledger_path, lock_path(ledger_path), Path(args.states), Path(args.questions)]
            with _open_output_apart(out_path, others) as fh:

                def write(rec: dict[str, Any]) -> None:
                    nonlocal written, failed, lost
                    try:
                        if not written:
                            fh.truncate(0)
                        fh.write(json.dumps(rec, sort_keys=True) + "\n")
                        fh.flush()
                    except OSError:
                        lost = True
                        raise
                    written += 1
                    failed += rec["model_served"] is None

                run_batch(
                    states,
                    q_map,
                    backend=backend,
                    ledger=ledger,
                    model_requested=args.model,
                    worst_usd_per_token=args.worst_usd_per_token,
                    reserve_overhead_tokens=args.reserve_overhead_tokens,
                    probe_mode=not args.no_expand_probes,
                    max_retries=args.max_retries,
                    on_record=write,
                )
                if not written:
                    fh.truncate(0)
    # Outside both `with` blocks, so a second failure while closing the output cannot replace this message.
    except (BudgetExceeded, ValueError, OSError) as e:
        if os.environ.get("S1_DEBUG") or not (written or lost):
            raise
        message = _os_error_message(e) if isinstance(e, OSError) else str(e)
        if lost:
            message += f"; the record of an already settled request was not written to {out_path}"
        raise CliError(f"{message}; {written} records kept in {out_path}") from None
    finally:
        close_error = _close_backend(backend)

    # Reached only when the run finished. The type alone: an SDK's text may repeat what a server sent.
    if close_error is not None:
        if os.environ.get("S1_DEBUG"):
            raise close_error
        raise CliError(f"closing the backend failed ({type(close_error).__name__}) after the run finished; {written} records kept in {out_path}, {failed} of them failed")

    st = ledger_read(args.ledger)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "n_records": written,
                "n_failed": failed,
                "spent_usd": round(st.spent_usd, 8),
                "cap_usd": args.cap_usd,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _gate_probs(text: str) -> dict:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        keys = [k for k, _ in items]
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            raise CliError(f"--probs repeats {repeated}")
        return dict(items)

    def constant(name: str) -> float:
        raise CliError(f"--probs: non-finite number {name} is not valid JSON")

    try:
        probs = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as e:
        raise CliError(f"--probs is not valid JSON: {e}") from None
    if not isinstance(probs, dict):
        raise CliError(f"--probs must be a JSON object of label to probability, got {text!r}")
    return probs


def _cmd_gate(args: argparse.Namespace) -> int:
    gate = DecisionGate.from_files(args.policy, args.fit)
    primitive = gate.primitive_of(args.question)
    common = {
        "question_id": args.question,
        "model_served": args.model,
        "expected_policy_sha256": args.expected_sha256,
    }
    try:
        if args.p is not None:
            if primitive in ("choice", "score"):
                raise CliError(f"{args.question} is a {primitive} question; pass --probs, not --p")
            dec = gate.evaluate_noul(p_raw=args.p, **common)
        else:
            probs = _gate_probs(args.probs)
            if primitive == "noul":
                raise CliError(f"{args.question} is a noul question; pass --p, not --probs")
            if primitive == "score":
                dec = gate.evaluate_score(probs_raw=probs, **common)
            else:
                dec = gate.evaluate_choice(probs_raw=probs, **common)
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
_H_FIT_IN_MEMORY = "fits.json from `s1 fit`; calibrates each input file in memory first, as `s1 apply --pass-through-unfitted` writes it: a record with a fit gets its `calib`, replacing a stored one, and any other keeps what it carries"
_H_ALLOW_IN_SAMPLE = "with --fit, calibrate items the fit was trained on rather than exiting 2"
_H_TRAINING_MANIFEST = (
    "training manifest from `s1 trainset export`, repeatable: refuse (exit 2) a selected group whose model trained on any of its items"
)
_H_USE_CALIB = "use the calibrated probabilities written by `s1 apply` instead of the raw ones"
_H_USE_CALIB_SLICE = (
    "predict from the calibrated probabilities written by `s1 apply` instead of the raw ones; every "
    "labeled record of a reported group must carry them, from one fit"
)
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
    p_aud.add_argument("--training-manifest", action="append", help=_H_TRAINING_MANIFEST)
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
    p_fit.add_argument("--training-manifest", action="append", help=_H_TRAINING_MANIFEST)
    p_fit.set_defaults(func=_cmd_fit)

    # apply
    p_app = sub.add_parser("apply", help="Apply calibration fit to held-out records", **_FMT)
    p_app.add_argument("records", help=f"held-out {_H_RECORDS}")
    p_app.add_argument("--fit", required=True, help=_H_FIT_IN)
    p_app.add_argument("-o", "--output", required=True, help="calibrated records JSONL to write; item `state` is never written (safe-v1), and a JSON summary with `state_dropped` is printed when any input record had one")
    p_app.add_argument("--allow-in-sample", action="store_true", help="apply even to items the fit was trained on (otherwise exit 2); they are marked in_fit_sample")
    p_app.add_argument("--pass-through-unfitted", action="store_true", help="write records the fit file has no fit for (another model, question or primitive) unchanged except for `state`, without adding or replacing `calib`, instead of exiting 3 or 1; prints a JSON summary of what was calibrated and passed through")
    p_app.set_defaults(func=_cmd_apply)

    # policy
    p_pol = sub.add_parser("policy", help="Threshold tuning & System 1->2 escalation compiler", **_FMT)
    sub_pol = p_pol.add_subparsers(dest="policy_cmd", required=True)
    p_pt = sub_pol.add_parser("tune", help="Compile error-budget & escalation policy.json", **_FMT)
    p_pt.add_argument("records", help=_H_RECORDS)
    p_pt.add_argument("-o", "--output", required=True, help="policy.json to write")
    p_pt.add_argument("--question", default=None, help=_H_QUESTION)
    p_pt.add_argument("--model", default=None, help=_H_MODEL)
    p_pt.add_argument("--error-budget", type=float, default=0.10, help="maximum error rate among auto-accepted items; the threshold is the lowest tie-safe one that meets it. A noul group is routed by the F1 band instead, which this does not bound (metrics_on_split.accepted_error says what it does), unless --noul-routing threshold (choice and score groups only under --max-false-pass)")
    p_pt.add_argument("--use-cp-upper", action="store_true", help="judge the budget (or --max-false-pass) against the Clopper-Pearson upper bound instead of the point estimate")
    p_pt.add_argument("--cp-alpha", type=float, default=0.05, help="significance level of the Clopper-Pearson bound")
    p_pt.add_argument("--cost-s1", type=float, default=0.00004, help=_H_COST_S1)
    p_pt.add_argument("--cost-sys2", type=float, default=0.004, help=_H_COST_SYS2)
    p_pt.add_argument("--max-escalation", type=float, default=None, help="largest fraction of items the noul escalation band may send to System 2 (default: 0.35). At 0 the band is empty, one cutoff that accepts every item, not the threshold")
    p_pt.add_argument(
        "--max-false-pass",
        type=float,
        default=None,
        help="for noul groups, in place of the F1 band: the smallest cutoff at which at most this share of label-false items pass (p >= cutoff), written as a band with band_lo == band_hi",
    )
    p_pt.add_argument(
        "--noul-routing",
        choices=("band", "threshold"),
        default=None,
        help="what routes a noul group: band (the default) is the F1 escalation band under --max-escalation; threshold writes mode error_budget, accepting when max(p, 1 - p) reaches the --error-budget threshold, so the budget bounds accepted_error on the split",
    )
    p_pt.add_argument("--use-calib", action="store_true", help=_H_USE_CALIB)
    p_pt.add_argument("--training-manifest", action="append", help=_H_TRAINING_MANIFEST)
    p_pc = sub_pol.add_parser("curve", help="Generate threshold operating curve: each point counts what an error_budget policy at its threshold accepts (for noul, what `policy tune --noul-routing threshold` writes, not the F1 band; a choice or score tie at the top is always escalated)", **_FMT)
    p_pc.add_argument("records", help=_H_RECORDS)
    p_pc.add_argument("--question", default=None, help=_H_QUESTION)
    p_pc.add_argument("--model", default=None, help=_H_MODEL)
    p_pc.add_argument("--cost-s1", type=float, default=0.00004, help=_H_COST_S1)
    p_pc.add_argument("--cost-sys2", type=float, default=0.004, help=_H_COST_SYS2)
    p_pc.add_argument("--use-calib", action="store_true", help=_H_USE_CALIB)
    p_pc.add_argument("--training-manifest", action="append", help=_H_TRAINING_MANIFEST)
    p_pol.set_defaults(func=_cmd_policy)

    # slice
    p_sl = sub.add_parser("slice", help="Answerability decomposition & BH-FDR error slice mining", **_FMT)
    sub_sl = p_sl.add_subparsers(dest="slice_cmd", required=True)
    p_sa = sub_sl.add_parser("answerable", help="Slice metrics vs majority & blind null, per (question, model, primitive) group", **_FMT)
    p_sa.add_argument("records", help=_H_RECORDS)
    p_sa.add_argument("--features", default=None, help="features JSONL tagging each item with a `slice`; untagged items fall in `default`")
    p_sa.add_argument("--question", default=None, help=_H_QUESTION)
    p_sa.add_argument("--model", default=None, help=_H_MODEL)
    p_sa.add_argument("--min-n", type=int, default=10, help="slices with fewer items are reported but never marked answerable")
    p_sa.add_argument("--use-calib", action="store_true", help=_H_USE_CALIB_SLICE)
    p_sm = sub_sl.add_parser("mine", help="Mine high-error feature slices with BH-FDR control, per (question, model, primitive) group", **_FMT)
    p_sm.add_argument("records", help=_H_RECORDS)
    p_sm.add_argument("--features", required=True, help="features JSONL with pre-declared per-item feature tags")
    p_sm.add_argument("--question", default=None, help=_H_QUESTION)
    p_sm.add_argument("--model", default=None, help=_H_MODEL)
    p_sm.add_argument("--fdr-q", type=float, default=0.10, help="Benjamini-Hochberg false discovery rate")
    p_sm.add_argument("--min-support", type=int, default=5, help="minimum items in a feature slice to test it")
    p_sm.add_argument("--use-calib", action="store_true", help=_H_USE_CALIB_SLICE)
    p_sl.set_defaults(func=_cmd_slice)

    # labelaudit
    p_la = sub.add_parser("labelaudit", help="Out-of-fold label disagreement & blind review sheets", **_FMT)
    sub_la = p_la.add_subparsers(dest="la_cmd", required=True)
    p_lar = sub_la.add_parser("rank", help="Rank candidate label errors by calibrated disagreement", **_FMT)
    p_lar.add_argument("records", help=_H_RECORDS)
    p_lar.add_argument("--fit", default=None, help=f"{_H_FIT_IN}; each record is ranked on its own group's calibrated distribution (temperature, platt or isotonic), and a record whose group has no fit is refused")
    p_lar.add_argument("--top", type=int, default=20, help="candidates to print")
    p_lar.add_argument("--question", default=None, help=_H_QUESTION)
    p_lar.add_argument("--model", default=None, help=_H_MODEL)
    p_las = sub_la.add_parser("sheet", help="Generate blinded markdown review sheet + secret key", **_FMT)
    p_las.add_argument("records", help=_H_RECORDS)
    p_las.add_argument("--states", required=True, help="local-only states JSONL (`dataset`, `item_id`, `state`; no `dataset` means \"default\", as for `s1 probe`) to show reviewers; never written to records")
    p_las.add_argument("--fit", default=None, help=f"{_H_FIT_IN}; each record is ranked on its own group's calibrated distribution (temperature, platt or isotonic), and a record whose group has no fit is refused")
    p_las.add_argument("-o", "--output", required=True, help="markdown review sheet to write")
    p_las.add_argument("-k", "--key", required=True, help="secret answer key JSON to write; keep it away from reviewers")
    p_las.add_argument("--top", type=int, default=10, help="flagged candidates on the sheet")
    p_las.add_argument("--controls", type=int, default=5, help="random control items mixed in to measure baseline disagreement")
    p_las.add_argument("--options", default=None, help="comma-separated answer options shown to reviewers, in this order (default: the question's classes, `True,False` for noul); they must be exactly those classes")
    p_las.add_argument("--question", default=None, help=f"{_H_QUESTION}; required when the records hold more than one question, since a sheet asks one")
    p_las.add_argument("--model", default=None, help=_H_MODEL)
    p_las.add_argument("--questions", default=None, help="questions JSON file, as `s1 probe` reads it, to print the question's wording and option criteria on the sheet")
    p_las.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p_lae = sub_la.add_parser("estimate", help="Score completed blind sheet against secret key", **_FMT)
    p_lae.add_argument("sheet", help="review sheet completed by the reviewer")
    p_lae.add_argument("--key", required=True, help="secret key JSON from `s1 labelaudit sheet`")
    p_la.set_defaults(func=_cmd_labelaudit)

    # compare
    p_cmp = sub.add_parser("compare", help="Multi-backend shadow comparison (McNemar-Holm)", **_FMT)
    p_cmp.add_argument("primary", help=f"primary backend {_H_RECORDS}")
    p_cmp.add_argument("shadow", help=f"shadow backend {_H_RECORDS} on the same items")
    p_cmp.add_argument(
        "--training-manifest",
        action="append",
        help="training manifest from `s1 trainset export`, repeatable: refuse (exit 2) a compared pair in which a side's model trained on the item",
    )
    p_cmp.set_defaults(func=_cmd_compare)

    # trainset
    p_ts = sub.add_parser("trainset", help="Training sets for an external trainer (s1ops starts no training)", **_FMT)
    sub_ts = p_ts.add_subparsers(dest="trainset_cmd", required=True)
    p_tse = sub_ts.add_parser("export", help="Write the items and labels a checkpoint will train on (ids only, no state or wording)", **_FMT)
    p_tse.add_argument("records", help=_H_RECORDS)
    p_tse.add_argument("-o", "--output", required=True, help="training manifest to write")
    p_tse.add_argument("--split", required=True, help="the `split` to train on; the other splits stay for fitting and evaluation")
    p_tse.add_argument("--checkpoint", required=True, help="the name the new checkpoint will answer under in `s1 probe --model`; unique per checkpoint")
    p_tse.add_argument("--parent", default=None, help="manifest of the checkpoint this one starts from, whose trained items it inherits")
    p_ts.set_defaults(func=_cmd_trainset)

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
    p_drf_dist = p_drf.add_mutually_exclusive_group()
    p_drf_dist.add_argument("--calibrated", action="store_true", help="compare calibrated probabilities (PSI, JS divergence, quantiles, entropy); every answered record in both files must carry `calib` from `s1 apply`, made by one fit per (question, model) group (or by none, for records applied before s1ops recorded it). The escalation rate always routes what each policy was tuned on")
    p_drf_dist.add_argument("--raw-only", action="store_true", help="compare raw probabilities; the default, kept for scripts written for s1ops 0.1.0")
    p_drf.add_argument("--fit", default=None, help=f"{_H_FIT_IN_MEMORY}; needs --policy or --calibrated, and the report gains a `fit` block saying what was calibrated")
    p_drf.add_argument("--allow-in-sample", action="store_true", help=_H_ALLOW_IN_SAMPLE)
    p_drf.add_argument("--fail-on-critical", action="store_true", help="exit 4 if any question is CRITICAL")
    p_drf.set_defaults(func=_cmd_drift)

    # probe
    p_prb = sub.add_parser("probe", help="Budget-guarded System 1 probe & batch runner", **_FMT)
    p_prb.add_argument("questions", help="questions JSON file")
    p_prb.add_argument("--states", required=True, help="local-only states JSONL; only state_sha256 reaches the output")
    p_prb.add_argument("-o", "--output", required=True, help="probe records JSONL to write")
    p_prb.add_argument("--ledger", required=True, help="budget journal; created if absent")
    p_prb.add_argument("--cap-usd", type=float, default=1.00, help="hard spend cap; the run stops before a call whose worst-case cost would exceed it")
    p_prb.add_argument("--model", default=None, help="model_requested sent to the backend; required for a live backend (the mock defaults to jev-1.13.0)")
    p_prb.add_argument("--backend", choices=["mock", "systemone", "openai"], default="mock", help="mock: offline deterministic answers; systemone: live POST /v1/systemone (typesafe extra, TYPESAFE_API_KEY); openai: live POST /v1/decisions (openai extra, OPENAI_API_KEY)")
    p_prb.add_argument("--served-model", default=None, help="model_served the mock backend reports (defaults to --model); set it differently to simulate a silent upgrade")
    p_prb.add_argument("--usd-per-input-token", type=float, default=None, help="price per billable input token; required for a live backend, which reports tokens, not dollars")
    p_prb.add_argument("--base-url", default=None, help="live API root (defaults to TYPESAFE_BASE_URL or OPENAI_BASE_URL, then the SDK default)")
    p_prb.add_argument("--timeout", type=float, default=None, help="per-request timeout in seconds (defaults to the SDK's)")
    p_prb.add_argument("--max-retries", type=int, default=2, help="retries after a timeout, connection error, 408, 429 or 5xx; each attempt reserves and settles on its own")
    p_prb.add_argument("--worst-usd-per-token", type=float, default=2e-6, help="price used for each attempt's worst-case reservation")
    p_prb.add_argument("--reserve-overhead-tokens", type=float, default=None, help="tokens added to each reservation for server-side prompt overhead (default: 300 systemone, 200 openai, 0 mock; measured per request); a measured cost above the reservation stops the run")
    p_prb.add_argument("--no-expand-probes", action="store_true", help="send only the questions as written, without twin and inverse probes")
    p_prb.set_defaults(func=_cmd_probe)

    # gate
    p_gte = sub.add_parser("gate", help="Evaluate runtime DecisionGate on a live System 1 output", **_FMT)
    p_gte.add_argument("policy", help=_H_POLICY_IN)
    p_gte.add_argument("--fit", default=None, help=f"{_H_FIT_IN}; calibrates --p or --probs; required when the policy was compiled with --use-calib, and must be the fit it was tuned through")
    p_gte.add_argument("--question", required=True, help="question_id of the live call")
    p_gte.add_argument("--model", required=True, help="model_served of the live call; a mismatch with the policy exits 3")
    p_gte_answer = p_gte.add_mutually_exclusive_group(required=True)
    p_gte_answer.add_argument("--p", type=float, default=None, help="raw noul probability returned by the model")
    p_gte_answer.add_argument(
        "--probs",
        default=None,
        help='raw choice or score probabilities as a JSON object, e.g. \'{"A": 0.7, "B": 0.3}\' or \'{"0": 0.1, "1": 0.9}\'',
    )
    p_gte.add_argument("--expected-sha256", default=None, help="expected policy_sha256; a mismatch exits 1")
    p_gte.set_defaults(func=_cmd_gate)

    # export
    p_exp = sub.add_parser("export", help="Safe-v1 exporters for Langfuse and Arize Phoenix", **_FMT)
    sub_exp = p_exp.add_subparsers(dest="export_cmd", required=True)
    p_el = sub_exp.add_parser("langfuse", help="Export safe-v1 Langfuse Dataset Run & Scores", **_FMT)
    p_el.add_argument("records", help=_H_RECORDS)
    p_el.add_argument("-o", "--output", required=True, help="Langfuse export JSON to write")
    p_el.add_argument("--policy", default=None, help=f"{_H_POLICY_IN}; routes each record with the compiled threshold or band instead of a fixed 0.5")
    p_el.add_argument("--fit", default=None, help=f"{_H_FIT_IN_MEMORY}; the printed summary gains a `fit` block saying what was calibrated")
    p_el.add_argument("--allow-in-sample", action="store_true", help=_H_ALLOW_IN_SAMPLE)
    p_el.add_argument("--run-name", default="s1ops-eval", help="dataset run name")
    p_ep = sub_exp.add_parser("phoenix", help="Export safe-v1 OpenInference spans JSONL", **_FMT)
    p_ep.add_argument("records", help=_H_RECORDS)
    p_ep.add_argument("-o", "--output", required=True, help="OpenInference spans JSONL to write")
    p_ep.add_argument("--policy", default=None, help=f"{_H_POLICY_IN}; routes with the compiled threshold or band and adds s1.policy_sha256, s1.threshold, s1.band_lo, s1.band_hi")
    p_ep.add_argument("--fit", default=None, help=f"{_H_FIT_IN_MEMORY}; the printed summary gains a `fit` block saying what was calibrated")
    p_ep.add_argument("--allow-in-sample", action="store_true", help=_H_ALLOW_IN_SAMPLE)
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
    except FitApplyError as e:
        if debug:
            raise
        return _fail(str(e), _FIT_APPLY_EXIT[e.kind])
    except TrainedItemError as e:
        if debug:
            raise
        return _fail(str(e), EXIT_IN_SAMPLE)
    except OSError as e:
        if debug:
            raise
        return _fail(_os_error_message(e))
    except (ValueError, PolicyIntegrityError, BudgetExceeded) as e:
        # ValueError covers RecordError, FitError, LedgerCorrupt and json.JSONDecodeError.
        if debug:
            raise
        return _fail(str(e))


def _silence_stdout() -> None:
    """Close and drop the stream, so neither shutdown nor its finalizer has a flush left to fail on."""
    stream, sys.stdout = sys.stdout, None
    try:
        stream.close()
    except (OSError, ValueError):  # the failed flush, once more; the stream is closed regardless
        pass


def _flushed(code: int) -> int:
    """Write what was printed while the exit code can still be chosen. The first failure stands."""
    try:
        if sys.stdout is not None:
            sys.stdout.flush()
    except BrokenPipeError:  # the reader left early, as `| head` does: not this command's failure
        _silence_stdout()
    except OSError as e:
        _silence_stdout()
        if code == EXIT_OK:
            return _fail(_os_error_message(e))
    return code


def console_main() -> None:
    """The `s1` and `s1ops` commands: `main`, plus what only a process's owner may do to its stdout."""
    try:
        code = main()
    except SystemExit as e:  # argparse: `-h`, `--version`, or a usage error
        code = int(e.code or EXIT_OK)
    except BaseException:  # a traceback is still exit 1, not 120
        _flushed(EXIT_ERROR)
        raise
    sys.exit(_flushed(code))


if __name__ == "__main__":
    console_main()

