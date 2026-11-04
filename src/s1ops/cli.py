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
from collections import defaultdict
from pathlib import Path
from typing import Any

from s1ops import __version__
from s1ops.bridges.langfuse import export_langfuse_file
from s1ops.bridges.openinference import export_openinference_jsonl
from s1ops.contracts._validation import UNDECODABLE
from s1ops.contracts.questions import parse_wire_question
from s1ops.contracts.questions import question_id as question_id_of
from s1ops.contracts.records import Group, GroupKey, Record, Selection, has_calib, read_records, select, short_repr, without_state
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
    SheetInputError,
    build_sheet,
    check_key,
    estimate,
    rank_with_skips,
    score_sheet,
)
from s1ops.improve.policy import (
    Band,
    DevRecord,
    Prices,
    choose,
    grid as band_grid,
    select_false_pass_cutoff,
    select_threshold,
    sweep,
    threshold_curve,
)
from s1ops.improve.slices import FeatureKey, mine_error_slices, parse_feature_rows, slice_inputs_by_group, slice_report
from s1ops.ledger.journal import (
    BudgetExceeded,
    Ledger,
    lock_path,
    migrate as ledger_migrate,
    read as ledger_read,
    recover as ledger_recover,
)
from s1ops.lint.lint import ProbeInputError, audit_probe_records, format_text, lint_path
from s1ops.lint.rules import RULES
from s1ops.monitor.drift import audit_drift
from s1ops.monitor.backends import HttpSystemOneBackend, OpenAIDecisionsBackend
from s1ops.monitor.runner import DeterministicMockBackend, check_reservation, check_states, run_batch
from s1ops.runtime.gate import DecisionGate, ModelServedMismatchError, PolicyIntegrityError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_IN_SAMPLE = 2
EXIT_MODEL_MISMATCH = 3
EXIT_DRIFT_CRITICAL = 4


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
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"s1: {_one_line(message)}\n")


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
        if value is not None:
            _require_file(value, what, producer)


def _tuned_rows(records: list[Record]) -> dict[GroupKey, list[Record]]:
    """The records `select(records, allow_model_mismatch=True)` tunes on, by group: answered and
    labeled (`select` refuses a repeated item within a group)."""
    rows: dict[GroupKey, list[Record]] = defaultdict(list)
    for r in records:
        if r.model_served is not None and r.label is not None:
            rows[r.key].append(r)
    return rows


def _tuned_fit_sha256(grp: Group, rows: list[Record]) -> str | None:
    """The `fit_sha256` that every record tuned into `grp` was calibrated through, or `None` for
    records from before `s1 apply` wrote it. A record without the calibrated distribution would be
    tuned on its raw one (`parse_record` falls back), so it is refused rather than counted."""
    name = f"{grp.question_id} ({grp.model_served})"
    for r in rows:
        if not has_calib(r.raw, grp.primitive):
            raise CliError(
                f"{name}: line {r.line} has no calibrated probabilities; `--use-calib` tunes on the "
                "`calib` that `s1 apply` writes, so apply the fit to every record first"
            )
    for r in rows:
        h = r.raw["calib"].get("fit_sha256")
        if h is not None and not isinstance(h, str):
            raise CliError(f"{name}: line {r.line}: calib.fit_sha256 must be a string, got {short_repr(h)}")
    hashes = {r.raw["calib"].get("fit_sha256") for r in rows}
    if len(hashes) > 1:
        shown = ", ".join(sorted(repr(h) for h in hashes))
        raise CliError(f"{name}: records were calibrated through different fits ({shown}); apply one fit file")
    return hashes.pop() if hashes else None


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


def _read_json(path: Path) -> Any:
    try:
        return json.loads(_read_text(path))
    except json.JSONDecodeError as e:
        raise CliError(f"{path}: not valid JSON: {e}") from None


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, or a JSON array of objects; every row must be an object."""
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
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
    # read_text already turned \r\n and \r into \n; str.splitlines would also split on \x0b, \x1c
    # or U+2028, which a JSON string may hold raw, and count lines the file does not have.
    for n, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if UNDECODABLE.search(line):
            raise CliError(f"{path}: line {n}: not valid UTF-8")
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
    calibrated = 0
    # `safe-v1`: no row is written with its item `state` (every path below leaves it out).
    state_dropped = sum(1 for rec in records if "state" in rec.raw)
    passed: dict[tuple, dict[str, Any]] = {}
    for rec in records:
        if rec.model_served is None:
            out_objs.append(without_state(rec.raw))
            continue
        fit = fit_for(fits, rec.key)
        if fit is None:
            msg = refusal(fits, rec.key, rec.classes) or f"no fit for {rec.key}"
            if args.pass_through_unfitted:
                qid, served, primitive = rec.key
                group = passed.setdefault(rec.key, {"question_id": qid, "model_served": served, "primitive": primitive, "n": 0, "reason": msg})
                group["n"] += 1
                out_objs.append(without_state(rec.raw))
                continue
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
        calibrated += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for o in out_objs:
            fh.write(json.dumps(o, sort_keys=True) + "\n")
    summary: dict[str, Any] = {"calibrated": calibrated}
    if args.pass_through_unfitted:
        summary["passed_through"] = list(passed.values())
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
        if not curves:
            raise _no_groups(args, sel, args.records)
        print(json.dumps({"groups": curves}, indent=2, sort_keys=True))
        return 0

    false_pass = args.max_false_pass
    if false_pass is not None and args.max_escalation is not None:
        raise CliError("--max-escalation sizes the F1 band, which --max-false-pass replaces; pass one of them")
    max_escalation = 0.35 if args.max_escalation is None else args.max_escalation
    src_sha = sha256_file(args.records)
    tuned_rows = _tuned_rows(records) if args.use_calib else {}
    selected = [
        grp
        for grp in sel.groups
        if not (args.question and grp.question_id != args.question) and not (args.model and grp.model_served != args.model)
    ]
    if false_pass is not None and selected and not any(grp.primitive == "noul" for grp in selected):
        raise CliError("--max-false-pass applies to noul groups, and none is selected")
    policies: list[PolicyEntry] = []
    for grp in selected:
        if false_pass is not None and grp.primitive == "noul":
            policies.append(_false_pass_policy(args, grp, src_sha, tuned_rows))
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
            best = choose(swept, metric="f1", max_escalation_rate=max_escalation)
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
                fit_sha256=_tuned_fit_sha256(grp, tuned_rows[grp.key]) if args.use_calib else None,
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


def _false_pass_policy(args: argparse.Namespace, grp: Group, src_sha: str, tuned_rows: dict) -> PolicyEntry:
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
        fit_sha256=_tuned_fit_sha256(grp, tuned_rows[grp.key]) if args.use_calib else None,
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
        records, features_by_item, slice_by_item, question_id=args.question, model_served=args.model
    )
    if not groups:
        raise _no_groups(args, sel, args.records)

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
    print(json.dumps({"groups": out}, indent=2, sort_keys=True))
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


def _question_from_file(path: str, question_id: str):
    """The question with `question_id` in a questions file (names to questions, or
    `{"questions": {...}}`, as `s1 probe` reads it)."""
    doc = _read_json(Path(path))
    if isinstance(doc, dict) and isinstance(doc.get("questions"), dict):
        entries = doc["questions"]
    elif isinstance(doc, dict):
        entries = doc
    else:
        raise CliError(f"{path}: must be a JSON object mapping names to questions")
    found = None
    for name, raw in entries.items():  # every entry is checked, wherever the match is
        try:
            parsed = parse_wire_question(raw)
        except ValueError as e:
            raise CliError(f"{path}: question {name!r}: {e}") from None
        if found is None and question_id_of(parsed) == question_id:
            found = parsed
    if found is None:
        raise CliError(f"{path}: no question has question_id {question_id}")
    return found


def _cmd_labelaudit(args: argparse.Namespace) -> int:
    if args.la_cmd == "estimate":
        _require_file(args.key, "secret key", "s1 labelaudit sheet")
        form_text = _read_text(Path(args.sheet))
        key_doc = _read_json(Path(args.key))
        try:
            check_key(key_doc)
        except ValueError as e:
            raise CliError(f"{args.key}: {e}") from None
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

    records = _filtered_records(args)
    # `--fit` given means calibrate, even when the file holds no fits: every record is then refused.
    fits = load_fits(args.fit) if getattr(args, "fit", None) else None
    if args.la_cmd == "rank":
        flags, skipped = rank_with_skips([r.raw for r in records], temperature=fits)
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
    except SheetInputError as e:
        where = args.questions if e.source == "question" else args.records
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
        use_calib=args.calibrated,
    )
    print(json.dumps(rep, indent=2, sort_keys=True))
    if args.fail_on_critical and rep["summary"]["critical"] > 0:
        return EXIT_DRIFT_CRITICAL
    return EXIT_OK


_DEFAULT_MODEL = {"mock": "jev-1.13.0", "systemone": "jev-1.13.0", "openai": "gpt-6-luna"}


def _probe_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return DeterministicMockBackend(served_model=args.served_model or args.model)
    if args.served_model is not None:
        raise CliError("--served-model applies only to --backend mock")
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
    q_doc = _read_json(Path(args.questions))
    if isinstance(q_doc, dict) and "questions" in q_doc and isinstance(q_doc["questions"], dict):
        q_map = q_doc["questions"]
    elif isinstance(q_doc, dict):
        q_map = {k: v for k, v in q_doc.items() if isinstance(v, dict) and "type" in v}
    else:
        raise CliError(f"{args.questions}: must be a JSON object mapping names to questions")
    if args.max_retries < 0:
        raise CliError("--max-retries must be >= 0")
    # None takes the backend's measured default inside run_batch; check an explicit value up front.
    check_reservation(args.worst_usd_per_token, args.reserve_overhead_tokens or 0.0)

    states = _load_jsonl_dicts(Path(args.states))
    try:
        check_states(states)
    except ValueError as e:
        raise CliError(f"{args.states}: {e}") from None
    if args.model is None:
        args.model = _DEFAULT_MODEL[args.backend]
    backend = _probe_backend(args)
    out_path = Path(args.output)
    ledger_path = Path(args.ledger)
    written = 0
    failed = 0
    lost = False
    try:
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
        close = getattr(backend, "close", None)
        if close is not None:
            close()

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
    p_pt.add_argument("--error-budget", type=float, default=0.10, help="maximum error rate among auto-accepted items; the threshold is the lowest tie-safe one that meets it (choice and score groups only under --max-false-pass)")
    p_pt.add_argument("--use-cp-upper", action="store_true", help="judge the budget (or --max-false-pass) against the Clopper-Pearson upper bound instead of the point estimate")
    p_pt.add_argument("--cp-alpha", type=float, default=0.05, help="significance level of the Clopper-Pearson bound")
    p_pt.add_argument("--cost-s1", type=float, default=0.00004, help=_H_COST_S1)
    p_pt.add_argument("--cost-sys2", type=float, default=0.004, help=_H_COST_SYS2)
    p_pt.add_argument("--max-escalation", type=float, default=None, help="largest fraction of items the noul escalation band may send to System 2 (default: 0.35)")
    p_pt.add_argument(
        "--max-false-pass",
        type=float,
        default=None,
        help="for noul groups, in place of the F1 band: the smallest cutoff at which at most this share of label-false items pass (p >= cutoff), written as a band with band_lo == band_hi",
    )
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
    p_sa = sub_sl.add_parser("answerable", help="Slice metrics vs majority & blind null, per (question, model, primitive) group", **_FMT)
    p_sa.add_argument("records", help=_H_RECORDS)
    p_sa.add_argument("--features", default=None, help="features JSONL tagging each item with a `slice`; untagged items fall in `default`")
    p_sa.add_argument("--question", default=None, help=_H_QUESTION)
    p_sa.add_argument("--model", default=None, help=_H_MODEL)
    p_sa.add_argument("--min-n", type=int, default=10, help="slices with fewer items are reported but never marked answerable")
    p_sm = sub_sl.add_parser("mine", help="Mine high-error feature slices with BH-FDR control, per (question, model, primitive) group", **_FMT)
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
    p_drf_dist = p_drf.add_mutually_exclusive_group()
    p_drf_dist.add_argument("--calibrated", action="store_true", help="compare calibrated probabilities (PSI, JS divergence, quantiles, entropy); every answered record in both files must carry `calib` from `s1 apply`, made by one fit per (question, model) group (or by none, for records applied before s1ops recorded it). The escalation rate always routes what each policy was tuned on")
    p_drf_dist.add_argument("--raw-only", action="store_true", help="compare raw probabilities; the default, kept for scripts written for s1ops 0.1.0")
    p_drf.add_argument("--fail-on-critical", action="store_true", help="exit 4 if any question is CRITICAL")
    p_drf.set_defaults(func=_cmd_drift)

    # probe
    p_prb = sub.add_parser("probe", help="Budget-guarded System 1 probe & batch runner", **_FMT)
    p_prb.add_argument("questions", help="questions JSON file")
    p_prb.add_argument("--states", required=True, help="local-only states JSONL; only state_sha256 reaches the output")
    p_prb.add_argument("-o", "--output", required=True, help="probe records JSONL to write")
    p_prb.add_argument("--ledger", required=True, help="budget journal; created if absent")
    p_prb.add_argument("--cap-usd", type=float, default=1.00, help="hard spend cap; the run stops before a call whose worst-case cost would exceed it")
    p_prb.add_argument("--model", default=None, help="model_requested sent to the backend (default: jev-1.13.0; gpt-6-luna for --backend openai)")
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
        return _fail(_os_error_message(e))
    except (ValueError, PolicyIntegrityError, BudgetExceeded) as e:
        # ValueError covers RecordError, FitError, LedgerCorrupt and json.JSONDecodeError.
        if debug:
            raise
        return _fail(str(e))


if __name__ == "__main__":
    sys.exit(main())

