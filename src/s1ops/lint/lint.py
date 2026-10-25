"""Run static lint rules over a questions file, apply suppressions, and analyze probe records."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from s1ops.contracts.records import read_records
from s1ops.lint.loader import QuestionsFile, load_questions
from s1ops.lint.rules import RULES, UNSUPPRESSIBLE, LintConfig, Severity, check_pairs, check_question

KNOWN_RULES = frozenset(RULES)


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    question: str | None
    question_id: str | None
    message: str
    fix: str
    evidence: str | None
    evidence_url: str | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LintResult:
    path: str
    findings: list[Finding] = field(default_factory=list)
    suppressed: list[Finding] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c = Counter(f.severity for f in self.findings)
        return {"error": c["error"], "warning": c["warning"], "info": c["info"], "suppressed": len(self.suppressed)}

    def exit_code(self, *, strict: bool = False) -> int:
        failing = {"error", "warning"} if strict else {"error"}
        return 1 if any(f.severity in failing for f in self.findings) else 0

    def as_dict(self) -> dict:
        return {
            "schema": "s1-lint-report/1",
            "file": self.path,
            "counts": self.counts(),
            "findings": [f.as_dict() for f in self.findings],
            "suppressed": [f.as_dict() for f in self.suppressed],
        }


def _finding(rule_id: str, question: str | None, qid: str | None, message: str) -> Finding:
    rule = RULES[rule_id]
    return Finding(rule_id, rule.severity, question, qid, message, rule.fix, rule.evidence, rule.evidence_url)


def lint(
    qfile: QuestionsFile,
    path: str,
    config: LintConfig = LintConfig(),
    ignore: frozenset[str] = frozenset(),
) -> LintResult:
    blocked = sorted(ignore & UNSUPPRESSIBLE)
    if blocked:
        raise ValueError(f"rules {blocked} cannot be suppressed")
    result = LintResult(path)
    for message in qfile.errors:
        result.findings.append(_finding("JL000", None, None, message))
    by_name = {spec.name: spec for spec in qfile.specs}
    raw: list[tuple[str, str, str]] = []
    for spec in qfile.specs:
        raw.extend(("JL000", spec.name, e) for e in spec.errors)
        attempted = sorted(spec.ignore & UNSUPPRESSIBLE)
        if attempted:
            raw.append(("JL000", spec.name, f"ignore lists {attempted}, which cannot be suppressed"))
        raw.extend((rule_id, spec.name, message) for rule_id, message in check_question(spec, config))
    raw.extend((rule_id, name, message) for name, (rule_id, message) in check_pairs(qfile.specs))
    for rule_id, name, message in raw:
        spec = by_name[name]
        finding = _finding(rule_id, name, spec.qid, message)
        if rule_id not in UNSUPPRESSIBLE and (rule_id in ignore or rule_id in spec.ignore):
            result.suppressed.append(finding)
        else:
            result.findings.append(finding)
    order = {"error": 0, "warning": 1, "info": 2}
    names = list(by_name)
    result.findings.sort(
        key=lambda f: (-1 if f.question is None else names.index(f.question), order[f.severity], f.rule)
    )
    return result


def lint_path(
    path: str | Path, config: LintConfig = LintConfig(), ignore: frozenset[str] = frozenset()
) -> LintResult:
    return lint(load_questions(path, KNOWN_RULES), str(path), config, ignore)


def format_text(result: LintResult) -> str:
    lines = []
    current: object = object()
    for f in result.findings:
        if f.question != current:
            current = f.question
            header = "(file)" if f.question is None else f.question
            if f.question_id:
                header += f"  [{f.question_id}]"
            lines.append(header)
        lines.append(f"  {f.rule} {f.severity:<7} {RULES[f.rule].title}: {f.message}")
        lines.append(f"      fix: {f.fix}")
        if f.evidence:
            lines.append(f"      evidence: {f.evidence} {f.evidence_url}")
    c = result.counts()
    lines.append(
        f"{result.path}: {c['error']} error(s), {c['warning']} warning(s), {c['info']} info, {c['suppressed']} suppressed"
    )
    return "\n".join(lines)


def audit_probe_records(path: str | Path) -> list[dict]:
    """Offline coherence analysis of probe records (`probe_role`: `noul`, `twin`, `inverse`)."""
    records = read_records(path)
    by_q_item: dict[tuple[str, str], dict[tuple[str | None, str], dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if r.model_served is None:
            continue
        role = r.raw.get("probe_role")
        base_q = r.raw.get("probe_question") or r.raw.get("probe_base_question_id") or r.question_id
        if role in ("noul", "twin", "inverse"):
            by_q_item[(str(base_q), r.model_served)][r.item][role] = r.raw

    reports: list[dict] = []
    for (base_q, model_served), items in sorted(by_q_item.items()):
        twin_gaps: list[float] = []
        twin_flips = 0
        inv_errors: list[float] = []
        inv_flips = 0
        points: list[dict] = []
        for (dataset, item_id), roles in sorted(items.items()):
            if "noul" not in roles:
                continue
            p_noul = float(roles["noul"]["p"])
            point: dict = {"dataset": dataset, "item_id": item_id, "p_noul": p_noul}
            if "twin" in roles:
                p_yes = float(roles["twin"]["probabilities"]["yes"])
                twin_gaps.append(abs(p_noul - p_yes))
                twin_flips += (p_noul >= 0.5) != (p_yes >= 0.5)
                point["p_twin_yes"] = p_yes
                point["twin_gap"] = abs(p_noul - p_yes)
            if "inverse" in roles:
                p_inv = float(roles["inverse"]["p"])
                inv_errors.append(abs(p_noul + p_inv - 1.0))
                inv_flips += (p_noul >= 0.5) == (p_inv >= 0.5)
                point["p_inverse"] = p_inv
                point["inverse_sum"] = p_noul + p_inv
            points.append(point)
        reports.append(
            {
                "probe_question": base_q,
                "model_served": model_served,
                "n_states": len(points),
                "mean_twin_gap": sum(twin_gaps) / len(twin_gaps) if twin_gaps else None,
                "max_twin_gap": max(twin_gaps) if twin_gaps else None,
                "twin_flip_rate": twin_flips / len(twin_gaps) if twin_gaps else None,
                "mean_inverse_error": sum(inv_errors) / len(inv_errors) if inv_errors else None,
                "max_inverse_error": max(inv_errors) if inv_errors else None,
                "inverse_flip_rate": inv_flips / len(inv_errors) if inv_errors else None,
                "points": points,
            }
        )
    return reports
