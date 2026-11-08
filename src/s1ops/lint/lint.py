"""Run static lint rules over a questions file, apply suppressions, and analyze probe records."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from s1ops.contracts.records import Record, read_records
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


PROBE_ROLES = ("noul", "twin", "inverse")
INVERSE_SOURCES = ("supplied", "generated")


class ProbeInputError(ValueError):
    """A probe record `audit_probe_records` cannot pair: an unknown role, a mistyped tag, a role
    answered with the wrong primitive, one role twice for an item, or a second wording for one role for
    a question and model. Names the line and field."""


def _probe_base(line: int, raw: dict, question_id: str) -> str:
    for name in ("probe_question", "probe_base_question_id"):
        if name in raw and (not isinstance(raw[name], str) or not raw[name]):
            raise ProbeInputError(f"line {line}: '{name}' must be a non-empty string, got {type(raw[name]).__name__}")
    return raw.get("probe_question") or raw.get("probe_base_question_id") or question_id


def _check_probe_answer(line: int, role: str, primitive: str, classes: tuple) -> None:
    if role in ("noul", "inverse") and primitive != "noul":
        raise ProbeInputError(f"line {line}: probe_role '{role}' needs a noul answer, got {primitive}")
    if role == "twin" and (primitive != "choice" or "yes" not in classes):
        raise ProbeInputError(f"line {line}: probe_role 'twin' needs a choice answer with a 'yes' option, got {primitive}")


def _probe_inverse(line: int, role: str, raw: dict) -> str | None:
    if "probe_inverse" not in raw:
        return None
    tag = raw["probe_inverse"]
    if role != "inverse":
        raise ProbeInputError(f"line {line}: 'probe_inverse' is only for probe_role 'inverse', got '{role}'")
    if not isinstance(tag, str) or tag not in INVERSE_SOURCES:
        raise ProbeInputError(f"line {line}: 'probe_inverse' must be one of {', '.join(INVERSE_SOURCES)}")
    return tag


def _one_source(tags: set[str | None]) -> str | None:
    return next(iter(tags)) if len(tags) == 1 else None


def audit_probe_records(path: str | Path) -> list[dict]:
    """`audit_probe` over the records read from `path`."""
    return audit_probe(read_records(path))


def audit_probe(records: list[Record]) -> list[dict]:
    """Offline coherence analysis of probe records (`probe_role`: `noul`, `twin`, `inverse`).

    Records without `probe_role` are not probes and are skipped, as are failed probes
    (`model_served: null`) once their tags check out. A tagged record that cannot be paired
    raises `ProbeInputError` before anything is computed, and so does a question and model whose
    probes of one role carry more than one `question_id`: two wordings would share one rate.

    `inverse_source` is the `probe_inverse` tag every inverse record of the group carries
    (`supplied`: the question's `x-s1-lint.inverse`; `generated`: "Is it false that: ..."), or
    `None` when they differ, any is untagged, or there are none.
    """
    by_q_item: dict[tuple[str, str], dict[tuple[str | None, str], dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    first_line: dict[tuple[str, str, tuple[str | None, str], str], int] = {}
    role_ids: dict[tuple[tuple[str, str], str], dict[str, int]] = defaultdict(dict)
    inverse_tags: dict[tuple[str, str], set[str | None]] = defaultdict(set)
    for r in records:
        line = r.line
        if "probe_role" not in r.raw:
            if "probe_inverse" in r.raw:
                raise ProbeInputError(f"line {line}: 'probe_inverse' is only for probe_role 'inverse', and this record has no probe_role")
            continue
        role = r.raw["probe_role"]
        if role not in PROBE_ROLES:
            raise ProbeInputError(f"line {line}: 'probe_role' must be one of {', '.join(PROBE_ROLES)}")
        base_q = _probe_base(line, r.raw, r.question_id)
        tag = _probe_inverse(line, role, r.raw)
        if r.model_served is None:
            continue
        _check_probe_answer(line, role, r.primitive, r.classes)
        key = (base_q, r.model_served, r.item, role)
        if key in first_line:
            raise ProbeInputError(f"line {line}: a second '{role}' probe for the same item and model as line {first_line[key]}")
        first_line[key] = line
        group = (base_q, r.model_served)
        seen = role_ids[(group, role)]
        if r.question_id not in seen and seen:
            other_id, other_line = next(iter(seen.items()))
            raise ProbeInputError(
                f"line {line}: {role} probe {r.question_id} for question {base_q!r} and model "
                f"{r.model_served!r} is worded differently from line {other_line} ({other_id}); "
                "audit one wording at a time"
            )
        seen.setdefault(r.question_id, line)
        if role == "inverse":
            inverse_tags[group].add(tag)
        by_q_item[group][r.item][role] = r.raw

    reports: list[dict] = []
    for (base_q, model_served), items in sorted(by_q_item.items()):
        twin_gaps: list[float] = []
        twin_flips = 0
        inv_errors: list[float] = []
        inv_flips = 0
        points: list[dict] = []
        for (dataset, item_id), roles in sorted(items.items(), key=lambda kv: (kv[0][0] is not None, kv[0][0] or "", kv[0][1])):
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
                "inverse_source": _one_source(inverse_tags.get((base_q, model_served), set())),
                "points": points,
            }
        )
    return reports
