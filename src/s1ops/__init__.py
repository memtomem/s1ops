"""s1ops: Dedicated System 1 LLMOps SDK for Typed Decision Models."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.4.0"

if TYPE_CHECKING:
    from s1ops.contracts.questions import (
        ChoiceQuestion,
        NoulQuestion,
        QuestionModel,
        ScoreQuestion,
        parse_wire_question,
        question_id,
        to_openai_decision,
    )
    from s1ops.contracts.records import (
        Group,
        Record,
        RecordError,
        Selection,
        parse_record,
        read_records,
        select,
    )
    from s1ops.contracts.schemas import (
        PolicyEntry,
        PolicyFingerprintError,
        load_policies,
        save_policies,
    )
    from s1ops.eval.audit import audit_group
    from s1ops.eval.compare import compare_records
    from s1ops.improve.calibrate import (
        Fit,
        FitError,
        fit_for,
        fit_group,
        load_fits,
        save_fits,
        transform,
    )
    from s1ops.improve.policy import select_threshold
    from s1ops.lint.lint import lint_path
    from s1ops.monitor.drift import (
        DriftReport,
        audit_drift,
        jensen_shannon,
        psi,
    )
    from s1ops.monitor.backends import HttpSystemOneBackend, OpenAIDecisionsBackend
    from s1ops.monitor.runner import (
        Backend,
        BackendError,
        BackendResponse,
        DeterministicMockBackend,
        expand_probes,
        run_batch,
    )
    from s1ops.runtime.gate import (
        DecisionGate,
        GateDecision,
        ModelServedMismatchError,
        PolicyIntegrityError,
    )

# Imported on first use (PEP 562): `from s1ops.runtime.gate import DecisionGate` then loads only
# the runtime path, not the fitting stack (scipy.optimize, scikit-learn) or the web app.
_LAZY = {
    "ChoiceQuestion": "s1ops.contracts.questions",
    "NoulQuestion": "s1ops.contracts.questions",
    "QuestionModel": "s1ops.contracts.questions",
    "ScoreQuestion": "s1ops.contracts.questions",
    "parse_wire_question": "s1ops.contracts.questions",
    "question_id": "s1ops.contracts.questions",
    "to_openai_decision": "s1ops.contracts.questions",
    "Group": "s1ops.contracts.records",
    "Record": "s1ops.contracts.records",
    "RecordError": "s1ops.contracts.records",
    "Selection": "s1ops.contracts.records",
    "parse_record": "s1ops.contracts.records",
    "read_records": "s1ops.contracts.records",
    "select": "s1ops.contracts.records",
    "PolicyEntry": "s1ops.contracts.schemas",
    "PolicyFingerprintError": "s1ops.contracts.schemas",
    "load_policies": "s1ops.contracts.schemas",
    "save_policies": "s1ops.contracts.schemas",
    "audit_group": "s1ops.eval.audit",
    "compare_records": "s1ops.eval.compare",
    "Fit": "s1ops.improve.calibrate",
    "FitError": "s1ops.improve.calibrate",
    "fit_for": "s1ops.improve.calibrate",
    "fit_group": "s1ops.improve.calibrate",
    "load_fits": "s1ops.improve.calibrate",
    "save_fits": "s1ops.improve.calibrate",
    "transform": "s1ops.improve.calibrate",
    "select_threshold": "s1ops.improve.policy",
    "lint_path": "s1ops.lint.lint",
    "DriftReport": "s1ops.monitor.drift",
    "audit_drift": "s1ops.monitor.drift",
    "jensen_shannon": "s1ops.monitor.drift",
    "psi": "s1ops.monitor.drift",
    "HttpSystemOneBackend": "s1ops.monitor.backends",
    "OpenAIDecisionsBackend": "s1ops.monitor.backends",
    "Backend": "s1ops.monitor.runner",
    "BackendError": "s1ops.monitor.runner",
    "BackendResponse": "s1ops.monitor.runner",
    "DeterministicMockBackend": "s1ops.monitor.runner",
    "expand_probes": "s1ops.monitor.runner",
    "run_batch": "s1ops.monitor.runner",
    "DecisionGate": "s1ops.runtime.gate",
    "GateDecision": "s1ops.runtime.gate",
    "ModelServedMismatchError": "s1ops.runtime.gate",
    "PolicyIntegrityError": "s1ops.runtime.gate",
}

__all__ = [
    "__version__",
    "NoulQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "QuestionModel",
    "question_id",
    "parse_wire_question",
    "to_openai_decision",
    "Record",
    "RecordError",
    "Group",
    "Selection",
    "parse_record",
    "read_records",
    "select",
    "PolicyEntry",
    "PolicyFingerprintError",
    "save_policies",
    "load_policies",
    "Fit",
    "FitError",
    "fit_group",
    "transform",
    "load_fits",
    "save_fits",
    "fit_for",
    "audit_group",
    "compare_records",
    "select_threshold",
    "lint_path",
    "DriftReport",
    "audit_drift",
    "psi",
    "jensen_shannon",
    "Backend",
    "BackendError",
    "BackendResponse",
    "DeterministicMockBackend",
    "HttpSystemOneBackend",
    "OpenAIDecisionsBackend",
    "expand_probes",
    "run_batch",
    "DecisionGate",
    "GateDecision",
    "ModelServedMismatchError",
    "PolicyIntegrityError",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
