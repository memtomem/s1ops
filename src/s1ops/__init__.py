"""s1ops: Dedicated System 1 LLMOps SDK for Typed Decision Models."""

from __future__ import annotations

__version__ = "0.2.0"

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
    load_policies,
    save_policies,
)
from s1ops.monitor.drift import (
    DriftReport,
    audit_drift,
    jensen_shannon,
    psi,
)
from s1ops.monitor.runner import (
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
    "save_policies",
    "load_policies",
    "DriftReport",
    "audit_drift",
    "psi",
    "jensen_shannon",
    "DeterministicMockBackend",
    "expand_probes",
    "run_batch",
    "DecisionGate",
    "GateDecision",
    "ModelServedMismatchError",
    "PolicyIntegrityError",
]
