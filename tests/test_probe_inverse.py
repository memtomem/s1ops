"""The inverse probe's wording: supplied through `x-s1-lint.inverse`, or generated with its criteria."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import NoulQuestion, question_id
from s1ops.ledger.journal import Ledger
from s1ops.lint.lint import INVERSE_SOURCES, ProbeInputError, audit_probe, audit_probe_records, lint_path
from s1ops.lint.loader import load_questions
from s1ops.lint.rules import RULES
from s1ops.monitor.runner import DeterministicMockBackend, expand_probes, probe_tasks, run_batch
from s1ops.contracts.records import parse_record, Record

BASE = {
    "type": "noul",
    "instructions": "Is this movie review positive?",
    "criteria": {"true": "positive", "false": "negative"},
}
SUPPLIED = {"instructions": "Is this movie review negative?", "criteria": {"true": "negative", "false": "positive"}}


def _write(path: Path, doc: object) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _states(path: Path, n: int = 2) -> Path:
    path.write_text("".join(json.dumps({"item_id": f"i{i}", "state": f"text {i}"}) + "\n" for i in range(n)), encoding="utf-8")
    return path


def _probe(tmp_path: Path, questions: dict, *extra: str) -> tuple[int, Path, Path]:
    q = _write(tmp_path / "q.json", questions)
    out, ledger = tmp_path / "out.jsonl", tmp_path / "journal.jsonl"
    rc = cli_main(["probe", str(q), "--states", str(_states(tmp_path / "s.jsonl")), "-o", str(out), "--ledger", str(ledger), *extra])
    return rc, out, ledger


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("key", ["x-s1-lint", "x-jev-lint"])
def test_supplied_inverse_is_sent_and_tagged(tmp_path: Path, key: str) -> None:
    questions = {"positive": {**BASE, key: {"inverse": SUPPLIED}}}
    rc, out, _ = _probe(tmp_path, questions)
    assert rc == 0
    rows = _rows(out)
    by_role = {(r["item_id"], r["probe_role"]): r for r in rows}
    assert len(by_role) == 6
    lint_qid = load_questions(tmp_path / "q.json", frozenset(RULES)).specs[0].qid
    for item in ("i0", "i1"):
        assert by_role[(item, "noul")]["question_id"] == lint_qid == question_id(BASE)
        assert by_role[(item, "inverse")]["question_id"] == question_id({"type": "noul", **SUPPLIED})
        assert by_role[(item, "inverse")]["probe_inverse"] == "supplied"
        assert "probe_inverse" not in by_role[(item, "noul")] and "probe_inverse" not in by_role[(item, "twin")]
    (report,) = audit_probe_records(out)
    assert report["inverse_source"] == "supplied" and report["n_states"] == 2


@pytest.mark.parametrize(
    ("criteria", "swapped"),
    [
        ({"true": "positive", "false": "negative"}, {"true": "negative", "false": "positive"}),
        ({"false": "negative", "true": "positive"}, {"true": "negative", "false": "positive"}),
        ({"true": "positive"}, {"false": "positive"}),
        ({"false": "negative"}, {"true": "negative"}),
        ({}, {}),
        (None, None),
        ({"true": {"label": "pos", "examples": [1, 2]}, "false": None}, {"true": None, "false": {"label": "pos", "examples": [1, 2]}}),
    ],
    ids=["both", "both-reordered", "true-only", "false-only", "empty", "absent", "structured"],
)
def test_generated_inverse_swaps_the_criteria_it_has(criteria, swapped) -> None:
    base = {"type": "noul", "instructions": "  Is it good?  "}
    if criteria is not None:
        base["criteria"] = criteria
    tasks = probe_tasks({"good": base}, probe_mode=True)
    assert [(t.role, t.inverse_source) for t in tasks] == [("noul", None), ("twin", None), ("inverse", "generated")]
    inverse = tasks[2].question
    assert inverse == NoulQuestion(instructions="Is it false that: Is it good?", criteria=swapped)
    if swapped is not None:
        assert list(inverse.criteria) == [k for k in ("true", "false") if k in swapped]
    assert {t.inverse_source for t in tasks} - {None} <= set(INVERSE_SOURCES)


def test_generated_inverse_records_are_tagged_generated(tmp_path: Path) -> None:
    rc, out, _ = _probe(tmp_path, {"positive": BASE})
    assert rc == 0
    inverse = [r for r in _rows(out) if r["probe_role"] == "inverse"]
    assert [r["probe_inverse"] for r in inverse] == ["generated", "generated"]
    (report,) = audit_probe_records(out)
    assert report["inverse_source"] == "generated"


def test_no_expand_probes_tags_nothing(tmp_path: Path) -> None:
    rc, out, _ = _probe(tmp_path, {"positive": {**BASE, "x-s1-lint": {"inverse": SUPPLIED}}}, "--no-expand-probes")
    assert rc == 0
    rows = _rows(out)
    assert len(rows) == 2 and all("probe_role" not in r and "probe_inverse" not in r for r in rows)
    assert {r["question_id"] for r in rows} == {question_id(BASE)}


def test_supplied_inverse_expands_object_instructions() -> None:
    base = {"type": "noul", "instructions": {"task": "is it positive?"}}
    assert [t.role for t in probe_tasks({"obj": base}, probe_mode=True)] == ["primary"]
    supplied = {**base, "x-s1-lint": {"inverse": {"instructions": {"task": "is it negative?"}}}}
    tasks = probe_tasks({"obj": supplied}, probe_mode=True)
    assert [(t.role, t.inverse_source) for t in tasks] == [("noul", None), ("twin", None), ("inverse", "supplied")]
    assert tasks[1].question.instructions == {"task": "is it positive?"}


def test_expand_probes_keeps_its_triples() -> None:
    expanded = expand_probes({"positive": {**BASE, "x-s1-lint": {"inverse": SUPPLIED}}})
    assert [len(t) for t in expanded] == [3, 3, 3]
    assert expanded[2][2] == NoulQuestion(**SUPPLIED)


def test_run_batch_library_call_takes_dicts_with_extensions(tmp_path: Path) -> None:
    with Ledger(tmp_path / "j.jsonl", cap_usd=1.0) as ledger:
        recs = run_batch(
            [{"item_id": "a", "state": "s"}],
            {"positive": {**BASE, "x-s1-lint": {"inverse": SUPPLIED, "ignore": ["JL040"]}}},
            backend=DeterministicMockBackend(served_model="m"),
            ledger=ledger,
            model_requested="jev-1.13.0",
            probe_mode=True,
        )
    assert [(r["probe_role"], r.get("probe_inverse")) for r in recs] == [("noul", None), ("twin", None), ("inverse", "supplied")]


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({**BASE, "x-s1-lint": {"inverse": SUPPLIED, "colour": 1}}, "x-s1-lint: unknown keys ['colour']"),
        ({**BASE, "x-s1-lint": {"ignore": ["JL999"]}}, "x-s1-lint.ignore: unknown rule ids ['JL999']"),
        ({**BASE, "x-s1-lint": {"inverse": "Is it negative?"}}, "x-s1-lint.inverse must be a noul question object"),
        ({**BASE, "x-s1-lint": {"inverse": {"instructions": "x", "criteria": {"maybe": 1}}}}, "x-s1-lint.inverse.criteria.maybe"),
        (
            {"type": "choice", "instructions": "Which?", "criteria": {"a": 1, "b": 2}, "x-s1-lint": {"inverse": SUPPLIED}},
            "x-s1-lint.inverse is only supported on noul/predicate questions",
        ),
        ({**BASE, "x-s1-lint": {}, "x-jev-lint": {"inverse": SUPPLIED}}, "use one of x-s1-lint or x-jev-lint, not both"),
        ({"type": "noul"}, "instructions: Field required"),
    ],
    ids=["unknown-key", "unknown-rule", "inverse-not-object", "inverse-bad-criteria", "inverse-on-choice", "both-keys", "body"],
)
def test_bad_question_is_one_line_before_ledger_or_output(tmp_path: Path, capsys, question: dict, message: str) -> None:
    out = tmp_path / "out.jsonl"
    out.write_text("previous\n", encoding="utf-8")
    rc, out, ledger = _probe(tmp_path, {"ok": BASE, "bad": question})
    assert rc == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith(f"s1: {tmp_path / 'q.json'}: question 'bad': "), err
    assert message in err[0]
    assert out.read_text(encoding="utf-8") == "previous\n"
    assert not ledger.exists() and not Path(f"{ledger}.lock").exists()


def test_backend_refusal_of_a_supplied_inverse_is_one_line_before_ledger_or_output(tmp_path: Path, capsys, monkeypatch) -> None:
    pytest.importorskip("openai")
    from s1ops.monitor.backends import OpenAIDecisionsBackend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    closed = []
    real_close = OpenAIDecisionsBackend.close
    monkeypatch.setattr(OpenAIDecisionsBackend, "close", lambda self: (closed.append(True), real_close(self)))
    (tmp_path / "out.jsonl").write_text("previous\n", encoding="utf-8")
    predicate = {"type": "predicate", "instructions": "Is it positive?", "x-s1-lint": {"inverse": SUPPLIED}}
    rc, out, ledger = _probe(tmp_path, {"pos": predicate}, "--backend", "openai", "--model", "gpt-6-luna", "--usd-per-input-token", "1e-6")
    assert rc == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [f"s1: {tmp_path / 'q.json'}: question 'pos' (inverse probe): an OpenAI predicate has no criteria; fold them into the instructions"]
    assert out.read_text(encoding="utf-8") == "previous\n"
    assert not ledger.exists() and not Path(f"{ledger}.lock").exists()
    assert closed == [True]


def test_lint_check_reports_both_extension_keys_as_jl000(tmp_path: Path) -> None:
    q = _write(tmp_path / "q.json", {"positive": {**BASE, "x-s1-lint": {}, "x-jev-lint": {"ignore": ["JL040"]}}})
    result = lint_path(q)
    assert [(f.rule, f.question) for f in result.findings] == [("JL000", "positive")]
    assert "use one of x-s1-lint or x-jev-lint, not both" in result.findings[0].message


# audit_probe over hand-made records


def _rec(line: int, item: str, role: str, qid: str, p: float, tag: object = None, *, model: str = "m", base: str = "pos") -> Record:
    raw = {
        "schema": "s1-record/1", "item_id": item, "question_id": qid, "model_requested": model, "model_served": model,
        "probe_role": role, "probe_question": base,
    }
    if role == "twin":
        raw.update(primitive="choice", probabilities={"yes": p, "no": 1 - p}, choice="yes" if p >= 0.5 else "no", confidence=max(p, 1 - p))
    else:
        raw.update(primitive="noul", p=p)
    if tag is not None:
        raw["probe_inverse"] = tag
    return parse_record(raw, line)


def _group(inverse_tags: list[object], inverse_ids: list[str] | None = None) -> list[Record]:
    recs = []
    ids = inverse_ids or ["q-inv"] * len(inverse_tags)
    for i, (tag, qid) in enumerate(zip(inverse_tags, ids)):
        recs.append(_rec(3 * i + 1, f"i{i}", "noul", "q-base", 0.9))
        recs.append(_rec(3 * i + 2, f"i{i}", "inverse", qid, 0.2, tag))
    return recs


@pytest.mark.parametrize(
    ("tags", "source"),
    [
        (["supplied", "supplied"], "supplied"),
        (["generated", "generated"], "generated"),
        ([None, None], None),
        ([None, "generated"], None),
        (["generated", None], None),
        (["supplied", "generated"], None),
    ],
    ids=["supplied", "generated", "legacy", "legacy-then-tagged", "tagged-then-legacy", "same-id-both-tags"],
)
def test_inverse_source_is_the_one_tag_every_inverse_record_carries(tags, source) -> None:
    (report,) = audit_probe(_group(tags))
    assert report["inverse_source"] == source
    assert report["n_states"] == 2


def test_inverse_source_is_none_without_inverse_records() -> None:
    (report,) = audit_probe([_rec(1, "i0", "noul", "q-base", 0.9), _rec(2, "i0", "twin", "q-twin", 0.8)])
    assert report["inverse_source"] is None and report["mean_inverse_error"] is None


def test_two_inverse_wordings_for_one_question_and_model_are_refused() -> None:
    recs = _group(["generated", "generated", "generated"], ["q-inv-a", "q-inv-a", "q-inv-b"])
    with pytest.raises(ProbeInputError, match=r"^line 8: inverse probe q-inv-b .* from line 2 \(q-inv-a\)"):
        audit_probe(recs)


def test_two_inverse_wordings_under_different_models_are_separate_groups() -> None:
    recs = [
        _rec(1, "i0", "noul", "q-base", 0.9, model="a"),
        _rec(2, "i0", "inverse", "q-inv-a", 0.2, "generated", model="a"),
        _rec(3, "i0", "noul", "q-base", 0.9, model="b"),
        _rec(4, "i0", "inverse", "q-inv-b", 0.2, "supplied", model="b"),
    ]
    assert [(r["model_served"], r["inverse_source"]) for r in audit_probe(recs)] == [("a", "generated"), ("b", "supplied")]


def test_a_failed_inverse_probe_neither_counts_as_a_wording_nor_escapes_tag_checks() -> None:
    failed = parse_record(
        {"schema": "s1-record/1", "item_id": "i9", "question_id": "q-other", "primitive": "noul", "model_requested": "m",
         "model_served": None, "error_code": "timeout", "probe_role": "inverse", "probe_question": "pos", "probe_inverse": "generated"},
        9,
    )
    (report,) = audit_probe([*_group(["generated"]), failed])
    assert report["inverse_source"] == "generated"
    bad = parse_record({**failed.raw, "probe_inverse": "made-up"}, 9)
    with pytest.raises(ProbeInputError, match="^line 9: 'probe_inverse' must be one of supplied, generated$"):
        audit_probe([*_group(["generated"]), bad])


@pytest.mark.parametrize("tag", ["made-up", "", 1, True, None])
def test_mistyped_probe_inverse_is_refused(tag) -> None:
    recs = _group(["generated"])
    raw = {**recs[1].raw, "probe_inverse": tag}
    recs[1] = parse_record(raw, 2)
    with pytest.raises(ProbeInputError, match="^line 2: 'probe_inverse' must be one of supplied, generated$"):
        audit_probe(recs)


@pytest.mark.parametrize("model_served", ["m", None], ids=["answered", "failed"])
def test_probe_inverse_without_a_probe_role_is_refused(model_served) -> None:
    raw = {"schema": "s1-record/1", "item_id": "i0", "question_id": "q-x", "primitive": "noul", "model_requested": "m",
           "model_served": model_served, "probe_inverse": "generated"}
    if model_served is None:
        raw["error_code"] = "timeout"
    else:
        raw["p"] = 0.4
    with pytest.raises(ProbeInputError, match="^line 5: 'probe_inverse' is only for probe_role 'inverse', and this record has no probe_role$"):
        audit_probe([parse_record(raw, 5)])


@pytest.mark.parametrize("role", ["noul", "twin"])
def test_probe_inverse_on_another_role_is_refused(role) -> None:
    recs = _group(["generated"])
    index = 0 if role == "noul" else None
    if role == "twin":
        recs.append(_rec(3, "i0", "twin", "q-twin", 0.8, "generated"))
        index = 2
    else:
        recs[0] = _rec(1, "i0", "noul", "q-base", 0.9, "generated")
    with pytest.raises(ProbeInputError, match=f"^line {recs[index].line}: 'probe_inverse' is only for probe_role 'inverse', got '{role}'$"):
        audit_probe(recs)


def _run(tmp_path: Path, name: str, wording: str, item: str, *extension) -> list[dict]:
    q = {"type": "noul", "instructions": wording}
    if extension:
        q["x-s1-lint"] = {"inverse": extension[0]}
    run = tmp_path / name
    run.mkdir()
    states = run / "s.jsonl"
    states.write_text(json.dumps({"dataset": "d", "item_id": item, "state": "s"}) + "\n", encoding="utf-8")
    assert cli_main(["probe", str(_write(run / "q.json", {"pos": q})), "--states", str(states), "-o", str(run / "o.jsonl"), "--ledger", str(run / "j.jsonl")]) == 0
    return _rows(run / "o.jsonl")


def _audit(tmp_path: Path, rows: list[dict], capsys) -> tuple[int, list[str]]:
    capsys.readouterr()
    path = tmp_path / "pooled.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    rc = cli_main(["lint", "probe-audit", str(path)])
    return rc, capsys.readouterr().err.strip().splitlines()


def test_two_wordings_under_one_name_are_refused_with_a_shared_inverse(tmp_path: Path, capsys) -> None:
    shared = {"instructions": "Is this review negative?"}
    first = _run(tmp_path, "a", "Is this review positive?", "a", shared)
    second = _run(tmp_path, "b", "Was the reviewer pleased?", "b", shared)
    rc, err = _audit(tmp_path, first + second, capsys)
    assert rc == 1 and len(err) == 1
    assert "noul probe" in err[0] and "'pos'" in err[0] and "worded differently from line 1" in err[0]


@pytest.mark.parametrize("role", ["noul", "twin"])
def test_each_role_alone_is_checked(tmp_path: Path, capsys, role) -> None:
    first = [row for row in _run(tmp_path, "a", "Is this review positive?", "a") if row["probe_role"] == role]
    second = [row for row in _run(tmp_path, "b", "Was the reviewer pleased?", "b") if row["probe_role"] == role]
    assert first[0]["question_id"] != second[0]["question_id"]
    rc, err = _audit(tmp_path, first + second, capsys)
    assert rc == 1 and len(err) == 1
    assert f"{role} probe {second[0]['question_id']}" in err[0] and f"({first[0]['question_id']})" in err[0]


def test_the_same_wording_over_two_runs_pools(tmp_path: Path, capsys) -> None:
    rows = _run(tmp_path, "a", "Is this review positive?", "a") + _run(tmp_path, "b", "Is this review positive?", "b")
    path = tmp_path / "pooled.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (report,) = audit_probe_records(path)
    assert (report["probe_question"], report["n_states"]) == ("pos", 2)
