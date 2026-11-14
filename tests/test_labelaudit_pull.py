"""`s1 labelaudit pull`: reviews made in a Langfuse annotation queue, as `s1.<question_id>.label`
scores on an exported observation or its trace, applied to the labels of the records the export was
made from. s1ops reads files only: the scores are a Scores API v3 response the caller fetched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records
from s1ops.improve.labelaudit import LabelPatchError, pull_labels

QID, OTHER_QID = "q-aaaaaaaaaaaaaaaa", "q-bbbbbbbbbbbbbbbb"
SECRET = "SECRET-STATE-TEXT"
N = 6


def _noul(item: str, model: str, label: bool, qid: str = QID, **extra) -> dict:
    return {"schema": "s1-record/1", "dataset": "d", "item_id": item, "question_id": qid, "primitive": "noul",
            "model_requested": model, "model_served": model, "p": 0.3 + 0.05 * int(item[1:]), "label": label, **extra}  # fmt: skip


def _choice(item: str, model: str, label: str) -> dict:
    probs = {"bug": 0.6, "billing": 0.3, "other": 0.1}
    return {"schema": "s1-record/1", "dataset": "d", "item_id": item, "question_id": QID, "primitive": "choice",
            "model_requested": model, "model_served": model, "choice": "bug", "confidence": 0.6, "probabilities": probs, "label": label}  # fmt: skip


def _score(item: str, model: str, label: int) -> dict:
    probs = {"0": 0.2, "1": 0.5, "2": 0.3}
    return {"schema": "s1-record/1", "dataset": "d", "item_id": item, "question_id": QID, "primitive": "score",
            "model_requested": model, "model_served": model, "score": 1.0, "confidence": 0.5, "probabilities": probs, "label": label}  # fmt: skip


def _noul_rows() -> list[dict]:
    return [_noul(f"i{i}", m, label=i % 2 == 0, state=f"{SECRET} {i}") for m in ("m1", "m2") for i in range(N)]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _rows(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class Queue:
    """Records, the export `s1 export langfuse` made of them, and scores a reviewer gave in Langfuse."""

    def __init__(self, tmp_path: Path, rows: list[dict], capsys, run: str = "r1") -> None:
        self.dir = tmp_path
        self.records = _write(tmp_path / "records.jsonl", rows)
        self.export = tmp_path / f"export-{run}.json"
        assert cli_main(["export", "langfuse", self.records, "--run-name", run, "-o", str(self.export)]) == 0
        capsys.readouterr()
        self.doc = json.loads(self.export.read_text())

    def obs(self, item: str, model: str = "m1", qid: str = QID) -> dict:
        return next(o for o in self.doc["observations"] if o["input"]["item_id"] == item and o["model"] == model and o["input"]["question_id"] == qid)

    def score(self, item: str, value, data_type: str = "BOOLEAN", *, model: str = "m1", on: str = "observation", qid: str = QID, sid: str | None = None) -> dict:
        o = self.obs(item, model)
        subject = {"kind": "observation", "id": o["id"], "traceId": o["traceId"]} if on == "observation" else {"kind": "trace", "id": o["traceId"]}
        return {"id": sid or f"s-{item}-{model}-{on}", "projectId": "p", "name": f"s1.{qid}.label", "value": value, "dataType": data_type,
                "source": "ANNOTATION", "subject": subject}  # fmt: skip

    def scores_file(self, rows: list[dict], wrap: bool = True) -> str:
        path = self.dir / "scores.json"
        path.write_text(json.dumps({"data": rows, "meta": {"limit": 100}} if wrap else rows), encoding="utf-8")
        return str(path)

    def pull(self, scores: str, out: Path, records: str | None = None, export: Path | None = None) -> int:
        return cli_main(["labelaudit", "pull", records or self.records, "--export", str(export or self.export), "--scores", scores, "-o", str(out)])


@pytest.fixture
def queue(tmp_path: Path, capsys) -> Queue:
    return Queue(tmp_path, _noul_rows(), capsys)


def test_a_reviewers_answer_becomes_the_label_of_every_record_of_the_item(queue: Queue, tmp_path: Path, capsys) -> None:
    # i1 is labeled False: the reviewer says True. i2 is labeled True and confirmed. Unrelated scores are not read.
    other = {"id": "x", "name": "s1.q-aaaaaaaaaaaaaaaa.brier", "value": 0.1, "dataType": "NUMERIC", "subject": None}
    scores = queue.scores_file([queue.score("i1", True), queue.score("i2", True), other])
    out = tmp_path / "pulled.jsonl"
    assert queue.pull(scores, out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["scores"], summary["scores_elsewhere"], summary["reviewed"], summary["corrected"]) == (2, 0, 2, 1)
    assert (summary["records_patched"], summary["records_already"], summary["state_dropped"]) == (2, 0, 2 * N)
    assert summary["corrections"] == [{"dataset": "d", "item_id": "i1", "question_id": QID, "arm": None, "from": False, "to": True}]
    pulled = _rows(out)
    for old, new in zip(_noul_rows(), pulled):
        expected = {k: v for k, v in old.items() if k != "state"}
        if old["item_id"] == "i1":
            expected["label"] = True
        assert new == expected
    assert SECRET not in out.read_text()
    assert len(read_records(out)) == 2 * N


def test_a_score_on_the_trace_reviews_the_item(queue: Queue, tmp_path: Path, capsys) -> None:
    out = tmp_path / "pulled.jsonl"
    # i3 is labeled False; the trace is the item, so the score reviews both models' records of it.
    assert queue.pull(queue.scores_file([queue.score("i3", True, on="trace")], wrap=False), out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["corrected"], summary["records_patched"]) == (1, 2)
    assert {r["label"] for r in _rows(out) if r["item_id"] == "i3"} == {True}


def test_pulling_again_over_the_pulled_records_changes_nothing(queue: Queue, tmp_path: Path, capsys) -> None:
    scores = queue.scores_file([queue.score("i1", True)])
    once, twice = tmp_path / "once.jsonl", tmp_path / "twice.jsonl"
    assert queue.pull(scores, once) == 0
    capsys.readouterr()
    assert queue.pull(scores, twice, records=str(once)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["reviewed"], summary["corrected"], summary["records_patched"]) == (1, 0, 0)
    assert twice.read_bytes() == once.read_bytes()


def test_label_scores_of_another_export_are_left_out_and_counted(queue: Queue, tmp_path: Path, capsys) -> None:
    (tmp_path / "other").mkdir()
    other = Queue(tmp_path / "other", _noul_rows(), capsys, run="r2")
    out = tmp_path / "pulled.jsonl"
    elsewhere = [other.score("i1", True), other.score("i2", False, on="trace")]
    assert queue.pull(queue.scores_file([queue.score("i1", True), *elsewhere]), out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["scores"], summary["scores_elsewhere"], summary["corrected"]) == (1, 2, 1)
    # Only another run's scores: nothing to pull from this export.
    assert queue.pull(queue.scores_file(elsewhere), tmp_path / "none.jsonl") == 1
    err = capsys.readouterr().err
    assert "scores.json: no `s1.<question_id>.label` score is on an observation or trace of this export (2 on others)" in err
    assert not (tmp_path / "none.jsonl").exists()


def test_two_reviews_of_one_item_must_agree(queue: Queue, tmp_path: Path, capsys) -> None:
    out = tmp_path / "pulled.jsonl"
    agree = [queue.score("i1", True), queue.score("i1", True, model="m2"), queue.score("i1", True, on="trace")]
    assert queue.pull(queue.scores_file(agree), out) == 0
    assert json.loads(capsys.readouterr().out)["reviewed"] == 1
    differ = [queue.score("i1", True, sid="a"), queue.score("i1", False, model="m2", sid="b")]
    assert queue.pull(queue.scores_file(differ), tmp_path / "x.jsonl") == 1
    assert "score a and score b answer item ('d', 'i1') 'True' and 'False'; keep one" in capsys.readouterr().err
    assert not (tmp_path / "x.jsonl").exists()


@pytest.mark.parametrize(
    ("make", "value", "data_type", "label"),
    [
        (lambda i, m: _choice(f"i{i}", m, label="bug"), "billing", "CATEGORICAL", "billing"),
        (lambda i, m: _score(f"i{i}", m, label=1), 2, "NUMERIC", 2),
        (lambda i, m: _score(f"i{i}", m, label=1), 2.0, "NUMERIC", 2),
        (lambda i, m: _score(f"i{i}", m, label=1), "0", "CATEGORICAL", 0),
    ],
)
def test_choice_and_score_answers_become_the_questions_classes(tmp_path: Path, capsys, make, value, data_type, label) -> None:
    q = Queue(tmp_path, [make(i, m) for m in ("m1", "m2") for i in range(N)], capsys)
    out = tmp_path / "pulled.jsonl"
    assert q.pull(q.scores_file([q.score("i0", value, data_type)]), out) == 0
    assert json.loads(capsys.readouterr().out)["records_patched"] == 2
    got = [r["label"] for r in _rows(out) if r["item_id"] == "i0"]
    assert got == [label, label] and all(type(g) is type(label) for g in got)
    assert {r.label for r in read_records(out) if r.item_id == "i0"} == {label}


@pytest.mark.parametrize(
    ("value", "data_type", "takes"),
    [
        (1, "NUMERIC", "a BOOLEAN"),
        ("True", "CATEGORICAL", "a BOOLEAN"),
        (1, "BOOLEAN", "a BOOLEAN"),
        ("yes", "TEXT", "a BOOLEAN"),
    ],
)
def test_an_answer_the_question_cannot_take_is_refused(queue: Queue, tmp_path: Path, capsys, value, data_type, takes) -> None:
    assert queue.pull(queue.scores_file([queue.score("i1", value, data_type, sid="s9")]), tmp_path / "x.jsonl") == 1
    err = capsys.readouterr().err
    assert f"score s9 (s1.{QID}.label) is {data_type} {value!r}; a noul question takes {takes} of ['True', 'False']" in err
    assert not (tmp_path / "x.jsonl").exists()


def test_a_score_level_outside_the_classes_is_refused(tmp_path: Path, capsys) -> None:
    q = Queue(tmp_path, [_score(f"i{i}", m, label=1) for m in ("m1",) for i in range(N)], capsys)
    for value, kind in ((5, "NUMERIC"), (1.5, "NUMERIC"), ("7", "CATEGORICAL")):
        assert q.pull(q.scores_file([q.score("i0", value, kind)]), tmp_path / "x.jsonl") == 1
        assert "a score question takes a whole NUMERIC level or a CATEGORICAL naming one of ['0', '1', '2']" in capsys.readouterr().err


def test_a_score_without_its_subject_names_the_field_group(queue: Queue, tmp_path: Path, capsys) -> None:
    row = queue.score("i1", True, sid="s1")
    del row["subject"]
    assert queue.pull(queue.scores_file([row]), tmp_path / "x.jsonl") == 1
    assert f"score s1 (s1.{QID}.label) has no `subject`; fetch the scores with `fields=subject`" in capsys.readouterr().err


def test_a_label_score_of_another_question_or_on_a_session_is_refused(tmp_path: Path, capsys) -> None:
    rows = [_noul(f"i{i}", "m1", label=True, qid=qid) for qid in (QID, OTHER_QID) for i in range(N)]
    q = Queue(tmp_path, rows, capsys)
    wrong = q.score("i1", False, qid=OTHER_QID, sid="w")  # on QID's observation, named for OTHER_QID
    assert q.pull(q.scores_file([wrong]), tmp_path / "x.jsonl") == 1
    assert f"score w is named s1.{OTHER_QID}.label and is on observation {q.obs('i1')['id']}, which asked question {QID}" in capsys.readouterr().err
    session = dict(q.score("i1", False, sid="ss"), subject={"kind": "session", "id": "sess-1"})
    assert q.pull(q.scores_file([session]), tmp_path / "x.jsonl") == 1
    assert "score ss (s1.q-aaaaaaaaaaaaaaaa.label) is on a session; a label score is on an observation or a trace" in capsys.readouterr().err


def test_two_questions_are_pulled_together(tmp_path: Path, capsys) -> None:
    rows = [_noul(f"i{i}", "m1", label=True, qid=qid) for qid in (QID, OTHER_QID) for i in range(N)]
    q = Queue(tmp_path, rows, capsys)
    other = dict(q.score("i2", False, sid="o"), name=f"s1.{OTHER_QID}.label",
                 subject={"kind": "observation", "id": q.obs("i2", qid=OTHER_QID)["id"], "traceId": q.obs("i2", qid=OTHER_QID)["traceId"]})  # fmt: skip
    out = tmp_path / "pulled.jsonl"
    assert q.pull(q.scores_file([q.score("i1", False), other]), out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert [(c["question_id"], c["item_id"]) for c in summary["corrections"]] == [(QID, "i1"), (OTHER_QID, "i2")]
    changed = {(r["question_id"], r["item_id"]) for r in _rows(out) if r["label"] is False}
    assert changed == {(QID, "i1"), (OTHER_QID, "i2")}


def test_records_the_export_was_not_made_from_are_refused(queue: Queue, tmp_path: Path, capsys) -> None:
    scores = queue.scores_file([queue.score("i1", True)])
    no_item = _write(tmp_path / "no_item.jsonl", [r for r in _noul_rows() if r["item_id"] != "i1"])
    assert queue.pull(scores, tmp_path / "x.jsonl", records=no_item) == 1
    err = capsys.readouterr().err
    assert f"{no_item}: item ('d', 'i1') of question {QID} (score s-i1-m1-observation) has no answered, labeled record" in err
    # A failed probe's row of the item carries another label than the item's answered records.
    failed = {"schema": "s1-record/1", "dataset": "d", "item_id": "i1", "question_id": QID, "primitive": "noul",
              "model_requested": "m3", "model_served": None, "error_code": "timeout", "label": "maybe"}  # fmt: skip
    bad = _write(tmp_path / "bad.jsonl", [*_noul_rows(), failed])
    assert queue.pull(scores, tmp_path / "x.jsonl", records=bad) == 1
    assert f"{bad}: line {2 * N + 1}:" in capsys.readouterr().err
    assert not (tmp_path / "x.jsonl").exists()


def test_a_failed_probes_row_of_a_corrected_item_takes_the_answer_too(queue: Queue, tmp_path: Path, capsys) -> None:
    failed = {"schema": "s1-record/1", "dataset": "d", "item_id": "i1", "question_id": QID, "primitive": "noul",
              "model_requested": "m3", "model_served": None, "error_code": "timeout", "label": False}  # fmt: skip
    records = _write(tmp_path / "with_failed.jsonl", [*_noul_rows(), failed])
    out = tmp_path / "pulled.jsonl"
    assert queue.pull(queue.scores_file([queue.score("i1", True)]), out, records=records) == 0
    assert json.loads(capsys.readouterr().out)["records_patched"] == 3
    assert _rows(out)[-1]["label"] is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "s1-langfuse-export/0"}, "schema: Input should be 's1-langfuse-export/1'"),
        ({"redaction": "none"}, "redaction: Input should be 'safe-v1'"),
        ({"observations": "x"}, "observations: Input should be a valid list"),
    ],
)
def test_an_export_that_is_not_one_is_refused_by_file(queue: Queue, tmp_path: Path, capsys, change, message) -> None:
    bad = tmp_path / "bad_export.json"
    bad.write_text(json.dumps({**queue.doc, **change}))
    assert queue.pull(queue.scores_file([queue.score("i1", True)]), tmp_path / "x.jsonl", export=bad) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"s1: {bad}: ") and message in err


def test_a_scores_file_that_is_not_a_response_is_refused(queue: Queue, tmp_path: Path, capsys) -> None:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"rows": []}))
    assert queue.pull(str(path), tmp_path / "x.jsonl") == 1
    assert f"s1: {path}: must be a Scores API v3 response (an object with a `data` list) or a list of its rows" in capsys.readouterr().err
    path.write_text(json.dumps([{"name": f"s1.{QID}.label", "dataType": 3, "subject": None}]))
    assert queue.pull(str(path), tmp_path / "x.jsonl") == 1
    assert f"s1: {path}: data.0: dataType: Input should be a valid string" in capsys.readouterr().err


def test_inputs_and_output_are_checked_before_anything_is_read(queue: Queue, tmp_path: Path, capsys) -> None:
    scores = queue.scores_file([queue.score("i1", True)])
    assert queue.pull(scores, tmp_path / "x.jsonl", export=tmp_path / "missing.json") == 1
    assert "Langfuse export not found" in capsys.readouterr().err
    assert queue.pull(scores, Path(queue.records)) == 1
    assert "is the same file as" in capsys.readouterr().err
    assert _rows(queue.records)[0]["state"].startswith(SECRET)


def test_the_library_raises_label_patch_error_naming_the_input() -> None:
    with pytest.raises(LabelPatchError) as e:
        pull_labels([], {"schema": "s1-langfuse-export/1", "redaction": "safe-v1", "observations": []}, {"data": []})
    assert e.value.source == "scores"
