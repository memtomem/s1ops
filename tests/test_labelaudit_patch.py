"""`s1 labelaudit patch`: a finished blind review, applied to the labels of the records it reviewed (#89)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records
from s1ops.improve.labelaudit import LabelPatchError, patch_labels

QID, OTHER_QID = "q-aaaaaaaaaaaaaaaa", "q-bbbbbbbbbbbbbbbb"
SECRET = "SECRET-STATE-TEXT"
N = 12


def _noul(item: str, model: str, label: bool, p: float, qid: str = QID) -> dict:
    return {"schema": "s1-record/1", "dataset": "d", "item_id": item, "question_id": qid, "primitive": "noul",
            "model_requested": model, "model_served": model, "p": p, "label": label}  # fmt: skip


def _choice(item: str, model: str, label: str, top: str) -> dict:
    probs = {o: (0.8 if o == top else 0.1) for o in ("bug", "billing", "other")}
    return {"schema": "s1-record/1", "dataset": "d", "item_id": item, "question_id": QID, "primitive": "choice",
            "model_requested": model, "model_served": model, "choice": top, "confidence": 0.8, "probabilities": probs, "label": label}  # fmt: skip


def _score(item: str, model: str, label: int, top: int) -> dict:
    probs = {str(level): (0.8 if level == top else 0.1) for level in (0, 1, 2)}
    return {"schema": "s1-record/1", "dataset": "d", "item_id": item, "question_id": QID, "primitive": "score",
            "model_requested": model, "model_served": model, "score": float(top), "confidence": 0.8, "probabilities": probs, "label": label}  # fmt: skip


def _noul_rows() -> list[dict]:
    rows = []
    for model in ("m1", "m2"):
        rows += [_noul(f"i{i}", model, label=i % 2 == 0, p=0.1 + 0.07 * i) for i in range(N)]
    return rows


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _rows(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class Review:
    """A sheet built by the CLI from `rows`, and the key it was built with."""

    def __init__(self, tmp_path: Path, rows: list[dict], capsys) -> None:
        self.dir = tmp_path
        self.records = _write(tmp_path / "records.jsonl", rows)
        items = sorted({r["item_id"] for r in rows})
        states = _write(tmp_path / "states.jsonl", [{"dataset": "d", "item_id": i, "state": f"{SECRET} {i}"} for i in items])
        self.sheet, self.key_path = tmp_path / "sheet.md", tmp_path / "key.json"
        argv = ["labelaudit", "sheet", self.records, "--states", states, "-o", str(self.sheet), "-k", str(self.key_path), "--top", "4", "--controls", "4"]
        if len({r["question_id"] for r in rows}) > 1:
            argv += ["--question", QID]
        assert cli_main(argv) == 0
        capsys.readouterr()
        self.key = json.loads(self.key_path.read_text())

    def answer(self, overturn: dict[str, str] | None = None) -> Path:
        """The sheet filled in: each item's own label, except the items of `overturn`."""
        overturn = overturn or {}
        by_n = {str(i["n"]): overturn.get(i["item_id"], i["label"]) for i in self.key["items"]}
        n, out = None, []
        for line in self.sheet.read_text().split("\n"):
            m = re.fullmatch(r"## (\d+)", line)
            n = m.group(1) if m else n
            out.append(f"Answer: {by_n[n]}" if line == "Answer: " else line)
        done = self.dir / "done.md"
        done.write_text("\n".join(out))
        return done

    def reviewed(self, k: int) -> list[str]:
        return [i["item_id"] for i in self.key["items"]][:k]

    def patch(self, sheet: Path, out: Path, records: str | None = None) -> int:
        return cli_main(["labelaudit", "patch", records or self.records, "--sheet", str(sheet), "--key", str(self.key_path), "-o", str(out)])


@pytest.fixture
def review(tmp_path: Path, capsys) -> Review:
    return Review(tmp_path, _noul_rows(), capsys)


def test_the_reviewers_answer_becomes_the_label_of_every_record_of_the_item(review: Review, tmp_path: Path, capsys) -> None:
    a, b = review.reviewed(2)
    before = {r["item_id"]: r["label"] for r in _rows(review.records)}
    done = review.answer({a: str(not before[a]), b: str(not before[b])})
    out = tmp_path / "patched.jsonl"
    assert review.patch(done, out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["reviewed"], summary["corrected"], summary["records_patched"], summary["records_already"]) == (8, 2, 4, 0)
    assert summary["question_id"] == QID and "state_dropped" not in summary
    assert {(c["item_id"], c["from"], c["to"]) for c in summary["corrections"]} == {(a, before[a], not before[a]), (b, before[b], not before[b])}
    assert {c["arm"] for c in summary["corrections"]} <= {"flag", "control"}
    patched = _rows(out)
    assert len(patched) == 2 * N
    for old, new in zip(_rows(review.records), patched):
        expected = {**old, "label": not old["label"]} if old["item_id"] in (a, b) else old
        assert new == expected
    assert all(isinstance(r["label"], bool) for r in patched)
    assert len(read_records(out)) == 2 * N


def test_a_review_that_confirms_every_label_changes_nothing(review: Review, tmp_path: Path, capsys) -> None:
    out = tmp_path / "patched.jsonl"
    assert review.patch(review.answer(), out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["corrected"], summary["records_patched"], summary["corrections"]) == (0, 0, [])
    assert _rows(out) == _rows(review.records)


def test_applying_the_review_again_changes_nothing_more(review: Review, tmp_path: Path, capsys) -> None:
    (a,) = review.reviewed(1)
    label = next(i["label"] for i in review.key["items"] if i["item_id"] == a)
    done = review.answer({a: str(label != "True")})
    once, twice = tmp_path / "once.jsonl", tmp_path / "twice.jsonl"
    assert review.patch(done, once) == 0
    capsys.readouterr()
    assert review.patch(done, twice, records=str(once)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["corrected"], summary["records_patched"], summary["records_already"]) == (1, 0, 2)
    assert twice.read_bytes() == once.read_bytes()


@pytest.mark.parametrize(
    ("make", "wrong", "right"),
    [
        (lambda i, m: _choice(f"i{i}", m, label=("bug", "billing", "other")[i % 3], top=("bug", "billing", "other")[(i + i // 4) % 3]), "other", str),
        (lambda i, m: _score(f"i{i}", m, label=i % 3, top=(i + i // 4) % 3), "2", int),
    ],
    ids=["choice", "score"],
)
def test_a_corrected_label_has_the_type_its_primitive_labels_have(tmp_path: Path, capsys, make, wrong: str, right) -> None:
    review = Review(tmp_path, [make(i, m) for m in ("m1", "m2") for i in range(N)], capsys)
    target = next(i for i in review.key["items"] if i["label"] != wrong)
    out = tmp_path / "patched.jsonl"
    assert review.patch(review.answer({target["item_id"]: wrong}), out) == 0
    capsys.readouterr()
    labels = {r["label"] for r in _rows(out) if r["item_id"] == target["item_id"]}
    assert labels == {right(wrong)} and all(type(v) is right for v in labels)
    assert len(read_records(out)) == 2 * N  # the parser takes every corrected label


def test_only_labeled_rows_of_the_question_asked_are_touched(tmp_path: Path, capsys) -> None:
    extra = [_noul(f"i{i}", "m1", label=True, p=0.9, qid=OTHER_QID) for i in range(N)]
    unlabeled = [{k: v for k, v in _noul(f"i{i}", "m3", label=True, p=0.5).items() if k != "label"} for i in range(N)]
    review = Review(tmp_path, [*_noul_rows(), *extra, *unlabeled], capsys)
    (a,) = review.reviewed(1)
    label = next(i["label"] for i in review.key["items"] if i["item_id"] == a)
    out = tmp_path / "patched.jsonl"
    assert review.patch(review.answer({a: str(label != "True")}), out) == 0
    assert json.loads(capsys.readouterr().out)["records_patched"] == 2
    changed = [(old, new) for old, new in zip(_rows(review.records), _rows(out)) if old != new]
    assert [(new["item_id"], new["question_id"], new["model_served"]) for _, new in changed] == [(a, QID, "m1"), (a, QID, "m2")]


def test_state_text_is_not_copied_out(tmp_path: Path, capsys) -> None:
    review = Review(tmp_path, [{**r, "state": SECRET} for r in _noul_rows()], capsys)
    out = tmp_path / "patched.jsonl"
    assert review.patch(review.answer(), out) == 0
    assert json.loads(capsys.readouterr().out)["state_dropped"] == 2 * N
    assert SECRET not in out.read_text() and all("state" not in r for r in _rows(out))


# ---- refusals ----


def test_records_relabeled_since_the_sheet_are_refused_by_line(review: Review, tmp_path: Path, capsys) -> None:
    (a,) = review.reviewed(1)
    rows = _rows(review.records)
    line = next(n for n, r in enumerate(rows, start=1) if r["item_id"] == a)
    for r in rows:  # in every model's record: one item with two labels is refused before this
        if r["item_id"] == a:
            r["label"] = not r["label"]
    edited = _write(tmp_path / "edited.jsonl", rows)
    out = tmp_path / "patched.jsonl"
    assert review.patch(review.answer(), out, records=edited) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"s1: {edited}: line {line}: item ('d', '{a}') is labeled") and "not the records the sheet was built from" in err
    assert not out.exists()


def test_records_without_a_reviewed_item_are_refused(review: Review, tmp_path: Path, capsys) -> None:
    (a,) = review.reviewed(1)
    fewer = _write(tmp_path / "fewer.jsonl", [r for r in _rows(review.records) if r["item_id"] != a])
    out = tmp_path / "patched.jsonl"
    assert review.patch(review.answer(), out, records=fewer) == 1
    assert capsys.readouterr().err.startswith(f"s1: {fewer}: 1 of the 8 reviewed items have no labeled record, first ('d', '{a}')")
    assert not out.exists()


def test_records_of_another_question_are_refused(review: Review, tmp_path: Path, capsys) -> None:
    other = _write(tmp_path / "other.jsonl", [{**r, "question_id": OTHER_QID} for r in _rows(review.records)])
    assert review.patch(review.answer(), tmp_path / "patched.jsonl", records=other) == 1
    assert capsys.readouterr().err == f"s1: {other}: no answered, labeled record is for question {QID}, the one the sheet asked\n"


def test_an_unfinished_sheet_names_the_sheet(review: Review, tmp_path: Path, capsys) -> None:
    out = tmp_path / "patched.jsonl"
    assert review.patch(review.sheet, out) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"s1: {review.sheet}: item ") and SECRET not in err
    assert not out.exists()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda k: k.pop("question_id"), "question_id: Field required"),
        (lambda k: k["items"][0].pop("item_id"), "items.0.item_id: Field required"),
        (lambda k: k["items"][0].update(question_id=OTHER_QID), f"item {{n}} is for question {OTHER_QID}"),
        (lambda k: k["items"][0].update(label="Maybe"), "item {n}: label 'Maybe' is not one of the options"),
        (lambda k: k.update(items=[i for i in k["items"] if i["arm"] == "flag"]), "items: no control item"),
    ],
)
def test_a_key_that_cannot_place_its_items_names_the_key(review: Review, tmp_path: Path, capsys, edit, message: str) -> None:
    done = review.answer()
    n = review.key["items"][0]["n"]
    edit(review.key)
    review.key_path.write_text(json.dumps(review.key))
    assert review.patch(done, tmp_path / "patched.jsonl") == 1
    assert capsys.readouterr().err.startswith(f"s1: {review.key_path}: {message.format(n=n)}")
    assert not (tmp_path / "patched.jsonl").exists()


def test_the_output_cannot_replace_an_input(review: Review, tmp_path: Path, capsys) -> None:
    before = Path(review.records).read_bytes()
    assert review.patch(review.answer(), Path(review.records)) == 1
    assert capsys.readouterr().err == f"s1: --output {review.records} is the same file as {review.records}\n"
    assert Path(review.records).read_bytes() == before


def test_a_missing_key_names_the_command_that_writes_it(review: Review, tmp_path: Path, capsys) -> None:
    done = review.answer()
    review.key_path.unlink()
    assert review.patch(done, tmp_path / "patched.jsonl") == 1
    assert capsys.readouterr().err == f"s1: secret key not found: {review.key_path} (run `s1 labelaudit sheet` first)\n"


def test_the_library_says_which_input_is_at_fault(review: Review) -> None:
    records = read_records(review.records)
    with pytest.raises(LabelPatchError) as e:
        patch_labels(records, review.sheet.read_text(), review.key)
    assert e.value.source == "sheet"
    with pytest.raises(LabelPatchError) as e:
        patch_labels(records, review.answer().read_text(), {**review.key, "options": []})
    assert e.value.source == "key"
    patch = patch_labels(records, review.answer().read_text(), review.key)
    assert (patch.reviewed, patch.corrections, patch.records_patched) == (8, (), 0)
