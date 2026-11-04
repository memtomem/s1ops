"""`labelaudit sheet` asks one question, offers that question's answers, and can show its wording (#73).

Measured on 0.6.0 with a six-question records file: the sheet asked "True, False" for 10 items of
which 8 were score items (labels 0..4 in the key), and never showed the question.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import parse_wire_question, question_id
from s1ops.contracts.records import parse_record
from s1ops.demo import generate_workspace
from s1ops.improve import labelaudit
from s1ops.improve.labelaudit import SheetInputError, build_sheet

NOUL_Q = {"type": "noul", "instructions": "Is this review positive?",
          "criteria": {"true": "The reviewer liked it", "false": "The reviewer did not"}}
CHOICE_Q = {"type": "choice", "instructions": "What is the topic?",
            "criteria": {"sports": "Games and athletes", "world": "International news", "tech": "Gadgets"}}
SCORE_Q = {"type": "score", "instructions": "How satisfied is the reviewer?",
           "criteria": ["unhappy", "neutral", "happy"]}


def _row(item: int, q: dict, label, *, model: str = "m", dataset: str = "d", probs: dict | None = None) -> dict:
    qid = question_id(q)
    row = {"schema": "s1-record/1", "dataset": dataset, "item_id": f"i{item}", "question_id": qid,
           "primitive": q["type"], "model_requested": model, "model_served": model}
    if q["type"] == "noul":
        row["p"] = 0.8 if item % 2 else 0.2
    else:
        keys = list(probs) if probs else (list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))])
        dist = probs or {k: round(1 / len(keys), 4) for k in keys}
        if not probs:
            dist[keys[0]] = round(1 - sum(dist[k] for k in keys[1:]), 4)
        top = max(dist, key=dist.get)
        row.update({"probabilities": dist, "confidence": dist[top]})
        row["choice" if q["type"] == "choice" else "score"] = top if q["type"] == "choice" else float(int(top))
    if label is not None:
        row["label"] = label
    return row


def _files(tmp_path: Path, rows: list[dict], n_states: int = 12) -> tuple[str, str]:
    rec = tmp_path / "records.jsonl"
    rec.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    items = sorted({(r["dataset"], r["item_id"]) for r in rows})
    states = tmp_path / "states.jsonl"
    states.write_text("".join(json.dumps({"dataset": d, "item_id": i, "state": f"text of {i}"}) + "\n" for d, i in items), encoding="utf-8")
    return str(rec), str(states)


def _questions(tmp_path: Path, **qs) -> str:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(qs), encoding="utf-8")
    return str(path)


def _sheet(tmp_path: Path, rec: str, states: str, *extra: str) -> list[str]:
    return ["labelaudit", "sheet", rec, "--states", states, "-o", str(tmp_path / "sheet.md"),
            "-k", str(tmp_path / "key.json"), "--top", "2", "--controls", "2", *extra]


def _err(capsys) -> str:
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("s1: ") and err.count("\n") == 1, (out, err)
    return err


def _mixed(tmp_path: Path) -> tuple[str, str]:
    rows = [_row(i, NOUL_Q, bool(i % 2)) for i in range(6)]
    rows += [_row(i, SCORE_Q, i % 3) for i in range(6, 12)]
    return _files(tmp_path, rows)


def _filled(tmp_path: Path, answer_of=None) -> Path:
    key = json.loads((tmp_path / "key.json").read_text(encoding="utf-8"))
    label = {it["n"]: (answer_of(it) if answer_of else it["label"]) for it in key["items"]}
    out, n = [], None
    for line in (tmp_path / "sheet.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"## (\d+)$", line)
        if m:
            n = int(m.group(1))
        if line.startswith("답:") and n is not None:
            line = f"답: {label[n]}"
        out.append(line)
    filled = tmp_path / "filled.md"
    filled.write_text("\n".join(out) + "\n", encoding="utf-8")
    return filled


def test_a_multi_question_file_needs_a_question(tmp_path: Path, capsys) -> None:
    rec, states = _mixed(tmp_path)
    assert cli_main(_sheet(tmp_path, rec, states)) == 1
    err = _err(capsys)
    assert "records hold 2 questions" in err and "pass --question" in err


def test_a_score_question_gets_its_levels_as_options(tmp_path: Path, capsys) -> None:
    rec, states = _mixed(tmp_path)
    assert cli_main(_sheet(tmp_path, rec, states, "--question", question_id(SCORE_Q))) == 0
    summary = json.loads(capsys.readouterr().out)
    key = json.loads((tmp_path / "key.json").read_text(encoding="utf-8"))
    assert summary["options"] == key["options"] == ["0", "1", "2"]
    assert key["question_id"] == question_id(SCORE_Q) and summary["question_shown"] is False
    assert all(it["label"] in key["options"] for it in key["items"])
    assert all(it["question_id"] == question_id(SCORE_Q) for it in key["items"])
    assert f"질문 ID:\n\n> {question_id(SCORE_Q)}\n" in (tmp_path / "sheet.md").read_text(encoding="utf-8")


def test_options_that_cannot_answer_the_labels_are_refused(tmp_path: Path, capsys) -> None:
    rec, states = _mixed(tmp_path)
    assert cli_main(_sheet(tmp_path, rec, states, "--question", question_id(SCORE_Q), "--options", "True,False")) == 1
    assert "cannot answer labels ['0', '1', '2']" in _err(capsys)


@pytest.mark.parametrize("options", ["", ",", "0,1,2,2", "0,1, 2"])
def test_bad_given_options_are_refused(tmp_path: Path, capsys, options) -> None:
    rec, states = _mixed(tmp_path)
    assert cli_main(_sheet(tmp_path, rec, states, "--question", question_id(SCORE_Q), "--options", options)) == 1
    _err(capsys)


def test_noul_options_stay_true_then_false(tmp_path: Path, capsys) -> None:
    rec, states = _mixed(tmp_path)
    assert cli_main(_sheet(tmp_path, rec, states, "--question", question_id(NOUL_Q))) == 0
    assert json.loads((tmp_path / "key.json").read_text(encoding="utf-8"))["options"] == ["True", "False"]
    assert "고르세요: True, False." in (tmp_path / "sheet.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("flag", ["--question", "--model"])
def test_a_filter_matching_nothing_names_what_is_there(tmp_path: Path, capsys, flag) -> None:
    rec, states = _mixed(tmp_path)
    for cmd in (["labelaudit", "rank", rec, flag, "zzz"], _sheet(tmp_path, rec, states, flag, "zzz")):
        assert cli_main(cmd) == 1
        err = _err(capsys)
        assert "'zzz'" in err
        if flag == "--question":
            assert question_id(NOUL_Q) in err and question_id(SCORE_Q) in err
        else:
            assert err.rstrip().endswith("the records hold m")


def test_rank_question_keeps_only_that_question(tmp_path: Path, capsys) -> None:
    rec, _ = _mixed(tmp_path)
    assert cli_main(["labelaudit", "rank", rec, "--question", question_id(NOUL_Q), "--top", "50"]) == 0
    top = json.loads(capsys.readouterr().out)["top"]
    assert top and {r["question_id"] for r in top} == {question_id(NOUL_Q)}


def test_a_selection_with_nothing_to_rank_is_refused(tmp_path: Path, capsys) -> None:
    rows = [_row(i, NOUL_Q, None) for i in range(4)] + [_row(i, SCORE_Q, i % 3) for i in range(4, 10)]
    rec, states = _files(tmp_path, rows)
    assert cli_main(["labelaudit", "rank", rec, "--question", question_id(NOUL_Q)]) == 1
    assert "nothing to rank" in _err(capsys)


def test_the_questions_file_shows_wording_and_criteria(tmp_path: Path, capsys) -> None:
    rec, states = _mixed(tmp_path)
    qfile = _questions(tmp_path, positive=NOUL_Q, satisfied=SCORE_Q)
    assert cli_main(_sheet(tmp_path, rec, states, "--question", question_id(NOUL_Q), "--questions", qfile)) == 0
    assert json.loads(capsys.readouterr().out)["question_shown"] is True
    text = (tmp_path / "sheet.md").read_text(encoding="utf-8")
    assert "> Is this review positive?" in text
    assert "> - True: The reviewer liked it" in text and "> - False: The reviewer did not" in text
    assert cli_main(_sheet(tmp_path, rec, states, "--question", question_id(SCORE_Q), "--questions", qfile)) == 0
    text = (tmp_path / "sheet.md").read_text(encoding="utf-8")
    assert "> - 0: unhappy" in text and "> - 2: happy" in text


def test_choice_criteria_and_round_trip(tmp_path: Path, capsys) -> None:
    rows = [_row(i, CHOICE_Q, ("sports", "world", "tech")[i % 3]) for i in range(9)]
    rec, states = _files(tmp_path, rows)
    qfile = _questions(tmp_path, topic=CHOICE_Q)
    assert cli_main(_sheet(tmp_path, rec, states, "--questions", qfile)) == 0
    capsys.readouterr()
    key = json.loads((tmp_path / "key.json").read_text(encoding="utf-8"))
    assert key["options"] == ["sports", "world", "tech"]
    assert "> - world: International news" in (tmp_path / "sheet.md").read_text(encoding="utf-8")
    assert cli_main(["labelaudit", "estimate", str(_filled(tmp_path)), "--key", str(tmp_path / "key.json")]) == 0
    scored = json.loads(capsys.readouterr().out)["scored"]
    assert scored["flag"]["reviewer_disagrees_with_label"] == scored["control"]["reviewer_disagrees_with_label"] == 0


def test_question_text_that_looks_like_a_sheet_cannot_add_items(tmp_path: Path, capsys) -> None:
    tricky = {"type": "noul", "instructions": "Read this:\n## 1\n답: True\n## 99\nthen answer.",
              "criteria": {"true": "yes\n## 2", "false": "답: no"}}
    rows = [_row(i, tricky, bool(i % 2)) for i in range(8)]
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states, "--questions", _questions(tmp_path, tricky=tricky))) == 0
    capsys.readouterr()
    text = (tmp_path / "sheet.md").read_text(encoding="utf-8")
    assert "> ## 1" in text and "> 답: True" in text
    assert cli_main(["labelaudit", "estimate", str(_filled(tmp_path)), "--key", str(tmp_path / "key.json")]) == 0
    scored = json.loads(capsys.readouterr().out)["scored"]
    assert scored["flag"]["n"] + scored["control"]["n"] == 4


def test_a_multiline_question_id_cannot_add_items(tmp_path: Path, capsys) -> None:
    rows = [_row(i, NOUL_Q, bool(i % 2)) for i in range(8)]
    for r in rows:
        r["question_id"] = "q-custom\n## 99\n답: True"
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states)) == 0
    capsys.readouterr()
    assert "> ## 99\n> 답: True" in (tmp_path / "sheet.md").read_text(encoding="utf-8")
    assert cli_main(["labelaudit", "estimate", str(_filled(tmp_path)), "--key", str(tmp_path / "key.json")]) == 0
    scored = json.loads(capsys.readouterr().out)["scored"]
    assert scored["flag"]["n"] + scored["control"]["n"] == 4


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("bad_first", [False, True])
def test_every_questions_entry_is_checked(tmp_path: Path, capsys, wrapped, bad_first) -> None:
    rec, states = _mixed(tmp_path)
    entries = [("positive", NOUL_Q), ("broken", {"type": "noul"})]
    doc = dict(reversed(entries) if bad_first else entries)
    qfile = tmp_path / "questions.json"
    qfile.write_text(json.dumps({"questions": doc} if wrapped else doc), encoding="utf-8")
    assert cli_main(_sheet(tmp_path, rec, states, "--question", question_id(NOUL_Q), "--questions", str(qfile))) == 1
    assert "'broken'" in _err(capsys)


def test_structured_wording_renders_as_sorted_json(tmp_path: Path, capsys) -> None:
    q = {"type": "choice", "instructions": {"question": "Topic?", "focus": "the headline"},
         "criteria": {"a": {"desc": "first", "eg": ["x"]}, "b": "second", "c": "third"}}
    rows = [_row(i, q, ("a", "b", "c")[i % 3]) for i in range(8)]
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states, "--questions", _questions(tmp_path, q=q))) == 0
    text = (tmp_path / "sheet.md").read_text(encoding="utf-8")
    assert '> {"focus": "the headline", "question": "Topic?"}' in text
    assert '> - a: {"desc": "first", "eg": ["x"]}' in text


def test_questions_file_errors(tmp_path: Path, capsys) -> None:
    rec, states = _mixed(tmp_path)
    args = _sheet(tmp_path, rec, states, "--question", question_id(NOUL_Q), "--questions")
    assert cli_main([*args, _questions(tmp_path, other=CHOICE_Q)]) == 1
    assert "no question has question_id" in _err(capsys)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cli_main([*args, str(bad)]) == 1
    assert "bad.json" in _err(capsys)
    assert cli_main([*args, _questions(tmp_path, broken={"type": "noul"})]) == 1
    err = _err(capsys)
    assert "questions.json" in err and "'broken'" in err


def test_a_question_of_another_kind_is_refused(tmp_path: Path, capsys) -> None:
    # Records that carry the questions file's question_id but another option set, or another primitive.
    other = {**CHOICE_Q, "criteria": {"sports": "a", "world": "b", "science": "c"}}
    rows = [_row(i, CHOICE_Q, "sports") for i in range(8)]
    for r in rows:
        r["question_id"] = question_id(other)
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states, "--questions", _questions(tmp_path, other=other))) == 1
    assert "differ from the records'" in _err(capsys)
    for r in rows:
        r["question_id"] = question_id(SCORE_Q)
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states, "--questions", _questions(tmp_path, s=SCORE_Q))) == 1
    assert "is a score question, the records answer a choice one" in _err(capsys)


def test_noncontiguous_score_levels(tmp_path: Path, capsys) -> None:
    rows = [_row(i, SCORE_Q, (0, 2)[i % 2], probs={"00": 0.4, "02": 0.6}) for i in range(8)]
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states)) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "key.json").read_text(encoding="utf-8"))["options"] == ["0", "2"]
    assert cli_main(["labelaudit", "estimate", str(_filled(tmp_path)), "--key", str(tmp_path / "key.json")]) == 0


@pytest.mark.parametrize("bad_key", ["very  good", " yes "])
def test_choice_keys_a_reviewer_cannot_type_are_refused(tmp_path: Path, capsys, bad_key) -> None:
    q = {"type": "choice", "instructions": "Rate it", "criteria": {bad_key: "x", "bad": "y"}}
    rows = [_row(i, q, (bad_key, "bad")[i % 2]) for i in range(8)]
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states)) == 1
    assert repr(bad_key) in _err(capsys)


def test_two_models_of_one_question(tmp_path: Path, capsys) -> None:
    rows = [_row(i, NOUL_Q, bool(i % 2), model=m) for m in ("m1", "m2") for i in range(8)]
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states)) == 0
    key = json.loads((tmp_path / "key.json").read_text(encoding="utf-8"))
    items = [(it["dataset"], it["item_id"]) for it in key["items"]]
    assert len(items) == len(set(items)) == 4
    capsys.readouterr()
    rows[0]["label"] = not rows[0]["label"]  # m1 labels i0 differently from m2
    rec, states = _files(tmp_path, rows)
    assert cli_main(_sheet(tmp_path, rec, states)) == 1
    assert "('d', 'i0') is labeled" in _err(capsys)


def test_two_models_with_different_classes_or_primitives_are_refused(tmp_path: Path, capsys) -> None:
    a = [_row(i, CHOICE_Q, "sports", model="m1") for i in range(6)]
    b = [_row(i, CHOICE_Q, "sports", model="m2", probs={"sports": 0.5, "world": 0.5}) for i in range(6)]
    rec, states = _files(tmp_path, a + b)
    assert cli_main(_sheet(tmp_path, rec, states)) == 1
    assert "has classes" in _err(capsys)
    # noul (False, True) and score levels (0, 1) compare equal as sets: the primitive check comes first
    score2 = {"type": "score", "instructions": "x", "criteria": ["no", "yes"]}
    n = [_row(i, NOUL_Q, bool(i % 2), model="m1") for i in range(6)]
    s = [_row(i, score2, i % 2, model="m2") for i in range(6)]
    for r in s:
        r["question_id"] = question_id(NOUL_Q)
    rec, states = _files(tmp_path, n + s)
    assert cli_main(_sheet(tmp_path, rec, states)) == 1
    assert "answered as noul, score" in _err(capsys)


def _records(rows: list[dict]) -> list:
    return [parse_record(row, n) for n, row in enumerate(rows, 1)]


TEXT = {("d", f"i{i}"): f"text {i}" for i in range(12)}
TOPICS = ["sports", "world", "tech"]


def _topics(n: int = 6, **kw) -> list:
    return _records([_row(i, CHOICE_Q, TOPICS[i % 3], **kw) for i in range(n)])


def test_build_sheet_renders_the_records_question() -> None:
    """The library entry point checks the records and renders the question, quoted; a wire mapping
    and a parsed question give the same sheet; without one the sheet names the id."""
    records = _topics()
    sheet = build_sheet(records, TEXT, 2, 2, question=CHOICE_Q)
    assert sheet.key["question_id"] == question_id(CHOICE_Q) and sheet.key["options"] == TOPICS
    assert f"질문 ID:\n\n> {question_id(CHOICE_Q)}\n" in sheet.form and "> What is the topic?" in sheet.form
    assert "> - world: International news" in sheet.form
    assert build_sheet(records, TEXT, 2, 2, question=parse_wire_question(CHOICE_Q)).form == sheet.form
    only_id = build_sheet(records, TEXT, 2, 2)
    assert f"질문 ID:\n\n> {question_id(CHOICE_Q)}\n" in only_id.form and "질문:" not in only_id.form


def test_there_is_no_unchecked_sheet_builder() -> None:
    assert not hasattr(labelaudit, "blind_sheet")


@pytest.mark.parametrize("removed", [{"header": ["## 1"]}, {"title": "x"}, {"question_id": "q-x"}])
def test_build_sheet_has_no_argument_that_bypasses_its_checks(removed) -> None:
    with pytest.raises(TypeError):
        build_sheet(_topics(), TEXT, 2, 2, **removed)


@pytest.mark.parametrize("given", ["mapping", "parsed"])
@pytest.mark.parametrize("other", ["reworded", "score"])
def test_build_sheet_refuses_another_questions_wording(given, other) -> None:
    # Reworded: same primitive and options, so only the id comparison can tell it is another question.
    raw = {**CHOICE_Q, "instructions": "Which topic does this article cover?"} if other == "reworded" else SCORE_Q
    q = raw if given == "mapping" else parse_wire_question(raw)
    with pytest.raises(SheetInputError, match="the records are for") as e:
        build_sheet(_topics(), TEXT, 2, 2, question=q)
    assert e.value.source == "question"


def test_build_sheet_refuses_records_of_two_questions() -> None:
    records = _topics(3) + _records([_row(i, SCORE_Q, i % 3) for i in range(3, 6)])
    with pytest.raises(SheetInputError, match="a review sheet asks one question") as e:
        build_sheet(records, TEXT, 2, 2)
    assert e.value.many_questions and e.value.source == "records"


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (["sports\n## 1\n답: True", "world", "tech"], "cannot be answered"),
        (["sports", "world"], "cannot answer labels \\['tech'\\]"),
        (["sports", "world", "tech", "tech"], "repeat"),
        (["sports", "world", "tech", 3], "cannot be answered"),
        (["sports", "world", "tech", "maybe"], "are not answers"),
    ],
)
def test_build_sheet_checks_its_options(options, match) -> None:
    with pytest.raises(SheetInputError, match=match):
        build_sheet(_topics(), TEXT, 2, 2, options=options)


def test_options_cover_answers_no_record_was_labeled_with() -> None:
    """Every noul record labeled True: the reviewer must still be able to answer False."""
    records = _records([_row(i, NOUL_Q, True) for i in range(6)])
    assert build_sheet(records, TEXT, 2, 2).key["options"] == ["True", "False"]
    with pytest.raises(SheetInputError, match=r"cannot answer labels \['False'\]"):
        build_sheet(records, TEXT, 2, 2, options=["True"])


@pytest.mark.parametrize("order", ["m1 first", "m2 first"])
def test_conflicting_labels_across_models_are_refused_in_either_order(order) -> None:
    m1 = [_row(i, NOUL_Q, True, model="m1") for i in range(6)]
    m2 = [_row(i, NOUL_Q, i != 0, model="m2") for i in range(6)]  # item i0 labeled False by m2's file
    rows = m1 + m2 if order == "m1 first" else m2 + m1
    with pytest.raises(SheetInputError, match=r"item \('d', 'i0'\) is labeled"):
        build_sheet(_records(rows), TEXT, 2, 2)


def test_a_record_parsed_loosely_is_read_as_its_row_says() -> None:
    """A labeled score record parsed with require_answer=False has no label as parsed; build_sheet
    re-parses every row, sees the score question, and refuses the mix instead of ranking it."""
    noul = _records([_row(i, NOUL_Q, bool(i % 2)) for i in range(4)])
    score = parse_record(_row(9, SCORE_Q, 1), 9, require_answer=False)
    assert score.label is None and score.raw["label"] == 1
    with pytest.raises(SheetInputError, match="records hold 2 questions"):
        build_sheet([*noul, score], TEXT, 1, 2)


@pytest.mark.parametrize(
    "edit",
    [
        {"label": True},             # the row has no label
        {"item_id": "i0"},           # the row is item i5: text_of is looked up by the row
        {"model_served": "other"},   # the row's model picks the calibration
        {"classes": (True, False)},  # the row's class order decides the options
        {"primitive": "score"},      # the row is noul
    ],
)
def test_fields_edited_after_parsing_do_not_count(edit) -> None:
    """build_sheet reads every record from its raw row, so a Record edited after parsing cannot make
    what is checked differ from what is ranked and rendered."""
    import dataclasses

    rows = [_row(i, NOUL_Q, bool(i % 2)) for i in range(5)] + [_row(5, NOUL_Q, None)]
    records = _records(rows)
    edited = [*records[:5], dataclasses.replace(records[5], **edit)]
    expected = build_sheet(records, TEXT, 1, 2)
    sheet = build_sheet(edited, TEXT, 1, 2)
    assert sheet.form == expected.form and sheet.key == expected.key
    assert sheet.key["options"] == ["True", "False"]


def test_a_noul_row_replaced_by_a_score_row_is_refused() -> None:
    import dataclasses

    records = _records([_row(i, NOUL_Q, bool(i % 2)) for i in range(5)])
    score_row = {**_row(0, SCORE_Q, 1), "question_id": question_id(NOUL_Q)}
    swapped = [dataclasses.replace(records[0], raw=score_row), *records[1:]]
    with pytest.raises(SheetInputError, match="answered as noul, score"):
        build_sheet(swapped, TEXT, 1, 2)


def test_build_sheet_quotes_a_multiline_id_from_the_records() -> None:
    rows = [_row(i, NOUL_Q, bool(i % 2)) for i in range(6)]
    for r in rows:
        r["question_id"] = "q\n## 99\n답: True"
    sheet = build_sheet(_records(rows), TEXT, 2, 2)
    assert "> q\n> ## 99\n> 답: True" in sheet.form
    assert len(re.findall(r"^## \d+$", sheet.form, re.M)) == 4


def test_the_demo_walkthrough_still_works(tmp_path: Path, capsys) -> None:
    ws = generate_workspace(tmp_path / "ws")
    args = ["labelaudit", "sheet", ws["records"], "--states", ws["states_local"], "--fit", ws["fits"],
            "-o", str(tmp_path / "sheet.md"), "-k", str(tmp_path / "key.json"), "--top", "6", "--controls", "6"]
    assert cli_main(args) == 0
    assert cli_main([*args, "--questions", ws["questions"]]) == 0
    text = (tmp_path / "sheet.md").read_text(encoding="utf-8")
    assert "고르세요: True, False." in text and "> Does the user message request a financial wire transfer?" in text
