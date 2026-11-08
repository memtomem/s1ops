"""Issue 78's smaller items: valid JSON from `estimate`, the cap's precision, an English sheet, `s1 fit`'s refusal."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import FitError
from s1ops.improve.labelaudit import estimate, parse_answers
from s1ops.ledger.journal import BudgetExceeded, Ledger

_OLD_MARKER = chr(0xB2F5) + ":"


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def _sheet(ws, tmp_path: Path) -> tuple[Path, Path]:
    sheet, key = tmp_path / "sheet.md", tmp_path / "key.json"
    args = ["labelaudit", "sheet", ws["records"], "--states", ws["states_local"], "--fit", ws["fits"],
            "-o", str(sheet), "-k", str(key), "--top", "6", "--controls", "6"]
    assert cli_main(args) == 0
    return sheet, key


def _no_constant(name: str):
    raise AssertionError(f"{name} is not JSON")


def _answered(form: str, marker: str = "Answer:", answer: str = "True") -> str:
    return re.sub(r"^Answer: $", f"{marker} {answer}", form, flags=re.M)


def test_the_sheet_is_english(ws, tmp_path, capsys) -> None:
    sheet, _ = _sheet(ws, tmp_path)
    text = sheet.read_text(encoding="utf-8")
    assert text.startswith("# Label blind review\n\nChoose one answer for each item: True, False. Write it after `Answer:`.\n")
    assert "Question ID:\n\n> " in text
    assert len(re.findall(r"^Answer: $", text, re.M)) == 12
    assert not any(0xAC00 <= ord(c) <= 0xD7A3 for c in text if c != "\n")


@pytest.mark.parametrize("marker", ["Answer:", _OLD_MARKER], ids=["english", "korean-from-0.6.0"])
def test_a_finished_sheet_scores_with_either_marker(ws, tmp_path, capsys, marker) -> None:
    sheet, key = _sheet(ws, tmp_path)
    sheet.write_text(_answered(sheet.read_text(encoding="utf-8"), marker), encoding="utf-8")
    capsys.readouterr()
    assert cli_main(["labelaudit", "estimate", str(sheet), "--key", str(key)]) == 0
    report = json.loads(capsys.readouterr().out, parse_constant=_no_constant)
    assert report["scored"]["flag"]["n"] == 6 and report["scored"]["control"]["n"] == 6


@pytest.mark.parametrize(
    "form, message",
    [
        pytest.param("## 1\n\n> Answer: True\n\nAnswer: \n", "is not one of", id="a-quoted-marker-is-not-the-answer"),
        pytest.param("## 1\n\nAnswer: True\nAnswer: False\n", "expected one line starting with Answer:", id="two-english"),
        pytest.param(f"## 1\n\nAnswer: True\n{_OLD_MARKER} False\n", "expected one line starting with Answer:", id="one-of-each"),
        pytest.param("## 1\n\nno marker here\n", "expected one line starting with Answer:", id="none"),
    ],
)
def test_answer_parsing_refuses(form, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_answers(form, ["True", "False"])


@pytest.mark.parametrize("line", ["Answer: True", f"{_OLD_MARKER} True"], ids=["english", "korean"])
def test_answer_parsing_reads_a_marker_at_the_line_start_only(line) -> None:
    assert parse_answers(f"## 1\n\n> {line.replace('True', 'False')}\n\n{line}\n", ["True", "False"]) == {"1": "True"}


def test_estimate_prints_valid_json_when_the_control_rate_is_zero(ws, tmp_path, capsys) -> None:
    sheet, key = _sheet(ws, tmp_path)
    key_doc = json.loads(key.read_text(encoding="utf-8"))
    by_n = {item["n"]: item for item in key_doc["items"]}
    form = sheet.read_text(encoding="utf-8")

    def answer(match: re.Match) -> str:
        n = int(match.group(1))
        return f"{match.group(0)[: match.group(0).rindex('Answer:')]}Answer: {by_n[n]['label']}"

    # Every reviewer answer equals the recorded label, so nothing disagrees in either arm.
    agreed = re.sub(r"^## (\d+)$.*?^Answer: $", answer, form, flags=re.M | re.S)
    sheet.write_text(agreed, encoding="utf-8")
    capsys.readouterr()
    assert cli_main(["labelaudit", "estimate", str(sheet), "--key", str(key)]) == 0
    report = json.loads(capsys.readouterr().out, parse_constant=_no_constant)["estimate"]
    assert report["control_rate"] == 0
    assert report["lift"] is None and report["lift_note"] == "control rate is 0"


def test_estimate_dict_keeps_a_finite_lift_and_the_outcome() -> None:
    finite = estimate(4, 6, 1, 6)
    assert finite.as_dict()["lift"] == pytest.approx(4.0) and finite.as_dict()["lift_note"] is None
    undefined = estimate(6, 6, 0, 12)
    assert math.isinf(undefined.lift)
    as_dict = undefined.as_dict()
    assert as_dict["lift"] is None and as_dict["lift_note"] == "control rate is 0"
    assert as_dict["conservative_lift"] == pytest.approx(undefined.conservative_lift)
    assert as_dict["outcome"] == undefined.outcome() != "fail"


def test_the_budget_refusal_prints_the_cap_as_it_was_given(tmp_path) -> None:
    with Ledger(tmp_path / "j.jsonl", cap_usd=0.0012) as ledger:
        with pytest.raises(BudgetExceeded) as refused:
            ledger.reserve(0.01)
    assert "> cap $0.001200" in str(refused.value)


def test_fit_says_which_question_a_method_does_not_fit(ws, tmp_path, capsys) -> None:
    assert cli_main(["fit", ws["records_categorical_train"], "--method", "platt", "-o", str(tmp_path / "f.json")]) == 1
    line = capsys.readouterr().err.strip()
    assert re.fullmatch(
        r"s1: platt is not defined for (choice|score) \(question q-[0-9a-f]{16}\); "
        r"use one of \['temperature'\], or select a question this method fits with --question",
        line,
    )


def test_fit_keeps_its_other_refusals(ws, tmp_path, capsys) -> None:
    rows = [json.loads(line) for line in Path(ws["records_train"]).read_text(encoding="utf-8").splitlines()]
    records = tmp_path / "one_class.jsonl"
    records.write_text("".join(json.dumps({**row, "label": True}) + "\n" for row in rows), encoding="utf-8")
    assert cli_main(["fit", str(records), "--method", "platt", "-o", str(tmp_path / "f.json")]) == 1
    line = capsys.readouterr().err.strip()
    assert "every label is the same class" in line and "--question" not in line
    assert issubclass(FitError, ValueError)


def test_the_documented_rating_suppression_works(tmp_path, capsys) -> None:
    question = {"type": "score", "instructions": "How many stars did this reviewer give?",
                "criteria": [f"{n} star{'s' if n > 1 else ''}: {w}" for n, w in enumerate(("very negative", "negative", "mixed", "positive", "very positive"), 1)]}
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"stars": question}), encoding="utf-8")
    assert cli_main(["lint", "check", str(plain), "--format", "json"]) == 0
    fired = {f["rule"] for f in json.loads(capsys.readouterr().out)["findings"]}
    assert {"JL012", "JL031"} <= fired
    quiet = tmp_path / "quiet.json"
    quiet.write_text(json.dumps({"stars": {**question, "x-s1-lint": {"ignore": ["JL012", "JL031"]}}}), encoding="utf-8")
    assert cli_main(["lint", "check", str(quiet), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not {f["rule"] for f in report["findings"]} & {"JL012", "JL031"}
    assert report["counts"]["suppressed"] >= 2
