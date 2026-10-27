"""CLI error contract: one `s1:` line on stderr, no traceback, and the documented exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops import __version__
from s1ops.cli import main as cli_main
from s1ops.contracts.schemas import load_policies
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import FitError, load_fits


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _domain_error(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert err.startswith("s1: ")
    assert "Traceback" not in err
    assert out == ""
    return err


def _policy_qid(ws: dict[str, str]) -> str:
    return json.loads(Path(ws["policy"]).read_text())["policies"][0]["question_id"]


def test_missing_records_file_exits_1_with_prefix(tmp_path: Path, capsys) -> None:
    assert cli_main(["audit", str(tmp_path / "nope.jsonl")]) == 1
    assert "nope.jsonl" in _domain_error(capsys)


def test_missing_fit_names_producer(ws, tmp_path: Path, capsys) -> None:
    rc = cli_main(["apply", ws["records"], "--fit", str(tmp_path / "fits.json"), "-o", str(tmp_path / "o.jsonl")])
    assert rc == 1
    assert "`s1 fit`" in _domain_error(capsys)


def test_missing_policy_names_producer(tmp_path: Path, capsys) -> None:
    rc = cli_main(["gate", str(tmp_path / "policy.json"), "--question", "q", "--model", "m", "--p", "0.5"])
    assert rc == 1
    assert "`s1 policy tune`" in _domain_error(capsys)


def test_malformed_jsonl_reports_line_number(ws, tmp_path: Path, capsys) -> None:
    lines = Path(ws["records"]).read_text().splitlines()[:2]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(lines + ["{not json"]) + "\n")
    assert cli_main(["audit", str(bad)]) == 1
    assert "line 3" in _domain_error(capsys)


def test_jsonl_line_numbers_count_leading_blank_lines(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.jsonl"
    grades.write_text('\n\n{"judge": 1, "human": 1}\nnull\n')
    assert cli_main(["judge", "audit", str(grades)]) == 1
    assert "line 4: expected a JSON object" in _domain_error(capsys)


def test_json_array_rows_must_be_objects(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.json"
    grades.write_text('[{"judge": 1, "human": 1}, null]')
    assert cli_main(["judge", "audit", str(grades)]) == 1
    assert "element 1: expected a JSON object" in _domain_error(capsys)


def test_json_array_of_objects_still_accepted(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.json"
    grades.write_text(json.dumps([{"judge": j, "human": h} for j, h in [(0, 0), (1, 1), (2, 1), (3, 3)]]))
    assert cli_main(["judge", "audit", str(grades)]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["n"] == 4


def test_gate_model_mismatch_exits_3(ws, capsys) -> None:
    rc = cli_main(["gate", ws["policy"], "--question", _policy_qid(ws), "--model", "kev-1.0", "--p", "0.9"])
    assert rc == 3
    assert "model_served" in _domain_error(capsys)


def test_gate_unknown_question_exits_1(ws, capsys) -> None:
    rc = cli_main(["gate", ws["policy"], "--question", "q-0000000000000000", "--model", "m", "--p", "0.5"])
    assert rc == 1
    err = _domain_error(capsys)
    assert "No policy found" in err
    assert "KeyError" not in err


def test_gate_policy_sha_mismatch_exits_1(ws, capsys) -> None:
    rc = cli_main(
        [
            "gate", ws["policy"], "--question", _policy_qid(ws), "--model", "jev-1.13.0",
            "--p", "0.9", "--expected-sha256", "deadbeef",
        ]
    )
    assert rc == 1
    assert "SHA-256 mismatch" in _domain_error(capsys)


def test_drift_fail_on_critical_exits_4(ws, capsys) -> None:
    rc = cli_main(["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"], "--fail-on-critical"])
    assert rc == 4
    assert json.loads(capsys.readouterr().out)["summary"]["critical"] >= 1


def test_fit_zero_groups_exits_1_with_message(ws, tmp_path: Path, capsys) -> None:
    out = tmp_path / "fits.json"
    assert cli_main(["fit", ws["records"], "-o", str(out), "--model", "no-such-model"]) == 1
    assert "no-such-model" in _domain_error(capsys)
    assert not out.exists()


def test_policy_tune_zero_groups_exits_1_with_message(ws, tmp_path: Path, capsys) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws["records"], "-o", str(out), "--model", "no-such-model"]) == 1
    assert "no labeled groups" in _domain_error(capsys)
    assert not out.exists()


def test_audit_zero_groups_exits_1_no_stdout(ws, capsys) -> None:
    assert cli_main(["audit", ws["records"], "--question", "q-deadbeefdeadbeef"]) == 1
    _domain_error(capsys)


def test_judge_audit_empty_file_exits_1(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.jsonl"
    grades.write_text("")
    assert cli_main(["judge", "audit", str(grades)]) == 1
    assert "no grade rows" in _domain_error(capsys)


def test_judge_null_value_exits_1(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.jsonl"
    grades.write_text('{"judge": null, "human": 2}\n')
    assert cli_main(["judge", "audit", str(grades)]) == 1
    assert "must be non-negative integers" in _domain_error(capsys)


def test_web_missing_workspace_exits_1(tmp_path: Path, capsys) -> None:
    assert cli_main(["web", "--workspace", str(tmp_path / "nope")]) == 1
    assert "workspace" in _domain_error(capsys)


@pytest.mark.parametrize("policies", [[{}], [None]])
def test_policy_file_with_malformed_entry_exits_1(ws, tmp_path: Path, capsys, policies) -> None:
    bad = tmp_path / "policy.json"
    bad.write_text(json.dumps({"schema": "s1-policy/1", "policies": policies}))
    assert cli_main(["gate", str(bad), "--question", "q", "--model", "m", "--p", "0.5"]) == 1
    assert "policies.0" in _domain_error(capsys)


def test_fits_container_not_list_exits_1(ws, tmp_path: Path, capsys) -> None:
    bad = tmp_path / "fits.json"
    bad.write_text(json.dumps({"schema": "jev-calib-fit/2", "fits": None}))
    assert cli_main(["audit", ws["records"], "--fit", str(bad)]) == 1
    assert "fits: Input should be a valid list" in _domain_error(capsys)


def test_version_prints_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    out, err = capsys.readouterr()
    assert out.strip() == f"s1 {__version__}"
    assert err == ""


def test_usage_error_exits_1(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["fit"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("usage: ")
    assert "\ns1: the following arguments are required" in err


def test_help_shows_non_obvious_defaults(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["policy", "tune", "--help"])
    assert exc.value.code == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "(default: 0.35)" in out
    assert "(default: 0.004)" in out
    assert "(default: None)" not in out
    assert "(default: False)" not in out


def test_every_argument_has_help() -> None:
    import argparse

    from s1ops.cli import build_parser

    missing: list[str] = []

    def walk(p: argparse.ArgumentParser, path: str) -> None:
        for a in p._actions:
            if isinstance(a, argparse._SubParsersAction):
                for name, sp in a.choices.items():
                    walk(sp, f"{path} {name}".strip())
            elif not a.help:
                missing.append(f"{path}: {a.dest}")

    walk(build_parser(), "s1")
    assert missing == []


@pytest.mark.parametrize(
    "row, loc",
    [
        ({}, "policies.0.question_id"),
        (None, "policies.0"),
        ({"threshold": "0.8"}, "policies.0.threshold"),
        ({"mode": "bogus"}, "policies.0.mode"),
        ({"threshold": 1.5}, "policies.0.threshold"),
    ],
)
def test_load_policies_rejects_malformed_entries(tmp_path: Path, row, loc) -> None:
    base = {"question_id": "q", "model_served": "m", "primitive": "noul", "mode": "error_budget", "split": "eval", "n": 3}
    entry = None if row is None else ({} if row == {} else {**base, **row})
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema": "s1-policy/1", "policies": [entry]}))
    with pytest.raises(ValueError, match=loc):
        load_policies(path)


@pytest.mark.parametrize("fits", [None, [None]])
def test_load_fits_rejects_bad_container(tmp_path: Path, fits) -> None:
    path = tmp_path / "fits.json"
    path.write_text(json.dumps({"schema": "jev-calib-fit/2", "fits": fits}))
    with pytest.raises(FitError):
        load_fits(path)


def test_judge_negative_grade_exits_1(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.jsonl"
    grades.write_text('{"judge": -1, "human": 2}\n')
    assert cli_main(["judge", "audit", str(grades)]) == 1
    assert "row 1: 'judge' and 'human' must be non-negative integers" in _domain_error(capsys)


@pytest.mark.parametrize(
    "fields",
    [
        {"threshold": 1},
        {"threshold": 0},
        {"threshold": 0.8},
        {"mode": "escalation_band", "band_lo": 0, "band_hi": 1},
        {"error_budget": 0},
    ],
)
def test_policy_fingerprint_survives_save_load(tmp_path: Path, fields) -> None:
    from s1ops.contracts.schemas import PolicyEntry, save_policies

    kw = {"question_id": "q", "model_served": "m", "primitive": "noul", "mode": "error_budget", "split": "eval", "n": 3}
    entry = PolicyEntry(**{**kw, **fields})
    path = tmp_path / "policy.json"
    save_policies([entry], path)
    (loaded,) = load_policies(path)
    assert loaded.policy_sha256 == entry.policy_sha256
    assert loaded == entry


def test_lint_check_missing_file_still_reports_json(tmp_path: Path, capsys) -> None:
    assert cli_main(["lint", "check", str(tmp_path / "missing.json"), "--format", "json"]) == 1
    out, _ = capsys.readouterr()
    report = json.loads(out)
    assert report["schema"] == "s1-lint-report/1"
    assert [f["rule"] for f in report["findings"]] == ["JL000"]


def test_judge_constant_grades_emit_valid_json(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.jsonl"
    grades.write_text('{"judge": 1, "human": 1}\n{"judge": 1, "human": 1}\n')
    assert cli_main(["judge", "audit", str(grades)]) == 0
    out = capsys.readouterr().out
    assert "NaN" not in out
    assert json.loads(out)["overall"]["weighted_kappa"] is None


def test_judge_all_zero_grades_exit_1(tmp_path: Path, capsys) -> None:
    grades = tmp_path / "grades.jsonl"
    grades.write_text('{"judge": 0, "human": 0}\n')
    assert cli_main(["judge", "audit", str(grades)]) == 1
    assert "at least two levels" in _domain_error(capsys)


def test_s1_debug_reraises_library_errors_only(ws, tmp_path: Path, monkeypatch, capsys) -> None:
    from s1ops.contracts.records import RecordError

    monkeypatch.setenv("S1_DEBUG", "1")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    with pytest.raises(RecordError):
        cli_main(["audit", str(bad)])
    with pytest.raises(Exception, match="model_served"):
        cli_main(["gate", ws["policy"], "--question", _policy_qid(ws), "--model", "kev-1.0", "--p", "0.9"])
    assert cli_main(["audit", str(tmp_path / "nope.jsonl")]) == 1
    assert "not found" in _domain_error(capsys)


def test_fit_min_n_warns_instead_of_skipping(ws, tmp_path: Path, capsys) -> None:
    out = tmp_path / "fits.json"
    assert cli_main(["fit", ws["records_train"], "-o", str(out), "--min-n", "10000"]) == 0
    (fit,) = load_fits(out)
    assert any("warning threshold 10000" in w for w in fit.warnings)


def test_integer_beyond_float_range_is_a_record_error(tmp_path: Path, capsys) -> None:
    row = '{"schema": "s1-record/1", "item_id": "i", "question_id": "q-1", "primitive": "noul", '
    row += '"model_requested": "m", "model_served": "m", "p": 1' + "0" * 400 + "}"
    path = tmp_path / "records.jsonl"
    path.write_text(row + "\n", encoding="utf-8")
    assert cli_main(["audit", str(path)]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("s1: line 1: p must be finite")


@pytest.mark.parametrize(
    "row, message",
    [
        ({"item_id": "i1"}, "row 2 is missing 'item_id' or 'state'"),
        ({"state": "x"}, "row 2 is missing 'item_id' or 'state'"),
        ({"item_id": "i1", "state": None}, "row 2: 'state' must be a string, got NoneType"),
    ],
)
def test_probe_bad_states_row_is_one_line_before_ledger_or_output(tmp_path: Path, capsys, row, message) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"questions": {"q1": {"type": "noul", "instructions": "ok?"}}}), encoding="utf-8")
    states = tmp_path / "states.jsonl"
    states.write_text(json.dumps({"item_id": "i0", "state": "fine"}) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    out.write_text("previous\n", encoding="utf-8")
    ledger = tmp_path / "journal.jsonl"
    rc = cli_main(["probe", str(questions), "--states", str(states), "-o", str(out), "--ledger", str(ledger)])
    assert rc == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [f"s1: {states}: {message}"]
    assert out.read_text() == "previous\n" and not ledger.exists()


# One line, whatever a message interpolates (#32): `_fail` and the parser escape every
# non-printable character. Record ids may carry any of them (`parse_record` accepts them).
@pytest.mark.parametrize(
    ("message", "line"),
    [
        ("a\nb", "a\\nb"),
        ("a\r\nb", "a\\r\\nb"),
        ("a\x0bb\x0cc\x1cd\x85e\u2028f\u2029g", "a\\x0bb\\x0cc\\x1cd\\x85e\\u2028f\\u2029g"),
        ("q\x1b[31mRED\x1b[0m", "q\\x1b[31mRED\\x1b[0m"),
        ("q\u202eevil", "q\\u202eevil"),
        ("tab\there", "tab\\there"),
        ("nul\x00", "nul\\x00"),
        ("bom\ufeff", "bom\\ufeff"),
        ("tag\U000e0001", "tag\\U000e0001"),
        ("lone\ud800", "lone\\ud800"),
        ("unassigned\u0378", "unassigned\\u0378"),
        ("질문 é 漢字 🙂", "질문 é 漢字 🙂"),
        ("", ""),
    ],
    ids=["newline", "crlf", "separators", "ansi", "bidi", "tab", "nul", "bom", "non-bmp-format", "surrogate", "unassigned", "printable-unicode", "empty"],
)
def test_one_line_escapes_only_non_printable_characters(message: str, line: str) -> None:
    from s1ops.cli import _one_line

    assert _one_line(message) == line
    assert line.isprintable()


def test_one_line_is_one_printable_line_for_every_code_point() -> None:
    import codecs

    from s1ops.cli import _one_line

    for cp in range(0x110000):
        c = chr(cp)
        out = _one_line(f"a{c}b")
        assert out.isprintable() and len(f"s1: {out}".splitlines()) == 1, hex(cp)
        assert out[0] == "a" and out[-1] == "b", hex(cp)
        middle = out[1:-1]
        if c.isprintable():
            assert middle == c, hex(cp)
        else:
            # An ASCII escape that decodes back to the character, not some other printable stand-in.
            assert middle.isascii() and middle.startswith("\\"), hex(cp)
            assert codecs.decode(middle, "unicode_escape") == c, hex(cp)


def test_backslashes_are_kept_so_a_literal_backslash_n_looks_escaped(tmp_path: Path, monkeypatch, capsys) -> None:
    from s1ops.cli import CliError

    def raise_(args):
        raise CliError("path C:\\temp\\new and a\nbreak")

    monkeypatch.setattr("s1ops.cli._cmd_audit", raise_)
    records = tmp_path / "r.jsonl"
    records.write_text("")
    assert cli_main(["audit", str(records)]) == 1
    assert capsys.readouterr().err == "s1: path C:\\temp\\new and a\\nbreak\n"


def _branch_errors():
    from s1ops.cli import CliError
    from s1ops.contracts.records import RecordError
    from s1ops.ledger.journal import BudgetExceeded
    from s1ops.runtime.gate import ModelServedMismatchError, PolicyIntegrityError

    return [
        (CliError("a\nb"), 1, "a\\nb"),
        (CliError("a\nb", 2), 2, "a\\nb"),
        (ModelServedMismatchError("a\nb"), 3, "a\\nb"),
        (OSError(2, "No such\nfile", "/tmp/x\ny"), 1, "/tmp/x\\ny: No such\\nfile"),
        (RecordError(4, "a\nb"), 1, "line 4: a\\nb"),
        (FitError("a\nb"), 1, "a\\nb"),
        (PolicyIntegrityError("a\nb"), 1, "a\\nb"),
        (BudgetExceeded("a\nb"), 1, "a\\nb"),
    ]


@pytest.mark.parametrize("index", range(8), ids=["cli", "cli-code-2", "model-mismatch", "os", "record", "fit", "integrity", "budget"])
def test_every_mapped_error_is_one_escaped_line(tmp_path: Path, monkeypatch, capsys, index: int) -> None:
    error, code, line = _branch_errors()[index]

    def raise_(args):
        raise error

    monkeypatch.setattr("s1ops.cli._cmd_audit", raise_)
    records = tmp_path / "r.jsonl"
    records.write_text("")
    assert cli_main(["audit", str(records)]) == code
    out, err = capsys.readouterr()
    assert (out, err) == ("", f"s1: {line}\n")


def test_s1_debug_reraises_library_errors_with_their_own_text(tmp_path: Path, monkeypatch, capsys) -> None:
    from s1ops.cli import CliError
    from s1ops.contracts.records import RecordError

    records = tmp_path / "r.jsonl"
    records.write_text("")
    monkeypatch.setenv("S1_DEBUG", "1")
    raised: list[Exception] = [RecordError(1, "a\nb"), CliError("a\nb")]

    def raise_(args):
        raise raised.pop(0)

    monkeypatch.setattr("s1ops.cli._cmd_audit", raise_)
    with pytest.raises(RecordError) as exc:
        cli_main(["audit", str(records)])
    assert str(exc.value) == "line 1: a\nb"
    assert cli_main(["audit", str(records)]) == 1
    assert capsys.readouterr().err == "s1: a\\nb\n"


def test_a_usage_error_message_is_one_escaped_line(tmp_path: Path, capsys) -> None:
    path = tmp_path / "q.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        cli_main(["lint", "check", str(path), "x\ny\x1b[2J"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("usage: ")
    assert err.splitlines()[-1] == "s1: unrecognized arguments: x\\ny\\x1b[2J"
    assert err.endswith("s1: unrecognized arguments: x\\ny\\x1b[2J\n")


QID = "q\nextra"
FIT_A, FIT_B = "a" * 16, "b" * 16


def _record(i: int, **extra) -> dict:
    return {
        "schema": "s1-record/1", "item_id": f"i{i}", "question_id": QID, "primitive": "noul",
        "model_requested": "m", "model_served": "m", "p": 0.7, **extra,
    }


def _write_rows(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _policy_file(path: Path, **fields) -> str:
    row = {"question_id": QID, "model_served": "m", "primitive": "noul", "mode": "error_budget",
           "split": "eval", "n": 1, "threshold": 0.5, "calibrated": True, **fields}
    path.write_text(json.dumps({"schema": "s1-policy/1", "policies": [row]}))
    return str(path)


@pytest.mark.parametrize("path", ["missing-calib", "other-fit", "tune-two-fits"])
def test_the_issue_paths_print_one_line(tmp_path: Path, capsys, path: str) -> None:
    out = str(tmp_path / "o")
    if path == "missing-calib":
        args = ["export", "phoenix", _write_rows(tmp_path / "r.jsonl", [_record(1)]), "-o", out,
                "--policy", _policy_file(tmp_path / "p.json")]
        expected = ("s1: line 1: policy for q\\nextra (m) was tuned on calibrated probabilities and this record has "
                    "none; run `s1 apply` first (`--pass-through-unfitted` keeps records of models it has no fit for)\n")
    elif path == "other-fit":
        args = ["export", "phoenix", _write_rows(tmp_path / "r.jsonl", [_record(1, calib={"p": 0.6, "fit_sha256": FIT_B})]),
                "-o", out, "--policy", _policy_file(tmp_path / "p.json", fit_sha256=FIT_A)]
        expected = (f"s1: line 1: policy for q\\nextra (m) was tuned through fit '{FIT_A}', but this record was calibrated "
                    f"through '{FIT_B}'; re-run `s1 apply` with the fits.json the policy was tuned through\n")
    else:
        rows = [_record(i, label=i % 2 == 0, split="eval", calib={"p": 0.6, "fit_sha256": FIT_A if i < 5 else FIT_B}) for i in range(10)]
        args = ["policy", "tune", _write_rows(tmp_path / "r.jsonl", rows), "-o", out, "--error-budget", "0.1", "--use-calib"]
        expected = f"s1: q\\nextra (m): records were calibrated through different fits ('{FIT_A}', '{FIT_B}'); apply one fit file\n"
    assert cli_main(args) == 1
    assert capsys.readouterr() == ("", expected)
