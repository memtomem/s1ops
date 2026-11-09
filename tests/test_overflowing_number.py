"""A float literal beyond the float range is refused anywhere in a record row, when it is read.

`1e400` is a float literal, so `json.loads` turns it into `inf` without calling `parse_constant`. In a
field nothing validates it then reached `s1 apply`, both exporters and `s1 drift`, which wrote it as
`Infinity`: not valid JSON, and refused by the reader on the next command.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, parse_record, read_records, read_row
from s1ops.demo import generate_workspace
from s1ops.web.app import create_app

LONG = "1" + "0" * 400 + ".0"
RECORD_FILES = (
    "records_train", "records", "records_kev", "records_canary", "probe_records",
    "records_categorical_train", "records_categorical",
)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _with_field(source: str | Path, path: Path, field: str, literal: str) -> str:
    """`source` with `"field": literal` added to its first row, as text, since `json.dumps` cannot write it."""
    lines = Path(source).read_text().splitlines()
    lines[0] = f'{lines[0].rstrip()[:-1]}, "{field}": {literal}}}'
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize(
    "literal, shown",
    [
        ("1e400", "'1e400'"),
        ("-1e400", "'-1e400'"),
        ("1E+400", "'1E+400'"),
        ("1.8e308", "'1.8e308'"),
        pytest.param(LONG, repr(LONG[:40]) + "... (403 characters)", id="401-digits"),
    ],
)
def test_apply_refuses_it_and_writes_nothing(ws, tmp_path: Path, capsys, literal: str, shown: str) -> None:
    bad = _with_field(ws["records_canary"], tmp_path / "over.jsonl", "extra", literal)
    out = tmp_path / "out.jsonl"
    assert cli_main(["apply", bad, "--fit", ws["fits"], "-o", str(out), "--pass-through-unfitted"]) == 1
    assert capsys.readouterr().err == f"s1: {bad}: line 1: number {shown} is beyond the float range\n"
    assert not out.exists()


def test_it_is_refused_inside_a_nested_value(ws, tmp_path: Path, capsys) -> None:
    bad = _with_field(ws["records_canary"], tmp_path / "over.jsonl", "extra", '{"a": [0.5, 1e400]}')
    assert cli_main(["audit", bad]) == 1
    assert capsys.readouterr().err == f"s1: {bad}: line 1: number '1e400' is beyond the float range\n"


@pytest.mark.parametrize("field", ["latency_ms", "timestamp", "extra"])
@pytest.mark.parametrize(
    "argv",
    [
        ["audit", "{bad}"],
        ["fit", "{bad}", "-o", "{out}"],
        ["policy", "tune", "{bad}", "-o", "{out}"],
        ["slice", "answerable", "{bad}"],
        ["compare", "{bad}", "{canary}"],
        ["drift", "{train}", "{bad}"],
        ["drift", "{bad}", "{canary}"],
        ["drift", "{train}", "{bad}", "--fit", "{fits}", "--calibrated", "--allow-in-sample"],
        ["export", "phoenix", "{bad}", "-o", "{out}"],
        ["export", "langfuse", "{bad}", "-o", "{out}"],
        ["export", "phoenix", "{bad}", "-o", "{out}", "--fit", "{fits}"],
        ["trainset", "export", "{bad}", "-o", "{out}", "--split", "train", "--checkpoint", "c"],
    ],
    ids=lambda argv: " ".join(a for a in argv if not a.startswith("{")),
)
def test_every_command_refuses_it_on_the_input(ws, tmp_path: Path, capsys, argv: list[str], field: str) -> None:
    bad = _with_field(ws["records_canary"], tmp_path / "over.jsonl", field, "1e400")
    out = tmp_path / "out"
    names = {"bad": bad, "out": str(out), "canary": ws["records_canary"], "train": ws["records_train"], "fits": ws["fits"]}
    assert cli_main([a.format(**names) for a in argv]) == 1
    assert capsys.readouterr().err == f"s1: {bad}: line 1: number '1e400' is beyond the float range\n"
    assert not out.exists()


@pytest.mark.parametrize(
    "route, name",
    [("/api/drift", "records_canary.jsonl"), ("/api/audit?n_boot=20", "records.jsonl")],
)
def test_the_web_answers_400_naming_the_file_once(tmp_path: Path, route: str, name: str) -> None:
    ws = generate_workspace(tmp_path / "ws")
    path = Path(ws["records"]).parent / name
    _with_field(path, path, "latency_ms", "1e400")
    response = TestClient(create_app(workspace_dir=path.parent)).get(route)
    assert response.status_code == 400
    assert response.json()["detail"] == f"{path}: line 1: number '1e400' is beyond the float range"


@pytest.mark.parametrize(
    "literal, value",
    [
        ("1e-400", 0.0),
        ("-1e-400", -0.0),
        ("1e308", 1e308),
        ("1.7976931348623157e308", 1.7976931348623157e308),
        ('"1e400"', "1e400"),
        pytest.param("1" + "0" * 400, 10**400, id="401-digit-integer"),
    ],
)
def test_what_is_not_refused_passes_through_apply(ws, tmp_path: Path, literal: str, value: object) -> None:
    """The boundary of the rule: underflow is finite, a string is not a number, and an integer in a field
    nothing reads stays exact and writes as valid JSON."""
    source = _with_field(ws["records_canary"], tmp_path / "in.jsonl", "extra", literal)
    out = tmp_path / "out.jsonl"
    assert cli_main(["apply", source, "--fit", ws["fits"], "-o", str(out), "--pass-through-unfitted"]) == 0
    written = read_records(out)[0].raw["extra"]
    assert type(written) is type(value) and repr(written) == repr(value)


def test_a_long_literal_is_cut_in_the_message() -> None:
    with pytest.raises(RecordError) as caught:
        read_row('{"extra": 1' + "0" * 500 + ".0}", 7, None)
    assert caught.value.reason == "number '1" + "0" * 39 + "'... (503 characters) is beyond the float range"


@pytest.mark.parametrize(
    "text, reason",
    [
        ('{"p": 1e400}', "number '1e400' is beyond the float range"),
        ('{"extra": 1e400, ', "number '1e400' is beyond the float range"),
        ('{"extra": 1e400, "extra": 0}', "number '1e400' is beyond the float range"),
        ('{"extra": nope, "later": 1e400}', "not valid JSON: Expecting value: line 1 column 11 (char 10)"),
        ('{"extra": Infinity}', "not valid JSON: non-finite number Infinity is not valid JSON"),
    ],
)
def test_the_first_problem_the_decoder_meets_is_the_message(text: str, reason: str) -> None:
    with pytest.raises(RecordError) as caught:
        read_row(text, 7, "f.jsonl")
    assert caught.value.reason == reason
    assert str(caught.value) == f"f.jsonl: line 7: {reason}"


@pytest.mark.parametrize("name", RECORD_FILES)
def test_an_accepted_row_reads_as_it_did_without_the_hook(ws, name: str) -> None:
    """`raw` is outside `Record` equality, so it is compared on its own; `json.dumps` writes `repr`, so a
    changed last bit shows."""
    lines = [line for line in Path(ws[name]).read_text().splitlines() if line.strip()]
    assert lines
    for n, line in enumerate(lines, start=1):
        record = read_row(line, n, None)
        row = json.loads(line)
        assert record == parse_record(row, n)
        assert json.dumps(record.raw) == json.dumps(row)


@pytest.mark.parametrize("literal", ["-0.0", "1e-400", "-1e-400", "5e-324", "0.1"])
def test_a_float_in_a_field_nothing_reads_keeps_its_bits(ws, tmp_path: Path, literal: str) -> None:
    path = _with_field(ws["records_canary"], tmp_path / "in.jsonl", "extra", literal)
    line = Path(path).read_text().splitlines()[0]
    assert json.dumps(read_row(line, 1, None).raw) == json.dumps(json.loads(line))
