"""Both exporters refuse a malformed stored `calib.probabilities` with one line naming the file (#66).

`_predict_info` built `dict(calib["probabilities"])` before `reparse` validated it, so on 0.5.0
`null`, `[1, 2]` and `3` crashed `s1 export langfuse|phoenix` with a `TypeError` traceback, and
`"x"` and `[[]]` printed a pathless line. Values `dict()` accepts (`{}`, `[]`, `""`,
`[["a", 1]]`) already reached the parser and named the file; they stay here as coverage. The
parser now runs first, as drift's does, and a valid record's stored `p_calib` is still exported as
stored, not renormalized.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.bridges.langfuse import build_langfuse_events
from s1ops.bridges.openinference import build_openinference_spans
from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, read_records
from s1ops.contracts.schemas import PolicyEntry, save_policies
from s1ops.demo import generate_workspace


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> dict[str, dict]:
    """One calibrated choice and one calibrated score row from the demo."""
    ws = generate_workspace(tmp_path_factory.mktemp("demo") / "ws")
    found: dict[str, dict] = {}
    for line in Path(ws["records_categorical"]).read_text().splitlines():
        row = json.loads(line)
        if "calib" in row:
            found.setdefault(row["primitive"], row)
    assert set(found) == {"choice", "score"}
    return found


def _with_calib(row: dict, value: object) -> dict:
    row = json.loads(json.dumps(row))
    row["calib"]["probabilities"] = value
    return row


def _write(path: Path, row: dict) -> Path:
    path.write_text(json.dumps(row) + "\n")
    return path


def _policy(row: dict, *, calibrated: bool) -> PolicyEntry:
    return PolicyEntry(
        question_id=row["question_id"], model_served=row["model_served"], primitive=row["primitive"],
        mode="error_budget", split="eval", n=1, threshold=0.0, calibrated=calibrated,
    )


BAD = {
    "null": None, "list": [1, 2], "number": 3, "string": "x", "empty-pair": [[]],
    "empty": {}, "empty-list": [], "empty-string": "", "pairs": [["a", 1]],
}


@pytest.mark.parametrize("exporter", ["langfuse", "phoenix"])
@pytest.mark.parametrize("primitive", ["choice", "score"])
@pytest.mark.parametrize("case", list(BAD))
def test_cli_names_the_file(rows, tmp_path: Path, capsys, exporter: str, primitive: str, case: str) -> None:
    path = _write(tmp_path / "r.jsonl", _with_calib(rows[primitive], BAD[case]))
    assert cli_main(["export", exporter, str(path), "-o", str(tmp_path / "o")]) == 1
    expected = f"s1: {path}: line 1: probabilities must be a non-empty object, got {BAD[case]!r}\n"
    assert capsys.readouterr() == ("", expected)


@pytest.mark.parametrize("exporter", ["langfuse", "phoenix"])
@pytest.mark.parametrize("calibrated", [False, True], ids=["raw-policy", "calibrated-policy"])
def test_cli_names_the_file_under_a_policy(
    rows, tmp_path: Path, capsys, exporter: str, calibrated: bool
) -> None:
    row = rows["choice"]
    path = _write(tmp_path / "r.jsonl", _with_calib(row, None))
    save_policies([_policy(row, calibrated=calibrated)], tmp_path / "p.json")
    args = ["export", exporter, str(path), "-o", str(tmp_path / "o"), "--policy", str(tmp_path / "p.json")]
    assert cli_main(args) == 1
    assert capsys.readouterr() == ("", f"s1: {path}: line 1: probabilities must be a non-empty object, got None\n")


@pytest.mark.parametrize("build", [build_langfuse_events, build_openinference_spans], ids=["langfuse", "phoenix"])
@pytest.mark.parametrize("case", ["null", "list", "number"])
def test_sdk_raises_record_error_with_path(rows, tmp_path: Path, build, case: str) -> None:
    path = _write(tmp_path / "r.jsonl", _with_calib(rows["choice"], BAD[case]))
    with pytest.raises(RecordError) as info:
        build(read_records(path))
    assert info.value.path == path
    assert str(info.value) == f"{path}: line 1: probabilities must be a non-empty object, got {BAD[case]!r}"


def _p_calib(build, records, policies) -> object:
    if build is build_langfuse_events:
        (obs,) = build(records, policies=policies)["observations"]
        return obs["output"]["predicted"], obs["output"]["p_calib"]
    (span,) = build(records, policies=policies)
    return span["attributes"]["s1.predicted"], span["attributes"]["s1.p_calib"]


@pytest.mark.parametrize("build", [build_langfuse_events, build_openinference_spans], ids=["langfuse", "phoenix"])
@pytest.mark.parametrize("primitive", ["choice", "score"])
@pytest.mark.parametrize("policy", [None, "raw", "calibrated"])
def test_stored_p_calib_is_exported_as_stored(rows, tmp_path: Path, build, primitive: str, policy) -> None:
    row = rows[primitive]
    stored = {k: v * 0.99 for k, v in row["calib"]["probabilities"].items()}
    path = _write(tmp_path / "r.jsonl", _with_calib(row, stored))
    policies = None if policy is None else [_policy(row, calibrated=policy == "calibrated")]
    predicted, p_calib = _p_calib(build, read_records(path), policies)
    assert p_calib == round(stored[predicted], 6)
    assert p_calib != round(stored[predicted] / 0.99, 6)
