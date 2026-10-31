"""A record that cannot be routed the way its policy was tuned names its records file (#63).

`RoutingInputError` takes `Record.path` (#62), so `s1 export --policy` and SDK callers of
`route_record` / `policy_distribution` say which file, as `RecordError` does. `s1 drift` keeps
naming the side, which maps to one of its two arguments; the path stays on the error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import parse_record, read_records
from s1ops.contracts.schemas import PolicyEntry
from s1ops.demo import generate_workspace
from s1ops.monitor.drift import audit_drift
from s1ops.runtime.routing import RoutingInputError, policy_distribution, route_record

FIT_A = "a" * 64
FIT_B = "b" * 64
RE_APPLY = "re-run `s1 apply` with the fits.json the policy was tuned through"


@pytest.fixture(scope="module")
def noul(tmp_path_factory) -> dict:
    ws = generate_workspace(tmp_path_factory.mktemp("demo") / "ws")
    for line in Path(ws["records_train"]).read_text().splitlines():
        row = json.loads(line)
        if row["primitive"] == "noul" and row["model_served"] == row["model_requested"]:
            return {k: v for k, v in row.items() if k != "calib"}
    raise AssertionError("no noul row in the demo")


def _policy(row: dict) -> PolicyEntry:
    return PolicyEntry(
        question_id=row["question_id"], model_served=row["model_served"], primitive="noul",
        mode="error_budget", split="eval", n=1, threshold=0.5, calibrated=True, fit_sha256=FIT_A,
    )


def _name(row: dict) -> str:
    return f"policy for {row['question_id']} ({row['model_served']})"


# Each case reaches one of the four raise sites in `runtime/routing.py`.
CASES = {
    "no-calib": (None, lambda row: (
        f"{_name(row)} was tuned on calibrated probabilities and this record has none; run `s1 apply` first "
        "(`--pass-through-unfitted` keeps records of models it has no fit for)"
    )),
    "not-a-string": ({"p": 0.6, "fit_sha256": 7}, lambda row: "calib.fit_sha256 must be a string, got 7"),
    "no-fit": ({"p": 0.6}, lambda row: (
        f"{_name(row)} was tuned through fit {FIT_A!r}, and this record's calib names no fit; {RE_APPLY}"
    )),
    "null-fit": ({"p": 0.6, "fit_sha256": None}, lambda row: (
        f"{_name(row)} was tuned through fit {FIT_A!r}, and this record's calib names no fit; {RE_APPLY}"
    )),
    "other-fit": ({"p": 0.6, "fit_sha256": FIT_B}, lambda row: (
        f"{_name(row)} was tuned through fit {FIT_A!r}, but this record was calibrated through {FIT_B!r}; {RE_APPLY}"
    )),
}


def _row(noul: dict, case: str) -> dict:
    calib, _ = CASES[case]
    return noul if calib is None else {**noul, "calib": calib}


def _write(path: Path, *rows: dict) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _policy_file(path: Path, row: dict) -> Path:
    path.write_text(json.dumps({"schema": "s1-policy/1", "policies": [_policy(row).to_dict()]}))
    return path


def _one_line(capsys) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1, err
    return lines[0]


@pytest.mark.parametrize("fmt", ["langfuse", "phoenix"])
@pytest.mark.parametrize("case", list(CASES))
def test_cli_export_names_the_records_file(tmp_path: Path, capsys, noul: dict, fmt: str, case: str) -> None:
    records = _write(tmp_path / "r.jsonl", _row(noul, case))
    policy = _policy_file(tmp_path / "p.json", noul)
    assert cli_main(["export", fmt, str(records), "-o", str(tmp_path / "out"), "--policy", str(policy)]) == 1
    assert _one_line(capsys) == f"s1: {records}: line 1: {CASES[case][1](noul)}"


@pytest.mark.parametrize("case", list(CASES))
def test_sdk_routing_names_the_file_of_a_record_read_from_one(tmp_path: Path, noul: dict, case: str) -> None:
    row, pol, message = _row(noul, case), _policy(noul), CASES[case][1](noul)
    path = _write(tmp_path / "r.jsonl", noul if case == "no-calib" else {**noul, "calib": {"p": 0.6, "fit_sha256": FIT_A}}, row)
    rec = read_records(path)[1]
    for call in (route_record, policy_distribution):
        with pytest.raises(RoutingInputError) as e:
            call(pol, rec)
        assert str(e.value) == f"{path}: line 2: {message}"
        assert (e.value.line, e.value.message, e.value.side, e.value.path) == (2, message, None, path)
    # A record built without a file keeps the line-only message.
    with pytest.raises(RoutingInputError) as e:
        route_record(pol, parse_record(row, 4))
    assert str(e.value) == f"line 4: {message}"
    assert e.value.path is None


def test_drift_names_the_side_and_keeps_the_path(tmp_path: Path, capsys, noul: dict) -> None:
    good = _write(tmp_path / "ref.jsonl", {**noul, "calib": {"p": 0.6, "fit_sha256": FIT_A}})
    bad = _write(tmp_path / "cur.jsonl", _row(noul, "other-fit"))
    policy = _policy_file(tmp_path / "p.json", noul)
    message = CASES["other-fit"][1](noul)
    assert cli_main(["drift", str(good), str(bad), "--policy", str(policy)]) == 1
    assert _one_line(capsys) == f"s1: current records: line 1: {message}"
    with pytest.raises(RoutingInputError) as e:
        audit_drift(read_records(good), read_records(bad), policies=[_policy(noul)])
    assert str(e.value) == f"current records: line 1: {message}"
    assert (e.value.side, e.value.path) == ("current", bad)


def test_the_side_wins_over_the_path() -> None:
    e = RoutingInputError(3, "m", side="reference", path="r.jsonl")
    assert str(e) == "reference records: line 3: m"
    assert str(RoutingInputError(3, "m", path="r.jsonl")) == "r.jsonl: line 3: m"
    assert str(RoutingInputError(3, "m")) == "line 3: m"
    moved = RoutingInputError(3, "m", path="r.jsonl").with_side("current")
    assert (str(moved), moved.path) == ("current records: line 3: m", "r.jsonl")
