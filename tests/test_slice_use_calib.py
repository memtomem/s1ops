"""`s1 slice --use-calib` and `/api/slices?use_calib=true` predict from the calibrated distribution
and refuse what `s1 policy tune --use-calib` refuses in the groups they report; by default they
predict from the raw one, and a record's `calib` does not change its prediction (#130)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from s1ops.web.app import create_app
from tests.test_curve_use_calib import CASES, FIT_A, GOOD, OTHER, QID, _row, _tune_error
from tests.test_source_digest import _through_fifo


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _slice(capsys, *argv: str) -> tuple[int, str, str]:
    rc = cli_main(["slice", *argv])
    out, err = capsys.readouterr()
    return rc, out, err


def _client(records: Path) -> TestClient:
    return TestClient(create_app(records.parent, records_path=records))


def _accuracy(out: str) -> list[float]:
    return [g["slices"][0]["accuracy"] for g in json.loads(out)["groups"]]


@pytest.mark.parametrize("cmd", ["answerable", "mine"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_use_calib_refuses_what_tune_refuses(tmp_path, capsys, case, cmd) -> None:
    records = _write(tmp_path / "records.jsonl", CASES[case])
    message = _tune_error(capsys, records, tmp_path).removeprefix("s1: ")
    features = _write(tmp_path / "f.jsonl", [{"dataset": "d", "item_id": "i0", "features": {"x": True}}])
    flags = ["--features", str(features)] if cmd == "mine" else []
    # The error names the records file, which tune's line does not.
    assert _slice(capsys, cmd, str(records), *flags, "--use-calib") == (1, "", f"s1: {records}: {message}")
    rc, out, err = _slice(capsys, cmd, str(records), *flags)
    assert (rc, err, json.loads(out)["distribution"]) == (0, "", "raw")

    client = _client(records)
    r = client.get("/api/slices", params={"use_calib": True})
    assert (r.status_code, r.json()["detail"]) == (400, f"records.jsonl: {message.rstrip()}")
    r = client.get("/api/slices")
    assert (r.status_code, r.json()["distribution"]) == (200, "raw")


@pytest.mark.parametrize("flag", ["--question", "--model"])
def test_only_the_reported_groups_are_checked(tmp_path, capsys, flag) -> None:
    other_model = [{**r, "model_requested": "m2", "model_served": "m2"} for r in GOOD] if flag == "--model" else GOOD
    records = _write(tmp_path / "records.jsonl", CASES["two-fits"] + other_model)
    value = OTHER if flag == "--question" else "m2"
    rc, out, err = _slice(capsys, "answerable", str(records), "--use-calib", flag, value)
    assert (rc, err, _accuracy(out)) == (0, "", [1.0])
    assert _slice(capsys, "answerable", str(records), "--use-calib")[0] == 1

    client = _client(records)
    param = {"question_id": OTHER} if flag == "--question" else {"model_served": "m2"}
    r = client.get("/api/slices", params={**param, "use_calib": True})
    assert r.status_code == 200 and len(r.json()["groups"]) == 1
    assert client.get("/api/slices", params={"use_calib": True}).status_code == 400


def test_unanswered_and_unlabeled_records_are_not_checked(tmp_path, capsys) -> None:
    calibrated = [_row(n, calib={"p": 0.1, "fit_sha256": FIT_A}) for n in range(5)]
    unlabeled = {k: v for k, v in _row(5, calib=None).items() if k != "label"}
    unanswered = {**_row(6, calib=None), "model_served": None}
    unanswered.pop("p")
    records = _write(tmp_path / "records.jsonl", [*calibrated, unlabeled, unanswered])
    rc, out, err = _slice(capsys, "answerable", str(records), "--use-calib")
    # calib.p 0.1 answers false on five label-true records.
    assert (rc, err, _accuracy(out)) == (0, "", [0.0])
    rc, out, err = _slice(capsys, "answerable", str(records))
    assert (rc, err, _accuracy(out)) == (0, "", [1.0])


@pytest.mark.parametrize("hashes", [[None] * 6, [None, None, None, "missing", "missing", "missing"]])
def test_records_without_a_hash_count_as_one_fit(tmp_path, capsys, hashes) -> None:
    rows = []
    for n, h in enumerate(hashes):
        calib = {"p": 0.1} if h == "missing" else {"p": 0.1, "fit_sha256": h}
        rows.append(_row(n, calib=calib))
    records = _write(tmp_path / "records.jsonl", rows)
    rc, out, err = _slice(capsys, "answerable", str(records), "--use-calib")
    assert (rc, err, _accuracy(out)) == (0, "", [0.0])


def test_a_null_hash_mixed_with_a_named_one_is_refused(tmp_path, capsys) -> None:
    rows = [_row(n, calib={"p": 0.1, "fit_sha256": FIT_A if n < 3 else None}) for n in range(6)]
    records = _write(tmp_path / "records.jsonl", rows)
    rc, out, err = _slice(capsys, "answerable", str(records), "--use-calib")
    assert (rc, out) == (1, "") and err.startswith(f"s1: {records}: {QID} (m): records were calibrated through different fits")


def test_mine_finds_the_slice_the_calibration_breaks(tmp_path, capsys) -> None:
    """Six tagged items answer true raw (`p` 0.95, right) and false calibrated (`calib.p` 0.1,
    wrong); ten untagged ones are right in both. Only the calibrated report has errors in `hard`."""
    rows = [_row(n, calib={"p": 0.1 if n < 6 else 0.9, "fit_sha256": FIT_A}) for n in range(16)]
    records = _write(tmp_path / "records.jsonl", rows)
    features = _write(
        tmp_path / "features.jsonl",
        [{"dataset": "d", "item_id": f"i{n}", "features": {"hard": n < 6}} for n in range(16)],
    )
    reports = {}
    for flags, distribution in (((), "raw"), (("--use-calib",), "calibrated")):
        rc, out, err = _slice(capsys, "mine", str(records), "--features", str(features), *flags)
        report = json.loads(out)
        assert (rc, err, report["distribution"]) == (0, "", distribution)
        (group,) = report["groups"]
        reports[distribution] = {e["feature"]: e for e in group["error_slices"]}
    assert (reports["raw"]["hard"]["slice_error_rate"], reports["calibrated"]["hard"]["slice_error_rate"]) == (0.0, 1.0)
    assert reports["calibrated"]["hard"]["significant"] and not reports["raw"]["hard"]["significant"]

    client = TestClient(create_app(records.parent, records_path=records, features_path=features))
    for params, rate in (({}, 0.0), ({"use_calib": True}, 1.0)):
        (group,) = client.get("/api/slices", params=params).json()["groups"]
        assert {e["feature"]: e for e in group["error_slices"]}["hard"]["slice_error_rate"] == rate


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def test_a_choice_record_without_calib_is_refused(demo, tmp_path, capsys) -> None:
    rows = [json.loads(line) for line in Path(demo["records_categorical"]).read_text().splitlines() if line.strip()]
    choice = next(i for i, r in enumerate(rows) if r["primitive"] == "choice")
    del rows[choice]["calib"]
    records = _write(tmp_path / "records.jsonl", rows)
    rc, out, err = _slice(capsys, "answerable", str(records), "--use-calib")
    assert (rc, out) == (1, "") and err.startswith(f"s1: {records}: ") and "has no calibrated probabilities" in err
    rc, out, err = _slice(capsys, "answerable", str(records))
    assert (rc, err) == (0, "") and len(json.loads(out)["groups"]) == 2


def test_a_malformed_calib_is_refused_in_both_modes(tmp_path, capsys) -> None:
    rows = [_row(n, calib={"p": "bad" if n == 2 else 0.1, "fit_sha256": FIT_A}) for n in range(5)]
    records = _write(tmp_path / "records.jsonl", rows)
    for flags in ((), ("--use-calib",)):
        rc, out, err = _slice(capsys, "answerable", str(records), *flags)
        assert (rc, out) == (1, "") and err.startswith(f"s1: {records}") and "line 3" in err
    for params in ({}, {"use_calib": True}):
        assert _client(records).get("/api/slices", params=params).status_code == 400


@pytest.mark.parametrize("flags", [(), ("--use-calib",)])
def test_a_fifo_is_read_once(demo, tmp_path, capsys, flags) -> None:
    expected = _slice(capsys, "answerable", demo["records"], *flags)
    assert expected[0] == 0
    got = _through_fifo(tmp_path, demo["records"], lambda fifo: _slice(capsys, "answerable", str(fifo), *flags))
    assert got == expected
