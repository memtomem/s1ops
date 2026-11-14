"""`s1 policy curve --use-calib` and the web simulator under `use_calib` refuse what
`s1 policy tune --use-calib` refuses: a tuned record without the calibrated distribution, a
`calib.fit_sha256` that is not a string, and a group calibrated through more than one fit (#124)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.web.app import create_app

QID = "q-0000000000000000"
OTHER = "q-1111111111111111"
FIT_A = "a" * 64
FIT_B = "b" * 64


def _row(n: int, *, calib: dict | None, qid: str = QID) -> dict:
    row = {"schema": "s1-record/1", "dataset": "d", "item_id": f"i{n}", "question_id": qid, "primitive": "noul",
           "model_requested": "m", "model_served": "m", "p": 0.95, "label": True}
    if calib is not None:
        row["calib"] = calib
    return row


# The issue's inputs: 10 noul records, raw p 0.95, all labeled true.
CASES = {
    "missing-calib": [_row(n, calib={"p": 0.8, "fit_sha256": FIT_A} if n < 8 else None) for n in range(10)],
    "two-fits": [_row(n, calib={"p": 0.8, "fit_sha256": FIT_A if n < 5 else FIT_B}) for n in range(10)],
    "non-string-hash": [_row(n, calib={"p": 0.8, "fit_sha256": 7 if n == 3 else FIT_A}) for n in range(10)],
}
GOOD = [_row(n, calib={"p": 0.8, "fit_sha256": FIT_A}, qid=OTHER) for n in range(10)]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(capsys, argv: list[str]) -> tuple[int, str, str]:
    rc = cli_main(argv)
    out, err = capsys.readouterr()
    return rc, out, err


def _tune_error(capsys, records: Path, tmp_path: Path, *extra: str) -> str:
    """Tune's refusal line, which names the records file (#140)."""
    rc, out, err = _run(capsys, ["policy", "tune", str(records), "-o", str(tmp_path / "p.json"), "--use-calib", *extra])
    assert (rc, out) == (1, "") and err.startswith(f"s1: {records}: ") and err.count("\n") == 1
    return err


def _web_detail(err: str, records: Path) -> str:
    """The web's 400 for the refusal in tune's line `err`: the file's name, not its path."""
    return f"{records.name}: " + err.removeprefix(f"s1: {records}: ").rstrip("\n")


@pytest.mark.parametrize("case", sorted(CASES))
def test_curve_refuses_what_tune_refuses(tmp_path, capsys, case) -> None:
    records = _write(tmp_path / "r.jsonl", CASES[case])
    expected = _tune_error(capsys, records, tmp_path)
    assert _run(capsys, ["policy", "curve", str(records), "--use-calib"]) == (1, "", expected)
    # Raw, nothing is calibrated, so nothing is refused.
    rc, out, err = _run(capsys, ["policy", "curve", str(records)])
    assert (rc, err) == (0, "") and json.loads(out)["groups"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_simulator_refuses_what_tune_refuses(tmp_path, capsys, case) -> None:
    records = _write(tmp_path / "records.jsonl", CASES[case])
    expected = _web_detail(_tune_error(capsys, records, tmp_path), records)
    client = TestClient(create_app(records_path=records))
    r = client.get("/api/simulate-policy", params={"use_calib": True})
    assert r.status_code == 400
    assert r.json()["detail"] == expected
    # Unset is auto, which picks calibrated for a group where any record carries calib: still refused.
    assert client.get("/api/simulate-policy").status_code == 400
    assert client.get("/api/simulate-policy", params={"use_calib": False}).status_code == 200


@pytest.mark.parametrize("case", sorted(CASES))
def test_only_the_selected_group_is_checked(tmp_path, capsys, case) -> None:
    """As tune checks only the groups it selects, a bad group left out by a filter is not refused."""
    records = _write(tmp_path / "records.jsonl", CASES[case] + GOOD)
    rc, out, err = _run(capsys, ["policy", "tune", str(records), "-o", str(tmp_path / "p.json"), "--use-calib", "--question", OTHER])
    assert (rc, err) == (0, "")
    rc, out, err = _run(capsys, ["policy", "curve", str(records), "--use-calib", "--question", OTHER])
    assert (rc, err) == (0, "")
    assert [g["question_id"] for g in json.loads(out)["groups"]] == [OTHER]
    # Unfiltered, the bad group is selected too.
    assert _run(capsys, ["policy", "curve", str(records), "--use-calib"])[0] == 1

    client = TestClient(create_app(records_path=records))
    r = client.get("/api/simulate-policy", params={"question_id": OTHER})
    assert r.status_code == 200 and r.json()["group"]["question_id"] == OTHER
    assert client.get("/api/simulate-policy", params={"question_id": QID}).status_code == 400


def test_one_fit_throughout_is_accepted(tmp_path, capsys) -> None:
    records = _write(tmp_path / "records.jsonl", [_row(n, calib={"p": 0.8, "fit_sha256": FIT_A}) for n in range(10)])
    rc, out, err = _run(capsys, ["policy", "curve", str(records), "--use-calib"])
    assert (rc, err) == (0, "")
    # One point, at the calibrated 0.8, none at the raw 0.95.
    assert [(pt["threshold"], pt["accepted"]) for pt in json.loads(out)["groups"][0]["curve"]] == [(0.8, 10)]
    r = TestClient(create_app(records_path=records)).get("/api/simulate-policy")
    assert r.status_code == 200 and r.json()["tuned_threshold"]["threshold"] == 0.8


def test_rows_tune_does_not_count_are_not_checked(tmp_path, capsys) -> None:
    """An unlabeled or unanswered record is not tuned, so it needs no `calib`; records from before
    `s1 apply` wrote `fit_sha256` (missing or null throughout) count as one fit, as in tune."""
    rows = [_row(n, calib={"p": 0.8} if n % 2 else {"p": 0.8, "fit_sha256": None}) for n in range(10)]
    unlabeled = _row(10, calib=None)
    del unlabeled["label"]
    unanswered = {k: v for k, v in _row(11, calib=None).items() if k != "p"}
    unanswered.update(model_served=None, error_code="timeout")
    records = _write(tmp_path / "records.jsonl", rows + [unlabeled, unanswered])
    rc, out, err = _run(capsys, ["policy", "tune", str(records), "-o", str(tmp_path / "p.json"), "--use-calib"])
    assert (rc, err) == (0, "")
    rc, out, err = _run(capsys, ["policy", "curve", str(records), "--use-calib"])
    assert (rc, err) == (0, "")
    assert [(pt["threshold"], pt["accepted"]) for pt in json.loads(out)["groups"][0]["curve"]] == [(0.8, 10)]
    assert TestClient(create_app(records_path=records)).get("/api/simulate-policy").status_code == 200


def test_a_choice_group_is_checked_too(tmp_path, capsys) -> None:
    def choice(n: int, calib: bool) -> dict:
        row = {"schema": "s1-record/1", "dataset": "d", "item_id": f"i{n}", "question_id": QID, "primitive": "choice",
               "model_requested": "m", "model_served": "m", "probabilities": {"a": 0.9, "b": 0.1}, "choice": "a",
               "confidence": 0.9, "label": "a"}
        if calib:
            row["calib"] = {"probabilities": {"a": 0.7, "b": 0.3}, "fit_sha256": FIT_A}
        return row

    records = _write(tmp_path / "records.jsonl", [choice(n, calib=n != 4) for n in range(10)])
    expected = _tune_error(capsys, records, tmp_path)
    assert "line 5 has no calibrated probabilities" in expected
    assert _run(capsys, ["policy", "curve", str(records), "--use-calib"]) == (1, "", expected)
    r = TestClient(create_app(records_path=records)).get("/api/simulate-policy")
    assert (r.status_code, r.json()["detail"]) == (400, _web_detail(expected, records))


@pytest.mark.parametrize("extra", [(), ("--max-false-pass", "0.1"), ("--noul-routing", "threshold")])
def test_every_tune_path_names_the_records_file(tmp_path, capsys, extra) -> None:
    """Each way tune compiles a noul policy checks the group and names the file (#140); the message
    speaks of the calibrated distribution, not of a policy, since slices share it."""
    # Half the labels false, so --max-false-pass finds its cutoff and reaches the check.
    rows = [{**row, "label": n % 2 == 0} for n, row in enumerate(CASES["missing-calib"])]
    records = _write(tmp_path / "r.jsonl", rows)
    assert _tune_error(capsys, records, tmp_path, *extra) == (
        f"s1: {records}: {QID} (m): line 9 has no calibrated probabilities (the `calib` that `s1 apply` writes); "
        "apply the fit to every record first\n"
    )
