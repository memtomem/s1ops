"""`score_histograms` and `GET /api/drift/histograms`: the binned scores behind drift's PSI.

The histograms must be the ones the report's PSI was computed from (same inputs and refusals, same
answered records, same score, same bins), and the route serves them with the report from one read
of the files, so the page never pairs a report with another version's distributions.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from s1ops.contracts.records import read_records
from s1ops.demo import generate_workspace
from s1ops.monitor.drift import DriftInputError, audit_drift, score_histograms
from s1ops.web.app import create_app


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _row(primitive: str, item: int, score: float, *, qid: str = "q", served: str | None = "m", fit: str | None = "f") -> dict:
    """An answered record whose primary score is `score` (and a calib of the same shape), or a failed
    one when `served` is None."""
    row = {"schema": "s1-record/1", "item_id": f"i{item}", "question_id": qid, "primitive": primitive,
           "model_requested": "m", "model_served": served}
    if served is None:
        row["error_code"] = "timeout"
        return row
    if primitive == "noul":
        row.update({"p": score, "calib": {"p": round(1 - score, 4), "fit_sha256": fit}})
        return row
    keys = ("A", "B") if primitive == "choice" else ("0", "1")
    probs = {keys[0]: score, keys[1]: round(1 - score, 6)}
    top = max(probs, key=probs.get)
    row.update({"choice" if primitive == "choice" else "score": top if primitive == "choice" else int(top),
                "confidence": probs[top], "probabilities": probs,
                "calib": {"probabilities": {keys[0]: round(1 - score, 6), keys[1]: score}, "fit_sha256": fit}})
    return row


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _random_rows(primitive: str, seed: int, n: int, *, qid: str = "q", failed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = [_row(primitive, i, round(rng.random(), 4), qid=qid) for i in range(n)]
    return rows + [_row(primitive, n + i, 0.0, qid=qid, served=None) for i in range(failed)]


def _assert_matches(hist: dict, report: dict) -> None:
    assert [q["question_id"] for q in hist["questions"]] == [q["question_id"] for q in report["questions"]]
    assert hist["distribution"] == report["distribution"]
    for h, q in zip(hist["questions"], report["questions"]):
        assert h["primitive"] == q["primitive"]
        assert h["psi"] == q["psi"]
        assert h["ref"]["n_answered"] == q["n_ref_answered"]
        assert h["cur"]["n_answered"] == q["n_cur_answered"]
        assert sum(h["ref"]["counts"]) == q["n_ref_answered"] and sum(h["cur"]["counts"]) == q["n_cur_answered"]
        assert len(h["edges"]) == hist["bins"] + 1 and len(h["ref"]["counts"]) == hist["bins"]


@pytest.mark.parametrize("primitive", ["noul", "choice", "score"])
@pytest.mark.parametrize("use_calib", [False, True])
def test_the_histograms_reproduce_the_reports_psi(tmp_path, primitive, use_calib) -> None:
    ref = _write(tmp_path / "ref.jsonl", _random_rows(primitive, 1, 80, failed=3) + _random_rows(primitive, 2, 40, qid="q2"))
    cur = _write(tmp_path / "cur.jsonl", _random_rows(primitive, 3, 60, failed=9) + _random_rows(primitive, 4, 50, qid="q2"))
    report = audit_drift(ref, cur, use_calib=use_calib)
    hist = score_histograms(ref, cur, use_calib=use_calib)
    _assert_matches(hist, report)
    assert len(hist["questions"]) == 2
    assert {q["score"] for q in hist["questions"]} == {"p" if primitive == "noul" else "top_probability"}


def test_the_demo_canary_and_the_categorical_records(ws) -> None:
    _assert_matches(score_histograms(ws["records"], ws["records_canary"]), audit_drift(ws["records"], ws["records_canary"]))
    recs = read_records(ws["records_categorical"], require_answer=True, use_calib=True)
    for use_calib in (False, True):
        hist = score_histograms(recs[::2], recs[1::2], use_calib=use_calib)
        _assert_matches(hist, audit_drift(recs[::2], recs[1::2], use_calib=use_calib))
        assert {q["primitive"] for q in hist["questions"]} == {"choice", "score"}


def test_counts_are_the_bins_of_the_primary_score(tmp_path) -> None:
    # 0.1 opens the second bin, 1.0 closes the last one: np.histogram's half-open bins.
    scores = [0.0, 0.05, 0.1, 0.55, 0.999, 1.0]
    ref = _write(tmp_path / "ref.jsonl", [_row("noul", i, s) for i, s in enumerate(scores)])
    cur = _write(tmp_path / "cur.jsonl", [_row("noul", i, 0.5) for i in range(3)])
    q = score_histograms(ref, cur)["questions"][0]
    assert q["ref"]["counts"] == [2, 1, 0, 0, 0, 1, 0, 0, 0, 2]
    assert q["cur"]["counts"] == [0, 0, 0, 0, 0, 3, 0, 0, 0, 0]
    assert q["edges"] == [float(e) for e in np.linspace(0, 1, 11)]
    # PSI written out here, independently of the module: each side's share, floored at 1e-4.
    r = [max(c / 6, 1e-4) for c in q["ref"]["counts"]]
    c = [max(n / 3, 1e-4) for n in q["cur"]["counts"]]
    terms = [(b - a) * math.log(b / a) for a, b in zip(r, c)]
    assert q["psi_terms"] == pytest.approx([round(t, 6) for t in terms], abs=1e-12)
    assert q["psi"] == pytest.approx(round(math.fsum(terms), 6), abs=1e-6)


def test_a_side_without_an_answered_record(tmp_path) -> None:
    ref = _write(tmp_path / "ref.jsonl", _random_rows("choice", 1, 30))
    cur = _write(tmp_path / "cur.jsonl", [_row("choice", i, 0.0, served=None) for i in range(5)])
    q = score_histograms(ref, cur)["questions"][0]
    rq = audit_drift(ref, cur)["questions"][0]
    assert q["cur"] == {"n_answered": 0, "counts": [0] * 10}
    assert q["psi"] == rq["psi"] == 0.0 and q["psi_terms"] is None
    assert sum(q["ref"]["counts"]) == 30


def _mixed_primitive(tmp_path: Path) -> tuple[list[dict], list[dict], bool]:
    return [_row("noul", 0, 0.4)], [{**_row("choice", 0, 0.7), "question_id": "q"}], False


def _no_calib(tmp_path: Path) -> tuple[list[dict], list[dict], bool]:
    bare = {k: v for k, v in _row("noul", 1, 0.3).items() if k != "calib"}
    return [_row("noul", 0, 0.4)], [bare], True


def _two_fits(tmp_path: Path) -> tuple[list[dict], list[dict], bool]:
    return [_row("noul", 0, 0.4, fit="f1")], [_row("noul", 1, 0.4, fit="f2")], True


@pytest.mark.parametrize("case", [_mixed_primitive, _no_calib, _two_fits])
def test_refusals_are_the_reports(tmp_path, case) -> None:
    ref_rows, cur_rows, use_calib = case(tmp_path)
    ref, cur = _write(tmp_path / "ref.jsonl", ref_rows), _write(tmp_path / "cur.jsonl", cur_rows)
    with pytest.raises(DriftInputError) as from_report:
        audit_drift(ref, cur, use_calib=use_calib)
    with pytest.raises(DriftInputError) as from_hist:
        score_histograms(ref, cur, use_calib=use_calib)
    assert str(from_hist.value) == str(from_report.value)


# --- the route ---------------------------------------------------------------------------------


def _client(root: Path) -> TestClient:
    return TestClient(create_app(workspace_dir=root), raise_server_exceptions=False)


def test_the_route_serves_the_report_and_its_histograms(ws) -> None:
    client = _client(Path(ws["records"]).parent)
    both = client.get("/api/drift/histograms")
    assert both.status_code == 200
    assert both.json()["report"] == client.get("/api/drift").json()
    assert both.json()["histograms"] == score_histograms(ws["records"], ws["records_canary"])


def test_without_a_canary_both_are_empty(tmp_path, ws) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    both = _client(empty).get("/api/drift/histograms").json()
    assert both["report"] == _client(empty).get("/api/drift").json()
    assert both["histograms"] == {"schema": "s1-drift-histograms/1", "distribution": "raw", "bins": 10, "questions": []}


def _bad_canary_line(root: Path) -> None:
    with (root / "records_canary.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"schema": "s1-record/1", "item_id": "x"}\n')


def _bad_policy(root: Path) -> None:
    (root / "policy.json").write_text("{}", encoding="utf-8")


def _mixed_question(root: Path) -> None:
    first = json.loads((root / "records_canary.jsonl").read_text(encoding="utf-8").splitlines()[0])
    with (root / "records_canary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({**_row("choice", 999, 0.7), "question_id": first["question_id"]}) + "\n")


@pytest.mark.parametrize("break_it", [_bad_canary_line, _bad_policy, _mixed_question])
def test_errors_are_the_drift_routes(tmp_path, break_it) -> None:
    root = tmp_path / "w"
    generate_workspace(root)
    break_it(root)
    client = _client(root)
    plain, both = client.get("/api/drift"), client.get("/api/drift/histograms")
    assert plain.status_code == 400
    assert (both.status_code, both.json()) == (plain.status_code, plain.json())
