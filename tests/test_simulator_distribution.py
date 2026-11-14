"""`/api/simulate-policy` without `use_calib` (auto) simulates a group on its calibrated
distribution when any record tuned into it carries `calib`, and on its raw one when none does, so
a workspace that never ran `s1 apply` simulates rather than failing (#128); explicit `true` /
`false` are unchanged, and the response names the distribution used."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.demo import generate_workspace
from s1ops.web.app import create_app
from tests.test_curve_use_calib import FIT_A, OTHER, QID, _row

STATIC = Path(__file__).resolve().parents[1] / "src" / "s1ops" / "web" / "static" / "index.html"


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get(records: Path, **params):
    return TestClient(create_app(records.parent, records_path=records)).get("/api/simulate-policy", params=params)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


@pytest.fixture(scope="module")
def stripped(demo, tmp_path_factory) -> Path:
    """The demo's records.jsonl as a workspace that never ran `s1 apply`."""
    rows = [{k: v for k, v in r.items() if k != "calib"} for r in _rows(Path(demo["records"]))]
    return _write(tmp_path_factory.mktemp("raw") / "records.jsonl", rows)


def test_a_workspace_without_calib_simulates_raw(stripped) -> None:
    auto = _get(stripped)
    assert (auto.status_code, auto.json()["distribution"]) == (200, "raw")
    raw = _get(stripped, use_calib=False)
    assert raw.status_code == 200 and auto.json() == raw.json()
    r = _get(stripped, use_calib=True)
    assert r.status_code == 400 and "has no calibrated probabilities" in r.json()["detail"]


def test_explicit_raw_on_calibrated_records_equals_the_same_records_without_calib(demo, stripped) -> None:
    calibrated = _get(Path(demo["records"]))
    assert calibrated.json()["distribution"] == "calibrated"
    raw = _get(Path(demo["records"]), use_calib=False)
    assert raw.status_code == 200 and raw.json() == _get(stripped).json()
    assert raw.json()["threshold_curve"] != calibrated.json()["threshold_curve"]


def test_auto_decides_per_chosen_group_on_its_tuned_records(tmp_path) -> None:
    """A calibrated group beside a calib-free one, whose only `calib` is on an unlabeled record
    (not tuned, so not counted): each group gets its own distribution."""
    unlabeled = {k: v for k, v in _row(99, calib={"p": 0.1, "fit_sha256": FIT_A}).items() if k != "label"}
    rows = [
        *[_row(n, calib={"p": 0.8, "fit_sha256": FIT_A}, qid=OTHER) for n in range(10)],
        *[_row(n, calib=None) for n in range(10)],
        unlabeled,
    ]
    records = _write(tmp_path / "records.jsonl", rows)
    auto = _get(records, question_id=QID)
    assert (auto.status_code, auto.json()["distribution"]) == (200, "raw")
    assert auto.json() == _get(records, question_id=QID, use_calib=False).json()
    other = _get(records, question_id=OTHER)
    assert (other.status_code, other.json()["distribution"]) == (200, "calibrated")
    assert other.json()["tuned_threshold"]["threshold"] == 0.8


def test_auto_still_refuses_a_group_mixing_calibrated_and_raw_records(tmp_path) -> None:
    rows = [_row(n, calib={"p": 0.8, "fit_sha256": FIT_A} if n < 8 else None) for n in range(10)]
    records = _write(tmp_path / "records.jsonl", rows)
    r = _get(records)
    assert r.status_code == 400 and "line 9 has no calibrated probabilities" in r.json()["detail"]
    assert _get(records, use_calib=False).json()["distribution"] == "raw"


def test_explicit_raw_does_not_read_calib(tmp_path) -> None:
    """As before auto: `use_calib=false` reads records raw, so a malformed `calib` is not read."""
    rows = [_row(n, calib={"p": "bad", "fit_sha256": FIT_A} if n == 2 else None) for n in range(10)]
    records = _write(tmp_path / "records.jsonl", rows)
    r = _get(records, use_calib=False)
    assert (r.status_code, r.json()["distribution"]) == (200, "raw")
    assert _get(records).status_code == 400


def test_the_policy_tab_sends_use_calib_only_when_chosen() -> None:
    html = STATIC.read_text()
    body = html[html.index("async function loadPolicy()"):html.index('["inp-group", "inp-dist"].forEach')]
    assert 'const dist = document.getElementById("inp-dist").value;' in body
    assert 'if (dist) params.set("use_calib", dist);' in body
    assert body.index('params.set("use_calib"') < body.index("fetchJson(`/api/simulate-policy?${params}`)")
    select = html[html.index('<select id="inp-dist">'):html.index("</select>", html.index('<select id="inp-dist">'))]
    assert [part.split('"')[0] for part in select.split('value="')[1:]] == ["", "false", "true"]
    assert body.count("${distLine}") == 2 and "esc(d.distribution" in body
