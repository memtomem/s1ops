"""`s1 apply` never writes an item's `state` (#37): it copies input rows, so each path it writes
through (calibrated, passed through, failed) leaves the top-level `state` out and reports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import parse_record
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import apply_to_record, load_fits

FIT_MODEL, NEW_MODEL = "jev-1.13.0", "jev-1.14.0-rc1"


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _inputs(ws: dict, schema: str) -> list[dict]:
    """The demo canary without `calib`: 80 rows of the fitted model, 40 of another, and one failed row."""
    rows = [{**{k: v for k, v in r.items() if k != "calib"}, "schema": schema} for r in _rows(ws["records_canary"])]
    failed = {k: v for k, v in rows[0].items() if k not in ("p",)}
    failed.update(item_id="failed-0", model_served=None, error="timeout")
    return [*rows, failed]


def _with_state(rows: list[dict], value="SECRET TEXT") -> list[dict]:
    return [{**r, "state": value, "features": {"state": "kept, not the item's"}} for r in rows]


def _apply(records: str, fits: str, out: Path, *flags: str) -> int:
    return cli_main(["apply", records, "--fit", fits, "-o", str(out), *flags])


@pytest.mark.parametrize("schema", ["s1-record/1", "jev-record/1"])
@pytest.mark.parametrize("value", ["SECRET TEXT", "", None], ids=["text", "empty", "null"])
def test_every_written_row_leaves_state_out_and_nothing_else(ws, tmp_path: Path, capsys, schema: str, value) -> None:
    clean = _inputs(ws, schema)
    dirty = [{**r, "state": value} for r in clean]
    assert {r["model_served"] for r in clean} == {FIT_MODEL, NEW_MODEL, None}
    out_clean, out_dirty = tmp_path / "clean.jsonl", tmp_path / "dirty.jsonl"
    dirty_path = _write(tmp_path / "in.jsonl", dirty)
    before = Path(dirty_path).read_bytes()

    assert _apply(_write(tmp_path / "clean_in.jsonl", clean), ws["fits"], out_clean, "--pass-through-unfitted") == 0
    clean_summary = json.loads(capsys.readouterr().out)
    assert _apply(dirty_path, ws["fits"], out_dirty, "--pass-through-unfitted") == 0
    dirty_summary = json.loads(capsys.readouterr().out)

    # Every path (80 calibrated, 40 passed through, 1 failed) writes what it writes without state,
    # line for line identical to the run that never had one.
    assert out_dirty.read_bytes() == out_clean.read_bytes()
    assert all("state" not in r for r in _rows(out_dirty))
    assert dirty_summary == {**clean_summary, "state_dropped": len(dirty)}
    assert "state_dropped" not in clean_summary
    assert Path(dirty_path).read_bytes() == before


def test_state_sha256_and_a_nested_state_key_are_kept(ws, tmp_path: Path, capsys) -> None:
    rows = _with_state(_inputs(ws, "s1-record/1"))
    for r in rows:
        r["state_sha256"] = "0" * 64
    out = tmp_path / "out.jsonl"
    assert _apply(_write(tmp_path / "in.jsonl", rows), ws["fits"], out, "--pass-through-unfitted") == 0
    capsys.readouterr()
    written = _rows(out)
    assert len(written) == len(rows)
    assert all(r["state_sha256"] == "0" * 64 and r["features"] == {"state": "kept, not the item's"} for r in written)
    assert all("state" not in r for r in written)


def test_a_plain_run_prints_a_summary_only_when_it_dropped_state(ws, tmp_path: Path, capsys) -> None:
    own = [r for r in _inputs(ws, "s1-record/1") if r["model_served"] == FIT_MODEL]
    out = tmp_path / "out.jsonl"
    assert _apply(_write(tmp_path / "clean.jsonl", own), ws["fits"], out) == 0
    assert capsys.readouterr().out == ""
    dirty = [*own[:3], *_with_state(own[3:5]), *own[5:]]
    assert _apply(_write(tmp_path / "dirty.jsonl", dirty), ws["fits"], out) == 0
    assert json.loads(capsys.readouterr().out) == {"calibrated": len(own), "state_dropped": 2}


@pytest.mark.parametrize(("flags", "code"), [((), 3), (("--pass-through-unfitted",), 2), ((), 1)], ids=["model-mismatch", "in-sample", "no-fit"])
def test_a_refusal_prints_no_summary(ws, tmp_path: Path, capsys, flags: tuple, code: int) -> None:
    rows = _with_state(_inputs(ws, "s1-record/1"))
    if code == 2:
        (fit,) = load_fits(ws["fits"])
        dataset, item_id = fit.items[0]
        own = next(r for r in rows if r["model_served"] == FIT_MODEL)
        rows = [{**own, "dataset": dataset, "item_id": item_id}]
    elif code == 1:
        own = next(r for r in rows if r["model_served"] == FIT_MODEL)
        rows = [{**own, "question_id": "q-0000000000000000"}]
    out = tmp_path / "out.jsonl"
    assert _apply(_write(tmp_path / "in.jsonl", rows), ws["fits"], out, *flags) == code
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("s1: refusing")
    assert not out.exists()


def test_apply_to_record_leaves_state_out_and_the_record_alone(ws) -> None:
    (fit,) = load_fits(ws["fits"])
    row = next(r for r in _with_state(_inputs(ws, "s1-record/1")) if r["model_served"] == FIT_MODEL)
    rec = parse_record(row)
    raw_before = json.dumps(rec.raw, sort_keys=True)
    out = apply_to_record(fit, rec)
    assert "state" not in out and out["features"] == row["features"] and "calib" in out
    assert json.dumps(rec.raw, sort_keys=True) == raw_before
    assert {k: v for k, v in out.items() if k != "calib"} == {k: v for k, v in row.items() if k != "state"}
