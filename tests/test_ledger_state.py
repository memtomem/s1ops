"""safe-v1 in the ledger: `Ledger` metadata refuses item `state`, and `recover` will not republish it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.ledger.journal import STATE_REFUSAL, Ledger, read, recover


@pytest.mark.parametrize("value", ["SECRET", "", None])
def test_reserve_refuses_state_and_writes_nothing(tmp_path: Path, value) -> None:
    path = tmp_path / "journal.jsonl"
    with Ledger(path, cap_usd=1.0) as ledger:
        before = path.read_bytes() if path.exists() else b""
        with pytest.raises(ValueError) as e:
            ledger.reserve(0.01, item_id="i", state=value)
        assert str(e.value) == STATE_REFUSAL
        assert (path.read_bytes() if path.exists() else b"") == before
        assert ledger.outstanding == {}


@pytest.mark.parametrize("value", ["SECRET", "", None])
def test_settle_refuses_state_and_keeps_the_reservation_open(tmp_path: Path, value) -> None:
    path = tmp_path / "journal.jsonl"
    with Ledger(path, cap_usd=1.0) as ledger:
        reservation = ledger.reserve(0.01, item_id="i")
        before = path.read_bytes()
        with pytest.raises(ValueError) as e:
            ledger.settle(reservation, 0.005, outcome="measured", state=value)
        assert str(e.value) == STATE_REFUSAL
        assert path.read_bytes() == before
        assert list(ledger.outstanding) == [reservation.id]
        ledger.settle(reservation, 0.005, outcome="measured")
    assert read(path).spent_usd == 0.005
    assert b"state" not in path.read_bytes().replace(b"state_sha256", b"")


def test_state_sha256_and_nested_state_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    with Ledger(path, cap_usd=1.0) as ledger:
        reservation = ledger.reserve(0.01, item_id="state", state_sha256="ab" * 32, extra={"state": 1})
        ledger.settle(reservation, 0.0, outcome="measured")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["state_sha256"] == "ab" * 32 and events[0]["extra"] == {"state": 1}


def _event(kind: str, **fields) -> dict:
    base = {"v": 1, "event": kind, "ts": 1.0}
    if kind == "opening":
        base.update(usd=0.0, legacy_path="legacy.jsonl", legacy_sha256="0" * 64, legacy_rows=0)
    elif kind == "reserve":
        base.update(id="r1", worst_usd=0.01)
    else:
        base.update(id="r1", outcome="measured", usd=0.005)
    return {**base, **fields}


def _journal(path: Path, carrier: str | None, value) -> None:
    events = [_event(kind, **({"state": value} if kind == carrier else {})) for kind in ("opening", "reserve", "settle")]
    path.write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in events) + b'{"v": 1, "event": "res')


@pytest.mark.parametrize("value", ["SECRET", "", None])
@pytest.mark.parametrize("carrier, line", [("opening", 1), ("reserve", 2), ("settle", 3)])
def test_recover_refuses_a_journal_whose_events_carry_state(tmp_path: Path, carrier: str, line: int, value) -> None:
    journal, out = tmp_path / "journal.jsonl", tmp_path / "recovered.jsonl"
    _journal(journal, carrier, value)
    before = journal.read_bytes()
    with pytest.raises(ValueError) as e:
        recover(journal, out)
    assert str(e.value) == (
        f"{journal} line {line}: an event carries item state text; recover will not copy it "
        "into a new file (remove the top-level state from those events first)"
    )
    assert not out.exists()
    assert not out.with_name(out.name + ".recovery.json").exists()
    assert journal.read_bytes() == before


def test_recover_still_copies_a_journal_without_state(tmp_path: Path) -> None:
    journal, out = tmp_path / "journal.jsonl", tmp_path / "recovered.jsonl"
    _journal(journal, None, None)
    recover(journal, out)
    assert out.read_bytes() == journal.read_bytes().rsplit(b"\n", 1)[0] + b"\n"


def test_a_journal_written_with_state_before_the_refusal_still_opens(tmp_path: Path) -> None:
    # Deliberate: refusing on open would halt paid runs without removing the text, and appending
    # new events does not rewrite the old ones.
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps(_event("reserve", state="SECRET")) + "\n")
    with Ledger(path, cap_usd=1.0) as ledger:
        assert ledger.outstanding == {"r1": 0.01}


class _OddHash(str):
    """Equal to the plain string it wraps but hashed apart from it, so dict membership misses it."""

    def __hash__(self) -> int:
        return super().__hash__() ^ 1


@pytest.mark.parametrize("key", [_OddHash("state"), _OddHash("v"), _OddHash("item_id")])
def test_metadata_keys_must_be_plain_strings(tmp_path: Path, key: str) -> None:
    path = tmp_path / "journal.jsonl"
    with Ledger(path, cap_usd=1.0) as ledger:
        with pytest.raises(ValueError) as e:
            ledger.reserve(0.01, **{key: "SECRET"})
        assert str(e.value) == "metadata keys must be plain strings"
        reservation = ledger.reserve(0.01)
        before = path.read_bytes()
        with pytest.raises(ValueError):
            ledger.settle(reservation, 0.0, outcome="measured", **{key: "SECRET"})
        assert path.read_bytes() == before
    assert b"SECRET" not in path.read_bytes()
