"""Reserve-then-settle budget journal for paid System 1 and Escalator calls.

One append-only JSONL journal records every paid attempt twice: a `reserve` event with the
attempt's worst case before it is sent, and a `settle` event with what it cost once it returns
or fails. Only one paid process may hold the journal at a time (an exclusive `flock` on
`lock_path`, the journal's resolved path plus `.lock`, taken at open and failing fast), so two
runs can never each stay under the cap while together exceeding it; a journal with a second hard
link is refused, since its other name would lock a different file.
"""

import errno
import fcntl
import hashlib
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

VERSION = 1
OUTCOMES = ("measured", "unknown", "unsent")
ACCOUNTING_KEYS = frozenset({"v", "event", "id", "ts", "outcome", "usd", "worst_usd"})


def token_bound(text: str) -> float:
    """Korean-safe worst-case token count: 2 tokens per non-ASCII char + 0.5 per ASCII char."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return 0.5 * ascii_chars + 2.0 * (len(text) - ascii_chars)


class BudgetExceeded(RuntimeError):
    pass


class LedgerUnusable(BudgetExceeded):
    """This instance refuses further writes: a write failed, or a measured cost exceeded its reservation."""


class LedgerCorrupt(ValueError):
    """The file is not a valid journal or legacy ledger; nothing may be spent against it."""


def _amount(value, name: str) -> None:
    try:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite non-negative number")


def _metadata(meta: dict) -> dict:
    clash = ACCOUNTING_KEYS.intersection(meta)
    if clash:
        raise ValueError(f"metadata cannot set accounting fields {sorted(clash)}")
    try:
        json.dumps(meta, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from exc
    return meta


@dataclass(frozen=True)
class Reservation:
    id: str
    worst_usd: float


@dataclass(frozen=True)
class LedgerState:
    format: str
    opening_usd: float
    settlements: list[dict]
    outstanding: list[dict]
    spent_usd: float

    @property
    def outstanding_usd(self) -> float:
        return math.fsum(r["worst_usd"] for r in self.outstanding)

    @property
    def overruns(self) -> list[dict]:
        if self.format != "journal":
            return []
        return [r for r in self.settlements if r["outcome"] == "measured" and r["usd"] > r["worst_usd"]]


_BLANK = object()
_FOREIGN = {"opening": {"id", "worst_usd", "outcome"}, "reserve": {"usd", "outcome"}, "settle": {"worst_usd"}}


def _no_constant(name: str):
    raise ValueError(f"{name} is not a finite JSON number")


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite JSON number")
    return value


def _parse(path: Path, data: bytes) -> LedgerState:
    def bad(line_number: int, why: str) -> LedgerCorrupt:
        return LedgerCorrupt(f"invalid ledger {path} line {line_number}: {why}")

    lines = data.split(b"\n")
    if lines[-1]:
        raise bad(len(lines), "not newline-terminated: a partial write (see `s1 ledger recover`)")
    rows = []
    for line_number, line in enumerate(lines[:-1], 1):
        try:
            row = json.loads(line, parse_constant=_no_constant, parse_float=_finite) if line.strip() else _BLANK
        except ValueError as exc:
            raise bad(line_number, str(exc)) from None
        rows.append((line_number, row))
    if not rows:
        return LedgerState("journal", 0.0, [], [], 0.0)
    first = next((row for _, row in rows if row is not _BLANK), None)
    if isinstance(first, dict) and "v" in first:
        return _parse_journal(rows, bad)
    return _parse_legacy(rows, bad)


def _parse_legacy(rows, bad) -> LedgerState:
    settlements, spent = [], 0.0
    for line_number, row in rows:
        if row is _BLANK:
            continue
        try:
            if not isinstance(row, dict) or "usd" not in row:
                raise ValueError("expected an object with usd")
            if "v" in row:
                raise ValueError("a journal event in a legacy ledger")
            _amount(row["usd"], "usd")
            total = spent + row["usd"]
            _amount(total, "replayed total")
        except ValueError as exc:
            raise bad(line_number, str(exc)) from None
        settlements.append(row)
        spent = total
    return LedgerState("legacy", 0.0, settlements, [], spent)


def _parse_journal(rows, bad) -> LedgerState:
    opening, spent = 0.0, 0.0
    open_: dict[str, dict] = {}
    seen: set[str] = set()
    settlements = []
    for line_number, event in rows:
        try:
            if not isinstance(event, dict) or type(event.get("v")) is not int or event["v"] != VERSION:
                raise ValueError(f"expected a version {VERSION} journal event")
            kind, ts = event.get("event"), event.get("ts")
            if not isinstance(kind, str) or kind not in _FOREIGN:
                raise ValueError(f"unknown event {kind!r}")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                raise ValueError("ts must be a number")
            if _FOREIGN[kind].intersection(event):
                raise ValueError(f"a {kind} event cannot carry {sorted(_FOREIGN[kind].intersection(event))}")
            if kind == "opening":
                if line_number != 1:
                    raise ValueError("an opening event must be the first line")
                if not isinstance(event.get("legacy_path"), str) or not isinstance(event.get("legacy_sha256"), str):
                    raise ValueError("an opening event names its legacy_path and legacy_sha256")
                _amount(event.get("usd"), "usd")
                opening = spent = event["usd"]
            elif kind == "reserve":
                rid = event.get("id")
                if not isinstance(rid, str) or not rid:
                    raise ValueError("a reservation needs an id")
                if rid in seen:
                    raise ValueError(f"reservation {rid} appears twice")
                _amount(event.get("worst_usd"), "worst_usd")
                seen.add(rid)
                open_[rid] = {k: v for k, v in event.items() if k not in ("v", "event")}
            elif kind == "settle":
                rid = event.get("id")
                reservation = open_.pop(rid, None) if isinstance(rid, str) else None
                if reservation is None:
                    raise ValueError(f"settlement of {rid!r}, which has no outstanding reservation before it")
                outcome, usd, worst = event.get("outcome"), event.get("usd"), reservation["worst_usd"]
                if outcome not in OUTCOMES:
                    raise ValueError(f"outcome must be one of {OUTCOMES}")
                _amount(usd, "usd")
                if outcome == "unknown" and usd != worst:
                    raise ValueError("an unknown outcome is charged at its reservation")
                if outcome == "unsent" and usd != 0:
                    raise ValueError("an unsent outcome is charged 0")
                total = spent + usd
                _amount(total, "replayed total")
                spent = total
                settlements.append({
                    **{k: v for k, v in reservation.items() if k != "ts"},
                    **{k: v for k, v in event.items() if k not in ("v", "event")},
                    "measured": outcome == "measured",
                    "not_sent": outcome == "unsent",
                })
        except ValueError as exc:
            raise bad(line_number, str(exc)) from None
    try:
        _amount(math.fsum(r["worst_usd"] for r in open_.values()), "outstanding total")
    except (ValueError, OverflowError) as exc:
        raise bad(rows[-1][0], str(exc)) from None
    return LedgerState("journal", opening, settlements, list(open_.values()), spent)


def read(path) -> LedgerState:
    path = Path(path)
    return _parse(path, path.read_bytes() if path.exists() else b"")


def lock_path(path) -> Path:
    return Path(os.path.realpath(path) + ".lock")


def _data_paths(*paths) -> None:
    for path in paths:
        for name in (str(path), os.path.realpath(path)):
            if name.casefold().endswith(".lock"):
                raise ValueError(f"{path}: a journal or ledger path cannot end in .lock (that name is a lock file's)")


@contextmanager
def _locked(path: Path):
    held = lock_path(path)
    try:
        lockfile = open(held, "a", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW, 0o666))
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise
        raise ValueError(f"{held} is a symbolic link; a lock file must be a regular file") from None
    try:
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BudgetExceeded(f"another paid run holds {held}; run paid steps one at a time") from None
        yield lockfile
    finally:
        lockfile.close()


def _one_name(path: Path, lockfile) -> None:
    if str(lock_path(path)) != lockfile.name:
        raise LedgerUnusable(
            f"{path} resolved to {lockfile.name.removesuffix('.lock')} when it was locked and "
            "elsewhere now; give the journal one stable name"
        )
    try:
        links = os.stat(path).st_nlink
    except FileNotFoundError:
        return
    if links > 1:
        raise LedgerUnusable(
            f"{path} has {links} hard links, and each name would lock a different file; give the journal one name"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _publish(dest: Path, data: bytes) -> None:
    if os.path.lexists(dest):
        raise FileExistsError(f"{dest} exists")
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, dest)
        dirfd = os.open(dest.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        tmp.unlink(missing_ok=True)


def migrate(legacy, journal, *, acknowledge_lost_reservations: bool = False) -> LedgerState:
    if not acknowledge_lost_reservations:
        raise ValueError(
            "a legacy ledger cannot reveal reservations lost before migration; acknowledge_lost_reservations=True"
        )
    legacy, journal = Path(legacy), Path(journal)
    _data_paths(legacy, journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with _locked(legacy), _locked(journal):
        if journal.exists():
            raise FileExistsError(f"{journal} exists; migrate starts a new journal")
        if not legacy.exists():
            raise FileNotFoundError(legacy)
        data = legacy.read_bytes()
        state = _parse(legacy, data)
        if state.format != "legacy":
            raise ValueError(f"{legacy} is already a journal")
        opening = {
            "v": VERSION,
            "event": "opening",
            "ts": time.time(),
            "usd": state.spent_usd,
            "legacy_path": str(legacy),
            "legacy_sha256": _sha256(data),
            "legacy_rows": len(state.settlements),
        }
        _publish(journal, (json.dumps(opening, allow_nan=False) + "\n").encode())
    return read(journal)


def recover(journal, out) -> Path:
    journal, out = Path(journal), Path(out)
    sidecar = out.with_name(out.name + ".recovery.json")
    _data_paths(journal, out, sidecar)
    with _locked(journal) as held, _locked(out), _locked(sidecar):
        _one_name(journal, held)
        for existing in (out, sidecar):
            if existing.exists():
                raise FileExistsError(f"{existing} exists; remove it only if {out} does not")
        data = journal.read_bytes() if journal.exists() else b""
        if not data or data.endswith(b"\n"):
            raise ValueError(f"{journal} does not end in a partial line; nothing to recover")
        cut = data.rfind(b"\n") + 1
        prefix, dropped = data[:cut], data[cut:]
        state = _parse(journal, prefix)
        if state.format != "journal":
            raise ValueError(f"{journal} is a legacy ledger; recover only handles journals")
        provenance = {
            "source": str(journal),
            "source_sha256": _sha256(data),
            "out_sha256": _sha256(prefix),
            "dropped_bytes": len(dropped),
            "dropped_hex": dropped.hex(),
            "outstanding": len(state.outstanding),
        }
        _publish(sidecar, (json.dumps(provenance, indent=2) + "\n").encode())
        _publish(out, prefix)
    return out


class Ledger:
    def __init__(self, path, cap_usd: float) -> None:
        _amount(cap_usd, "cap_usd")
        _data_paths(path)
        self.path, self.cap = Path(path), cap_usd
        self.lock = threading.Lock()
        self._stopped: str | None = None
        self._overrun: str | None = None
        self._lock_cm = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_cm = _locked(self.path)
        lockfile = lock_cm.__enter__()
        self._lock_cm = lock_cm
        try:
            _one_name(self.path, lockfile)
            state = read(self.path)
            if state.format == "legacy":
                raise LedgerCorrupt(
                    f"{self.path} is a settlement-only ledger and is never appended to; start a journal with "
                    f"`s1 ledger migrate {self.path} <journal> --acknowledge-lost-reservations`"
                )
            self.spent = state.spent_usd
            self._open = {r["id"]: r["worst_usd"] for r in state.outstanding}
            self.reserved = math.fsum(self._open.values())
        except BaseException:
            self.close()
            raise

    @property
    def overrun(self) -> str | None:
        return self._overrun

    @property
    def outstanding(self) -> dict[str, float]:
        with self.lock:
            return dict(self._open)

    def _write(self, event: dict, apply) -> None:
        line = json.dumps(event, allow_nan=False) + "\n"
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            apply()
        except BaseException as exc:
            self._stopped = f"a write to {self.path} did not complete ({exc!r}); reopen it to replay what it holds"
            raise

    def reserve(self, worst_usd: float, **meta) -> Reservation:
        _amount(worst_usd, "worst_usd")
        _metadata(meta)
        with self.lock:
            if self._stopped or self._overrun:
                raise LedgerUnusable(self._stopped or self._overrun)
            if self.spent + self.reserved + worst_usd > self.cap:
                raise BudgetExceeded(
                    f"spent ${self.spent:.4f} + reserved ${self.reserved:.4f} + ${worst_usd:.6f} > cap ${self.cap:.2f}"
                )
            reservation = Reservation(uuid.uuid4().hex, worst_usd)

            def apply():
                self._open[reservation.id] = worst_usd
                self.reserved = math.fsum(self._open.values())

            self._write(
                {"v": VERSION, "event": "reserve", "id": reservation.id, "ts": time.time(), "worst_usd": worst_usd, **meta},
                apply,
            )
            return reservation

    def settle(self, reservation: Reservation, usd: float | None = None, *, outcome: str, **meta) -> None:
        if not isinstance(reservation, Reservation) or not isinstance(reservation.id, str):
            raise ValueError("settle takes the Reservation that reserve returned")
        _amount(reservation.worst_usd, "reservation.worst_usd")
        if outcome == "measured":
            _amount(usd, "usd")
        elif outcome == "unknown":
            if usd is not None:
                raise ValueError("an unknown outcome takes usd=None; it is charged at its reservation")
        elif outcome == "unsent":
            if usd != 0 or isinstance(usd, bool):
                raise ValueError("an unsent outcome takes usd=0")
        else:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        _metadata(meta)
        with self.lock:
            if self._stopped:
                raise LedgerUnusable(self._stopped)
            worst = self._open.get(reservation.id)
            if worst is None or worst != reservation.worst_usd:
                raise ValueError(f"no outstanding reservation {reservation.id} (unknown, or already settled)")
            charged = {"measured": usd, "unknown": worst, "unsent": 0.0}[outcome]
            total = self.spent + charged
            _amount(total, "settled total")

            def apply():
                del self._open[reservation.id]
                self.reserved = math.fsum(self._open.values())
                self.spent = total
                if outcome == "measured" and usd > worst:
                    self._overrun = (
                        f"measured ${usd:.6f} > reserved ${worst:.6f}: the bound is broken, so no further reservation"
                    )

            self._write(
                {"v": VERSION, "event": "settle", "id": reservation.id, "ts": time.time(), "outcome": outcome, "usd": charged, **meta},
                apply,
            )

    def close(self) -> None:
        with self.lock:
            self._stopped = self._stopped or f"{self.path} was closed"
            cm, self._lock_cm = self._lock_cm, None
        if cm is not None:
            cm.__exit__(None, None, None)

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
