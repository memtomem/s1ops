"""A backend `close()` that raises never replaces what `s1 probe` itself has to say."""

from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path

import httpx2
import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import NoulQuestion
from s1ops.ledger.journal import token_bound
from s1ops.monitor import backends as backends_mod
from s1ops.monitor.runner import BackendError, DeterministicMockBackend

CLOSE_TEXT = "TOP_SECRET close text"
STATE = "s" * 50
QUESTION = {"type": "noul", "instructions": "x?"}
EMPTY_CRITERIA = {"type": "choice", "instructions": "x?", "criteria": {}}
# Admits the first reservation; what the first request spent plus the next reservation exceeds it.
CAP_FOR_ONE = repr((token_bound(STATE) + token_bound(json.dumps(NoulQuestion(instructions="x?").wire_dict()))) * 2e-6 + 1e-7)

FAILURES = [pytest.param(RuntimeError, id="RuntimeError"), pytest.param(lambda text: OSError(5, text), id="OSError")]


class Probe:
    def __init__(self, tmp_path: Path, capsys, monkeypatch) -> None:
        self.tmp, self.capsys, self.monkeypatch = tmp_path, capsys, monkeypatch
        self.calls = 0
        self.run_dir = tmp_path / "run"
        self.out = self.run_dir / "out.jsonl"

    def close_with(self, make=None) -> None:
        def close(backend) -> None:
            self.calls += 1
            if make is not None:
                raise make(CLOSE_TEXT)

        self.monkeypatch.setattr(DeterministicMockBackend, "close", close, raising=False)

    def __call__(self, *extra: str, questions: dict | None = None, n: int = 3) -> tuple[int, str, list[str]]:
        q, states = self.tmp / "q.json", self.tmp / "s.jsonl"
        q.write_text(json.dumps(questions or {"q": QUESTION}), encoding="utf-8")
        states.write_text("".join(json.dumps({"dataset": "d", "item_id": f"i{i}", "state": STATE}) + "\n" for i in range(n)), encoding="utf-8")
        self.run_dir.mkdir(exist_ok=True)
        args = ["probe", str(q), "--states", str(states), "-o", str(self.out), "--ledger", str(self.run_dir / "journal.jsonl"), "--no-expand-probes", *extra]
        rc = cli_main(args)
        captured = self.capsys.readouterr()
        return rc, captured.out, captured.err.strip().splitlines()

    def records(self) -> list[dict]:
        return [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]

    def reset(self) -> None:
        shutil.rmtree(self.run_dir)


@pytest.fixture
def probe(tmp_path, capsys, monkeypatch) -> Probe:
    return Probe(tmp_path, capsys, monkeypatch)


def _full_disk(monkeypatch) -> None:
    real_fdopen = os.fdopen

    class Full(io.TextIOWrapper):
        def write(self, text):
            raise OSError(28, "No space left on device")

    monkeypatch.setattr("s1ops.cli.os.fdopen", lambda fd, mode, encoding: Full(real_fdopen(fd, mode.replace("w", "wb")), encoding=encoding))


@pytest.mark.parametrize("make", FAILURES)
def test_a_refusal_stays_the_refusal(probe, tmp_path, make) -> None:
    probe.close_with(make)
    rc, out, err = probe(questions={"c": EMPTY_CRITERIA})
    assert rc == 1 and out == ""
    assert err == [f"s1: {tmp_path / 'q.json'}: question 'c': choice criteria is empty"]
    assert probe.calls == 1
    assert not probe.out.exists()


@pytest.mark.parametrize("make", FAILURES)
@pytest.mark.parametrize(
    "extra, disk_full, kept",
    [
        pytest.param(("--cap-usd", CAP_FOR_ONE), False, 1, id="budget-stop"),
        pytest.param((), True, 0, id="output-write-failure"),
    ],
)
def test_an_error_that_ended_the_run_is_unchanged_by_a_failing_close(probe, monkeypatch, make, extra, disk_full, kept) -> None:
    if disk_full:
        _full_disk(monkeypatch)
    probe.close_with()
    working = probe(*extra)
    assert working[0] == 1 and len(working[2]) == 1 and f"{kept} records kept in {probe.out}" in working[2][0]
    assert probe.calls == 1
    probe.reset()

    probe.close_with(make)
    assert probe(*extra) == working
    assert probe.calls == 2


@pytest.mark.parametrize("make, name", [(RuntimeError, "RuntimeError"), (lambda text: OSError(5, text), "OSError")], ids=["RuntimeError", "OSError"])
def test_a_finished_run_fails_with_one_line_naming_only_the_type(probe, make, name) -> None:
    probe.close_with(make)
    rc, out, err = probe()
    assert rc == 1 and out == ""
    assert err == [f"s1: closing the backend failed ({name}) after the run finished; 3 records kept in {probe.out}, 0 of them failed"]
    assert len(probe.records()) == 3
    assert probe.calls == 1
    assert CLOSE_TEXT not in out + "".join(err)


def test_the_line_counts_failed_records(probe, monkeypatch) -> None:
    def infer(self, *args, **kwargs):
        raise BackendError("invalid_request", sent=False)

    monkeypatch.setattr(DeterministicMockBackend, "infer", infer)
    probe.close_with(RuntimeError)
    rc, out, err = probe()
    assert rc == 1 and out == ""
    assert err == [f"s1: closing the backend failed (RuntimeError) after the run finished; 3 records kept in {probe.out}, 3 of them failed"]
    assert sum(row["model_served"] is None for row in probe.records()) == 3


def test_no_states_with_a_stale_output_says_zero_records_and_leaves_it_empty(probe) -> None:
    probe.run_dir.mkdir()
    probe.out.write_text("previous\n", encoding="utf-8")
    probe.close_with(RuntimeError)
    rc, out, err = probe(n=0)
    assert rc == 1 and out == ""
    assert err == [f"s1: closing the backend failed (RuntimeError) after the run finished; 0 records kept in {probe.out}, 0 of them failed"]
    assert probe.out.read_text(encoding="utf-8") == ""


def test_debug_reraises_the_close_error_of_a_finished_run(probe, monkeypatch) -> None:
    monkeypatch.setenv("S1_DEBUG", "1")
    probe.close_with(RuntimeError)
    with pytest.raises(RuntimeError, match=CLOSE_TEXT):
        probe()
    assert len(probe.records()) == 3


def test_debug_keeps_the_refusal(probe, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("S1_DEBUG", "1")
    probe.close_with(RuntimeError)
    rc, _, err = probe(questions={"c": EMPTY_CRITERIA})
    assert rc == 1
    assert err == [f"s1: {tmp_path / 'q.json'}: question 'c': choice criteria is empty"]


def test_an_unexpected_body_exception_survives_a_failing_close(probe, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise KeyError("from the body")

    monkeypatch.setattr("s1ops.cli.run_batch", explode)
    probe.close_with(RuntimeError)
    with pytest.raises(KeyError, match="from the body"):
        probe()
    assert probe.calls == 1


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_during_close_is_not_swallowed(probe, interrupt) -> None:
    probe.close_with(lambda text: interrupt())
    with pytest.raises(interrupt):
        probe()


def test_a_close_that_works_changes_nothing(probe) -> None:
    probe.close_with()
    rc, out, err = probe()
    assert rc == 0 and err == []
    assert json.loads(out)["n_records"] == 3
    assert probe.calls == 1


def test_a_live_backend_takes_the_same_path(probe, monkeypatch) -> None:
    def echo(request: httpx2.Request) -> httpx2.Response:
        model = json.loads(request.content)["model"]
        return httpx2.Response(200, json={"model": model, "usage": {"input_tokens": 40, "output_tokens": 1}, "answers": {"q": {"type": "noul", "noul": 0.6}}})

    real = backends_mod.HttpSystemOneBackend

    def failing_close(self) -> None:
        raise RuntimeError(CLOSE_TEXT)

    monkeypatch.setattr(real, "close", failing_close)
    monkeypatch.setattr("s1ops.cli.HttpSystemOneBackend", lambda **kw: real(api_key="test-key", transport=httpx2.MockTransport(echo), **kw))
    rc, out, err = probe("--backend", "systemone", "--model", "kev-0.8b", "--usd-per-input-token", "1e-7")
    assert rc == 1 and out == ""
    assert err == [f"s1: closing the backend failed (RuntimeError) after the run finished; 3 records kept in {probe.out}, 0 of them failed"]
    assert len(probe.records()) == 3
