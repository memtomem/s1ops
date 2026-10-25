"""`HttpSystemOneBackend` and `run_batch` settlement, fully offline through an `httpx2.MockTransport`."""

from __future__ import annotations

import json
import os
import sys
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx2
import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from s1ops.contracts.records import read_records
from s1ops.ledger.journal import BudgetExceeded, Ledger, LedgerUnusable, read as ledger_read, token_bound
from s1ops.monitor import backends as backends_mod
from s1ops.monitor.backends import HttpSystemOneBackend, retry_after_s
from s1ops.monitor.runner import BackendError, BackendResponse, run_batch

SECRET = "TOP_SECRET_STATE\nsecond line"
PRICE = 1e-7
NOUL = NoulQuestion(instructions="Is this about billing?")
CHOICE = ChoiceQuestion(instructions="Tone?", criteria={"calm": None, "angry": None})
SCORE = ScoreQuestion(instructions="Severity?", criteria=["none", "minor", "major"])


def ok(answer: dict, *, model: str = "jev-1.13.0", tokens: int = 40) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={"model": model, "usage": {"input_tokens": tokens, "output_tokens": 1}, "answers": {"q": answer}},
    )


class Server:
    """Replays queued responses (or raises queued exceptions) and records each request body."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply) and not isinstance(reply, httpx2.Response):
            reply = reply(request)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def backend(server: Server) -> HttpSystemOneBackend:
    return HttpSystemOneBackend(
        usd_per_input_token=PRICE,
        api_key="test-key",
        base_url="https://api.test",
        transport=httpx2.MockTransport(server),
    )


def run(tmp_path: Path, server: Server, question=NOUL, **kw):
    journal = tmp_path / "journal.jsonl"
    emitted: list[dict] = []
    sleeps: list[float] = []
    with Ledger(journal, cap_usd=kw.pop("cap_usd", 1.0)) as ledger:
        recs = run_batch(
            [{"dataset": "d", "item_id": "i1", "state": SECRET}],
            {"q1": question},
            backend=backend(server),
            ledger=ledger,
            on_record=emitted.append,
            sleep=sleeps.append,
            **kw,
        )
    assert recs == emitted
    return recs, ledger_read(journal), sleeps, journal


def outcomes(state) -> list[str]:
    return [s["outcome"] for s in state.settlements]


@pytest.mark.parametrize(
    "question, answer, expected",
    [
        (NOUL, {"type": "noul", "noul": 0.8}, {"p": 0.8}),
        (
            CHOICE,
            {"type": "choice", "choice": "angry", "confidence": 0.7, "probabilities": {"angry": 0.7, "calm": 0.3}},
            {"choice": "angry", "confidence": 0.7, "probabilities": {"calm": 0.3, "angry": 0.7}},
        ),
        (
            SCORE,
            {
                "type": "score",
                "score": 1.2,
                "confidence": 0.6,
                "legend": {"0": "none", "1": "minor", "2": "major"},
                "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
            },
            {"score": 1.2, "confidence": 0.6, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}},
        ),
    ],
)
def test_answers_map_to_valid_records(tmp_path: Path, question, answer, expected) -> None:
    server = Server(ok(answer, model="jev-1.14.0"))
    recs, state, _, _ = run(tmp_path, server, question)
    (rec,) = recs
    assert {k: rec[k] for k in expected} == expected
    assert list(rec.get("probabilities", {})) == list(expected.get("probabilities", {}))
    assert rec["model_served"] == "jev-1.14.0" and rec["model_requested"] == "jev-1.13.0"
    assert server.bodies[0]["questions"] == {"q": question.wire_dict()}
    assert server.bodies[0]["model"] == "jev-1.13.0"
    assert outcomes(state) == ["measured"]
    assert state.spent_usd == pytest.approx(40 * PRICE)
    out = tmp_path / "recs.jsonl"
    out.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert read_records(out)[0].model_mismatch


def test_429_retries_after_server_wait_with_a_fresh_reservation(tmp_path: Path) -> None:
    server = Server(httpx2.Response(429, headers={"retry-after-ms": "250"}), ok({"type": "noul", "noul": 0.4}))
    recs, state, sleeps, _ = run(tmp_path, server, max_retries=2)
    assert recs[0]["p"] == 0.4
    assert sleeps == [0.25]
    assert outcomes(state) == ["unknown", "measured"]
    assert state.outstanding == []


def test_persistent_5xx_ends_in_a_no_answer_record(tmp_path: Path) -> None:
    server = Server(httpx2.Response(503))
    recs, state, sleeps, journal = run(tmp_path, server, max_retries=2)
    assert len(server.bodies) == 3
    assert sleeps == [0.5, 1.0]
    assert outcomes(state) == ["unknown"] * 3
    (rec,) = recs
    assert rec["model_served"] is None and rec["error_code"] == "http_503"
    out = tmp_path / "recs.jsonl"
    out.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert read_records(out)[0].probs == ()


def test_client_error_is_not_retried(tmp_path: Path) -> None:
    server = Server(httpx2.Response(400, json={"detail": "bad"}))
    recs, state, sleeps, _ = run(tmp_path, server, max_retries=2)
    assert len(server.bodies) == 1 and sleeps == []
    assert recs[0]["error_code"] == "http_400"
    assert outcomes(state) == ["unknown"]


@pytest.mark.parametrize(
    "exc, code",
    [(httpx2.ConnectError("refused"), "connect"), (httpx2.ReadTimeout("slow"), "timeout")],
)
def test_transport_failures_charge_the_reservation(tmp_path: Path, exc, code) -> None:
    recs, state, _, _ = run(tmp_path, Server(exc), max_retries=1)
    assert recs[0]["error_code"] == code
    assert outcomes(state) == ["unknown", "unknown"]


def test_retry_after_above_the_limit_ends_retries(tmp_path: Path) -> None:
    server = Server(httpx2.Response(429, headers={"retry-after": "3600"}), ok({"type": "noul", "noul": 0.4}))
    recs, state, sleeps, _ = run(tmp_path, server, max_retries=2)
    assert sleeps == [] and recs[0]["error_code"] == "http_429"
    assert outcomes(state) == ["unknown"]


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "7"}, 7.0),
        ({"retry-after-ms": "junk", "retry-after": "2"}, 2.0),
        ({"retry-after": "-1"}, None),
        ({"retry-after": "nan"}, None),
        ({"retry-after": "not a date"}, None),
        ({"retry-after": "Sun, 25 Oct 2026 12:00:00 +999999999999999999999999"}, None),
        ({}, None),
    ],
)
def test_retry_after_parsing(headers, expected) -> None:
    assert retry_after_s(httpx2.Headers(headers)) == expected


def test_retry_after_http_date(monkeypatch) -> None:
    now = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(backends_mod, "datetime", Frozen)
    when = format_datetime(now + timedelta(seconds=30), usegmt=True)
    assert retry_after_s(httpx2.Headers({"retry-after": when})) == 30.0
    past = format_datetime(now - timedelta(seconds=30), usegmt=True)
    assert retry_after_s(httpx2.Headers({"retry-after": past})) == 0.0


@pytest.mark.parametrize(
    "question, body, code, outcome",
    [
        (NOUL, {"model": "m", "usage": {}, "answers": {"q": {"type": "noul", "noul": 0.5}}}, "invalid_usage", "unknown"),
        (NOUL, {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {}}, "invalid_response", "measured"),
        (NOUL, {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "noul", "noul": 2}}}, "invalid_response", "measured"),
        (
            NOUL,
            {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "choice", "choice": "a", "confidence": 1, "probabilities": {"a": 1}}}},
            "invalid_response",
            "measured",
        ),
        (
            CHOICE,
            {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "choice", "choice": "calm", "confidence": 0.5, "probabilities": {"calm": 0.5, "sad": 0.5}}}},
            "invalid_response",
            "measured",
        ),
        (
            CHOICE,
            {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "choice", "choice": "calm", "confidence": 0.5, "probabilities": {"calm": 0.5, "angry": 0.1}}}},
            "invalid_response",
            "measured",
        ),
        (
            SCORE,
            {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "score", "score": 99, "confidence": 0.5, "legend": {"0": "a", "1": "b", "2": "c"}, "probabilities": {"0": 0.5, "1": 0.5, "2": 0}}}},
            "invalid_response",
            "measured",
        ),
        (
            SCORE,
            {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "score", "score": -1, "confidence": 0.5, "legend": {"0": "a", "1": "b", "2": "c"}, "probabilities": {"0": 0.5, "1": 0.5, "2": 0}}}},
            "invalid_response",
            "measured",
        ),
        (NOUL, {"model": "", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "noul", "noul": 0.5}}}, "invalid_response", "measured"),
        (NOUL, {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}}}, "invalid_response", "unknown"),
        (NOUL, {"model": "m", "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "noul", "noul": "bad"}}}, "invalid_response", "measured"),
        (NOUL, {"model": SECRET, "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "noul", "noul": 0.5}}}, "invalid_response", "measured"),
        (NOUL, {"model": "m" * 129, "usage": {"input_tokens": 9, "output_tokens": 0}, "answers": {"q": {"type": "noul", "noul": 0.5}}}, "invalid_response", "measured"),
    ],
)
def test_unusable_responses_keep_their_measured_usage(tmp_path: Path, question, body, code, outcome) -> None:
    recs, state, sleeps, _ = run(tmp_path, Server(httpx2.Response(200, json=body)), question, max_retries=2)
    assert recs[0]["error_code"] == code and recs[0]["model_served"] is None
    assert outcomes(state) == [outcome] and sleeps == []
    if outcome == "measured":
        assert state.spent_usd == pytest.approx(9 * PRICE)


def test_overrun_reported_inside_an_sdk_rejected_body_still_stops(tmp_path: Path) -> None:
    body = {"model": "m", "usage": {"input_tokens": 10**9, "output_tokens": 0}, "answers": {"q": {"type": "noul", "noul": "bad"}}}
    server = Server(httpx2.Response(200, json=body))
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1000.0) as ledger:
        with pytest.raises(LedgerUnusable):
            run_batch(
                [{"item_id": "i1", "state": "s"}, {"item_id": "i2", "state": "s"}],
                {"q1": NOUL},
                backend=backend(server),
                ledger=ledger,
            )
    assert len(server.bodies) == 1 and ledger_read(journal).overruns


def test_overrun_on_the_last_request_stops_after_keeping_its_record(tmp_path: Path) -> None:
    server = Server(ok({"type": "noul", "noul": 0.9}, tokens=10**9))
    journal = tmp_path / "journal.jsonl"
    emitted: list[dict] = []
    with Ledger(journal, cap_usd=1000.0) as ledger:
        with pytest.raises(LedgerUnusable, match="bound is broken"):
            run_batch(
                [{"item_id": "i1", "state": SECRET}],
                {"q1": NOUL},
                backend=backend(server),
                ledger=ledger,
                on_record=emitted.append,
            )
    assert [r["p"] for r in emitted] == [0.9]
    assert ledger_read(journal).overruns


class _Raising:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def infer(self, state, question, *, model_requested):
        self.calls += 1
        raise self.exc


def test_http_error_usage_is_measured_and_stops_on_overrun(tmp_path: Path) -> None:
    server = Server(httpx2.Response(500, json={"usage": {"input_tokens": 10**9, "output_tokens": 0}}))
    journal = tmp_path / "journal.jsonl"
    emitted: list[dict] = []
    with Ledger(journal, cap_usd=1000.0) as ledger:
        with pytest.raises(LedgerUnusable):
            run_batch(
                [{"item_id": "i1", "state": "s"}, {"item_id": "i2", "state": "s"}],
                {"q1": NOUL},
                backend=backend(server),
                ledger=ledger,
                max_retries=2,
                on_record=emitted.append,
                sleep=lambda s: None,
            )
    assert len(server.bodies) == 1
    assert [r["error_code"] for r in emitted] == ["http_500"]
    assert outcomes(ledger_read(journal)) == ["measured"]


def test_refused_retry_still_records_the_paid_failure(tmp_path: Path) -> None:
    server = Server(httpx2.Response(503))
    journal = tmp_path / "journal.jsonl"
    worst = (token_bound("s") + token_bound(json.dumps(NOUL.wire_dict()))) * 2e-6
    emitted: list[dict] = []
    # The first reservation fits; after it is charged as `unknown`, the retry's does not.
    with Ledger(journal, cap_usd=worst * 1.5) as ledger:
        with pytest.raises(BudgetExceeded):
            run_batch(
                [{"item_id": "i1", "state": "s"}],
                {"q1": NOUL},
                backend=backend(server),
                ledger=ledger,
                max_retries=2,
                on_record=emitted.append,
                sleep=lambda s: None,
            )
    assert [r["error_code"] for r in emitted] == ["http_503"]
    assert outcomes(ledger_read(journal)) == ["unknown"]


def test_interrupt_mid_request_charges_the_reservation(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        with pytest.raises(KeyboardInterrupt):
            run_batch([{"item_id": "i1", "state": "s"}], {"q1": NOUL}, backend=_Raising(KeyboardInterrupt()), ledger=ledger)
    st = ledger_read(journal)
    assert outcomes(st) == ["unknown"] and st.outstanding == []


def test_interrupt_after_the_answer_but_before_settling(tmp_path: Path) -> None:
    class Late:
        """A response whose cost read is interrupted: infer has returned, settle has not begun."""

        model_served = "m"
        answer = {"p": 0.5}
        latency_ms = 1.0

        @property
        def cost_usd(self):
            raise KeyboardInterrupt

    class Answering:
        def infer(self, state, question, *, model_requested):
            return Late()

    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        with pytest.raises(KeyboardInterrupt):
            run_batch([{"item_id": "i1", "state": "s"}], {"q1": NOUL}, backend=Answering(), ledger=ledger)
    st = ledger_read(journal)
    assert outcomes(st) == ["unknown"] and st.outstanding == []


def test_interrupt_inside_settle_is_not_settled_twice(tmp_path: Path, monkeypatch) -> None:
    class Cheap:
        def infer(self, state, question, *, model_requested):
            return BackendResponse(model_served="m", answer={"p": 0.5}, cost_usd=1e-7, latency_ms=1.0)

    with Ledger(tmp_path / "journal.jsonl", cap_usd=1.0) as ledger:
        calls = []

        def settle(reservation, usd=None, *, outcome, **meta):
            calls.append(outcome)
            raise KeyboardInterrupt

        monkeypatch.setattr(ledger, "settle", settle)
        with pytest.raises(KeyboardInterrupt):
            run_batch([{"item_id": "i1", "state": "s"}], {"q1": NOUL}, backend=Cheap(), ledger=ledger)
    assert calls == ["measured"]


def test_unsent_failure_is_charged_nothing(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        recs = run_batch(
            [{"item_id": "i1", "state": "s"}],
            {"q1": NOUL},
            backend=_Raising(BackendError("invalid_request", sent=False)),
            ledger=ledger,
            max_retries=3,
        )
    assert recs[0]["error_code"] == "invalid_request"
    st = ledger_read(journal)
    assert outcomes(st) == ["unsent"] and st.spent_usd == 0


@pytest.mark.parametrize("reached_journal", [False, True])
def test_failed_settlement_sends_nothing_more(tmp_path: Path, monkeypatch, reached_journal) -> None:
    class Cheap:
        calls = 0

        def infer(self, state, question, *, model_requested):
            Cheap.calls += 1
            return BackendResponse(model_served="m", answer={"p": 0.5}, cost_usd=1e-7, latency_ms=1.0)

    journal = tmp_path / "journal.jsonl"
    settles = []
    with Ledger(journal, cap_usd=1.0) as ledger:
        real = ledger.settle

        def broken(*a, **k):
            settles.append(k["outcome"])
            if reached_journal:
                real(*a, **k)
            raise OSError("disk full")

        monkeypatch.setattr(ledger, "settle", broken)
        with pytest.raises(OSError, match="disk full"):
            run_batch(
                [{"item_id": "i1", "state": "s"}, {"item_id": "i2", "state": "s"}],
                {"q1": NOUL},
                backend=Cheap(),
                ledger=ledger,
                max_retries=3,
            )
    assert Cheap.calls == 1 and settles == ["measured"]
    assert outcomes(ledger_read(journal)) == (["measured"] if reached_journal else [])


def test_empty_criteria_rejected_before_any_reservation(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        with pytest.raises(ValueError, match="criteria is empty"):
            run_batch(
                [{"item_id": "i1", "state": "s"}],
                {"ok": NOUL, "bad": ChoiceQuestion(instructions="x", criteria={})},
                backend=_Raising(AssertionError("must not be called")),
                ledger=ledger,
            )
    assert not journal.exists() or ledger_read(journal).settlements == []


def test_server_echo_of_state_never_leaves_the_process(tmp_path: Path, capsys) -> None:
    replies = (
        httpx2.Response(500, json={"detail": SECRET}, headers={"x-typesafe-request-id": "TOP_SECRET_STATE"}),
        httpx2.Response(200, json={"model": "m", "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": {"q": {"type": SECRET}}}),
    )
    echo = Server(*replies)
    for code in ("http_500", "invalid_response"):
        with pytest.raises(BackendError) as info:
            backend(echo).infer(SECRET, NOUL, model_requested="m")
        e = info.value
        assert e.error_code == code
        assert "TOP_SECRET" not in str(e) and e.__cause__ is None and e.__suppress_context__
    echo.replies = list(replies)
    recs, _, _, journal = run(tmp_path, echo, max_retries=1)
    assert "TOP_SECRET" not in json.dumps(recs) + journal.read_text(encoding="utf-8")
    assert "TOP_SECRET" not in "".join(capsys.readouterr())


def test_sdk_logs_during_infer_never_carry_the_state(tmp_path: Path, caplog) -> None:
    import logging

    echo = Server(
        httpx2.Response(500, json={"detail": SECRET}, headers={"x-typesafe-request-id": "TOP_SECRET_STATE"}),
        httpx2.Response(200, json={"model": "m", "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": {"q": {"type": SECRET}}}),
        ok({"type": "noul", "noul": 0.5}),
    )
    b = backend(echo)
    sdk_logger = logging.getLogger("typesafe_sdk")
    with caplog.at_level(logging.DEBUG, logger="typesafe_sdk"):
        for _ in range(3):
            try:
                b.infer(SECRET, NOUL, model_requested="m")
            except BackendError:
                pass
        sdk_logger.warning("outside any request")
    assert "TOP_SECRET" not in caplog.text
    # The filter is scoped to the call: the SDK's logs elsewhere in the process still pass.
    assert [r.getMessage() for r in caplog.records if r.name == "typesafe_sdk"] == ["outside any request"]


# CLI


@pytest.fixture
def probe_files(tmp_path: Path) -> dict[str, Path]:
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": {"q1": NOUL.wire_dict()}}), encoding="utf-8")
    states = tmp_path / "states.jsonl"
    states.write_text(
        "".join(json.dumps({"dataset": "d", "item_id": f"i{n}", "state": SECRET}) + "\n" for n in range(3)),
        encoding="utf-8",
    )
    return {"questions": questions, "states": states, "ledger": tmp_path / "journal.jsonl", "out": tmp_path / "out.jsonl"}


def probe(f: dict[str, Path], *extra: str, out: Path | None = None) -> int:
    return cli_main(
        [
            "probe",
            str(f["questions"]),
            "--states",
            str(f["states"]),
            "-o",
            str(out or f["out"]),
            "--ledger",
            str(f["ledger"]),
            "--no-expand-probes",
            *extra,
        ]
    )


def _one_line_error(capsys) -> str:
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("s1: ")
    return err[0]


@pytest.fixture
def live(monkeypatch) -> Server:
    server = Server(ok({"type": "noul", "noul": 0.6}))
    real = backends_mod.HttpSystemOneBackend

    def with_transport(**kw):
        return real(transport=httpx2.MockTransport(server), **kw)

    monkeypatch.setattr("s1ops.cli.HttpSystemOneBackend", with_transport)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    return server


def test_cli_live_probe_writes_records_and_hides_state(probe_files, live, capsys, monkeypatch) -> None:
    import logging

    monkeypatch.setattr(logging.getLogger("typesafe_sdk"), "level", logging.DEBUG)
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7") == 0
    out = capsys.readouterr()
    assert "TOP_SECRET" not in out.out + out.err
    summary = json.loads(out.out)
    assert summary["n_records"] == 3 and summary["n_failed"] == 0
    assert summary["spent_usd"] == pytest.approx(3 * 40 * 1e-7)
    assert len(read_records(probe_files["out"])) == 3
    assert "TOP_SECRET" not in probe_files["out"].read_text() + probe_files["ledger"].read_text()


def test_cli_live_needs_a_price(probe_files, live, capsys) -> None:
    assert probe(probe_files, "--backend", "systemone") == 1
    assert "--usd-per-input-token" in _one_line_error(capsys)
    assert live.bodies == [] and not probe_files["ledger"].exists()


def test_cli_live_rejects_served_model(probe_files, live, capsys) -> None:
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7", "--served-model", "x") == 1
    assert "--served-model" in _one_line_error(capsys)


def test_cli_live_missing_key_is_one_line(probe_files, live, capsys, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7") == 1
    assert "TYPESAFE_API_KEY" in _one_line_error(capsys)
    assert live.bodies == []


def test_cli_live_without_extra_names_it(probe_files, capsys, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7") == 1
    assert "s1ops[typesafe]" in _one_line_error(capsys)


def test_cli_budget_stop_keeps_paid_records(probe_files, live, capsys) -> None:
    worst = (token_bound(SECRET) + token_bound(json.dumps(NOUL.wire_dict()))) * 2e-6
    live.replies = [ok({"type": "noul", "noul": 0.6}, tokens=400)]  # 4e-5 measured per request, below `worst`
    # The cap admits the first reservation; spent 4e-5 plus the next reservation exceeds it.
    cap = worst + 1e-5
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7", "--cap-usd", repr(cap)) == 1
    line = _one_line_error(capsys)
    assert "1 records kept in" in line
    assert len(probe_files["out"].read_text().splitlines()) == 1


def _alias_cases(f: dict[str, Path]):
    yield f["ledger"], "is the same file as"
    link = f["ledger"].parent / "link.jsonl"
    link.symlink_to(f["ledger"])
    yield link, "is the same file as"
    hard = f["ledger"].parent / "hard.jsonl"
    os.link(f["states"], hard)
    yield hard, "is the same file as"
    yield f["questions"], "is the same file as"
    ledger_hard = f["ledger"].parent / "ledger_hard.jsonl"
    os.link(f["ledger"], ledger_hard)
    # Ledger already refuses a hard-linked journal, before any output is opened.
    yield ledger_hard, "hard links"


def test_cli_output_aliasing_an_input_or_ledger_is_refused(probe_files, capsys) -> None:
    assert probe(probe_files) == 0
    capsys.readouterr()
    for alias, message in _alias_cases(probe_files):
        before = [probe_files[k].read_bytes() for k in ("ledger", "states", "questions")]
        assert probe(probe_files, out=alias) == 1
        assert message in _one_line_error(capsys)
        assert [probe_files[k].read_bytes() for k in ("ledger", "states", "questions")] == before


def test_cli_output_case_alias_of_a_new_ledger(probe_files, capsys) -> None:
    probe_files["ledger"].parent.joinpath("CaseProbe").touch()
    if not probe_files["ledger"].parent.joinpath("caseprobe").exists():
        pytest.skip("case-sensitive filesystem")
    alias = probe_files["ledger"].with_name(probe_files["ledger"].name.upper())
    assert probe(probe_files, out=alias) == 1
    assert "is the same file as" in _one_line_error(capsys)
    assert not alias.exists() and not probe_files["ledger"].exists()


def test_cli_records_reach_disk_as_they_arrive(probe_files, live, capsys) -> None:
    probe_files["out"].write_text("stale\n" * 50, encoding="utf-8")
    seen: list[int] = []
    reply = live.replies[0]

    def counting(request):
        seen.append(len(probe_files["out"].read_text().splitlines()))
        return reply

    live.replies = [counting]
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7") == 0
    # The old output stands until the first record replaces it.
    assert seen == [50, 1, 2]
    assert len(probe_files["out"].read_text().splitlines()) == 3


@pytest.mark.parametrize("ok_lines", [0, 1])
def test_cli_output_failure_keeps_the_partial_output_message(probe_files, live, capsys, monkeypatch, ok_lines) -> None:
    import io

    real_fdopen = os.fdopen

    class Full(io.TextIOWrapper):
        """Fails every flush that has data once `ok_lines` lines are on disk, including on close."""

        lines = 0
        pending = False

        def write(self, text):
            Full.pending = True
            return super().write(text)

        def flush(self):
            if Full.pending and Full.lines >= ok_lines:
                raise OSError(28, "No space left on device")
            if Full.pending:
                Full.lines += 1
                Full.pending = False
            super().flush()

    def fdopen(fd, mode, encoding):
        raw = real_fdopen(fd, mode.replace("w", "wb"))
        return Full(raw, encoding=encoding)

    monkeypatch.setattr("s1ops.cli.os.fdopen", fdopen)
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7") == 1
    line = _one_line_error(capsys)
    assert "No space left on device" in line
    assert "already settled request was not written" in line and f"{ok_lines} records kept in" in line


def test_cli_overflowing_retry_after_never_prints_the_echoed_state(probe_files, live, capsys) -> None:
    live.replies = [
        httpx2.Response(
            503,
            json={"detail": SECRET},
            headers={"retry-after": "Sun, 25 Oct 2026 12:00:00 +999999999999999999999999"},
        )
    ]
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7", "--max-retries", "0") == 0
    out = capsys.readouterr()
    assert "TOP_SECRET" not in out.out + out.err
    assert json.loads(out.out)["n_failed"] == 3


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--worst-usd-per-token", "nan"),
        ("--worst-usd-per-token", "-1"),
        ("--worst-usd-per-token", "0"),
        ("--reserve-overhead-tokens", "nan"),
        ("--reserve-overhead-tokens", "-1"),
        ("--reserve-overhead-tokens", "inf"),
    ],
)
def test_cli_rejects_reservation_inputs_before_touching_files(probe_files, live, capsys, flag, value) -> None:
    probe_files["out"].write_text("keep\n", encoding="utf-8")
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7", flag, value) == 1
    assert flag.lstrip("-").replace("-", "_") in _one_line_error(capsys)
    assert live.bodies == [] and not probe_files["ledger"].exists()
    assert probe_files["out"].read_text() == "keep\n"


@pytest.mark.parametrize(
    "questions, extra",
    [
        ({"q1": {"type": "choice", "instructions": "x", "criteria": {}}}, ()),
        ({"q1": NOUL.wire_dict()}, ("--cap-usd", "1e-9")),
    ],
)
def test_cli_failure_before_any_request_keeps_previous_output(probe_files, live, capsys, questions, extra) -> None:
    probe_files["out"].write_text("previous paid results\n", encoding="utf-8")
    probe_files["questions"].write_text(json.dumps({"questions": questions}), encoding="utf-8")
    assert probe(probe_files, "--backend", "systemone", "--usd-per-input-token", "1e-7", *extra) == 1
    _one_line_error(capsys)
    assert live.bodies == []
    assert probe_files["out"].read_text() == "previous paid results\n"


def test_cli_empty_states_leave_an_empty_output(probe_files, capsys) -> None:
    probe_files["out"].write_text("old\n", encoding="utf-8")
    probe_files["states"].write_text("", encoding="utf-8")
    assert probe(probe_files) == 0
    assert probe_files["out"].read_text() == ""


def test_cli_mock_probe_unchanged(probe_files, capsys) -> None:
    assert probe(probe_files, "--served-model", "jev-9") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_records"] == 3 and summary["n_failed"] == 0
    assert {r.model_served for r in read_records(probe_files["out"])} == {"jev-9"}


def test_budget_exceeded_before_first_call(tmp_path: Path) -> None:
    with Ledger(tmp_path / "j.jsonl", cap_usd=1e-9) as ledger:
        with pytest.raises(BudgetExceeded):
            run_batch([{"item_id": "i1", "state": "s"}], {"q1": NOUL}, backend=_Raising(AssertionError()), ledger=ledger)
