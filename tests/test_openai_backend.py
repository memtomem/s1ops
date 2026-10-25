"""`OpenAIDecisionsBackend`, its question mapping, and SDK/transport log suppression, fully offline."""

from __future__ import annotations

import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx2
import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion, parse_wire_question, to_openai_decision
from s1ops.ledger.journal import Ledger, LedgerUnusable, read as ledger_read
from s1ops.lint.lint import lint_path
from s1ops.monitor import backends as backends_mod
from s1ops.monitor.backends import QUIET_LOGGERS, HttpSystemOneBackend, OpenAIDecisionsBackend
from s1ops.monitor.runner import BackendError, run_batch

SECRET = "TOP_SECRET_STATE"
PRICE = 1e-7
NOUL = NoulQuestion(instructions="Is this about billing?")
CHOICE = ChoiceQuestion(instructions="Tone?", criteria={"calm": "Even tone.", "angry": None})
SCORE = ScoreQuestion(instructions="Severity?", criteria=["none", "minor", {"label": "major", "description": "Blocks use."}])


def usage(tokens: int = 40, **details) -> dict:
    return {
        "input_tokens": tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, **details},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": tokens,
    }


def decision(answer: dict, *, model: str = "gpt-6-luna-2026-10-06", tokens: int = 40) -> httpx2.Response:
    return httpx2.Response(200, json={"model": model, "answers": [{"name": "q", **answer}], "usage": usage(tokens)})


class Server:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply


def backend(server: Server) -> OpenAIDecisionsBackend:
    return OpenAIDecisionsBackend(
        usd_per_input_token=PRICE,
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
    )


def run(tmp_path: Path, server: Server, question=NOUL, **kw):
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=kw.pop("cap_usd", 1.0)) as ledger:
        recs = run_batch(
            [{"item_id": "i1", "state": SECRET}],
            {"q1": question},
            backend=backend(server),
            ledger=ledger,
            model_requested="gpt-6-luna",
            sleep=lambda s: None,
            **kw,
        )
    return recs, ledger_read(journal), journal


def outcomes(state) -> list[str]:
    return [s["outcome"] for s in state.settlements]


# Question mapping


def test_score_round_trips_through_openai_levels() -> None:
    wire = to_openai_decision("q", SCORE)
    assert wire["levels"] == [{"label": "none"}, {"label": "minor"}, {"label": "major", "description": "Blocks use."}]
    back = parse_wire_question({"type": "score", "name": "q", "instructions": "Severity?", "levels": wire["levels"]})
    assert back == SCORE
    assert to_openai_decision("q", back) == wire


@pytest.mark.parametrize(
    "question, message",
    [
        (NoulQuestion(instructions="x", criteria={"true": "yes"}), "no criteria"),
        (NoulQuestion(instructions={"text": "x"}), "string instructions"),
        (ChoiceQuestion(instructions="x", criteria={"only": None}), "2 to 255"),
        (ChoiceQuestion(instructions="x", criteria={"a": {"rich": 1}, "b": None}), "descriptions must be strings"),
        (ScoreQuestion(instructions="x", criteria=[str(i) for i in range(11)]), "2 to 10"),
        (ScoreQuestion(instructions="x", criteria=["a", ["structured"]]), "score level"),
    ],
)
def test_questions_openai_cannot_take_are_refused(question, message) -> None:
    with pytest.raises(ValueError, match=message):
        to_openai_decision("q", question)


def test_duplicate_openai_choice_values_are_refused(tmp_path: Path) -> None:
    raw = {
        "type": "choice",
        "instructions": "x",
        "choices": [{"value": "A", "description": "first"}, {"value": "A", "description": "second"}, {"value": "B"}],
    }
    with pytest.raises(ValueError, match="duplicate choice value 'A'"):
        parse_wire_question(raw)
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"tone": raw}), encoding="utf-8")
    report = lint_path(path)
    assert [(f.rule, f.message) for f in report.findings if f.severity == "error"] == [("JL000", "duplicate choice value 'A'")]


def test_instruction_parts_pass_through_and_bare_objects_do_not() -> None:
    parts = [{"type": "input_text", "text": "Is this billing?"}, {"type": "input_json", "data": {"amount": 3}}]
    assert to_openai_decision("q", NoulQuestion(instructions=parts))["instructions"] == parts
    for bad in ([], [{"type": "image"}], ["plain"], [{"type": []}], [{"type": {}}]):
        with pytest.raises(ValueError, match="string instructions or a list"):
            to_openai_decision("q", NoulQuestion(instructions=bad))


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "predicate"},
        {"type": "choice", "choices": [{"value": "a"}, {"value": "b"}]},
        {"type": "score", "levels": [{"label": "a"}, {"label": "b"}]},
    ],
)
def test_openai_form_without_instructions_is_a_lint_finding(tmp_path: Path, raw) -> None:
    with pytest.raises(ValueError, match="missing 'instructions'"):
        parse_wire_question(raw)
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"q": raw}), encoding="utf-8")
    errors = [(f.rule, f.message) for f in lint_path(path).findings if f.severity == "error"]
    assert errors == [("JL000", f"{raw['type']} question is missing 'instructions'")]


def test_unsupported_question_is_refused_before_any_reservation(tmp_path: Path) -> None:
    server = Server(decision({"type": "predicate", "probability": 0.5}))
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        with pytest.raises(ValueError, match="'bad'.*no criteria"):
            run_batch(
                [{"item_id": "i1", "state": "s"}],
                {"ok": NOUL, "bad": NoulQuestion(instructions="x", criteria={"true": "yes"})},
                backend=backend(server),
                ledger=ledger,
            )
    assert server.bodies == [] and not journal.exists()


# Responses


@pytest.mark.parametrize(
    "question, answer, expected",
    [
        (NOUL, {"type": "predicate", "probability": 0.8}, {"p": 0.8}),
        (
            CHOICE,
            {
                "type": "choice",
                "choice": "angry",
                "confidence": 0.7,
                "probabilities": [{"value": "angry", "probability": 0.7}, {"value": "calm", "probability": 0.3}],
            },
            {"choice": "angry", "confidence": 0.7, "probabilities": {"calm": 0.3, "angry": 0.7}},
        ),
        (
            SCORE,
            {
                "type": "score",
                "score": 1.2,
                "confidence": 0.6,
                "probabilities": [
                    {"value": 2, "label": "major", "probability": 0.4},
                    {"value": 0, "label": "none", "probability": 0.2},
                    {"value": 1, "label": "minor", "probability": 0.4},
                ],
            },
            {"score": 1.2, "confidence": 0.6, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}},
        ),
    ],
)
def test_answers_map_and_requests_disable_caching(tmp_path: Path, question, answer, expected) -> None:
    server = Server(decision(answer))
    recs, state, _ = run(tmp_path, server, question)
    (rec,) = recs
    assert {k: rec[k] for k in expected} == expected
    assert list(rec.get("probabilities", {})) == list(expected.get("probabilities", {}))
    assert rec["model_served"] == "gpt-6-luna-2026-10-06"
    body = server.bodies[0]
    assert body["input"] == SECRET and body["model"] == "gpt-6-luna"
    assert body["questions"] == [to_openai_decision("q", question)]
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    assert outcomes(state) == ["measured"] and state.spent_usd == pytest.approx(40 * PRICE)


def test_refusal_is_a_measured_no_answer_record(tmp_path: Path) -> None:
    recs, state, _ = run(tmp_path, Server(decision({"type": "refusal"})), max_retries=2)
    assert recs[0]["error_code"] == "refusal" and recs[0]["model_served"] is None
    assert outcomes(state) == ["measured"]


PRED = {"type": "predicate", "probability": 0.5}


@pytest.mark.parametrize(
    "question, body, code, outcome",
    [
        (NOUL, {"model": "m", "answers": [{"name": "q", **PRED}], "usage": usage(9, cache_write_tokens=5)}, "invalid_usage", "unknown"),
        (NOUL, {"model": "m", "answers": [{"name": "q", **PRED}], "usage": {"input_tokens": 9}}, None, "measured"),
        (NOUL, {"model": "m", "answers": [{"name": "q", **PRED}], "usage": usage(9, cached_tokens=9)}, None, "measured"),
        (NOUL, {"model": "m", "answers": [{"name": "q", **PRED}]}, "invalid_usage", "unknown"),
        (NOUL, {"model": "m", "answers": [], "usage": usage(9)}, "invalid_response", "measured"),
        (NOUL, {"model": "m", "answers": [{"name": "q", **PRED}, {"name": "q", **PRED}], "usage": usage(9)}, "invalid_response", "measured"),
        (NOUL, {"model": "m", "answers": [{"name": "other", **PRED}], "usage": usage(9)}, "invalid_response", "measured"),
        (NOUL, {"model": SECRET + " x", "answers": [{"name": "q", **PRED}], "usage": usage(9)}, "invalid_response", "measured"),
        (NOUL, {"model": "m", "answers": [{"name": "q", "type": "predicate", "probability": 2}], "usage": usage(9)}, "invalid_response", "measured"),
        (NOUL, {"model": "m", "answers": [{"name": "q", "type": "predicate", "probability": 10**400}], "usage": usage(9)}, "invalid_response", "measured"),
        (
            CHOICE,
            {"model": "m", "usage": usage(9), "answers": [{"name": "q", "type": "choice", "choice": True, "confidence": 0.5,
                                                         "probabilities": [{"value": True, "probability": 0.5}, {"value": "angry", "probability": 0.5}]}]},
            "invalid_response",
            "measured",
        ),
        (
            CHOICE,
            {"model": "m", "usage": usage(9), "answers": [{"name": "q", "type": "choice", "choice": "calm", "confidence": 0.5,
                                                         "probabilities": [{"value": "calm", "probability": 0.5}, {"value": "calm", "probability": 0.5}]}]},
            "invalid_response",
            "measured",
        ),
        (
            SCORE,
            # Three entries for three levels, but a duplicated index: as a dict this collapses into a valid distribution.
            {"model": "m", "usage": usage(9), "answers": [{"name": "q", "type": "score", "score": 1.0, "confidence": 0.5,
                                                         "probabilities": [{"value": 0, "label": "a", "probability": 0.9},
                                                                           {"value": 0, "label": "a", "probability": 0.2},
                                                                           {"value": 1, "label": "b", "probability": 0.8}]}]},
            "invalid_response",
            "measured",
        ),
        (
            ScoreQuestion(instructions="x", criteria=["low", "high"]),
            # The reviewer's case: three entries for two levels, collapsing to {0: .2, 1: .8} as a dict.
            {"model": "m", "usage": usage(9), "answers": [{"name": "q", "type": "score", "score": 0.8, "confidence": 0.8,
                                                         "probabilities": [{"value": 0, "label": "low", "probability": 0.9},
                                                                           {"value": 0, "label": "low", "probability": 0.2},
                                                                           {"value": 1, "label": "high", "probability": 0.8}]}]},
            "invalid_response",
            "measured",
        ),
        (
            SCORE,
            {"model": "m", "usage": usage(9), "answers": [{"name": "q", "type": "score", "score": 7, "confidence": 0.5,
                                                         "probabilities": [{"value": i, "label": "x", "probability": p} for i, p in enumerate([0.5, 0.5, 0])]}]},
            "invalid_response",
            "measured",
        ),
        (
            SCORE,
            {"model": "m", "usage": usage(9), "answers": [{"name": "q", "type": "score", "score": 1, "confidence": 0.5,
                                                         "probabilities": [{"value": v, "label": "x", "probability": p} for v, p in [(False, 0.5), (1, 0.5), (2, 0)]]}]},
            "invalid_response",
            "measured",
        ),
    ],
)
def test_response_validation(tmp_path: Path, question, body, code, outcome) -> None:
    recs, state, _ = run(tmp_path, Server(httpx2.Response(200, json=body)), question, max_retries=2)
    assert recs[0].get("error_code") == code
    assert outcomes(state) == [outcome]


def test_non_json_success_body_is_invalid(tmp_path: Path) -> None:
    recs, state, _ = run(tmp_path, Server(httpx2.Response(200, content=b"<html>oops</html>")))
    assert recs[0]["error_code"] == "invalid_response" and outcomes(state) == ["unknown"]


@pytest.mark.parametrize(
    "reply",
    [
        httpx2.Response(503, content=b"<html>" + SECRET.encode() + b"</html>", headers={"content-type": "text/html"}),
        httpx2.Response(503, content=b"\xff\xfe\xfa" + SECRET.encode(), headers={"content-type": "application/json"}),
        httpx2.Response(503, json={"error": {"message": SECRET}}),
    ],
)
def test_unreadable_or_bare_error_bodies_keep_retrying(tmp_path: Path, reply) -> None:
    server = Server(reply, decision({"type": "predicate", "probability": 0.4}))
    recs, state, _ = run(tmp_path, server, max_retries=2)
    assert recs[0]["p"] == 0.4
    assert outcomes(state) == ["unknown", "measured"]


def test_error_with_top_level_usage_is_measured_and_overrun_stops(tmp_path: Path) -> None:
    body = {"error": {"message": "server error"}, "usage": usage(10**9)}
    server = Server(httpx2.Response(500, json=body))
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1000.0) as ledger:
        with pytest.raises(LedgerUnusable):
            run_batch(
                [{"item_id": "i1", "state": "s"}, {"item_id": "i2", "state": "s"}],
                {"q1": NOUL},
                backend=backend(server),
                ledger=ledger,
                max_retries=2,
                sleep=lambda s: None,
            )
    assert len(server.bodies) == 1 and outcomes(ledger_read(journal)) == ["measured"]


@pytest.mark.parametrize("status", [200, 500])
def test_absurd_token_counts_never_escape_as_overflow(tmp_path: Path, status) -> None:
    body = {"model": "m", "answers": [{"name": "q", **PRED}], "usage": usage(10**400), "error": {"message": SECRET}}
    server = Server(httpx2.Response(status, json=body))
    with pytest.raises(BackendError) as info:
        backend(server).infer("s", NOUL, model_requested="gpt-6-luna")
    e = info.value
    assert e.error_code == ("invalid_usage" if status == 200 else "http_500")
    assert e.usage_usd is None and SECRET not in str(e) and e.__cause__ is None
    assert e.__context__ is None or e.__suppress_context__
    assert e.retryable is (status == 500)


def test_429_waits_for_retry_after(tmp_path: Path) -> None:
    sleeps: list[float] = []
    server = Server(httpx2.Response(429, headers={"retry-after-ms": "300"}), decision(PRED))
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        recs = run_batch([{"item_id": "i1", "state": "s"}], {"q1": NOUL}, backend=backend(server), ledger=ledger, max_retries=1, sleep=sleeps.append)
    assert sleeps == [0.3] and recs[0]["p"] == 0.5
    assert outcomes(ledger_read(journal)) == ["unknown", "measured"]


@pytest.mark.parametrize(
    "exc, code",
    [(httpx2.ConnectError("refused"), "connect"), (httpx2.ReadTimeout("slow"), "timeout")],
)
def test_transport_failures(tmp_path: Path, exc, code) -> None:
    recs, state, _ = run(tmp_path, Server(exc))
    assert recs[0]["error_code"] == code and outcomes(state) == ["unknown"]


def test_error_messages_never_carry_server_text() -> None:
    server = Server(httpx2.Response(400, json={"error": {"message": SECRET}}, headers={"x-request-id": SECRET}))
    with pytest.raises(BackendError) as info:
        backend(server).infer("s", NOUL, model_requested="gpt-6-luna")
    assert SECRET not in str(info.value) and info.value.__cause__ is None and info.value.__suppress_context__


def test_timeout_none_keeps_the_sdk_default() -> None:
    default = OpenAIDecisionsBackend(usd_per_input_token=PRICE, api_key="k")
    assert default._client.timeout is not None
    assert OpenAIDecisionsBackend(usd_per_input_token=PRICE, api_key="k", timeout=3.0)._client.timeout == 3.0


# Logs, through a real httpx2/httpcore2 transport


class _Echo(BaseHTTPRequestHandler):
    """Reflects the state into the reason phrase, a request-id header and the body."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.endswith("/v1/decisions"):
            body = {"model": "m", "answers": [{"name": "q", "type": SECRET}], "usage": usage(1)}
        else:
            body = {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": {"q": {"type": SECRET}}}
        out = json.dumps(body).encode()
        self.send_response(200, SECRET)
        for name in ("x-request-id", "x-typesafe-request-id"):
            self.send_header(name, SECRET)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def echo_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _live_backends(echo_url: str):
    return [
        OpenAIDecisionsBackend(usd_per_input_token=PRICE, api_key="k", base_url=f"{echo_url}/v1"),
        HttpSystemOneBackend(usd_per_input_token=PRICE, api_key="k", base_url=echo_url, transport=httpx2.HTTPTransport()),
    ]


def _infer_all(backends) -> None:
    for b in backends:
        with pytest.raises(BackendError):
            b.infer(SECRET, NOUL, model_requested="m")


def test_sdk_and_transport_logs_during_infer_never_carry_the_state(echo_url, caplog) -> None:
    backends = _live_backends(echo_url)
    with caplog.at_level(logging.DEBUG):
        _infer_all(backends)
        logging.getLogger("httpx2").info("outside any request")
    assert SECRET not in caplog.text
    # The filter is scoped to the call: the same loggers still pass records outside it.
    assert [r.getMessage() for r in caplog.records if r.name in QUIET_LOGGERS] == ["outside any request"]


def test_every_listed_logger_is_filtered_before_its_module_loads() -> None:
    # httpcore2.http2 and httpcore2.socks exist only after a lazy import; they must already carry
    # the filter, so that the import receives the filtered logger.
    OpenAIDecisionsBackend(usd_per_input_token=PRICE, api_key="k")
    assert {"httpcore2.http2", "httpcore2.socks"} <= set(QUIET_LOGGERS)
    for name in QUIET_LOGGERS:
        assert backends_mod._DROP_DURING_INFER in logging.getLogger(name).filters, name


def test_every_logger_that_speaks_during_a_request_is_quieted(echo_url, caplog, monkeypatch) -> None:
    # Without the filter, this is what a request logs and that the state would reach; a library
    # upgrade that logs from a new logger fails here until QUIET_LOGGERS names it.
    monkeypatch.setattr(backends_mod._DropDuringInfer, "filter", lambda self, record: True)
    backends = _live_backends(echo_url)
    with caplog.at_level(logging.DEBUG):
        _infer_all(backends)
    speakers = {r.name for r in caplog.records}
    assert SECRET in caplog.text  # the threat is real, not hypothetical
    assert {"httpx2", "httpcore2.http11", "openai._base_client", "typesafe_sdk"} <= speakers
    assert speakers <= set(QUIET_LOGGERS), sorted(speakers - set(QUIET_LOGGERS))


# CLI


@pytest.fixture
def probe_files(tmp_path: Path) -> dict[str, Path]:
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": {"q1": NOUL.wire_dict()}}), encoding="utf-8")
    states = tmp_path / "states.jsonl"
    states.write_text(json.dumps({"item_id": "i1", "state": SECRET}) + "\n", encoding="utf-8")
    return {"questions": questions, "states": states, "ledger": tmp_path / "journal.jsonl", "out": tmp_path / "out.jsonl"}


def probe(f: dict[str, Path], *extra: str) -> int:
    return cli_main(
        ["probe", str(f["questions"]), "--states", str(f["states"]), "-o", str(f["out"]), "--ledger", str(f["ledger"]), *extra]
    )


@pytest.fixture
def live(monkeypatch) -> Server:
    server = Server(decision(PRED))
    real = backends_mod.OpenAIDecisionsBackend

    def with_client(**kw):
        return real(http_client=httpx2.Client(transport=httpx2.MockTransport(server)), **kw)

    monkeypatch.setattr("s1ops.cli.OpenAIDecisionsBackend", with_client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return server


def _one_line_error(capsys) -> str:
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("s1: ")
    return err[0]


def test_cli_openai_probe_defaults_to_its_model(probe_files, live, capsys) -> None:
    assert probe(probe_files, "--backend", "openai", "--usd-per-input-token", "1e-7", "--no-expand-probes") == 0
    assert json.loads(capsys.readouterr().out)["n_records"] == 1
    assert [b["model"] for b in live.bodies] == ["gpt-6-luna"]
    assert SECRET not in probe_files["out"].read_text() + probe_files["ledger"].read_text()


def test_cli_openai_probe_expands_probes(probe_files, live, capsys) -> None:
    assert probe(probe_files, "--backend", "openai", "--usd-per-input-token", "1e-7") == 0
    assert [b["questions"][0]["type"] for b in live.bodies] == ["predicate", "choice", "predicate"]


def test_cli_openai_refuses_an_unsupported_question_before_the_ledger(probe_files, live, capsys) -> None:
    probe_files["questions"].write_text(
        json.dumps({"questions": {"q1": {"type": "noul", "instructions": "x", "criteria": {"true": "yes"}}}}), encoding="utf-8"
    )
    assert probe(probe_files, "--backend", "openai", "--usd-per-input-token", "1e-7") == 1
    assert "no criteria" in _one_line_error(capsys)
    assert live.bodies == [] and not probe_files["ledger"].exists()


def test_cli_openai_missing_key_is_one_line(probe_files, live, capsys, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    assert probe(probe_files, "--backend", "openai", "--usd-per-input-token", "1e-7") == 1
    assert "OPENAI_API_KEY" in _one_line_error(capsys)


def test_cli_openai_without_extra_names_it(probe_files, capsys, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    assert probe(probe_files, "--backend", "openai", "--usd-per-input-token", "1e-7") == 1
    assert "s1ops[openai]" in _one_line_error(capsys)


def test_cli_mock_and_systemone_keep_their_default_model(probe_files, capsys) -> None:
    assert probe(probe_files, "--no-expand-probes") == 0
    assert {json.loads(line)["model_requested"] for line in probe_files["out"].read_text().splitlines()} == {"jev-1.13.0"}
