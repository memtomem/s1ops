"""A live probe names its model: `model_served` is what the server reports, and an echoing server reports what was sent."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.questions import NoulQuestion
from s1ops.contracts.records import read_records
from s1ops.ledger.journal import Ledger
from s1ops.monitor import backends as backends_mod
from s1ops.monitor.runner import DeterministicMockBackend, run_batch


def _echo_systemone(model: str) -> dict:
    return {"model": model, "usage": {"input_tokens": 40, "output_tokens": 1}, "answers": {"q": {"type": "noul", "noul": 0.6}}}


def _echo_openai(model: str) -> dict:
    return {
        "model": model,
        "answers": [{"name": "q", "type": "predicate", "probability": 0.6}],
        "usage": {"input_tokens": 40, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 0}},
    }


@pytest.fixture
def echoing(monkeypatch) -> list[str]:
    """Both live backends talk to a server that returns the requested name; the list holds what it was sent."""
    sent: list[str] = []

    def handler(reply):
        def handle(request: httpx2.Request) -> httpx2.Response:
            model = json.loads(request.content)["model"]
            sent.append(model)
            return httpx2.Response(200, json=reply(model))

        return handle

    systemone, openai = backends_mod.HttpSystemOneBackend, backends_mod.OpenAIDecisionsBackend
    monkeypatch.setattr(
        "s1ops.cli.HttpSystemOneBackend",
        lambda **kw: systemone(api_key="test-key", transport=httpx2.MockTransport(handler(_echo_systemone)), **kw),
    )
    monkeypatch.setattr(
        "s1ops.cli.OpenAIDecisionsBackend",
        lambda **kw: openai(api_key="test-key", http_client=httpx2.Client(transport=httpx2.MockTransport(handler(_echo_openai))), **kw),
    )
    return sent


def _probe(tmp_path: Path, *extra: str) -> tuple[int, Path, Path]:
    """Output and journal sit in a directory that only an opened ledger or output would create."""
    q, states = tmp_path / "q.json", tmp_path / "s.jsonl"
    q.write_text(json.dumps({"q1": {"type": "noul", "instructions": "Is this about billing?"}}), encoding="utf-8")
    states.write_text(json.dumps({"dataset": "d", "item_id": "i1", "state": "s"}) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    out, ledger = run / "out.jsonl", run / "journal.jsonl"
    rc = cli_main(["probe", str(q), "--states", str(states), "-o", str(out), "--ledger", str(ledger), "--no-expand-probes", *extra])
    return rc, out, ledger


def _models(out: Path) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return [(row["model_requested"], row["model_served"]) for row in rows]


@pytest.mark.parametrize("backend", ["systemone", "openai"])
@pytest.mark.parametrize("given", [(), ("--model", "")], ids=["absent", "empty"])
def test_live_probe_without_a_model_is_refused_before_anything_exists(tmp_path, echoing, capsys, backend, given) -> None:
    rc, _, ledger = _probe(tmp_path, "--backend", backend, "--usd-per-input-token", "1e-7", *given)
    assert rc == 1
    assert capsys.readouterr().err.strip().splitlines() == [
        f"s1: --backend {backend} needs --model: there is no default, since a server that echoes the request would report it as the model that answered"
    ]
    assert echoing == []
    assert not ledger.parent.exists()  # so no journal, `.lock` or output either


@pytest.mark.parametrize("backend", ["systemone", "openai"])
def test_live_probe_records_the_name_an_echoing_server_returns(tmp_path, echoing, capsys, backend) -> None:
    rc, out, _ = _probe(tmp_path, "--backend", backend, "--usd-per-input-token", "1e-7", "--model", "kev-0.8b")
    assert rc == 0
    assert echoing == ["kev-0.8b"]
    assert _models(out) == [("kev-0.8b", "kev-0.8b")]


def test_mock_keeps_its_default_model(tmp_path, capsys) -> None:
    rc, out, _ = _probe(tmp_path)
    assert rc == 0
    assert _models(out) == [("jev-1.13.0", "jev-1.13.0")]


def test_mock_takes_a_given_model(tmp_path, capsys) -> None:
    rc, out, _ = _probe(tmp_path, "--model", "kev-1.0", "--served-model", "kev-1.1")
    assert rc == 0
    assert _models(out) == [("kev-1.0", "kev-1.1")]


@pytest.mark.parametrize("given", [(), ("--served-model", "m")], ids=["alone", "with-served-model"])
def test_mock_refuses_an_empty_model_before_anything_exists(tmp_path, capsys, given) -> None:
    rc, _, ledger = _probe(tmp_path, "--model", "", *given)
    assert rc == 1
    assert capsys.readouterr().err.strip().splitlines() == ["s1: --model must not be empty"]
    assert not ledger.parent.exists()  # so no journal, `.lock` or output either


@pytest.mark.parametrize("value", ["", None, 7, b"m"], ids=["empty", "none", "int", "bytes"])
def test_run_batch_refuses_a_name_read_records_would_refuse(tmp_path, value) -> None:
    class NeverCalled(DeterministicMockBackend):
        def infer(self, *args, **kwargs):
            raise AssertionError("no request may be sent")

    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        with pytest.raises(ValueError) as refused:
            run_batch(
                [{"item_id": "i1", "state": "s"}],
                {"q1": NoulQuestion(instructions="x?")},
                backend=NeverCalled(),
                ledger=ledger,
                model_requested=value,
            )
    assert str(refused.value) == f"model_requested must be a non-empty string, got {value!r}"
    assert not journal.exists()


def test_run_batch_records_read_back_under_the_rule_it_checks(tmp_path) -> None:
    out = tmp_path / "out.jsonl"
    with Ledger(tmp_path / "journal.jsonl", cap_usd=1.0) as ledger:
        rows = run_batch(
            [{"item_id": "i1", "state": "s"}],
            {"q1": NoulQuestion(instructions="x?")},
            backend=DeterministicMockBackend(served_model="m"),
            ledger=ledger,
            model_requested=" ",
        )
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert [r.raw["model_requested"] for r in read_records(out)] == [" "]


def test_run_batch_needs_a_model_and_reserves_nothing_without_one(tmp_path) -> None:
    journal = tmp_path / "journal.jsonl"
    with Ledger(journal, cap_usd=1.0) as ledger:
        with pytest.raises(TypeError, match="model_requested"):
            run_batch(
                [{"item_id": "i1", "state": "s"}],
                {"q1": NoulQuestion(instructions="x?")},
                backend=DeterministicMockBackend(),
                ledger=ledger,
            )
    assert not journal.exists()
