"""A failing `close()` does not replace the error raised inside a `with backend:` block."""

from __future__ import annotations

import httpx2
import pytest

from s1ops.monitor.backends import HttpSystemOneBackend, OpenAIDecisionsBackend


def _transport() -> httpx2.MockTransport:
    return httpx2.MockTransport(lambda request: httpx2.Response(500))


def _systemone() -> HttpSystemOneBackend:
    return HttpSystemOneBackend(usd_per_input_token=0.0, api_key="k", base_url="https://x.test", transport=_transport())


def _openai() -> OpenAIDecisionsBackend:
    return OpenAIDecisionsBackend(usd_per_input_token=0.0, api_key="k", base_url="https://x.test", http_client=httpx2.Client(transport=_transport()))


BACKENDS = [pytest.param(_systemone, HttpSystemOneBackend, id="systemone"), pytest.param(_openai, OpenAIDecisionsBackend, id="openai")]


def _failing_close(monkeypatch, cls, error: BaseException) -> list[bool]:
    calls: list[bool] = []

    def close(self) -> None:
        calls.append(True)
        raise error

    monkeypatch.setattr(cls, "close", close)
    return calls


@pytest.mark.parametrize("build, cls", BACKENDS)
def test_the_block_error_wins(monkeypatch, build, cls) -> None:
    calls = _failing_close(monkeypatch, cls, RuntimeError("close failed"))
    with pytest.raises(ValueError, match="the caller's error"), build():
        raise ValueError("the caller's error")
    assert calls == [True]


@pytest.mark.parametrize("build, cls", BACKENDS)
def test_a_clean_block_still_reports_the_close_failure(monkeypatch, build, cls) -> None:
    calls = _failing_close(monkeypatch, cls, RuntimeError("close failed"))
    with pytest.raises(RuntimeError, match="close failed"), build():
        pass
    assert calls == [True]


@pytest.mark.parametrize("build, cls", BACKENDS)
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_during_close_is_not_swallowed(monkeypatch, build, cls, interrupt) -> None:
    _failing_close(monkeypatch, cls, interrupt())
    with pytest.raises(interrupt), build():
        raise ValueError("the caller's error")


@pytest.mark.parametrize("build, cls", BACKENDS)
def test_a_close_that_works_changes_nothing(monkeypatch, build, cls) -> None:
    calls: list[bool] = []
    real = cls.close
    monkeypatch.setattr(cls, "close", lambda self: (calls.append(True), real(self))[1])
    with build() as backend:
        assert isinstance(backend, cls)
    assert calls == [True]
    with pytest.raises(ValueError, match="the caller's error"), build():
        raise ValueError("the caller's error")
