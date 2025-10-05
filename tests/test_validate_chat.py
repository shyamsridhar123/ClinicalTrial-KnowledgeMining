from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from docintel.validate_chat import ValidationConfig, run_chat_validation


class _StubHTTPResponse:
    def __init__(self, status_code: int, text: str = "{}") -> None:
        self.status_code = status_code
        self.text = text


class _StubHTTPClient:
    def __init__(self, *args, **kwargs) -> None:
        self.calls = []

    async def __aenter__(self) -> "_StubHTTPClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    async def get(self, url: str) -> _StubHTTPResponse:
        self.calls.append(url)
        return _StubHTTPResponse(200, text="{\"data\": []}")


class _StubCompletions:
    def __init__(self, parent: "_StubOpenAI") -> None:
        self.parent = parent
        self.calls = []

    async def create(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="chatcmpl-test",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ACK"))],
        )


class _StubOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        self.chat = SimpleNamespace(completions=_StubCompletions(self))
        self.closed = False

    async def close(self) -> None:  # type: ignore[override]
        self.closed = True


@pytest.mark.asyncio
async def test_run_chat_validation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docintel.validate_chat.httpx.AsyncClient", lambda *a, **k: _StubHTTPClient())
    monkeypatch.setattr("docintel.validate_chat.AsyncOpenAI", lambda *a, **k: _StubOpenAI())

    config = ValidationConfig(
        base_url="http://testserver/v1",
        api_key="EMPTY",
        model="test-model",
        prompt="ping",
        timeout=5.0,
        probe_models=True,
        max_tokens=32,
        temperature=0.0,
    )

    exit_code = await run_chat_validation(config)

    assert exit_code == 0


@pytest.mark.asyncio
async def test_run_chat_validation_handles_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingHTTPClient(_StubHTTPClient):
        async def get(self, url: str) -> _StubHTTPResponse:  # type: ignore[override]
            self.calls.append(url)
            return _StubHTTPResponse(503, text="service unavailable")

    monkeypatch.setattr("docintel.validate_chat.httpx.AsyncClient", lambda *a, **k: _FailingHTTPClient())
    monkeypatch.setattr("docintel.validate_chat.AsyncOpenAI", lambda *a, **k: _StubOpenAI())

    config = ValidationConfig(
        base_url="http://testserver/v1",
        api_key="EMPTY",
        model="test-model",
        prompt="ping",
        timeout=5.0,
        probe_models=True,
        max_tokens=None,
        temperature=0.0,
    )

    exit_code = await run_chat_validation(config)

    assert exit_code == 1
