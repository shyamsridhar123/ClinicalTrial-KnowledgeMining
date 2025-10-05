from __future__ import annotations

import pytest

from docintel.docling_health import main, run


class _StubClient:
    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy
        self.calls = 0

    async def health_check(self) -> bool:
        self.calls += 1
        return self._healthy


def test_run_returns_true_when_endpoint_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient(True)
    assert run(verbose=False, client=stub)
    assert stub.calls == 1


def test_run_returns_false_when_endpoint_unhealthy() -> None:
    stub = _StubClient(False)
    assert not run(verbose=False, client=stub)
    assert stub.calls == 1


def test_main_exits_with_error_when_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docintel.docling_health.run", lambda verbose=True, client=None: False)
    with pytest.raises(SystemExit) as exc:
        main(["--quiet"])
    assert exc.value.code == 1
