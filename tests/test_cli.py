from __future__ import annotations

from typing import Any, Dict

import pytest

from docintel import cli


def test_run_ingestion_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_prompt_int(*args: Any, **kwargs: Any) -> int:
        return 42

    def fake_run(*, max_studies: int) -> Dict[str, Any]:
        captured["max_studies"] = max_studies
        return {"status": "ok"}

    monkeypatch.setattr(cli, "_prompt_int", fake_prompt_int)
    monkeypatch.setattr(cli.ingest, "run", fake_run)

    report = cli._run_ingestion_interactive()

    assert report == {"status": "ok"}
    assert captured["max_studies"] == 42


def test_run_parsing_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    monkeypatch.setattr(cli, "_prompt_bool", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "_prompt_optional_int", lambda *args, **kwargs: 8)

    def fake_run(*, force_reparse: bool, max_workers: int | None) -> Dict[str, Any]:
        captured["force_reparse"] = force_reparse
        captured["max_workers"] = max_workers
        return {"status": "parsed"}

    monkeypatch.setattr(cli.parse, "run", fake_run)

    report = cli._run_parsing_interactive()

    assert report == {"status": "parsed"}
    assert captured == {"force_reparse": True, "max_workers": 8}


def test_run_embedding_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    monkeypatch.setattr(cli, "_prompt_bool", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "_prompt_optional_int", lambda *args, **kwargs: 64)
    monkeypatch.setattr(cli, "_prompt_choice", lambda *args, **kwargs: "int8")

    def fake_run(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"status": "embedded"}

    monkeypatch.setattr(cli.embed, "run", fake_run)

    report = cli._run_embedding_interactive()

    assert report == {"status": "embedded"}
    assert captured == {
        "force_reembed": True,
        "batch_size": 64,
        "quantization_encoding": "int8",
        "store_float32": True,
    }


def test_run_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    def fake_ingestion() -> Dict[str, Any]:
        order.append("ingestion")
        return {"status": "ingested"}

    def fake_parsing() -> Dict[str, Any]:
        order.append("parsing")
        return {"status": "parsed"}

    def fake_embedding() -> Dict[str, Any]:
        order.append("embedding")
        return {"status": "embedded"}

    monkeypatch.setattr(cli, "_run_ingestion_interactive", fake_ingestion)
    monkeypatch.setattr(cli, "_run_parsing_interactive", fake_parsing)
    monkeypatch.setattr(cli, "_run_embedding_interactive", fake_embedding)

    results = cli._run_full_pipeline()

    assert order == ["ingestion", "parsing", "embedding"]
    assert results == {
        "ingestion": {"status": "ingested"},
        "parsing": {"status": "parsed"},
        "embedding": {"status": "embedded"},
    }
