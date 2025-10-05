from __future__ import annotations

from pathlib import Path

import pytest

from docintel import config


@pytest.fixture(autouse=True)
def clear_settings_cache():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_settings_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage_root = tmp_path / "ingestion"
    monkeypatch.setenv("DOCINTEL_STORAGE_ROOT", str(storage_root))

    settings = config.get_settings()

    assert settings.storage_root.resolve() == storage_root.resolve()
    directories = settings.storage_directories()
    for key in ("root", "documents", "metadata", "logs", "temp"):
        assert directories[key].exists()
        assert directories[key].is_dir()

    assert str(settings.clinicaltrials_api_base) == "https://clinicaltrials.gov/api/v2"
    assert settings.max_concurrent_downloads == 5
    assert settings.target_therapeutic_areas == []
    assert settings.target_phases == []
    assert settings.search_query_term is None
    assert settings.search_overfetch_multiplier == 4
    assert settings.desired_page_size(5) == 20