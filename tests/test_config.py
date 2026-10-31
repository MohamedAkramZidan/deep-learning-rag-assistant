import pytest
from pydantic import ValidationError

from rag_app.core.config import Settings


def test_settings_accept_valid_values():
    settings = Settings(
        qdrant_url="https://example.qdrant.io",
        qdrant_api_key="test-api-key",
    )

    assert str(settings.qdrant_url) == "https://example.qdrant.io"
    assert settings.qdrant_api_key.get_secret_value() == "test-api-key"


def test_settings_require_qdrant_credentials(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)