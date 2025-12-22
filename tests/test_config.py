import os
import pytest
from unittest.mock import patch


def test_settings_loads_from_env():
    """Settings should load values from environment variables."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test-key",
        "UPSTASH_REDIS_URL": "rediss://default:pass@test.upstash.io:6379"
    }):
        from app.config import Settings
        settings = Settings()
        assert settings.openai_api_key == "sk-test-key"
        assert settings.upstash_redis_url == "rediss://default:pass@test.upstash.io:6379"


def test_settings_has_defaults():
    """Settings should have sensible defaults for optional values."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test-key",
        "UPSTASH_REDIS_URL": "rediss://default:pass@test.upstash.io:6379"
    }):
        from app.config import Settings
        settings = Settings()
        assert settings.session_ttl == 300
        assert settings.conversation_ttl == 3600
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
