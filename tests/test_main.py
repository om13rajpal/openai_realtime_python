import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = MagicMock()
    settings.host = "localhost"
    settings.port = 8000
    settings.openai_api_key = "test-key"
    settings.upstash_redis_url = "redis://test"
    return settings


@pytest.fixture
def client(mock_settings):
    """Create test client with mocked Redis."""
    with patch("app.redis_client.init_redis", new_callable=AsyncMock):
        with patch("app.redis_client.close_redis", new_callable=AsyncMock):
            with patch("app.config.get_settings", return_value=mock_settings):
                from app.main import app
                return TestClient(app)


def test_health_endpoint(client):
    """GET /health should return ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_session_endpoint(client):
    """POST /session should create session and return token."""
    with patch("app.main.create_session", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {
            "token": "sess_abc123",
            "conversation_id": "conv_xyz789",
            "expires_in": 300,
            "websocket_url": "/ws?token=sess_abc123"
        }

        response = client.post("/session", json={
            "voice": "shimmer",
            "instructions": "Be helpful"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["token"] == "sess_abc123"
        assert "websocket_url" in data


def test_create_session_with_defaults(client):
    """POST /session should work with empty body."""
    with patch("app.main.create_session", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {
            "token": "sess_abc123",
            "conversation_id": "conv_xyz789",
            "expires_in": 300,
            "websocket_url": "/ws?token=sess_abc123"
        }

        response = client.post("/session", json={})
        assert response.status_code == 200
