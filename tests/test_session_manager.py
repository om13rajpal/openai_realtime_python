import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import time


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    mock = AsyncMock()
    mock.setex = AsyncMock()
    mock.get = AsyncMock()
    mock.delete = AsyncMock()
    mock.expire = AsyncMock()
    mock.persist = AsyncMock()
    mock.set = AsyncMock()
    return mock


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.session_ttl = 300
    settings.conversation_ttl = 3600
    return settings


@pytest.mark.asyncio
async def test_create_session_generates_token(mock_redis, mock_settings):
    """create_session should generate a unique token."""
    with patch("app.session_manager.get_redis", return_value=mock_redis), \
         patch("app.session_manager.get_settings", return_value=mock_settings):
        from app.session_manager import create_session
        from app.models import SessionCreateRequest

        request = SessionCreateRequest(voice="alloy", instructions="Be helpful")
        result = await create_session(request)

        assert result["token"].startswith("sess_")
        assert result["conversation_id"].startswith("conv_")
        assert result["expires_in"] == 300
        assert "websocket_url" in result


@pytest.mark.asyncio
async def test_get_session_returns_data(mock_redis):
    """get_session should return session data for valid token."""
    session_data = {
        "voice": "alloy",
        "instructions": "Be helpful",
        "conversation_id": "conv_123",
        "created_at": time.time(),
        "status": "pending"
    }
    mock_redis.get = AsyncMock(return_value=json.dumps(session_data))

    with patch("app.session_manager.get_redis", return_value=mock_redis):
        from app.session_manager import get_session

        result = await get_session("sess_abc")
        assert result is not None
        assert result.voice == "alloy"


@pytest.mark.asyncio
async def test_get_session_returns_none_for_invalid(mock_redis):
    """get_session should return None for invalid token."""
    mock_redis.get = AsyncMock(return_value=None)

    with patch("app.session_manager.get_redis", return_value=mock_redis):
        from app.session_manager import get_session

        result = await get_session("invalid_token")
        assert result is None


@pytest.mark.asyncio
async def test_delete_session(mock_redis, mock_settings):
    """delete_session should remove session from Redis."""
    with patch("app.session_manager.get_redis", return_value=mock_redis), \
         patch("app.session_manager.get_settings", return_value=mock_settings):
        from app.session_manager import delete_session

        await delete_session("sess_abc", "conv_123")
        mock_redis.delete.assert_called()
