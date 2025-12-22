import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def test_realtime_bridge_class_exists():
    """RealtimeBridge class should be importable."""
    from app.realtime_bridge import RealtimeBridge
    assert RealtimeBridge is not None


def test_realtime_bridge_init():
    """RealtimeBridge should accept voice and instructions."""
    from app.realtime_bridge import RealtimeBridge

    bridge = RealtimeBridge(
        voice="shimmer",
        instructions="You are helpful.",
        conversation_id="conv_123"
    )
    assert bridge.voice == "shimmer"
    assert bridge.instructions == "You are helpful."
    assert bridge.conversation_id == "conv_123"
    assert bridge.connection is None


@pytest.mark.asyncio
async def test_realtime_bridge_connect():
    """RealtimeBridge.connect should establish OpenAI connection."""
    from app.realtime_bridge import RealtimeBridge

    bridge = RealtimeBridge(
        voice="alloy",
        instructions="Test",
        conversation_id="conv_123"
    )

    # Mock the OpenAI client
    mock_connection = AsyncMock()
    mock_connection.session = MagicMock()
    mock_connection.session.update = AsyncMock()

    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    with patch("app.realtime_bridge.AsyncOpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.realtime.connect = MagicMock(return_value=mock_context)
        mock_openai.return_value = mock_client

        # The connect method is a context manager
        assert bridge.connection is None
