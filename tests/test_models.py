import pytest
from pydantic import ValidationError


def test_session_create_request_valid():
    """SessionCreateRequest should accept valid data."""
    from app.models import SessionCreateRequest

    req = SessionCreateRequest(
        voice="alloy",
        instructions="You are helpful.",
        conversation_id=None
    )
    assert req.voice == "alloy"
    assert req.instructions == "You are helpful."
    assert req.conversation_id is None


def test_session_create_request_defaults():
    """SessionCreateRequest should have sensible defaults."""
    from app.models import SessionCreateRequest

    req = SessionCreateRequest()
    assert req.voice == "alloy"
    assert req.instructions == ""
    assert req.conversation_id is None


def test_session_create_response():
    """SessionCreateResponse should contain all required fields."""
    from app.models import SessionCreateResponse

    resp = SessionCreateResponse(
        token="sess_abc123",
        conversation_id="conv_xyz789",
        expires_in=300,
        websocket_url="wss://example.com/ws?token=sess_abc123"
    )
    assert resp.token == "sess_abc123"
    assert resp.conversation_id == "conv_xyz789"


def test_websocket_message_audio():
    """WebSocketMessage should handle audio type."""
    from app.models import WebSocketMessage

    msg = WebSocketMessage(type="audio", data="base64encodedaudio==")
    assert msg.type == "audio"
    assert msg.data == "base64encodedaudio=="


def test_websocket_message_status():
    """WebSocketMessage should handle status type."""
    from app.models import WebSocketMessage

    msg = WebSocketMessage(type="status", status="speaking")
    assert msg.type == "status"
    assert msg.status == "speaking"
