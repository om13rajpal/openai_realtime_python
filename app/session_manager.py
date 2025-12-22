import json
import time
import uuid
from typing import Optional

from app.config import get_settings
from app.models import (
    SessionCreateRequest,
    SessionData,
    ConversationData,
    ConversationMessage,
)
from app.redis_client import get_redis


def _generate_token() -> str:
    """Generate a unique session token."""
    return f"sess_{uuid.uuid4().hex[:16]}"


def _generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return f"conv_{uuid.uuid4().hex[:16]}"


async def create_session(request: SessionCreateRequest, base_url: str = "") -> dict:
    """Create a new session and store in Redis."""
    settings = get_settings()
    redis = get_redis()

    token = _generate_token()
    conversation_id = request.conversation_id or _generate_conversation_id()

    # Create session data
    session = SessionData(
        voice=request.voice,
        instructions=request.instructions,
        conversation_id=conversation_id,
        created_at=time.time(),
        status="pending"
    )

    # Store session with TTL
    await redis.setex(
        f"session:{token}",
        settings.session_ttl,
        session.model_dump_json()
    )

    # Create conversation if new
    if not request.conversation_id:
        conv_data = ConversationData(
            history=[],
            created_at=time.time(),
            last_active=time.time()
        )
        await redis.set(
            f"conversation:{conversation_id}",
            conv_data.model_dump_json()
        )

    ws_url = f"{base_url}/ws?token={token}" if base_url else f"/ws?token={token}"

    return {
        "token": token,
        "conversation_id": conversation_id,
        "expires_in": settings.session_ttl,
        "websocket_url": ws_url
    }


async def get_session(token: str) -> Optional[SessionData]:
    """Get session data by token."""
    redis = get_redis()
    data = await redis.get(f"session:{token}")

    if not data:
        return None

    return SessionData.model_validate_json(data)


async def update_session_status(token: str, status: str) -> None:
    """Update session status."""
    redis = get_redis()
    settings = get_settings()

    data = await redis.get(f"session:{token}")
    if data:
        session = SessionData.model_validate_json(data)
        session.status = status
        await redis.setex(
            f"session:{token}",
            settings.session_ttl,
            session.model_dump_json()
        )


async def delete_session(token: str, conversation_id: str) -> None:
    """Delete session and set conversation TTL."""
    redis = get_redis()
    settings = get_settings()

    # Delete session
    await redis.delete(f"session:{token}")

    # Set conversation TTL (1 hour)
    await redis.expire(f"conversation:{conversation_id}", settings.conversation_ttl)


async def get_conversation(conversation_id: str) -> Optional[ConversationData]:
    """Get conversation history."""
    redis = get_redis()
    data = await redis.get(f"conversation:{conversation_id}")

    if not data:
        return None

    return ConversationData.model_validate_json(data)


async def add_to_conversation(
    conversation_id: str,
    role: str,
    transcript: str,
    msg_type: str = "audio"
) -> None:
    """Add a message to conversation history."""
    redis = get_redis()

    data = await redis.get(f"conversation:{conversation_id}")
    if not data:
        return

    conv = ConversationData.model_validate_json(data)
    conv.history.append(ConversationMessage(
        role=role,
        type=msg_type,
        transcript=transcript
    ))
    conv.last_active = time.time()

    # Remove TTL while conversation is active
    await redis.persist(f"conversation:{conversation_id}")
    await redis.set(f"conversation:{conversation_id}", conv.model_dump_json())


async def delete_conversation(conversation_id: str) -> None:
    """Immediately delete conversation (on close message)."""
    redis = get_redis()
    await redis.delete(f"conversation:{conversation_id}")
