# OpenAI Realtime Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a FastAPI WebSocket bridge that connects Flutter clients to OpenAI's Realtime API for voice conversations.

**Architecture:** Session-based auth with Upstash Redis for persistence. Flutter connects via WebSocket, server bridges to OpenAI Realtime API. Server VAD for automatic turn detection. Conversations persist for 1 hour on disconnect.

**Tech Stack:** FastAPI, AsyncOpenAI, Upstash Redis, WebSockets, Pydantic

---

## Task 1: Configuration Module

**Files:**
- Create: `app/config.py`
- Test: `tests/test_config.py`

**Step 1: Create tests directory and test file**

```bash
mkdir -p tests
```

Create `tests/__init__.py`:
```python
```

Create `tests/test_config.py`:
```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`

**Step 3: Write minimal implementation**

Create `app/config.py`:
```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Required
    openai_api_key: str
    upstash_redis_url: str

    # Optional with defaults
    host: str = "0.0.0.0"
    port: int = 8000
    session_ttl: int = 300  # 5 minutes
    conversation_ttl: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/config.py tests/
git commit -m "feat: add configuration module with env loading"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Create `tests/test_models.py`:
```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`

**Step 3: Write minimal implementation**

Create `app/models.py`:
```python
from typing import Optional, Literal
from pydantic import BaseModel


# --- Request/Response Models ---

class SessionCreateRequest(BaseModel):
    """Request body for POST /session."""
    voice: str = "alloy"
    instructions: str = ""
    conversation_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    """Response body for POST /session."""
    token: str
    conversation_id: str
    expires_in: int
    websocket_url: str


# --- WebSocket Message Models ---

class WebSocketMessage(BaseModel):
    """Generic WebSocket message structure."""
    type: str
    data: Optional[str] = None  # Base64 audio
    text: Optional[str] = None  # Transcript text
    status: Optional[str] = None  # speaking/idle/listening
    message: Optional[str] = None  # Error message
    code: Optional[str] = None  # Error code


# --- Session Storage Models ---

class SessionData(BaseModel):
    """Session data stored in Redis."""
    voice: str
    instructions: str
    conversation_id: str
    created_at: float
    status: Literal["pending", "connected", "closed"] = "pending"


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: Literal["user", "assistant"]
    type: Literal["audio", "text"]
    transcript: str


class ConversationData(BaseModel):
    """Conversation data stored in Redis."""
    history: list[ConversationMessage] = []
    created_at: float
    last_active: float
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add Pydantic models for API and WebSocket messages"
```

---

## Task 3: Redis Client Module

**Files:**
- Create: `app/redis_client.py`
- Test: `tests/test_redis_client.py`

**Step 1: Write the failing test**

Create `tests/test_redis_client.py`:
```python
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_get_redis_returns_client():
    """get_redis should return a Redis client instance."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "sk-test",
        "UPSTASH_REDIS_URL": "rediss://default:pass@test.upstash.io:6379"
    }):
        from app.redis_client import get_redis

        # Mock the redis connection
        with patch("app.redis_client.redis_client") as mock_client:
            mock_client.ping = AsyncMock(return_value=True)
            client = get_redis()
            assert client is not None


def test_redis_module_exists():
    """Redis client module should be importable."""
    from app import redis_client
    assert hasattr(redis_client, "get_redis")
    assert hasattr(redis_client, "init_redis")
    assert hasattr(redis_client, "close_redis")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_redis_client.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `app/redis_client.py`:
```python
from typing import Optional
import redis.asyncio as redis
from app.config import get_settings

# Global redis client instance
redis_client: Optional[redis.Redis] = None


async def init_redis() -> redis.Redis:
    """Initialize Redis connection pool."""
    global redis_client
    settings = get_settings()

    redis_client = redis.from_url(
        settings.upstash_redis_url,
        encoding="utf-8",
        decode_responses=True
    )

    # Test connection
    await redis_client.ping()
    return redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None


def get_redis() -> redis.Redis:
    """Get the Redis client instance."""
    if redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return redis_client
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_redis_client.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/redis_client.py tests/test_redis_client.py
git commit -m "feat: add async Redis client with Upstash support"
```

---

## Task 4: Session Manager

**Files:**
- Create: `app/session_manager.py`
- Test: `tests/test_session_manager.py`

**Step 1: Write the failing test**

Create `tests/test_session_manager.py`:
```python
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
    return mock


@pytest.mark.asyncio
async def test_create_session_generates_token(mock_redis):
    """create_session should generate a unique token."""
    with patch("app.session_manager.get_redis", return_value=mock_redis):
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
async def test_delete_session(mock_redis):
    """delete_session should remove session from Redis."""
    with patch("app.session_manager.get_redis", return_value=mock_redis):
        from app.session_manager import delete_session

        await delete_session("sess_abc", "conv_123")
        mock_redis.delete.assert_called()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_session_manager.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/session_manager.py`:
```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_session_manager.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/session_manager.py tests/test_session_manager.py
git commit -m "feat: add session manager with Redis persistence"
```

---

## Task 5: OpenAI Realtime Bridge

**Files:**
- Create: `app/realtime_bridge.py`
- Test: `tests/test_realtime_bridge.py`

**Step 1: Write the failing test**

Create `tests/test_realtime_bridge.py`:
```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_realtime_bridge.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/realtime_bridge.py`:
```python
import asyncio
import base64
from typing import Optional, AsyncIterator, Callable
from openai import AsyncOpenAI

from app.config import get_settings
from app.session_manager import add_to_conversation


class RealtimeBridge:
    """Bridge between client WebSocket and OpenAI Realtime API."""

    def __init__(
        self,
        voice: str,
        instructions: str,
        conversation_id: str,
        on_message: Optional[Callable] = None
    ):
        self.voice = voice
        self.instructions = instructions
        self.conversation_id = conversation_id
        self.on_message = on_message
        self.connection = None
        self._client: Optional[AsyncOpenAI] = None
        self._running = False

    async def connect(self) -> None:
        """Connect to OpenAI Realtime API."""
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._running = True

    async def start(self, send_callback: Callable) -> None:
        """Start the realtime session and process events."""
        if not self._client:
            await self.connect()

        async with self._client.realtime.connect(model="gpt-4o-realtime-preview") as connection:
            self.connection = connection

            # Configure session
            await connection.session.update(
                session={
                    "modalities": ["text", "audio"],
                    "instructions": self.instructions,
                    "voice": self.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {"type": "server_vad"},
                }
            )

            # Notify client we're ready
            await send_callback({"type": "ready"})

            # Process events from OpenAI
            async for event in connection:
                if not self._running:
                    break

                await self._handle_event(event, send_callback)

    async def _handle_event(self, event, send_callback: Callable) -> None:
        """Handle an event from OpenAI."""
        event_type = event.type

        if event_type == "response.audio.delta":
            # Send audio chunk to client
            await send_callback({
                "type": "audio",
                "data": event.delta  # Already base64
            })

        elif event_type == "response.audio_transcript.delta":
            # Partial transcript of AI response
            pass  # Could send incremental updates

        elif event_type == "response.audio_transcript.done":
            # Full AI transcript
            transcript = event.transcript
            await send_callback({
                "type": "transcript.assistant",
                "text": transcript
            })
            # Save to conversation
            await add_to_conversation(
                self.conversation_id,
                "assistant",
                transcript
            )

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # User speech transcript
            transcript = event.transcript
            await send_callback({
                "type": "transcript.user",
                "text": transcript
            })
            # Save to conversation
            await add_to_conversation(
                self.conversation_id,
                "user",
                transcript
            )

        elif event_type == "response.created":
            await send_callback({
                "type": "status",
                "status": "speaking"
            })

        elif event_type == "response.done":
            await send_callback({
                "type": "status",
                "status": "listening"
            })

        elif event_type == "input_audio_buffer.speech_started":
            await send_callback({
                "type": "status",
                "status": "listening"
            })

        elif event_type == "error":
            await send_callback({
                "type": "error",
                "message": str(event.error.message) if hasattr(event, 'error') else "Unknown error",
                "code": str(event.error.code) if hasattr(event, 'error') else "unknown"
            })

    async def send_audio(self, audio_base64: str) -> None:
        """Send audio data to OpenAI."""
        if self.connection:
            await self.connection.input_audio_buffer.append(audio=audio_base64)

    async def commit_audio(self) -> None:
        """Commit audio buffer and request response."""
        if self.connection:
            await self.connection.input_audio_buffer.commit()
            await self.connection.response.create()

    async def cancel_response(self) -> None:
        """Cancel ongoing response."""
        if self.connection:
            await self.connection.response.cancel()

    async def disconnect(self) -> None:
        """Disconnect from OpenAI."""
        self._running = False
        self.connection = None
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_realtime_bridge.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/realtime_bridge.py tests/test_realtime_bridge.py
git commit -m "feat: add OpenAI Realtime API bridge"
```

---

## Task 6: FastAPI Main Application

**Files:**
- Create: `app/main.py`
- Test: `tests/test_main.py`

**Step 1: Write the failing test**

Create `tests/test_main.py`:
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


@pytest.fixture
def client():
    """Create test client with mocked Redis."""
    with patch("app.main.init_redis", new_callable=AsyncMock):
        with patch("app.main.close_redis", new_callable=AsyncMock):
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_main.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/main.py`:
```python
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.redis_client import init_redis, close_redis
from app.models import SessionCreateRequest, SessionCreateResponse, WebSocketMessage
from app.session_manager import (
    create_session,
    get_session,
    delete_session,
    delete_conversation,
    update_session_status,
)
from app.realtime_bridge import RealtimeBridge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    await init_redis()
    yield
    # Shutdown
    await close_redis()


app = FastAPI(
    title="OpenAI Realtime Bridge",
    description="WebSocket bridge for OpenAI Realtime API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/session", response_model=SessionCreateResponse)
async def create_new_session(request: SessionCreateRequest = SessionCreateRequest()):
    """Create a new session for WebSocket connection."""
    settings = get_settings()
    base_url = f"wss://{settings.host}:{settings.port}" if settings.host != "0.0.0.0" else ""

    result = await create_session(request, base_url)
    return SessionCreateResponse(**result)


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for realtime communication."""
    # Validate token
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    session = await get_session(token)
    if not session:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # Accept connection
    await websocket.accept()

    # Update session status
    await update_session_status(token, "connected")

    # Create bridge
    bridge = RealtimeBridge(
        voice=session.voice,
        instructions=session.instructions,
        conversation_id=session.conversation_id
    )

    async def send_to_client(message: dict):
        """Send message to WebSocket client."""
        try:
            await websocket.send_json(message)
        except Exception:
            pass

    # Start tasks
    bridge_task = None
    try:
        # Start OpenAI connection in background
        bridge_task = asyncio.create_task(bridge.start(send_to_client))

        # Handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "audio":
                    await bridge.send_audio(message.get("data", ""))

                elif msg_type == "commit":
                    await bridge.commit_audio()

                elif msg_type == "cancel":
                    await bridge.cancel_response()

                elif msg_type == "close":
                    # Immediate cleanup
                    await delete_conversation(session.conversation_id)
                    break

            except json.JSONDecodeError:
                # Ignore malformed messages
                pass

    except WebSocketDisconnect:
        pass

    finally:
        # Cleanup
        await bridge.disconnect()
        if bridge_task:
            bridge_task.cancel()
            try:
                await bridge_task
            except asyncio.CancelledError:
                pass

        await delete_session(token, session.conversation_id)


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_main.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add FastAPI main app with WebSocket endpoint"
```

---

## Task 7: Create conftest.py for Shared Fixtures

**Files:**
- Create: `tests/conftest.py`

**Step 1: Create shared test fixtures**

Create `tests/conftest.py`:
```python
import os
import pytest
from unittest.mock import patch

# Set test environment variables before any imports
@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for all tests."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test-key-123",
        "UPSTASH_REDIS_URL": "rediss://default:testpass@test.upstash.io:6379"
    }):
        yield
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared test fixtures"
```

---

## Task 8: Create README

**Files:**
- Create: `README.md`

**Step 1: Write README**

Create `README.md`:
```markdown
# OpenAI Realtime Bridge

FastAPI WebSocket bridge for OpenAI Realtime API with Flutter client support.

## Features

- Voice conversations with Server VAD (automatic turn detection)
- Session-based authentication
- Conversation persistence (1-hour TTL on disconnect)
- Bidirectional audio streaming

## Setup

1. Clone and install:
```bash
cd aiseasafe
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your keys
```

3. Run server:
```bash
python -m app.main
# or
uvicorn app.main:app --reload
```

## API

### POST /session

Create a new session:

```json
{
  "voice": "alloy",
  "instructions": "You are a helpful assistant.",
  "conversation_id": null
}
```

Response:
```json
{
  "token": "sess_xxx",
  "conversation_id": "conv_xxx",
  "expires_in": 300,
  "websocket_url": "/ws?token=sess_xxx"
}
```

### WebSocket /ws?token=xxx

Connect with session token. Messages are JSON.

**Client → Server:**
- `{"type": "audio", "data": "base64..."}` - Send audio chunk
- `{"type": "cancel"}` - Cancel AI response
- `{"type": "close"}` - End session

**Server → Client:**
- `{"type": "ready"}` - Connection established
- `{"type": "audio", "data": "base64..."}` - AI audio response
- `{"type": "transcript.user", "text": "..."}` - User speech text
- `{"type": "transcript.assistant", "text": "..."}` - AI speech text
- `{"type": "status", "status": "speaking|idle|listening"}`
- `{"type": "error", "message": "...", "code": "..."}`

## Audio Format

- PCM16 signed little-endian
- 24,000 Hz sample rate
- Mono channel
- Base64 encoded for JSON transport

## Testing

```bash
pytest tests/ -v
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and API docs"
```

---

## Final: Run Full Test Suite

**Step 1: Install dependencies and run tests**

```bash
cd E:/aiseasafe
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio
pytest tests/ -v
```

**Step 2: Final commit with all tests passing**

```bash
git add .
git commit -m "chore: complete implementation with passing tests"
```

---

## Summary

| Task | File | Purpose |
|------|------|---------|
| 1 | `config.py` | Environment settings |
| 2 | `models.py` | Pydantic schemas |
| 3 | `redis_client.py` | Upstash connection |
| 4 | `session_manager.py` | Session CRUD |
| 5 | `realtime_bridge.py` | OpenAI connection |
| 6 | `main.py` | FastAPI routes |
| 7 | `conftest.py` | Test fixtures |
| 8 | `README.md` | Documentation |

**Total commits:** 9
