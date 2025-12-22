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
