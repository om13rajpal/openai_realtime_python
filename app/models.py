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
