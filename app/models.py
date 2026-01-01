from typing import Optional, Literal, Any
from pydantic import BaseModel, Field


# --- Request/Response Models ---

class SessionCreateRequest(BaseModel):
    """Request body for POST /session.

    Frontend only sends auth_token and optional location/coordinates.
    Backend controls voice, language, and system prompt.
    """
    auth_token: Optional[str] = None  # User's auth token for external APIs
    user_location: Optional[str] = None  # User's current location name
    coordinates: Optional[list[list[float]]] = None  # Route coordinates for weather
    conversation_id: Optional[str] = None  # For resuming conversations
    preload_weather: Optional[bool] = True  # Whether to preload weather into voice session context


class SessionCreateResponse(BaseModel):
    """Response body for POST /session."""
    token: str
    conversation_id: str
    expires_in: int
    websocket_url: str


# --- Maritime Chat Models ---

class ChatMessage(BaseModel):
    """Single message in a chat conversation."""
    role: Literal["user", "assistant", "system"]
    content: str


class MaritimeChatRequest(BaseModel):
    """Request body for POST /maritime-chat.

    Supports text-based chat with tool-augmented responses.
    """
    message: str = Field(..., description="User's message/query")
    auth_token: Optional[str] = Field(
        default=None,
        description="User's auth token for fetching preferences and external API calls"
    )
    conversation_history: Optional[list[ChatMessage]] = Field(
        default=None,
        description="Previous conversation history for context"
    )
    user_location: Optional[str] = Field(
        default=None,
        description="User's current location name"
    )
    coordinates: Optional[list[list[float]]] = Field(
        default=None,
        description="User's current coordinates [[lat, lon]]"
    )
    vessels: Optional[list[dict]] = Field(
        default=None,
        description="User's vessel information [{'make': '', 'model': '', 'year': ''}]"
    )
    # New options for controlling behavior
    include_audio: Optional[bool] = Field(
        default=False,
        description="Whether to include TTS audio in the response"
    )
    audio_voice: Optional[str] = Field(
        default="nova",
        description="Voice for TTS audio (alloy, echo, fable, onyx, nova, shimmer)"
    )
    preload_weather: Optional[bool] = Field(
        default=True,
        description="Whether to preload weather data into context (set to False to disable)"
    )


class ToolCallInfo(BaseModel):
    """Information about a tool call made during chat."""
    tool_name: str
    arguments: dict
    result: dict


class MaritimeChatResponse(BaseModel):
    """Response body for POST /maritime-chat."""
    success: bool
    message: str = Field(..., description="AI assistant's response")
    tool_calls: Optional[list[ToolCallInfo]] = Field(
        default=None,
        description="List of tool calls made to generate the response"
    )
    route_data: Optional[dict] = Field(
        default=None,
        description="Route data if a route was planned (for map rendering)"
    )


class AudioData(BaseModel):
    """Audio data for TTS responses."""
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded audio data"
    )
    audio_format: Optional[str] = Field(
        default="mp3",
        description="Audio format (mp3, opus, aac, flac, wav, pcm)"
    )
    audio_mime_type: Optional[str] = Field(
        default="audio/mpeg",
        description="MIME type of the audio"
    )
    voice: Optional[str] = Field(
        default="nova",
        description="Voice used for TTS"
    )


class StructuredChatResponse(BaseModel):
    """Structured response body for POST /maritime-chat with type field.

    Response structure follows the specification:
    - weather: includes 'report' field with weather details
    - assistance: includes 'local_assistance' array
    - route: includes 'trip_plan' with route and analysis
    - normal: just message text

    Optionally includes audio data when include_audio=True in request.
    """
    response: dict = Field(..., description="Structured response with type field")
    audio: Optional[AudioData] = Field(
        default=None,
        description="TTS audio data (only present when include_audio=True)"
    )


# --- Vessel Details Models ---

class VesselInput(BaseModel):
    """Input for vessel details generation - can be partial or text description."""
    text_description: Optional[str] = Field(
        default=None,
        description="Free-text description of the vessel (e.g., '2018 Catalina 315 sailboat')"
    )
    make: Optional[str] = Field(default=None, description="Vessel manufacturer")
    model: Optional[str] = Field(default=None, description="Vessel model")
    year: Optional[str] = Field(default=None, description="Vessel year")
    vessel_type: Optional[str] = Field(
        default=None,
        description="Type of vessel (e.g., 'sailboat', 'powerboat', 'yacht')"
    )


class AddVesselDetailsRequest(BaseModel):
    """Request body for POST /add_vessel_details."""
    auth_token: str = Field(..., description="User's auth token (required)")
    vessel_input: VesselInput = Field(..., description="Vessel information to process")


class VesselDetails(BaseModel):
    """Complete vessel details generated by LLM."""
    make: str
    model: str
    year: str
    vessel_type: Optional[str] = None
    length_feet: Optional[float] = None
    beam_feet: Optional[float] = None
    draft_feet: Optional[float] = None
    displacement_lbs: Optional[int] = None
    engine_type: Optional[str] = None
    fuel_capacity_gallons: Optional[float] = None
    water_capacity_gallons: Optional[float] = None
    max_speed_knots: Optional[float] = None
    cruising_speed_knots: Optional[float] = None
    sleeping_capacity: Optional[int] = None
    recommended_crew: Optional[int] = None
    safety_features: Optional[list[str]] = None
    suitable_conditions: Optional[str] = None


class AddVesselDetailsResponse(BaseModel):
    """Response body for POST /add_vessel_details."""
    success: bool
    message: str
    vessel_details: Optional[VesselDetails] = None
    vessel_id: Optional[str] = Field(
        default=None,
        description="ID of the vessel if successfully added to user's account"
    )


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
    auth_token: Optional[str] = None  # For external API calls


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
