# AiSeaSafe Backend Documentation

## Overview

AiSeaSafe Backend is a FastAPI-based Python application that provides AI-powered maritime safety assistance. It serves as a bridge between the Flutter mobile app and various AI/external services, offering:

- **Text-based chat** with structured AI responses
- **Voice-based real-time conversations** via WebSocket
- **Weather data retrieval** for any location worldwide
- **Route planning** for marine vessels
- **Local maritime assistance** information
- **Text-to-Speech (TTS)** audio generation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Flutter Mobile App                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
         ┌──────────────────┐        ┌──────────────────┐
         │   HTTP REST API  │        │    WebSocket     │
         │  /maritime-chat  │        │      /ws         │
         │  /session        │        │  (Voice Chat)    │
         └────────┬─────────┘        └────────┬─────────┘
                  │                           │
                  └─────────────┬─────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    FastAPI Server     │
                    │      (main.py)        │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OpenAI API  │     │   External API  │     │      Redis      │
│  GPT-4o, TTS  │     │  (NestJS @3000) │     │   (Sessions)    │
└───────────────┘     └─────────────────┘     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
            ┌─────────────┐     ┌─────────────┐
            │   Weather   │     │   User      │
            │     API     │     │  Settings   │
            └─────────────┘     └─────────────┘
```

---

## File Structure

```
aiseasafe/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & endpoints
│   ├── config.py            # Configuration & environment variables
│   ├── models.py            # Pydantic models for requests/responses
│   ├── tools.py             # LLM function calling tools (weather, route, assistance)
│   ├── geocoding.py         # OpenStreetMap Nominatim geocoding service
│   ├── tts_service.py       # OpenAI Text-to-Speech service
│   ├── external_api.py      # External NestJS API client
│   ├── session_manager.py   # Redis session management
│   ├── redis_client.py      # Redis connection management
│   ├── realtime_bridge.py   # OpenAI Realtime API WebSocket bridge
│   ├── response_formatter.py # Response formatting utilities
│   └── system_prompt.py     # Maritime AI system prompt
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_*.py            # Unit tests
├── .env                     # Environment variables (not in repo)
├── requirements.txt         # Python dependencies
└── DOCUMENTATION.md         # This file
```

---

## Configuration

### Environment Variables (.env)

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...                    # Required: OpenAI API key

# Redis Configuration
REDIS_URL=redis://localhost:6379         # Redis connection URL

# Server Configuration
HOST=0.0.0.0                             # Server host
PORT=8000                                # Server port

# External API (NestJS Backend)
EXTERNAL_API_BASE=http://52.5.27.89:3000 # NestJS API base URL

# Session Configuration
SESSION_TTL=3600                         # Session TTL in seconds (default: 1 hour)
CONVERSATION_TTL=3600                    # Conversation TTL in seconds
```

### Configuration Module (config.py)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    redis_url: str = "redis://localhost:6379"
    host: str = "0.0.0.0"
    port: int = 8000
    session_ttl: int = 3600
    conversation_ttl: int = 3600
```

---

## API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

### 2. Maritime Chat (Text)

```http
POST /maritime-chat
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "What's the weather in Miami?",
  "auth_token": "eyJhbGciOiJIUzI1NiIs...",
  "user_location": "Miami, FL",
  "coordinates": [[25.77, -80.19]],
  "vessels": [
    {"make": "Beneteau", "model": "Oceanis 38", "year": "2020"}
  ],
  "conversation_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ],
  "include_audio": true,
  "audio_voice": "nova",
  "preload_weather": false
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `message` | string | Yes | - | User's message |
| `auth_token` | string | No | null | JWT token for external API authentication |
| `user_location` | string | No | null | User's current location name |
| `coordinates` | array | No | null | Route coordinates `[[lat, lon], ...]` |
| `vessels` | array | No | null | User's vessels information |
| `conversation_history` | array | No | null | Previous conversation messages |
| `include_audio` | boolean | No | false | Include TTS audio in response |
| `audio_voice` | string | No | "nova" | TTS voice (alloy, echo, fable, onyx, nova, shimmer) |
| `preload_weather` | boolean | No | true | Preload weather data into AI context |

**Response:**
```json
{
  "response": {
    "message": "Current conditions in Miami show calm seas with light winds...",
    "type": "weather",
    "report": {
      "location": "Miami, FL",
      "conditions": {...},
      "safety_assessment": "Low risk"
    }
  },
  "audio": {
    "audio_base64": "//PkxAAA...",
    "audio_format": "mp3",
    "audio_mime_type": "audio/mpeg",
    "voice": "nova"
  }
}
```

**Response Types:**

| Type | Description | Additional Fields |
|------|-------------|-------------------|
| `weather` | Weather information | `report` object with weather data |
| `assistance` | Local maritime assistance | `local_assistance` array |
| `route` | Route planning | `trip_plan` object with route data |
| `normal` | General conversation | None |

---

### 3. Maritime Chat Streaming

```http
POST /maritime-chat/stream
Content-Type: application/json
```

Same request body as `/maritime-chat`. Returns Server-Sent Events (SSE).

**SSE Event Types:**

```
data: {"type": "status", "status": "processing_tools"}
data: {"type": "tool_start", "tool": "get_marine_weather"}
data: {"type": "tool_complete", "tool": "get_marine_weather", "success": true}
data: {"type": "status", "status": "generating_response"}
data: {"type": "message_delta", "content": "The "}
data: {"type": "message_delta", "content": "weather "}
data: {"type": "complete", "response": {...}, "audio": {...}}
```

---

### 4. Chat with Context

```http
POST /chat/with-context?conversation_id=conv_abc123
Content-Type: application/json
```

Maintains conversation history in Redis across text and voice sessions.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `conversation_id` | string | No | Existing conversation ID to resume |

---

### 5. Create Voice Session

```http
POST /session
Content-Type: application/json
```

**Request Body:**
```json
{
  "auth_token": "eyJhbGciOiJIUzI1NiIs...",
  "user_location": "Miami, FL",
  "coordinates": [[25.77, -80.19]],
  "conversation_id": "conv_abc123",
  "preload_weather": false
}
```

**Response:**
```json
{
  "token": "sess_abc123def456",
  "conversation_id": "conv_xyz789",
  "expires_in": 3600,
  "websocket_url": "/ws?token=sess_abc123def456"
}
```

---

### 6. WebSocket Voice Chat

```
WS /ws?token=sess_abc123def456
```

**Client → Server Messages:**

```json
// Send audio data
{"type": "audio", "data": "base64_encoded_audio"}

// Commit audio buffer (end of speech)
{"type": "commit"}

// Cancel current response
{"type": "cancel"}

// Close session
{"type": "close"}
```

**Server → Client Messages:**

```json
// Session connected
{"type": "session.created"}

// Audio response
{"type": "response.audio.delta", "delta": "base64_audio"}

// Transcript
{"type": "response.text.delta", "delta": "Hello, how can..."}

// Response complete
{"type": "response.done"}
```

---

### 7. Add Vessel Details

```http
POST /add_vessel_details
Content-Type: application/json
```

**Request Body:**
```json
{
  "auth_token": "eyJhbGciOiJIUzI1NiIs...",
  "vessel_input": {
    "make": "Beneteau",
    "model": "Oceanis 38",
    "year": "2020",
    "vessel_type": "sailboat",
    "text_description": "My 38-foot sailing yacht"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Vessel Beneteau Oceanis 38 (2020) added successfully.",
  "vessel_details": {
    "make": "Beneteau",
    "model": "Oceanis 38",
    "year": "2020",
    "vessel_type": "sailboat",
    "length_feet": 38.0,
    "beam_feet": 12.8,
    ...
  },
  "vessel_id": "vessel_123"
}
```

---

## Core Modules

### 1. Tools (tools.py)

The tools module implements LLM function calling for maritime operations.

#### Available Tools:

**get_marine_weather**
```python
async def get_marine_weather(
    location: str,
    coordinates: Optional[list[list[float]]] = None,
    auth_token: Optional[str] = None
) -> dict
```

- Geocodes location using Nominatim (OpenStreetMap)
- Fetches weather from external NestJS API
- Returns weather conditions with safety assessment

**get_local_assistance**
```python
async def get_local_assistance(
    location: str,
    assistance_type: Optional[str] = None,
    auth_token: Optional[str] = None
) -> dict
```

- Uses GPT-4o to generate local maritime assistance info
- Types: marina, fuel, repair, emergency, supplies, general

**plan_and_analyze_marine_route**
```python
async def plan_and_analyze_marine_route(
    origin: str,
    origin_coordinates: list[float],
    destination: str,
    destination_coordinates: list[float],
    vessel_make: Optional[str] = None,
    vessel_model: Optional[str] = None,
    vessel_year: Optional[str] = None,
    auth_token: Optional[str] = None
) -> dict
```

- Uses `searoute` package for marine route calculation
- Fetches weather along route points
- Provides safety assessment and estimated travel time

---

### 2. Geocoding (geocoding.py)

Uses OpenStreetMap Nominatim API for worldwide geocoding.

```python
async def geocode_location(location: str) -> Optional[list[float]]
```

- **Input:** Location name (e.g., "Miami, FL", "Sydney, Australia")
- **Output:** Coordinates as `[longitude, latitude]` (searoute format)
- **Rate Limiting:** 1 request per second (Nominatim policy)
- **No API Key Required**

**Example:**
```python
coords = await geocode_location("Malibu, California")
# Returns: [-118.689423, 34.035591]
```

---

### 3. TTS Service (tts_service.py)

OpenAI Text-to-Speech integration for audio responses.

```python
async def generate_speech_base64(
    text: str,
    voice: str = "nova",
    model: str = "tts-1",
    response_format: str = "mp3"
) -> Optional[str]
```

**Available Voices:**
- `alloy` - Neutral, balanced
- `echo` - Warm, conversational
- `fable` - Expressive, storytelling
- `onyx` - Deep, authoritative
- `nova` - Friendly, upbeat (default)
- `shimmer` - Clear, professional

**Output:** Base64-encoded MP3 audio

---

### 4. External API (external_api.py)

Client for the NestJS backend API.

**Functions:**

```python
async def fetch_user_settings(auth_token: str) -> Optional[dict]
```
- Fetches user preferences, vessels, language settings

```python
async def fetch_weather_data(
    auth_token: str,
    coordinates: list[list[float]]
) -> Optional[dict]
```
- Fetches weather for given coordinates
- Requires valid JWT token

```python
async def build_enriched_instructions(
    auth_token: Optional[str],
    user_location: Optional[str],
    coordinates: Optional[list[list[float]]],
    preload_weather: bool = True
) -> str
```
- Builds AI system prompt with user context
- Optionally preloads weather data into context

---

### 5. Session Manager (session_manager.py)

Redis-based session and conversation management for maintaining state across voice and text interactions.

#### Why Redis?

Redis is used for session management for several key reasons:

1. **Speed**: In-memory storage provides sub-millisecond read/write operations, essential for real-time voice conversations
2. **TTL Support**: Built-in key expiration automatically cleans up stale sessions
3. **Persistence**: Data survives server restarts (configurable)
4. **Scalability**: Supports horizontal scaling with Redis Cluster for production deployments
5. **Atomic Operations**: Ensures data consistency in concurrent WebSocket connections
6. **Cross-Service Access**: Multiple server instances can share session state
7. **Cloud-Ready**: Works with managed services like Upstash Redis for serverless deployments

#### Redis Connection (redis_client.py)

```python
import redis.asyncio as redis

# Global connection pool (singleton pattern)
redis_client: Optional[redis.Redis] = None

async def init_redis() -> redis.Redis:
    """Initialize Redis connection pool."""
    global redis_client
    settings = get_settings()

    redis_client = redis.from_url(
        settings.upstash_redis_url,  # e.g., "redis://localhost:6379"
        encoding="utf-8",
        decode_responses=True
    )

    await redis_client.ping()  # Verify connection
    return redis_client

def get_redis() -> redis.Redis:
    """Get the Redis client instance (must call init_redis first)."""
    if redis_client is None:
        raise RuntimeError("Redis not initialized")
    return redis_client
```

#### Key Patterns

| Pattern | Example | Description |
|---------|---------|-------------|
| `session:{token}` | `session:sess_abc123def456` | Voice session data |
| `conversation:{id}` | `conversation:conv_xyz789abc` | Conversation history |

#### Session Data Structure

```python
class SessionData(BaseModel):
    voice: str              # AI voice (alloy, nova, etc.)
    instructions: str       # Enriched system prompt with user context
    conversation_id: str    # Links to conversation history
    created_at: float       # Unix timestamp
    status: str             # "pending", "active", "closed"
    auth_token: Optional[str]  # JWT for external API calls
```

**Stored in Redis as:**
```json
{
  "voice": "alloy",
  "instructions": "You are a maritime assistant...[User location: Miami]...",
  "conversation_id": "conv_xyz789abc",
  "created_at": 1703980800.123,
  "status": "active",
  "auth_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

#### Conversation Data Structure

```python
class ConversationData(BaseModel):
    history: list[ConversationMessage]  # Message history
    created_at: float                    # Unix timestamp
    last_active: float                   # Last activity timestamp

class ConversationMessage(BaseModel):
    role: str       # "user" or "assistant"
    type: str       # "audio" or "text"
    transcript: str # The actual message content
```

**Stored in Redis as:**
```json
{
  "history": [
    {"role": "user", "type": "audio", "transcript": "What's the weather in Miami?"},
    {"role": "assistant", "type": "audio", "transcript": "Current conditions show..."}
  ],
  "created_at": 1703980800.123,
  "last_active": 1703981400.456
}
```

#### Session Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SESSION LIFECYCLE                             │
└──────────────────────────────────────────────────────────────────────┘

1. CREATE SESSION (POST /session)
   ┌─────────────────────────────────────────────────────────────────┐
   │ Client sends: auth_token, user_location, coordinates            │
   │                                                                  │
   │ Server:                                                          │
   │   1. Generate unique token (sess_xxx) and conversation_id        │
   │   2. Fetch user settings (voice preference) from external API    │
   │   3. Build enriched instructions with user context               │
   │   4. Store SessionData in Redis with TTL (default: 1 hour)       │
   │   5. Create ConversationData in Redis (no TTL while active)      │
   │   6. Return token, conversation_id, websocket_url                │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. CONNECT WEBSOCKET (WS /ws?token=xxx)
   ┌─────────────────────────────────────────────────────────────────┐
   │ Server:                                                          │
   │   1. Validate token exists in Redis                              │
   │   2. Update session status to "active"                           │
   │   3. Connect to OpenAI Realtime API                              │
   │   4. Start bidirectional audio streaming                         │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. DURING CONVERSATION
   ┌─────────────────────────────────────────────────────────────────┐
   │ Each message:                                                    │
   │   1. Add to conversation history in Redis                        │
   │   2. Remove TTL from conversation (keep alive)                   │
   │   3. Update last_active timestamp                                │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. SESSION ENDS (close message or disconnect)
   ┌─────────────────────────────────────────────────────────────────┐
   │ Server:                                                          │
   │   1. Delete session from Redis                                   │
   │   2. Set conversation TTL (1 hour) for potential resume          │
   │   3. OR immediately delete conversation (on explicit close)      │
   └─────────────────────────────────────────────────────────────────┘
```

#### TTL Behavior

| Key Type | TTL | Behavior |
|----------|-----|----------|
| Session | `SESSION_TTL` (default: 3600s) | Set on creation, expires if unused |
| Conversation (active) | None | Persists while session is active |
| Conversation (after close) | `CONVERSATION_TTL` (default: 3600s) | Allows resuming within 1 hour |

#### Session Resumption

Users can resume previous conversations by passing the `conversation_id`:

```python
# Create new session with existing conversation
request = SessionCreateRequest(
    conversation_id="conv_xyz789abc",  # Resume this conversation
    auth_token="eyJhbG...",
    user_location="Miami, FL"
)
```

This enables:
- Switching between voice and text chat seamlessly
- Resuming interrupted conversations
- Maintaining context across sessions

#### Key Functions

```python
async def create_session(request: SessionCreateRequest, base_url: str) -> dict
```
- Generates unique token and conversation_id
- Fetches user preferences (voice) from external API
- Builds enriched system prompt with user context
- Stores session in Redis with TTL
- Creates conversation if new

```python
async def get_session(token: str) -> Optional[SessionData]
```
- Retrieves session by token
- Returns None if expired or not found

```python
async def add_to_conversation(
    conversation_id: str,
    role: str,
    transcript: str,
    msg_type: str = "audio"
) -> None
```
- Appends message to conversation history
- Updates `last_active` timestamp
- Removes TTL to keep conversation alive

```python
async def delete_session(token: str, conversation_id: str) -> None
```
- Deletes session immediately
- Sets TTL on conversation for potential resume

```python
async def delete_conversation(conversation_id: str) -> None
```
- Immediately deletes conversation (explicit close)

---

### 6. Realtime Bridge (realtime_bridge.py)

WebSocket bridge to OpenAI Realtime API for voice conversations.

```python
class RealtimeBridge:
    def __init__(self, voice: str, instructions: str, conversation_id: str)
    async def start(self, on_message: Callable)
    async def send_audio(self, audio_base64: str)
    async def commit_audio(self)
    async def cancel_response(self)
    async def disconnect(self)
```

- Maintains bidirectional WebSocket connection to OpenAI
- Handles audio streaming in both directions
- Manages conversation state

---

### 7. Response Formatter (response_formatter.py)

Formats AI responses into structured JSON.

```python
def format_response_from_tool_calls(
    message: str,
    tool_calls: list,
    route_data: Optional[dict] = None
) -> dict
```

**Response Structure:**
```json
{
  "response": {
    "message": "...",
    "type": "weather|assistance|route|normal",
    "report": {...},           // For weather type
    "local_assistance": [...], // For assistance type
    "trip_plan": {...}         // For route type
  }
}
```

---

## Data Models (models.py)

### Request Models

```python
class MaritimeChatRequest(BaseModel):
    message: str
    auth_token: Optional[str] = None
    user_location: Optional[str] = None
    coordinates: Optional[list[list[float]]] = None
    vessels: Optional[list[dict]] = None
    conversation_history: Optional[list[ChatMessage]] = None
    include_audio: Optional[bool] = False
    audio_voice: Optional[str] = "nova"
    preload_weather: Optional[bool] = True

class SessionCreateRequest(BaseModel):
    auth_token: Optional[str] = None
    user_location: Optional[str] = None
    coordinates: Optional[list[list[float]]] = None
    conversation_id: Optional[str] = None
    preload_weather: Optional[bool] = True
```

### Response Models

```python
class StructuredChatResponse(BaseModel):
    response: dict
    audio: Optional[AudioData] = None

class AudioData(BaseModel):
    audio_base64: Optional[str] = None
    audio_format: Optional[str] = "mp3"
    audio_mime_type: Optional[str] = "audio/mpeg"
    voice: Optional[str] = "nova"

class SessionCreateResponse(BaseModel):
    token: str
    conversation_id: str
    expires_in: int
    websocket_url: str
```

---

## System Prompt (system_prompt.py)

The AI assistant's behavior is defined by a comprehensive system prompt that includes:

1. **Role Definition:** Maritime weather assistant for recreational boating
2. **Location Tracking:** Base location vs. context location logic
3. **Vessel Verification:** How to handle vessel selection
4. **Weather Queries:** When and how to use weather tools
5. **Route Planning:** Two-step confirmation process
6. **Local Assistance:** How to provide service information
7. **Off-Topic Policy:** Redirect non-maritime questions
8. **Safety Focus:** Always prioritize safety in recommendations

---

## Error Handling

### HTTP Errors

| Status | Description |
|--------|-------------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid/expired token |
| 404 | Not Found - Invalid endpoint |
| 500 | Internal Server Error |

### Tool Errors

Tools return structured error responses:
```json
{
  "success": false,
  "error": "Weather API returned status 401",
  "location": "Miami, FL",
  "details": "Unauthorized"
}
```

---

## Running the Server

### Development

```bash
cd aiseasafe
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Testing

```bash
cd aiseasafe
pytest tests/ -v
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Maritime chat
curl -X POST http://localhost:8000/maritime-chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "weather of Miami",
    "user_location": "Miami, FL",
    "include_audio": true
  }'

# Streaming chat
curl -N -X POST http://localhost:8000/maritime-chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "user_location": "Miami, FL"}'
```

---

## Dependencies

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
openai>=1.0.0
httpx>=0.25.0
redis>=5.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
searoute>=1.0.0
```

---

## Security Considerations

1. **JWT Tokens:** All external API calls require valid JWT tokens
2. **CORS:** Currently allows all origins (configure for production)
3. **Rate Limiting:** Consider adding rate limiting for production
4. **Secrets:** Never commit `.env` files or API keys
5. **Input Validation:** Pydantic models validate all inputs

---

## Troubleshooting

### Weather API Returns 401

- **Cause:** Expired or invalid JWT token
- **Solution:** Get a fresh token from the authentication endpoint

### Geocoding Fails

- **Cause:** Nominatim rate limiting or invalid location
- **Solution:** Wait 1 second between requests, verify location name

### WebSocket Connection Fails

- **Cause:** Invalid or expired session token
- **Solution:** Create a new session via POST /session

### Redis Connection Error

- **Cause:** Redis server not running
- **Solution:** Start Redis or check REDIS_URL configuration

---

## Changelog

### Version 1.1.0 (December 2025)
- Added OpenStreetMap Nominatim geocoding (worldwide support)
- Added TTS audio output with `include_audio` parameter
- Added `preload_weather` parameter to control weather preloading
- Added detailed logging for debugging
- Fixed Authorization header passing to weather API

### Version 1.0.0 (Initial Release)
- Maritime chat with GPT-4o
- Voice chat via WebSocket
- Weather, route planning, and assistance tools
- Redis session management
