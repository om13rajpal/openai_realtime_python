# AiSeaSafe Backend

AI-powered maritime safety and weather assistant backend built with FastAPI and OpenAI GPT-4o.

## Features

- **Text Chat API** - Structured AI responses for weather, routes, and assistance
- **Streaming Chat** - Real-time SSE streaming for progressive responses
- **Voice Chat WebSocket** - Real-time voice conversations via OpenAI Realtime API
- **Weather Data** - Marine weather for any location worldwide (OpenStreetMap geocoding)
- **Route Planning** - Vessel-aware marine route analysis with weather integration
- **Local Assistance** - Maritime service information (marinas, fuel, repairs)
- **Text-to-Speech** - Audio responses via OpenAI TTS API
- **Session Management** - Redis-based session and conversation persistence

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Flutter Mobile App                             │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
    ┌──────────────────┐               ┌──────────────────┐
    │   REST API       │               │    WebSocket     │
    │  /maritime-chat  │               │      /ws         │
    │  /session        │               │  (Voice Chat)    │
    └────────┬─────────┘               └────────┬─────────┘
             │                                  │
             └─────────────┬────────────────────┘
                           │
               ┌───────────▼───────────┐
               │     FastAPI Server    │
               └───────────┬───────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐   ┌────────────────┐   ┌─────────────┐
│ OpenAI API  │   │ External API   │   │    Redis    │
│ GPT-4o, TTS │   │ (Weather Data) │   │ (Sessions)  │
│ Realtime    │   │                │   │             │
└─────────────┘   └────────────────┘   └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (local or Upstash for cloud)
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/aiseasafe-backend.git
cd aiseasafe-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create `.env` file:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key

# Redis (local or Upstash)
UPSTASH_REDIS_URL=redis://localhost:6379

# External API (optional - for weather data)
EXTERNAL_API_BASE=http://your-weather-api:3000

# Server
HOST=0.0.0.0
PORT=8000
SESSION_TTL=3600
CONVERSATION_TTL=3600
```

### Run Server

```bash
# Development (with auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check
```http
GET /health
```

### Text Chat
```http
POST /maritime-chat
Content-Type: application/json

{
  "message": "What's the weather in Miami?",
  "user_location": "Miami, FL",
  "include_audio": true,
  "audio_voice": "nova",
  "preload_weather": false
}
```

### Streaming Chat (SSE)
```http
POST /maritime-chat/stream
Content-Type: application/json

{
  "message": "Plan a route from Miami to Key West",
  "user_location": "Miami, FL"
}
```

**SSE Events:**
```
data: {"type": "status", "status": "processing_tools"}
data: {"type": "tool_start", "tool": "get_marine_weather"}
data: {"type": "tool_complete", "tool": "get_marine_weather", "success": true}
data: {"type": "message_delta", "content": "Current "}
data: {"type": "message_delta", "content": "conditions "}
data: {"type": "complete", "response": {...}, "audio": {...}}
```

### Create Voice Session
```http
POST /session
Content-Type: application/json

{
  "user_location": "Miami, FL",
  "auth_token": "optional-jwt-token",
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

### Voice WebSocket
```
WS /ws?token=sess_abc123def456
```

**Client → Server:**
```json
{"type": "audio", "data": "base64_encoded_pcm16_audio"}
{"type": "commit"}   // End of speech
{"type": "cancel"}   // Cancel AI response
{"type": "close"}    // End session
```

**Server → Client:**
```json
{"type": "ready"}
{"type": "audio", "data": "base64_audio"}
{"type": "transcript", "role": "user", "text": "..."}
{"type": "transcript_delta", "delta": "..."}
{"type": "status", "status": "speaking|listening"}
{"type": "error", "message": "...", "code": "..."}
```

## Response Types

| Type | Description | Data Fields |
|------|-------------|-------------|
| `weather` | Marine weather report | `report` with conditions |
| `route` | Trip plan with analysis | `trip_plan` with waypoints |
| `assistance` | Local maritime services | `local_assistance` array |
| `normal` | General conversation | Message only |

### Example Weather Response

```json
{
  "response": {
    "message": "Current conditions in Miami show calm seas with light winds...",
    "type": "weather",
    "report": {
      "weather": "Clear",
      "temperature": "28",
      "wind_speed": "12",
      "wind_direction": "SE",
      "wave_height": "0.5",
      "wave_direction": "180",
      "visibility": "10",
      "risk_level": "low"
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

### Example Route Response

```json
{
  "response": {
    "message": "I've planned your route from Miami to Key West...",
    "type": "route",
    "trip_plan": {
      "route": {
        "source": {"name": "Miami", "coordinates": [-80.19, 25.77]},
        "destination": {"name": "Key West", "coordinates": [-81.78, 24.55]},
        "route_path": [[-80.19, 25.77], [-80.5, 25.2], ...],
        "distance_nautical_miles": 85.3,
        "total_waypoints": 12
      },
      "trip_analysis": {
        "status": "SAFE",
        "vessel_compatible": true,
        "summary": "Good conditions for sailing...",
        "issues": [],
        "recommendation": "Recommended departure time: 6 AM"
      }
    }
  }
}
```

## Project Structure

```
aiseasafe/
├── app/
│   ├── main.py              # FastAPI application & endpoints
│   ├── config.py            # Configuration & environment variables
│   ├── models.py            # Pydantic models for requests/responses
│   ├── tools.py             # LLM function calling tools
│   ├── geocoding.py         # OpenStreetMap Nominatim geocoding
│   ├── tts_service.py       # OpenAI Text-to-Speech service
│   ├── external_api.py      # External API client (weather)
│   ├── session_manager.py   # Redis session management
│   ├── redis_client.py      # Redis connection pooling
│   ├── realtime_bridge.py   # OpenAI Realtime API WebSocket bridge
│   ├── response_formatter.py # Response formatting utilities
│   └── system_prompt.py     # Maritime AI system prompt
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
├── DOCUMENTATION.md        # Detailed technical documentation
└── README.md               # This file
```

## Key Technologies

### Geocoding (Worldwide)
Uses OpenStreetMap Nominatim API - no API key required:
```python
coords = await geocode_location("Sydney, Australia")
# Returns: [151.209, -33.868]
```

### Session Management (Redis)
Why Redis?
- **Speed**: Sub-millisecond operations for real-time voice
- **TTL**: Automatic session expiration
- **Scalability**: Supports horizontal scaling
- **Persistence**: Data survives restarts

Key patterns:
- `session:{token}` - Voice session data
- `conversation:{id}` - Conversation history

### Function Calling (GPT-4o Tools)
AI uses tools for real data:
- `get_marine_weather` - Weather conditions for any location
- `plan_and_analyze_marine_route` - Marine route with weather analysis
- `get_local_assistance` - Maritime services information

### Audio Format
- PCM 16-bit signed little-endian
- 24,000 Hz sample rate
- Mono channel
- Base64 encoded for JSON transport

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Health check
curl http://localhost:8000/health

# Chat test
curl -X POST http://localhost:8000/maritime-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "weather in Miami", "include_audio": false}'

# Streaming test
curl -N -X POST http://localhost:8000/maritime-chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'
```

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t aiseasafe-backend .
docker run -p 8000:8000 --env-file .env aiseasafe-backend
```

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for:
- Complete API reference with all parameters
- Data models and response structures
- Session lifecycle and Redis key patterns
- System prompt and AI behavior
- Error handling and troubleshooting

## Security Notes

- Never commit `.env` file
- JWT tokens required for external API access
- Session tokens expire after 1 hour
- Rate limiting recommended for production

## License

MIT License
