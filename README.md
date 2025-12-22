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
