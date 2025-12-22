# OpenAI Realtime API Bridge - Design Document

**Date:** 2024-12-23
**Status:** Approved

## Overview

FastAPI WebSocket bridge for OpenAI Realtime API with Flutter client support.

**Stack:**
- FastAPI + WebSockets
- AsyncOpenAI (Realtime API)
- Upstash Redis (sessions + conversations)
- Flutter mobile client

**Key Features:**
- Voice conversations with Server VAD (automatic turn detection)
- Session-based authentication
- Conversation persistence (1-hour TTL on disconnect)
- Bidirectional audio streaming

---

## Architecture

```
┌─────────────┐      POST /session       ┌─────────────────┐
│   Flutter   │ ───────────────────────► │     FastAPI     │
│    App      │ ◄─────────────────────── │     Server      │
│             │   { token, conv_id }     │                 │
│             │                          │                 │
│             │   WebSocket /ws?token=   │                 │      ┌─────────────┐
│             │ ◄───────────────────────►│                 │◄────►│   Upstash   │
│  Audio In   │   Audio + JSON messages  │  realtime_      │      │    Redis    │
│  Audio Out  │                          │  bridge.py      │      └─────────────┘
│             │                          │                 │
└─────────────┘                          │                 │      ┌─────────────┐
                                         │                 │◄────►│   OpenAI    │
                                         │                 │  WS  │  Realtime   │
                                         └─────────────────┘      └─────────────┘
```

---

## Project Structure

```
aiseasafe/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes
│   ├── config.py            # Settings from env
│   ├── models.py            # Pydantic schemas
│   ├── redis_client.py      # Upstash connection
│   ├── session_manager.py   # Session + conversation CRUD
│   └── realtime_bridge.py   # OpenAI WebSocket handler
├── docs/
│   └── plans/
│       └── 2024-12-23-openai-realtime-bridge-design.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/session` | Create session, get token |
| `GET` | `/health` | Health check |
| `WS` | `/ws?token=xxx` | WebSocket connection |

---

## Message Protocol

### Flutter → Server

| Type | Payload | Purpose |
|------|---------|---------|
| `audio` | `{ type, data }` | Base64 PCM16 audio chunk |
| `commit` | `{ type }` | Force response (optional) |
| `cancel` | `{ type }` | Interrupt AI response |
| `close` | `{ type }` | End session, purge data |

### Server → Flutter

| Type | Payload | Purpose |
|------|---------|---------|
| `ready` | `{ type }` | Connection established |
| `audio` | `{ type, data }` | Base64 PCM16 response |
| `transcript.user` | `{ type, text }` | User speech transcription |
| `transcript.assistant` | `{ type, text }` | AI speech transcription |
| `status` | `{ type, status }` | `speaking` / `idle` / `listening` |
| `error` | `{ type, message, code }` | Error occurred |

---

## Session Configuration

### POST /session Request

```json
{
  "voice": "alloy",
  "instructions": "You are a helpful assistant...",
  "conversation_id": null
}
```

### POST /session Response

```json
{
  "token": "sess_xxx",
  "conversation_id": "conv_abc",
  "expires_in": 300,
  "websocket_url": "wss://yourserver.com/ws?token=sess_xxx"
}
```

### Available Voices

| Voice | Description |
|-------|-------------|
| `alloy` | Neutral, balanced |
| `echo` | Warm, conversational |
| `shimmer` | Clear, expressive |
| `ash` | Calm, thoughtful |
| `ballad` | Soft, gentle |
| `coral` | Friendly, upbeat |
| `sage` | Wise, measured |
| `verse` | Dynamic, engaging |

---

## Session Lifecycle

| Event | Action |
|-------|--------|
| WebSocket connects | Session active, conversation stored |
| WebSocket disconnects | Start 1-hour TTL countdown |
| Reconnects within 1 hour | Resume conversation (same conversation_id) |
| 1 hour passes | Auto-purge from Redis |
| Client sends `close` | Immediate purge |

### Redis Keys

```
session:{token}        → TTL: 5 min (for initial connect)
conversation:{id}      → TTL: 1 hour (set on disconnect)
```

---

## Audio Specifications

| Property | Value |
|----------|-------|
| Format | PCM16 signed little-endian |
| Sample rate | 24,000 Hz |
| Channels | Mono |
| Chunk size | ~100ms (4,800 bytes) |
| Encoding | Base64 for JSON transport |

---

## Environment Variables

```
OPENAI_API_KEY=sk-...
UPSTASH_REDIS_URL=rediss://default:xxx@xxx.upstash.io:6379
```

---

## Dependencies

```
fastapi
uvicorn[standard]
websockets
openai
redis[hiredis]
pydantic-settings
python-dotenv
```

---

## Error Handling

| Scenario | Server Action | Client Message |
|----------|---------------|----------------|
| Invalid/expired token | Reject WebSocket | `{"type": "error", "code": "invalid_token"}` |
| OpenAI connection fails | Close WebSocket | `{"type": "error", "code": "openai_connect_failed"}` |
| OpenAI disconnects | Attempt reconnect, then close | `{"type": "error", "code": "openai_disconnect"}` |
| Rate limited | Forward error | `{"type": "error", "code": "rate_limited"}` |
| Malformed message | Ignore, log | Silent drop |

---

## Reconnection Strategy (Flutter)

1. On disconnect → wait 1 second
2. Call `POST /session` with existing `conversation_id`
3. Connect to WebSocket with new token
4. Conversation context restored from Redis
