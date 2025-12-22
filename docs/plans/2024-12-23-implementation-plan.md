# Implementation Plan - OpenAI Realtime Bridge

**Date:** 2024-12-23
**Design:** [2024-12-23-openai-realtime-bridge-design.md](./2024-12-23-openai-realtime-bridge-design.md)

---

## Implementation Order

### Phase 1: Core Infrastructure

1. **`app/config.py`** - Settings management
   - Load environment variables
   - Pydantic Settings model
   - Validation for required vars

2. **`app/redis_client.py`** - Redis connection
   - Async connection pool to Upstash
   - Connection lifecycle (startup/shutdown)
   - Helper functions for common operations

3. **`app/models.py`** - Pydantic schemas
   - SessionCreateRequest / SessionCreateResponse
   - WebSocket message types
   - Conversation history model

### Phase 2: Session Management

4. **`app/session_manager.py`** - Session CRUD
   - `create_session()` - Generate token, store config
   - `get_session()` - Retrieve and validate
   - `delete_session()` - Cleanup
   - `save_conversation()` - Store history
   - `get_conversation()` - Retrieve history

### Phase 3: OpenAI Bridge

5. **`app/realtime_bridge.py`** - OpenAI connection handler
   - `RealtimeBridge` class
   - Connect to OpenAI Realtime API
   - Configure session (voice, instructions, VAD)
   - Handle incoming events from OpenAI
   - Forward audio/events to client
   - Manage conversation context

### Phase 4: WebSocket & Routes

6. **`app/main.py`** - FastAPI application
   - FastAPI app with lifespan (Redis connect/disconnect)
   - `POST /session` - Create session endpoint
   - `GET /health` - Health check
   - `WebSocket /ws` - Main WebSocket handler
   - Bidirectional message routing

### Phase 5: Testing & Polish

7. **Testing**
   - Manual WebSocket testing with websocat/wscat
   - Test session creation and expiry
   - Test reconnection flow

8. **Documentation**
   - README.md with setup instructions
   - Flutter integration example

---

## File Implementation Details

### 1. config.py

```python
# Key functionality:
- class Settings(BaseSettings):
    openai_api_key: str
    upstash_redis_url: str
    host: str = "0.0.0.0"
    port: int = 8000
    session_ttl: int = 300  # 5 min
    conversation_ttl: int = 3600  # 1 hour
```

### 2. redis_client.py

```python
# Key functionality:
- redis_pool: ConnectionPool
- get_redis() -> Redis
- startup_redis() / shutdown_redis()
```

### 3. models.py

```python
# Key models:
- SessionConfig (voice, instructions, conversation_id)
- SessionCreateRequest / SessionCreateResponse
- WebSocketMessage (type, data, text, etc.)
- ConversationHistory
```

### 4. session_manager.py

```python
# Key functions:
- create_session(config) -> (token, conversation_id)
- validate_session(token) -> SessionConfig | None
- delete_session(token)
- update_conversation(conv_id, message)
- get_conversation(conv_id) -> list
- set_conversation_expiry(conv_id, ttl)
```

### 5. realtime_bridge.py

```python
# Key class:
class RealtimeBridge:
    def __init__(self, config: SessionConfig)
    async def connect()
    async def disconnect()
    async def send_audio(data: bytes)
    async def handle_openai_events() -> AsyncIterator[dict]
    async def restore_conversation(history: list)
```

### 6. main.py

```python
# Key endpoints:
@app.post("/session") -> SessionCreateResponse
@app.get("/health") -> {"status": "ok"}
@app.websocket("/ws")
    - Validate token
    - Create RealtimeBridge
    - Run two tasks:
      - client_to_openai: read from WS, forward to OpenAI
      - openai_to_client: read from OpenAI, forward to WS
```

---

## Dependencies Between Files

```
config.py          (standalone)
     ↓
redis_client.py    (imports config)
     ↓
models.py          (standalone)
     ↓
session_manager.py (imports redis_client, models)
     ↓
realtime_bridge.py (imports config, models)
     ↓
main.py            (imports all above)
```

---

## Estimated Effort

| File | Complexity | Lines (approx) |
|------|------------|----------------|
| config.py | Low | ~30 |
| redis_client.py | Low | ~40 |
| models.py | Low | ~60 |
| session_manager.py | Medium | ~100 |
| realtime_bridge.py | High | ~150 |
| main.py | Medium | ~120 |
| **Total** | | **~500** |

---

## Ready to Implement

Start with Phase 1 (config, redis, models) as these are dependencies for everything else.
