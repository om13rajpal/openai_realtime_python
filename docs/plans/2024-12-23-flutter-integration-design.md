# Flutter Integration Design - Bridge Connection & UI Redesign

**Date:** 2024-12-23
**Related:** [OpenAI Realtime Bridge Design](./2024-12-23-openai-realtime-bridge-design.md)

---

## Overview

Modify the AiSeaSafe Flutter app to connect to the FastAPI bridge instead of directly to OpenAI, with a complete voice UI redesign following Siri-style interaction patterns.

---

## Architecture

**Current Flow (Direct):**
```
Flutter App → OpenAI Realtime API (API key exposed)
```

**New Flow (Via Bridge):**
```
Flutter App → FastAPI Bridge → OpenAI Realtime API (API key secured)
```

---

## Configuration

### Environment Variables

Using `flutter_dotenv` with environment-based URLs:

| Environment | URL |
|-------------|-----|
| Dev | `http://localhost:8000` |
| Staging | `https://staging-api.aiseasafe.com` |
| Prod | `https://api.aiseasafe.com` |

### Token Management

- **Strategy:** Lazy with caching
- Get token on first use via `POST /session`
- Cache token in memory
- Auto-refresh if server rejects (401/expired)

---

## Session & Conversation Flow

### Session Lifecycle

1. User taps wave
2. Check cached token → if none/expired: `POST /session`
3. Receive `{ token, conversation_id }`
4. Show "Resume" / "New" inline buttons if history exists
5. Connect WebSocket: `ws://bridge/ws?token=xxx`
6. Stream audio bidirectionally
7. User taps wave to stop → close WebSocket

### Conversation Persistence

- Bridge stores conversations for 1 hour
- User chooses via inline UI: "Resume last" or "Start new"

---

## Message Protocol

### Flutter → Bridge

| Type | Purpose | Payload |
|------|---------|---------|
| `audio` | Send audio chunk | `{ "data": "<base64>" }` |
| `commit` | Force end of speech | `{}` |
| `cancel` | Stop AI mid-response | `{}` |
| `close` | End session cleanly | `{}` |

### Bridge → Flutter

| Type | Purpose | Payload |
|------|---------|---------|
| `transcript` | User/AI speech text | `{ "text": "...", "role": "user/assistant" }` |
| `audio` | AI audio response | `{ "data": "<base64>" }` |
| `status` | State changes | `{ "status": "listening/speaking/idle" }` |
| `error` | Error occurred | `{ "message": "...", "code": "..." }` |

---

## UI Design

### Visual Style

- **Theme:** Minimal/Clean (Apple-inspired)
- **Core Element:** Lottie waveform animation
- **Lottie URL:** `https://lottie.host/85e40b76-0c0e-45ae-a330-c436b197b24e/lDbtQSrqFQ.json`

### Interaction States

| State | Wave Position | Wave Size | Text Display |
|-------|---------------|-----------|--------------|
| Idle | Bottom | Small | None |
| Listening | Center | Large, pulsing | "Listening..." |
| AI Speaking | Bottom | Small | Streaming response (center) |
| Re-activated (with history) | Bottom | Medium | Previous response above |

### State Flow Diagram

```
1. IDLE - Wave at bottom (small, decorative)
┌─────────────────────┐
│                     │
│                     │
│                     │
│                     │
│     ∿∿∿∿∿∿∿∿∿      │  ← Tap to start
└─────────────────────┘

2. LISTENING - Wave animates to center, expands
┌─────────────────────┐
│                     │
│    ∿∿∿∿∿∿∿∿∿∿∿     │
│   ∿∿∿∿∿∿∿∿∿∿∿∿∿    │  ← Pulsing, active
│    ∿∿∿∿∿∿∿∿∿∿∿     │
│                     │
└─────────────────────┘

3. AI RESPONDING - Wave shrinks, text takes focus
┌─────────────────────┐
│                     │
│  "The weather will  │
│   be clear with     │
│   calm seas..."     │  ← Streaming text
│                     │
│      ∿∿∿∿∿∿        │  ← Small again
└─────────────────────┘

4. TAP AGAIN (with history) - Wave expands slightly
┌─────────────────────┐
│  "The weather..."   │  ← Previous response
│                     │
│                     │
│    ∿∿∿∿∿∿∿∿∿∿      │  ← Slightly expanded
│   ∿∿∿∿∿∿∿∿∿∿∿∿     │  ← Shows "active"
└─────────────────────┘
```

### Gestures

- **Tap wave** → Start/stop listening
- **Swipe up** → Reveal history drawer
- **Swipe down** → Dismiss history drawer

### History Drawer

- Simple transcript list (You/AI exchanges)
- No timestamps, no grouping
- Clean dividers between exchanges
- Accessible via swipe-up gesture

### Settings

- No settings exposed to users
- Sensible defaults for voice, language, VAD

---

## Error Handling

### Retry Strategy

Auto-retry with exponential backoff:

```
Attempt 1 → fail → wait 1s
Attempt 2 → fail → wait 2s
Attempt 3 → fail → wait 4s
Attempt 4 → fail → Show error to user
```

### Error States

| Error Type | UI Behavior |
|------------|-------------|
| No internet | Wave turns gray, show "No connection" |
| Server unreachable | Retry silently, then "Server unavailable" |
| Token expired | Auto-refresh token, reconnect seamlessly |
| Session rejected | Clear cache, start fresh session |
| OpenAI error (via bridge) | Show "Try again" with wave reset |

### Visual Error Indicator

- Wave animation pauses
- Subtle gray overlay
- Small error text below wave
- Tap wave to retry

---

## File Structure

### Files to Create/Modify

```
lib/
├── services/
│   └── voice/
│       ├── bridge_service.dart        # NEW - Replaces openai_realtime_service.dart
│       ├── session_manager.dart       # NEW - Token caching & session handling
│       ├── audio_stream_service.dart  # KEEP - Minor updates
│       └── realtime_events.dart       # SIMPLIFY - Remove OpenAI-specific events
│
├── controllers/
│   └── voice_controller.dart          # REPLACE - Simpler state management
│
├── screens/
│   └── voice_screen.dart              # REWRITE - New Siri-style UI
│
├── widgets/
│   └── voice/
│       ├── wave_animation.dart        # NEW - Lottie wave with state animations
│       ├── streaming_text.dart        # NEW - Real-time text display
│       └── history_drawer.dart        # NEW - Swipe-up transcript list
│
└── .env                               # UPDATE - Add BRIDGE_URL
```

### Dependencies to Add

```yaml
lottie: ^2.7.0  # For wave animation
```

### Files to Delete

- `openai_realtime_service.dart` (replaced by `bridge_service.dart`)
- All mock mode code

---

## Removed Features

- Mock mode (bridge handles everything)
- Direct OpenAI connection
- Client-side API key exposure
- Settings screen (using defaults)

---

## Ready for Implementation

This design covers:
- Bridge connection architecture
- Token management with caching
- Siri-style voice UI with wave animations
- History drawer with transcripts
- Error handling with retry logic
