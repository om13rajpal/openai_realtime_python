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
