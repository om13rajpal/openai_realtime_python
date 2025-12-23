"""
OpenAI Realtime API Bridge for Flutter/Python WebSocket Communication

This module provides a bridge between a Flutter client and OpenAI's Realtime API,
enabling real-time voice conversations with transcription support.

Updated for OpenAI Realtime API GA (General Availability) version.
"""

import asyncio
import logging
import traceback
import json
from typing import Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from openai import AsyncOpenAI

from app.config import get_settings
from app.session_manager import add_to_conversation

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Enum representing the connection states of the Realtime bridge."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CONFIGURING = "configuring"
    READY = "ready"
    ERROR = "error"


@dataclass
class AudioConfig:
    """Configuration for audio input/output settings."""
    input_format: str = "pcm"  # OpenAI expects 'audio/pcm' for 16-bit signed LE PCM
    output_format: str = "pcm"  # OpenAI expects 'audio/pcm' for 16-bit signed LE PCM
    sample_rate: int = 24000
    transcription_model: str = "whisper-1"
    turn_detection_type: str = "server_vad"
    turn_detection_threshold: float = 0.5
    turn_detection_prefix_padding_ms: int = 300
    turn_detection_silence_duration_ms: int = 500


@dataclass
class RealtimeBridgeConfig:
    """Configuration for the Realtime Bridge."""
    voice: str = "alloy"
    instructions: str = ""
    conversation_id: str = ""
    model: str = "gpt-4o-realtime-preview-2024-12-17"  # Use beta model for stability
    audio: AudioConfig = field(default_factory=AudioConfig)
    max_output_tokens: Optional[int] = None


class RealtimeBridge:
    """
    Bridge class for OpenAI Realtime API communication.

    Handles WebSocket connection, session management, audio streaming,
    and event processing for real-time voice conversations.

    Attributes:
        config: Configuration for the bridge
        on_message: Optional callback for custom message handling
        connection: The active OpenAI Realtime connection
    """

    def __init__(
        self,
        voice: str,
        instructions: str,
        conversation_id: str,
        on_message: Optional[Callable] = None,
        model: str = "gpt-4o-realtime-preview-2024-12-17",
        audio_config: Optional[AudioConfig] = None
    ):
        """
        Initialize the Realtime Bridge.

        Args:
            voice: Voice to use for audio output (e.g., 'alloy', 'echo', 'shimmer', 'ash', 'coral', 'sage', 'marin')
            instructions: System instructions for the AI assistant
            conversation_id: Unique identifier for the conversation
            on_message: Optional callback function for message handling
            model: OpenAI model to use (default: 'gpt-4o-realtime-preview-2024-12-17')
            audio_config: Optional audio configuration settings
        """
        self.config = RealtimeBridgeConfig(
            voice=voice,
            instructions=instructions,
            conversation_id=conversation_id,
            model=model,
            audio=audio_config or AudioConfig()
        )
        self.on_message = on_message
        self.connection = None
        self._client: Optional[AsyncOpenAI] = None
        self._running = False
        self._session_configured = False
        self._has_active_response = False
        self._state = ConnectionState.DISCONNECTED
        self._pending_audio_chunks: list = []
        self._current_transcript: str = ""
        self._lock = asyncio.Lock()

        logger.info(
            "RealtimeBridge initialized: voice=%s, model=%s, conv_id=%s",
            voice, model, conversation_id
        )

    @property
    def state(self) -> ConnectionState:
        """Get the current connection state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if the bridge is ready for communication."""
        return self._state == ConnectionState.READY

    async def connect(self) -> None:
        """
        Establish connection to OpenAI API.

        Creates the AsyncOpenAI client instance.
        """
        if self._state != ConnectionState.DISCONNECTED:
            logger.warning("Already connected or connecting, state: %s", self._state)
            return

        self._state = ConnectionState.CONNECTING
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._running = True
        logger.info("OpenAI client created")

    async def start(self, send_callback: Callable) -> None:
        """
        Start the Realtime API session and begin processing events.

        Args:
            send_callback: Async callback function to send messages to the client
        """
        if not self._client:
            await self.connect()

        try:
            logger.info("Connecting to OpenAI Realtime API with model: %s", self.config.model)
            self._state = ConnectionState.CONNECTED

            async with self._client.realtime.connect(model=self.config.model) as connection:
                self.connection = connection
                logger.info("Connected to OpenAI Realtime API")

                # Start event loop - wait for session.created before configuring
                logger.info("Starting event loop, waiting for session.created...")
                async for event in connection:
                    if not self._running:
                        logger.info("Stopping event loop - running flag is False")
                        break

                    await self._process_event(event, send_callback)

        except asyncio.CancelledError:
            logger.info("Realtime bridge task was cancelled")
            raise
        except Exception as e:
            self._state = ConnectionState.ERROR
            logger.error("Realtime bridge error: %s", e)
            logger.error("Exception type: %s", type(e).__name__)
            logger.error("Full traceback:\n%s", traceback.format_exc())
            await send_callback({
                "type": "error",
                "message": str(e),
                "code": "connection_error"
            })
        finally:
            self._state = ConnectionState.DISCONNECTED
            self.connection = None

    async def _process_event(self, event: Any, send_callback: Callable) -> None:
        """
        Process incoming events from the Realtime API.

        Args:
            event: The event object from OpenAI
            send_callback: Callback to send messages to the client
        """
        event_type = event.type
        logger.debug("Received event: %s", event_type)

        # Event handlers mapping
        # Note: OpenAI Realtime API GA uses "response.output_audio.*" event names
        handlers = {
            "session.created": self._handle_session_created,
            "session.updated": self._handle_session_updated,
            # Audio output events (GA API uses "output_audio" naming)
            "response.output_audio.delta": self._handle_audio_delta,
            "response.output_audio.done": self._handle_audio_done,
            "response.output_audio_transcript.delta": self._handle_transcript_delta,
            "response.output_audio_transcript.done": self._handle_transcript_done,
            # Also support legacy naming (for backwards compatibility)
            "response.audio.delta": self._handle_audio_delta,
            "response.audio.done": self._handle_audio_done,
            "response.audio_transcript.delta": self._handle_transcript_delta,
            "response.audio_transcript.done": self._handle_transcript_done,
            "conversation.item.input_audio_transcription.completed": self._handle_input_transcription_completed,
            "conversation.item.input_audio_transcription.failed": self._handle_input_transcription_failed,
            "response.created": self._handle_response_created,
            "response.done": self._handle_response_done,
            "input_audio_buffer.speech_started": self._handle_speech_started,
            "input_audio_buffer.speech_stopped": self._handle_speech_stopped,
            "input_audio_buffer.committed": self._handle_buffer_committed,
            "rate_limits.updated": self._handle_rate_limits_updated,
            "error": self._handle_error,
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(event, send_callback)
        else:
            # Log unhandled events at debug level
            logger.debug("Unhandled event type: %s", event_type)

    async def _handle_session_created(self, event: Any, send_callback: Callable) -> None:
        """Handle session.created event - configure session."""
        logger.info("Received session.created, now configuring session...")
        await self._configure_session(send_callback)

    async def _handle_session_updated(self, event: Any, send_callback: Callable) -> None:
        """Handle session.updated event - confirm configuration."""
        logger.info("Session updated confirmed by OpenAI")
        if hasattr(event, 'session'):
            session = event.session
            if hasattr(session, 'instructions') and session.instructions:
                instr_preview = session.instructions[:100]
                logger.info("Session instructions (preview): %s...", instr_preview)
            if hasattr(session, 'audio') and session.audio:
                logger.debug("Audio config: %s", session.audio)

    async def _handle_audio_delta(self, event: Any, send_callback: Callable) -> None:
        """Handle response.audio.delta event - stream audio to client."""
        await send_callback({"type": "audio", "data": event.delta})

    async def _handle_audio_done(self, event: Any, send_callback: Callable) -> None:
        """Handle response.audio.done event - audio streaming complete."""
        logger.debug("Audio streaming completed for response")
        await send_callback({"type": "audio_done"})

    async def _handle_transcript_delta(self, event: Any, send_callback: Callable) -> None:
        """Handle response.audio_transcript.delta event - streaming transcript."""
        if hasattr(event, 'delta'):
            self._current_transcript += event.delta
            await send_callback({
                "type": "transcript_delta",
                "delta": event.delta,
                "role": "assistant"
            })

    async def _handle_transcript_done(self, event: Any, send_callback: Callable) -> None:
        """Handle response.audio_transcript.done event - complete AI transcript."""
        transcript = getattr(event, 'transcript', self._current_transcript)
        logger.info("AI transcript: %s", transcript[:100] if len(transcript) > 100 else transcript)

        await send_callback({
            "type": "transcript",
            "text": transcript,
            "role": "assistant"
        })
        await add_to_conversation(self.config.conversation_id, "assistant", transcript)

        # Reset current transcript
        self._current_transcript = ""

    async def _handle_input_transcription_completed(self, event: Any, send_callback: Callable) -> None:
        """Handle user input audio transcription completion."""
        transcript = event.transcript
        logger.info("User transcript: %s", transcript[:100] if len(transcript) > 100 else transcript)

        await send_callback({
            "type": "transcript",
            "text": transcript,
            "role": "user"
        })
        await add_to_conversation(self.config.conversation_id, "user", transcript)

    async def _handle_input_transcription_failed(self, event: Any, send_callback: Callable) -> None:
        """Handle input transcription failure (non-critical)."""
        error_msg = "Transcription failed"
        if hasattr(event, "error") and event.error:
            error_msg = str(getattr(event.error, "message", error_msg))
        logger.warning("Transcription failed (non-critical): %s", error_msg)

        # Optionally notify client about transcription failure
        await send_callback({
            "type": "transcription_failed",
            "message": error_msg
        })

    async def _handle_response_created(self, event: Any, send_callback: Callable) -> None:
        """Handle response.created event - AI is generating response."""
        async with self._lock:
            self._has_active_response = True
        await send_callback({"type": "status", "status": "speaking"})

    async def _handle_response_done(self, event: Any, send_callback: Callable) -> None:
        """Handle response.done event - AI response complete."""
        async with self._lock:
            self._has_active_response = False

        # Check if response failed
        if hasattr(event, 'response') and hasattr(event.response, 'status'):
            if event.response.status == "failed":
                error_msg = "Response failed"
                error_code = "response_failed"

                if hasattr(event.response, 'status_details') and event.response.status_details:
                    status_details = event.response.status_details
                    if hasattr(status_details, 'error') and status_details.error:
                        error_msg = str(getattr(status_details.error, 'message', error_msg))
                        error_code = str(getattr(status_details.error, 'code', error_code))

                logger.error("Response failed: %s - %s", error_code, error_msg)
                await send_callback({
                    "type": "error",
                    "message": error_msg,
                    "code": error_code
                })
                return

            # Log usage if available
            if hasattr(event.response, 'usage') and event.response.usage:
                usage = event.response.usage
                logger.info(
                    "Response usage - input tokens: %s, output tokens: %s",
                    getattr(usage, 'input_tokens', 'N/A'),
                    getattr(usage, 'output_tokens', 'N/A')
                )

        await send_callback({"type": "status", "status": "listening"})

    async def _handle_speech_started(self, event: Any, send_callback: Callable) -> None:
        """Handle input_audio_buffer.speech_started event."""
        logger.debug("Speech started detected")
        await send_callback({"type": "status", "status": "user_speaking"})

    async def _handle_speech_stopped(self, event: Any, send_callback: Callable) -> None:
        """Handle input_audio_buffer.speech_stopped event."""
        logger.debug("Speech stopped detected")
        await send_callback({"type": "status", "status": "processing"})

    async def _handle_buffer_committed(self, event: Any, send_callback: Callable) -> None:
        """Handle input_audio_buffer.committed event."""
        logger.debug("Audio buffer committed")
        if hasattr(event, 'item_id'):
            logger.debug("Committed item ID: %s", event.item_id)

    async def _handle_rate_limits_updated(self, event: Any, send_callback: Callable) -> None:
        """Handle rate_limits.updated event."""
        if hasattr(event, 'rate_limits'):
            for limit in event.rate_limits:
                logger.debug(
                    "Rate limit - %s: %s/%s remaining",
                    getattr(limit, 'name', 'unknown'),
                    getattr(limit, 'remaining', 'N/A'),
                    getattr(limit, 'limit', 'N/A')
                )

    async def _handle_error(self, event: Any, send_callback: Callable) -> None:
        """Handle error events from OpenAI."""
        error_msg = "Unknown error"
        error_code = "unknown"

        if hasattr(event, "error"):
            error_msg = str(getattr(event.error, "message", error_msg))
            error_code = str(getattr(event.error, "code", error_code))

        logger.error("OpenAI error event: %s - %s", error_code, error_msg)
        await send_callback({
            "type": "error",
            "message": error_msg,
            "code": error_code
        })

    async def _configure_session(self, send_callback: Callable) -> None:
        """
        Configure the Realtime session after receiving session.created event.

        Args:
            send_callback: Callback to send messages to the client
        """
        if self._session_configured:
            logger.warning("Session already configured, skipping")
            return

        self._state = ConnectionState.CONFIGURING
        audio_config = self.config.audio

        # Build session configuration for GA API
        session_config = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self.config.instructions,
            "audio": {
                "input": {
                    "format": {
                        "type": f"audio/{audio_config.input_format}",
                        "rate": audio_config.sample_rate
                    },
                    "transcription": {
                        "model": audio_config.transcription_model
                    },
                    "turn_detection": {
                        "type": audio_config.turn_detection_type,
                        "threshold": audio_config.turn_detection_threshold,
                        "prefix_padding_ms": audio_config.turn_detection_prefix_padding_ms,
                        "silence_duration_ms": audio_config.turn_detection_silence_duration_ms,
                        "create_response": True,
                        "interrupt_response": True
                    }
                },
                "output": {
                    "format": {
                        "type": f"audio/{audio_config.output_format}",
                        "rate": audio_config.sample_rate
                    },
                    "voice": self.config.voice
                }
            }
        }

        # Add optional max_output_tokens if specified
        if self.config.max_output_tokens:
            session_config["max_output_tokens"] = self.config.max_output_tokens

        logger.info("Session config: %s", json.dumps(session_config, indent=2, default=str))

        try:
            logger.info("Calling session.update()...")
            await self.connection.session.update(session=session_config)
            logger.info("Session update sent")

            self._session_configured = True
            self._state = ConnectionState.READY

            # Send ready to client after session is configured
            await send_callback({"type": "ready"})
            logger.info("Sent ready to client")

        except Exception as e:
            logger.error("Failed to configure session: %s", e)
            self._state = ConnectionState.ERROR
            await send_callback({
                "type": "error",
                "message": f"Failed to configure session: {str(e)}",
                "code": "session_config_error"
            })

    async def send_audio(self, audio_base64: str) -> None:
        """
        Send audio data to the Realtime API.

        Args:
            audio_base64: Base64 encoded audio data (PCM16 format expected)
        """
        if self.connection and self._state == ConnectionState.READY:
            try:
                await self.connection.input_audio_buffer.append(audio=audio_base64)
            except Exception as e:
                logger.error("Failed to send audio: %s", e)
                raise

    async def commit_audio(self) -> None:
        """
        Manually commit the audio buffer and trigger response generation.

        Note: When using server_vad turn detection, this is typically
        not needed as the server automatically handles turn detection.
        Use this for manual turn detection or push-to-talk implementations.
        """
        if self.connection and self._state == ConnectionState.READY:
            try:
                logger.info("Committing audio buffer and creating response")
                await self.connection.input_audio_buffer.commit()
                await self.connection.response.create()
            except Exception as e:
                logger.error("Failed to commit audio: %s", e)
                raise

    async def clear_audio_buffer(self) -> None:
        """
        Clear the audio input buffer.

        Useful when you need to discard buffered audio.
        """
        if self.connection and self._state == ConnectionState.READY:
            try:
                logger.info("Clearing audio buffer")
                await self.connection.input_audio_buffer.clear()
            except Exception as e:
                logger.error("Failed to clear audio buffer: %s", e)
                raise

    async def cancel_response(self) -> None:
        """
        Cancel the current response if one is active.

        Useful for implementing interruption handling.
        """
        async with self._lock:
            if self.connection and self._has_active_response:
                try:
                    logger.info("Cancelling response")
                    await self.connection.response.cancel()
                    self._has_active_response = False
                except Exception as e:
                    logger.error("Failed to cancel response: %s", e)
                    raise

    async def send_text_message(self, text: str, create_response: bool = True) -> None:
        """
        Send a text message to the conversation.

        Args:
            text: The text message to send
            create_response: Whether to automatically trigger a response
        """
        if not self.connection or self._state != ConnectionState.READY:
            logger.warning("Cannot send text: connection not ready")
            return

        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            )

            if create_response:
                await self.connection.response.create()

            logger.info("Text message sent: %s", text[:50] if len(text) > 50 else text)
        except Exception as e:
            logger.error("Failed to send text message: %s", e)
            raise

    async def update_instructions(self, instructions: str) -> None:
        """
        Update the session instructions dynamically.

        Args:
            instructions: New instructions for the AI assistant
        """
        if not self.connection or self._state != ConnectionState.READY:
            logger.warning("Cannot update instructions: connection not ready")
            return

        try:
            await self.connection.session.update(
                session={
                    "type": "realtime",
                    "instructions": instructions
                }
            )
            self.config.instructions = instructions
            logger.info("Instructions updated")
        except Exception as e:
            logger.error("Failed to update instructions: %s", e)
            raise

    async def disconnect(self) -> None:
        """
        Gracefully disconnect from the Realtime API.

        Stops the event loop and cleans up resources.
        """
        logger.info("Disconnecting from Realtime API")

        async with self._lock:
            self._running = False

            # Cancel any active response
            if self._has_active_response and self.connection:
                try:
                    await self.connection.response.cancel()
                except Exception as e:
                    logger.debug("Error cancelling response during disconnect: %s", e)

            self._has_active_response = False

        # Clear state
        self.connection = None
        self._session_configured = False
        self._state = ConnectionState.DISCONNECTED
        self._current_transcript = ""

        logger.info("Disconnected successfully")

    def get_status(self) -> dict:
        """
        Get the current status of the bridge.

        Returns:
            Dictionary containing the current state and configuration
        """
        return {
            "state": self._state.value,
            "is_ready": self.is_ready,
            "has_active_response": self._has_active_response,
            "session_configured": self._session_configured,
            "model": self.config.model,
            "voice": self.config.voice,
            "conversation_id": self.config.conversation_id
        }
