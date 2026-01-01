"""
Text-to-Speech Service for AiSeaSafe

This module provides TTS functionality using OpenAI's TTS API
to convert text responses to natural-sounding speech audio.

Uses OpenAI TTS API: https://platform.openai.com/docs/guides/text-to-speech
"""

import base64
import logging
from typing import Optional, Literal
from io import BytesIO

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# TTS Configuration
TTS_MODEL = "tts-1"  # Use "tts-1-hd" for higher quality (slower, more expensive)
TTS_VOICE = "nova"  # Options: alloy, echo, fable, onyx, nova, shimmer
TTS_RESPONSE_FORMAT = "mp3"  # Options: mp3, opus, aac, flac, wav, pcm
TTS_SPEED = 1.0  # Range: 0.25 to 4.0

# Maximum text length for TTS (OpenAI limit is 4096 characters)
MAX_TTS_LENGTH = 4000


async def generate_speech(
    text: str,
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = TTS_VOICE,
    model: Literal["tts-1", "tts-1-hd"] = TTS_MODEL,
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = TTS_RESPONSE_FORMAT,
    speed: float = TTS_SPEED,
) -> Optional[bytes]:
    """
    Generate speech audio from text using OpenAI TTS API.

    Args:
        text: Text to convert to speech (max 4096 characters)
        voice: Voice to use for synthesis
        model: TTS model (tts-1 for speed, tts-1-hd for quality)
        response_format: Audio format
        speed: Speech speed (0.25 to 4.0)

    Returns:
        Audio data as bytes, or None if generation fails.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for TTS")
        return None

    # Truncate text if too long
    if len(text) > MAX_TTS_LENGTH:
        logger.warning("Text truncated from %d to %d characters for TTS", len(text), MAX_TTS_LENGTH)
        text = text[:MAX_TTS_LENGTH]

    try:
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        response = await client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=speed,
        )

        # Get the audio bytes
        audio_bytes = response.content

        logger.info("Generated TTS audio: %d bytes, format: %s, voice: %s",
                    len(audio_bytes), response_format, voice)

        return audio_bytes

    except Exception as e:
        logger.error("TTS generation error: %s", e)
        return None


async def generate_speech_base64(
    text: str,
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = TTS_VOICE,
    model: Literal["tts-1", "tts-1-hd"] = TTS_MODEL,
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = TTS_RESPONSE_FORMAT,
    speed: float = TTS_SPEED,
) -> Optional[str]:
    """
    Generate speech audio and return as base64-encoded string.

    This is convenient for embedding audio in JSON responses.

    Args:
        text: Text to convert to speech
        voice: Voice to use for synthesis
        model: TTS model
        response_format: Audio format
        speed: Speech speed

    Returns:
        Base64-encoded audio data, or None if generation fails.
    """
    audio_bytes = await generate_speech(
        text=text,
        voice=voice,
        model=model,
        response_format=response_format,
        speed=speed,
    )

    if audio_bytes:
        return base64.b64encode(audio_bytes).decode("utf-8")
    return None


def get_audio_mime_type(response_format: str) -> str:
    """Get the MIME type for an audio format."""
    mime_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    return mime_types.get(response_format, "audio/mpeg")


def get_audio_data_url(audio_base64: str, response_format: str = "mp3") -> str:
    """
    Create a data URL from base64-encoded audio.

    This can be used directly in HTML audio elements or for playback.
    """
    mime_type = get_audio_mime_type(response_format)
    return f"data:{mime_type};base64,{audio_base64}"
