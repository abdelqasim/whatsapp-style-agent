"""
Audio Transcriber — WhatsApp Voice Note Support
─────────────────────────────────────────────────
Downloads voice notes from WhatsApp via the Meta Graph API,
then transcribes them using OpenAI Whisper API.

Supports: English, Arabic, Turkish (and 50+ other languages).

Flow:
  1. n8n detects audio message → calls POST /transcribe with media_id & sender
  2. This module downloads the audio from Meta's CDN
  3. Sends it to OpenAI Whisper for transcription
  4. Returns the transcribed text (which then goes through the normal pipeline)
"""

import logging
import os
import tempfile

import httpx
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)


async def download_whatsapp_media(media_id: str) -> tuple[bytes, str]:
    """
    Download media from WhatsApp via Meta Graph API.

    Step 1: GET the media URL from Meta (using media_id)
    Step 2: Download the actual file from the CDN URL

    Returns: (audio_bytes, mime_type)
    """
    headers = {"Authorization": f"Bearer {settings.whatsapp_token}"}

    async with httpx.AsyncClient(timeout=30.0) as http:
        # Step 1: Get media URL
        meta_url = f"{settings.whatsapp_api_url}/{media_id}"
        resp = await http.get(meta_url, headers=headers)
        resp.raise_for_status()
        media_info = resp.json()

        download_url = media_info["url"]
        mime_type = media_info.get("mime_type", "audio/ogg")

        logger.info(f"[audio] Media URL obtained for {media_id}, type={mime_type}")

        # Step 2: Download the actual audio file
        audio_resp = await http.get(download_url, headers=headers)
        audio_resp.raise_for_status()

        logger.info(f"[audio] Downloaded {len(audio_resp.content)} bytes")
        return audio_resp.content, mime_type


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/ogg") -> dict:
    """
    Transcribe audio bytes using OpenAI Whisper API.

    Returns: {
        "text": "transcribed text",
        "language": "en" | "ar" | "tr" | ...,
        "duration": float (seconds)
    }
    """
    # Map MIME type to file extension for Whisper
    ext_map = {
        "audio/ogg": ".ogg",
        "audio/ogg; codecs=opus": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/wav": ".wav",
        "audio/webm": ".webm",
        "audio/amr": ".amr",
    }
    ext = ext_map.get(mime_type, ".ogg")

    # Write to temp file (Whisper API needs a file)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Call Whisper API
        with open(tmp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
            )

        text = response.text.strip()
        language = getattr(response, "language", "unknown")
        duration = getattr(response, "duration", 0.0)

        logger.info(
            f"[audio] Transcribed: lang={language}, "
            f"duration={duration:.1f}s, "
            f"text_length={len(text)} chars"
        )

        return {
            "text": text,
            "language": language,
            "duration": duration,
        }

    finally:
        # Clean up temp file
        os.unlink(tmp_path)


async def handle_audio_message(media_id: str, sender: str = "unknown") -> dict:
    """
    Full pipeline: download WhatsApp audio → transcribe → return text.

    This is the main entry point called by the /transcribe endpoint.

    Returns: {
        "text": "transcribed text",
        "language": "en",
        "duration": 5.2,
        "sender": "905316339030"
    }
    """
    try:
        # Download audio from WhatsApp
        audio_bytes, mime_type = await download_whatsapp_media(media_id)

        # Transcribe with Whisper
        result = transcribe_audio(audio_bytes, mime_type)
        result["sender"] = sender

        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"[audio] Failed to download media {media_id}: {e}")
        return {
            "text": "",
            "language": "unknown",
            "duration": 0.0,
            "sender": sender,
            "error": f"Failed to download audio: {e.response.status_code}",
        }
    except Exception as e:
        logger.error(f"[audio] Transcription error: {e}")
        return {
            "text": "",
            "language": "unknown",
            "duration": 0.0,
            "sender": sender,
            "error": str(e),
        }
