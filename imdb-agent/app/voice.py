"""
Voice I/O: OpenAI Whisper (STT) + OpenAI TTS wrappers.

Both functions are optional — the Streamlit UI shows them only when
enable_voice=True in .env. This avoids mandatory API cost for the demo.

WHISPER (STT): Transcribes audio bytes to text. Called when the user submits
an audio recording via st.audio_input.

TTS: Converts agent response text to audio bytes. Returned and played via
st.audio() in the UI.

SCALE NOTE: For low-latency TTS in production, use streaming TTS
(openai.audio.speech.with_streaming_response.create()) and chunk the audio
to the browser as it generates, rather than waiting for the full file.
"""
import io
import logging
from openai import AsyncOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI = None


def _get_client() -> AsyncOpenAI:
    """Lazy-initialise OpenAI async client (reuse across calls)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> tuple[str, str | None]:
    """
    Transcribe audio to text using OpenAI Whisper.

    Args:
        audio_bytes: Raw audio bytes (WebM/WAV/MP3/M4A supported)
        filename:    Filename with extension — Whisper uses this to detect the codec.
                     Defaults to "audio.webm" which is what browsers record in.

    Returns:
        (transcribed_text, error_message) — error_message is None on success.
    """
    if not audio_bytes:
        return "", "No audio data received."

    client = _get_client()
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        transcript = await client.audio.transcriptions.create(
            model=settings.stt_model,
            file=audio_file,
            language="en",
        )
        text = transcript.text.strip()
        logger.info(f"STT transcription ({filename}): '{text[:80]}'")
        return text, None
    except Exception as e:
        logger.error(f"Whisper STT error: {e}")
        return "", str(e)


async def text_to_speech(text: str) -> bytes:
    """
    Convert text to audio using OpenAI TTS.

    Args:
        text: Response text to speak (trimmed to 4096 chars — TTS limit)

    Returns:
        MP3 audio bytes, or empty bytes on error
    """
    client = _get_client()
    try:
        response = await client.audio.speech.create(
            model=settings.tts_model,
            voice="alloy",        # Neutral voice — good for information delivery
            input=text[:4096],    # TTS API char limit
            response_format="mp3",
        )
        audio_bytes = response.content
        logger.info(f"TTS generated: {len(audio_bytes)} bytes")
        return audio_bytes
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""
