"""
Streaming Speech-to-Text (STT) Router & Factory.
Routes audio streams to Sarvam AI, ElevenLabs, Groq Whisper LPU, or local Mock STT.
"""

from typing import AsyncGenerator, Dict, Any, Optional
from app.stt.base import BaseSTT
from app.stt.sarvam_stt import SarvamSTT
from app.stt.elevenlabs_stt import ElevenLabsSTT
from app.stt.groq_whisper import GroqWhisperSTT
from app.stt.mock_stt import MockSTT
from app.config import settings

class StreamingSTT:
    def __init__(self, provider: Optional[str] = None):
        self.provider_name = (provider or settings.STT_PROVIDER).lower()
        self.client: BaseSTT = self._initialize_client()

    def _initialize_client(self) -> BaseSTT:
        if self.provider_name == "sarvam" and settings.SARVAM_API_KEY:
            return SarvamSTT(api_key=settings.SARVAM_API_KEY)
        elif self.provider_name == "elevenlabs" and settings.ELEVENLABS_API_KEY:
            return ElevenLabsSTT(api_key=settings.ELEVENLABS_API_KEY)
        elif (self.provider_name in ["groq", "whisper", "mock"] or not self.provider_name) and settings.GROQ_API_KEY:
            return GroqWhisperSTT(api_key=settings.GROQ_API_KEY)
        elif self.provider_name == "mock":
            return MockSTT()
        else:
            return MockSTT()

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "en",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Routes audio stream to the configured STT engine."""
        async for event in self.client.stream_audio_to_text(audio_stream, language_code=language_code):
            yield event

def get_stt_engine(provider: Optional[str] = None) -> BaseSTT:
    """Helper factory returning the underlying STT engine instance."""
    stt = StreamingSTT(provider=provider)
    return stt.client
