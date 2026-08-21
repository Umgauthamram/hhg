"""
Unified Streaming Speech-to-Text Manager & Provider Factory.
Routes audio streams to Sarvam AI, ElevenLabs, or local mock engines based on configuration.
"""

from typing import Optional, AsyncGenerator, Dict, Any
from app.stt.base import BaseSTT
from app.stt.sarvam_stt import SarvamSTT
from app.stt.elevenlabs_stt import ElevenLabsSTT
from app.stt.mock_stt import MockSTT
from app.config import settings

def get_stt_engine(provider: Optional[str] = None) -> BaseSTT:
    """
    Factory function returning the appropriate STT implementation.
    """
    prov = (provider or settings.STT_PROVIDER).lower()

    if prov == "sarvam":
        return SarvamSTT()
    elif prov == "elevenlabs":
        return ElevenLabsSTT()
    else:
        return MockSTT()

class StreamingSTT:
    def __init__(self, provider: Optional[str] = None):
        self.provider_name = provider or settings.STT_PROVIDER
        self.engine: BaseSTT = get_stt_engine(self.provider_name)

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "hi-IN",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Transcribes streaming binary PCM / Opus audio chunks into partial and final text transcripts.
        """
        async for event in self.engine.stream_audio_to_text(audio_stream, language_code=language_code):
            yield event
