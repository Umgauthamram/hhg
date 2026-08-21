"""
Abstract Base Class for Streaming Speech-to-Text (STT) engines.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Union

class BaseSTT(ABC):
    @abstractmethod
    async def stream_audio_to_text(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "hi-IN",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Consumes raw audio chunks (PCM 16-bit 16kHz or Opus/WAV)
        and yields transcript events:
        {
            "type": "transcript_partial" | "transcript_final",
            "text": str,
            "latency_ms": float,
            "language": str,
            "is_final": bool
        }
        """
        pass
