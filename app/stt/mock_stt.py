"""
Local Mock Streaming Speech-to-Text (STT) Engine for deterministic low-latency testing.
Simulates sub-50ms streaming transcription of audio frames.
"""

import time
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from app.stt.base import BaseSTT

class MockSTT(BaseSTT):
    def __init__(self, default_text: str = "What is a corporation?", simulated_latency_ms: float = 35.0):
        self.default_text = default_text
        self.simulated_latency_ms = simulated_latency_ms

    async def stream_audio_to_text(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "en",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.perf_counter()
        
        # Consume incoming audio chunks
        chunk_count = 0
        async for chunk in audio_stream:
            chunk_count += 1
            if chunk_count % 2 == 0:
                partial_words = self.default_text.split()[:min(chunk_count + 1, len(self.default_text.split()))]
                partial_text = " ".join(partial_words)
                yield {
                    "type": "transcript_partial",
                    "text": partial_text,
                    "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                    "language": language_code,
                    "is_final": False,
                }

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        yield {
            "type": "transcript_final",
            "text": self.default_text,
            "latency_ms": total_latency_ms,
            "language": language_code,
            "is_final": True,
        }
