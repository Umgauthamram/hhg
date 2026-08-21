"""
High-Speed Whisper LPU Speech-to-Text Engine via Groq Cloud API.
Provides sub-100ms accurate transcription for recorded audio blobs using 'whisper-large-v3-turbo'.
"""

import time
import os
import io
from typing import AsyncGenerator, Dict, Any, Optional
from groq import AsyncGroq
from app.stt.base import BaseSTT
from app.config import settings

class GroqWhisperSTT(BaseSTT):
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-large-v3-turbo"):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model
        self.client = AsyncGroq(api_key=self.api_key)

    async def transcribe_audio_bytes(self, audio_bytes: bytes, filename: str = "audio.webm", language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribes audio bytes using Groq Whisper LPU.
        """
        start_t = time.perf_counter()
        if not audio_bytes or len(audio_bytes) < 100:
            return {"text": "", "latency_ms": 0.0, "error": "Audio stream too short"}

        try:
            audio_file = (filename, audio_bytes)
            kwargs = {
                "file": audio_file,
                "model": self.model,
                "response_format": "json",
                "temperature": 0.0,
            }
            if language and language != "auto":
                kwargs["language"] = language

            transcription = await self.client.audio.transcriptions.create(**kwargs)
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            
            return {
                "text": transcription.text.strip(),
                "latency_ms": latency_ms,
                "model": self.model,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            print(f"[GroqWhisper] Transcription error: {e}")
            return {
                "text": "",
                "latency_ms": latency_ms,
                "error": str(e)
            }

    async def stream_audio_to_text(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "en",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Collects audio stream buffer and transcribes with Whisper LPU."""
        start_t = time.perf_counter()
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)

        res = await self.transcribe_audio_bytes(bytes(buffer), language=language_code if language_code != "auto" else None)
        
        yield {
            "type": "transcript_final",
            "text": res.get("text", ""),
            "latency_ms": res.get("latency_ms", (time.perf_counter() - start_t) * 1000.0),
            "language": language_code,
            "is_final": True,
        }
