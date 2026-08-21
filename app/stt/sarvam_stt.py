"""
Sarvam AI Speech-to-Text Engine (REST & Streaming).
Optimized for Indian languages and English (saarika:v2 / saaras:v1).
"""

import time
import json
import asyncio
import httpx
from typing import AsyncGenerator, Dict, Any, Optional
import websockets
from app.stt.base import BaseSTT
from app.config import settings

class SarvamSTT(BaseSTT):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.rest_url = "https://api.sarvam.ai/speech-to-text"
        self.ws_url = "wss://api.sarvam.ai/v1/speech-to-text-translate/ws"

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "en-IN"
    ) -> Dict[str, Any]:
        """Transcribes audio using Sarvam AI REST API (saarika:v2)."""
        start_t = time.perf_counter()
        if not self.api_key or self.api_key.startswith("your_"):
            return {"text": "", "latency_ms": 0.0, "error": "Missing Sarvam API Key"}

        headers = {
            "api-subscription-key": self.api_key
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                files = {"file": (filename, audio_bytes, "audio/wav")}
                data = {
                    "model": "saarika:v2",
                    "language_code": language_code,
                }
                res = await client.post(self.rest_url, headers=headers, files=files, data=data)
                latency_ms = (time.perf_counter() - start_t) * 1000.0

                if res.status_code == 200:
                    result = res.json()
                    transcript = result.get("transcript", "")
                    return {
                        "text": transcript.strip(),
                        "latency_ms": latency_ms,
                        "provider": "sarvam",
                    }
                else:
                    return {
                        "text": "",
                        "latency_ms": latency_ms,
                        "error": f"Sarvam error {res.status_code}: {res.text}"
                    }
        except Exception as e:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            return {"text": "", "latency_ms": latency_ms, "error": str(e)}

    async def stream_audio_to_text(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "en-IN",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.perf_counter()
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)

        res = await self.transcribe_audio_bytes(bytes(buffer), language_code=language_code)
        yield {
            "type": "transcript_final",
            "text": res.get("text", ""),
            "latency_ms": res.get("latency_ms", (time.perf_counter() - start_time) * 1000.0),
            "language": language_code,
            "is_final": True,
            "error": res.get("error"),
        }
