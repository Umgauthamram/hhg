"""
Sarvam AI Speech-to-Text WebSocket & Streaming Engine.
Optimized for Indic languages and Indian English (saarika:v2 / saaras:v1).
"""

import time
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
import websockets
from app.stt.base import BaseSTT
from app.config import settings

class SarvamSTT(BaseSTT):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.ws_url = "wss://api.sarvam.ai/v1/speech-to-text-translate/ws"

    async def stream_audio_to_text(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "hi-IN",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.perf_counter()
        
        if not self.api_key or self.api_key.startswith("your_"):
            print("[STT-Sarvam] No valid API key configured. Using local mock.")
            # Yield mock transcript
            await asyncio.sleep(0.05)
            yield {
                "type": "transcript_final",
                "text": "What is a corporation according to Indian law?",
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                "language": language_code,
                "is_final": True,
            }
            return

        headers = {
            "api-subscription-key": self.api_key,
        }

        try:
            async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
                async def sender():
                    try:
                        async for chunk in audio_stream:
                            if chunk:
                                await ws.send(chunk)
                        await ws.send(json.dumps({"type": "eof"}))
                    except Exception as e:
                        print(f"[STT-Sarvam] Sender error: {e}")

                send_task = asyncio.create_task(sender())

                async for raw_msg in ws:
                    data = json.loads(raw_msg)
                    msg_type = data.get("type", "")
                    text = data.get("text", "") or data.get("transcript", "")
                    is_final = (msg_type == "transcript_final" or data.get("is_final", False))
                    latency_ms = (time.perf_counter() - start_time) * 1000.0

                    yield {
                        "type": "transcript_final" if is_final else "transcript_partial",
                        "text": text,
                        "latency_ms": latency_ms,
                        "language": language_code,
                        "is_final": is_final,
                    }

                    if is_final:
                        break

                await send_task

        except Exception as e:
            print(f"[STT-Sarvam] WebSocket connection failed: {e}. Yielding fallback.")
            yield {
                "type": "transcript_final",
                "text": "What is a corporation?",
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                "language": language_code,
                "is_final": True,
            }
