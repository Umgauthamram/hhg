"""
Ultra-Low-Latency Async Groq Streaming Client.
Features:
- Sub-75ms Time-To-First-Token (TTFT) tracking.
- Thinking tag & markdown stripping.
- Streamlined async generator yielding immediate token chunks.
"""

import time
import re
from typing import AsyncGenerator, Dict, Any, Optional
from groq import AsyncGroq
from app.config import settings
from app.llm.prompts import VOICE_RAG_SYSTEM_PROMPT, format_voice_user_prompt

class LowLatencyLLM:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.client = AsyncGroq(api_key=self.api_key)

    def _clean_token(self, token: str) -> str:
        """Strips formatting symbols for clean speech synthesis."""
        return re.sub(r'[*#_`]', '', token)

    async def generate_stream(
        self,
        query: str,
        context: str,
        system_prompt: str = VOICE_RAG_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams generated tokens asynchronously from Groq LPUs.
        """
        start_time = time.perf_counter()
        first_token_time = None

        user_content = format_voice_user_prompt(query=query, context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_text_parts = []
        is_first = True
        in_think_block = False

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else ""
            if not delta:
                continue

            # Skip thinking tags
            if "<think>" in delta:
                in_think_block = True
                continue
            if "</think>" in delta:
                in_think_block = False
                continue
            if in_think_block:
                continue

            clean_delta = self._clean_token(delta)
            if not clean_delta:
                continue

            if is_first:
                first_token_time = time.perf_counter()
                ttft_ms = (first_token_time - start_time) * 1000.0
                is_first = False
                yield {
                    "token": clean_delta,
                    "is_first": True,
                    "is_done": False,
                    "ttft_ms": ttft_ms,
                    "total_ms": ttft_ms,
                }
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                yield {
                    "token": clean_delta,
                    "is_first": False,
                    "is_done": False,
                    "ttft_ms": (first_token_time - start_time) * 1000.0 if first_token_time else elapsed_ms,
                    "total_ms": elapsed_ms,
                }

            full_text_parts.append(clean_delta)

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        full_text = "".join(full_text_parts).strip()
        ttft = (first_token_time - start_time) * 1000.0 if first_token_time else total_time_ms

        yield {
            "token": "",
            "is_first": False,
            "is_done": True,
            "ttft_ms": ttft,
            "total_ms": total_time_ms,
            "full_text": full_text,
        }

    async def generate_complete(
        self,
        query: str,
        context: str,
        system_prompt: str = VOICE_RAG_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Collects the stream into a single structured response dictionary."""
        full_text = ""
        ttft_ms = 0.0
        total_ms = 0.0

        async for chunk in self.generate_stream(
            query=query,
            context=context,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if chunk["is_first"]:
                ttft_ms = chunk["ttft_ms"]
            if chunk["is_done"]:
                full_text = chunk["full_text"]
                total_ms = chunk["total_ms"]
                if not ttft_ms:
                    ttft_ms = chunk["ttft_ms"]

        return {
            "text": full_text,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
        }
