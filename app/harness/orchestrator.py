"""
End-to-End Voice RAG Pipeline Orchestrator & Harness.
Connects: STT -> Input Guardrails -> Hybrid Retrieval -> Grounding Gate -> Low-Latency LLM Streaming.
Tracks per-node microsecond latency breakdowns and percentile distributions.
"""

import time
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.chunking.hybrid_indexer import HybridIndexer
from app.llm.groq_client import LowLatencyLLM
from app.stt.streaming_stt import StreamingSTT
from app.harness.guardrails import GuardrailEngine
from app.harness.latency_tracker import LatencyTracker, LatencyBreakdown
from app.config import settings

class RAGRequest(BaseModel):
    query: str
    language: str = Field(default="en")
    top_k: int = Field(default=3)

class RAGResponse(BaseModel):
    query: str
    answer: str
    is_safe: bool = True
    is_grounded: bool = True
    refusal_reason: str = ""
    latency: LatencyBreakdown
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class RAGOrchestrator:
    def __init__(
        self,
        hybrid_indexer: Optional[HybridIndexer] = None,
        llm_client: Optional[LowLatencyLLM] = None,
        stt_client: Optional[StreamingSTT] = None,
        guardrail_engine: Optional[GuardrailEngine] = None,
    ):
        self.indexer = hybrid_indexer or HybridIndexer()
        self.llm = llm_client or LowLatencyLLM()
        self.stt = stt_client or StreamingSTT()
        self.guardrail = guardrail_engine or GuardrailEngine()
        self.latency_tracker = LatencyTracker()

    async def execute_query(
        self,
        query: str,
        language: str = "en",
        top_k: int = 3,
    ) -> RAGResponse:
        """
        Synchronous / non-streaming execution with full latency breakdown.
        """
        pipeline_start = time.perf_counter()
        breakdown = LatencyBreakdown()

        # 1. Input Guardrail
        g_start = time.perf_counter()
        is_safe, refusal_reason = self.guardrail.validate_input(query)
        breakdown.guardrail_ms = (time.perf_counter() - g_start) * 1000.0

        if not is_safe:
            breakdown.total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0
            return RAGResponse(
                query=query,
                answer=f"I cannot process this request: {refusal_reason}",
                is_safe=False,
                is_grounded=False,
                refusal_reason=refusal_reason,
                latency=breakdown,
                sources=[],
            )

        # 2. Hybrid Retrieval
        retrieved_docs, retrieval_ms = self.indexer.search_hybrid(query, top_k=top_k)
        breakdown.retrieval_ms = retrieval_ms

        # 3. Grounding Gate
        is_grounded, ground_reason = self.guardrail.validate_retrieval_grounding(retrieved_docs)
        if not is_grounded:
            breakdown.total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0
            return RAGResponse(
                query=query,
                answer="I cannot find this information in the provided records.",
                is_safe=True,
                is_grounded=False,
                refusal_reason=ground_reason,
                latency=breakdown,
                sources=[],
            )

        # 4. LLM Generation
        context_text = "\n---\n".join([
            d.get("parent_context") or d.get("text", "") for d in retrieved_docs
        ])
        
        llm_res = await self.llm.generate_complete(query=query, context=context_text)
        breakdown.llm_ttft_ms = llm_res["ttft_ms"]
        breakdown.llm_total_ms = llm_res["total_ms"]
        breakdown.total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0

        self.latency_tracker.record(breakdown)

        return RAGResponse(
            query=query,
            answer=llm_res["text"],
            is_safe=True,
            is_grounded=True,
            latency=breakdown,
            sources=[
                {
                    "doc_id": d.get("doc_id", ""),
                    "score": d.get("fused_score", 0.0),
                    "text": d.get("text", ""),
                    "language": d.get("language", "en"),
                }
                for d in retrieved_docs
            ],
        )

    async def execute_audio_query(
        self,
        audio_bytes: bytes,
        language: str = "en",
        top_k: int = 3,
    ) -> RAGResponse:
        """
        Transcribes raw audio bytes and processes through RAG pipeline.
        """
        start_t = time.perf_counter()
        
        async def single_chunk_gen():
            yield audio_bytes

        transcribed_text = ""
        stt_latency = 0.0
        async for evt in self.stt.transcribe_stream(single_chunk_gen(), language_code=language):
            if evt.get("is_final"):
                transcribed_text = evt.get("text", "")
                stt_latency = evt.get("latency_ms", (time.perf_counter() - start_t) * 1000.0)

        if not transcribed_text:
            return RAGResponse(
                query="",
                answer="Could not transcribe audio.",
                is_safe=True,
                is_grounded=False,
                latency=LatencyBreakdown(stt_ms=stt_latency, total_pipeline_ms=(time.perf_counter() - start_t) * 1000.0)
            )

        resp = await self.execute_query(query=transcribed_text, language=language, top_k=top_k)
        resp.latency.stt_ms = stt_latency
        resp.latency.total_pipeline_ms += stt_latency
        return resp

    async def execute_stream(
        self,
        query: str,
        language: str = "en",
        top_k: int = 3,
        stt_latency_ms: float = 0.0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes streaming RAG pipeline over WebSockets, yielding real-time tokens and latency stats.
        """
        pipeline_start = time.perf_counter()
        breakdown = LatencyBreakdown(stt_ms=stt_latency_ms)

        # 1. Input Guardrail
        g_start = time.perf_counter()
        is_safe, refusal_reason = self.guardrail.validate_input(query)
        breakdown.guardrail_ms = (time.perf_counter() - g_start) * 1000.0

        if not is_safe:
            breakdown.total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0 + stt_latency_ms
            yield {
                "event": "error",
                "token": f"I cannot process this request: {refusal_reason}",
                "is_safe": False,
                "is_grounded": False,
                "latency": breakdown.model_dump(),
            }
            return

        # 2. Hybrid Retrieval
        retrieved_docs, retrieval_ms = self.indexer.search_hybrid(query, top_k=top_k)
        breakdown.retrieval_ms = retrieval_ms

        # 3. Grounding Gate
        is_grounded, ground_reason = self.guardrail.validate_retrieval_grounding(retrieved_docs)
        if not is_grounded:
            breakdown.total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0 + stt_latency_ms
            yield {
                "event": "grounding_rejection",
                "token": "I cannot find this information in the provided records.",
                "is_safe": True,
                "is_grounded": False,
                "reason": ground_reason,
                "latency": breakdown.model_dump(),
                "sources": [],
            }
            return

        # 4. Stream LLM Response
        context_text = "\n---\n".join([
            d.get("parent_context") or d.get("text", "") for d in retrieved_docs
        ])

        async for chunk in self.llm.generate_stream(query=query, context=context_text):
            if chunk["is_first"]:
                breakdown.llm_ttft_ms = chunk["ttft_ms"]
                breakdown.total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0 + stt_latency_ms
                yield {
                    "event": "token",
                    "token": chunk["token"],
                    "is_first": True,
                    "is_done": False,
                    "latency": breakdown.model_dump(),
                }
            elif not chunk["is_done"]:
                yield {
                    "event": "token",
                    "token": chunk["token"],
                    "is_first": False,
                    "is_done": False,
                    "latency": breakdown.model_dump(),
                }
            else:
                breakdown.llm_total_ms = chunk["total_ms"]
                breakdown.total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0 + stt_latency_ms
                self.latency_tracker.record(breakdown)

                yield {
                    "event": "complete",
                    "token": "",
                    "is_done": True,
                    "is_safe": True,
                    "is_grounded": True,
                    "full_answer": chunk["full_text"],
                    "latency": breakdown.model_dump(),
                    "sources": [
                        {"doc_id": d.get("doc_id", ""), "score": d.get("fused_score", 0.0), "language": d.get("language", "en")}
                        for d in retrieved_docs
                    ],
                }
