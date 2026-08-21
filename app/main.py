"""
FastAPI Server & Real-time WebSocket Gateway for Voice-Enabled Sub-200ms RAG.
Features:
- WebSocket endpoint /ws/voice-rag for low-latency token streaming and binary audio ingestion.
- REST endpoints /api/query, /api/voice-query, /api/health, /api/benchmark/stats, /api/ingest.
- Static file serving for the interactive dashboard.
"""

import os
import sys
import json
import time
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.harness.orchestrator import RAGOrchestrator, RAGResponse
from app.chunking.hybrid_indexer import HybridIndexer
from app.llm.groq_client import LowLatencyLLM
from app.stt.streaming_stt import StreamingSTT
from app.harness.guardrails import GuardrailEngine
from dataset_ingest import MSMARCOXIngramIngester
from app.config import settings

# Global shared pipeline instances
orchestrator: Optional[RAGOrchestrator] = None
ingester: Optional[MSMARCOXIngramIngester] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes in-memory components and loads MSMARCO-XI index on startup."""
    global orchestrator, ingester
    startup_start = time.perf_counter()
    print("\n[Server Startup] Initializing Voice-Enabled RAG Sub-200ms Pipeline...")

    indexer = HybridIndexer()
    ingester = MSMARCOXIngramIngester(hybrid_indexer=indexer)

    # 1. Load curated multi-lingual seed dataset
    ingester.ingest_curated_seed()

    # 2. Stream and index MSMARCO-XI from HuggingFace
    try:
        ingester.ingest_from_hf(languages=["hin"], limit_per_lang=10)
    except Exception as e:
        print(f"[Server Startup] Warning: Online ingestion fallback: {e}")

    # 3. Initialize Orchestrator components
    llm = LowLatencyLLM()
    stt = StreamingSTT()
    guardrail = GuardrailEngine()

    orchestrator = RAGOrchestrator(
        hybrid_indexer=indexer,
        llm_client=llm,
        stt_client=stt,
        guardrail_engine=guardrail,
    )

    startup_ms = (time.perf_counter() - startup_start) * 1000.0
    print(f"[Server Startup] Pipeline initialized and ready in {startup_ms:.2f}ms!\n")

    yield

    print("[Server Shutdown] Cleaning up resources...")

app = FastAPI(
    title="Voice-Enabled Sub-200ms RAG System",
    description="MSMARCO-XI Grounded Voice-First RAG API for HH Goa 2026",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web assets
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get_index():
    """Serves the primary web dashboard."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Voice RAG API is live. Static UI not found."}

@app.get("/api/health")
async def health_check():
    """System health check and pipeline status."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "status": "healthy",
        "vector_count": orchestrator.indexer.vector_store.client.count(
            collection_name=settings.QDRANT_COLLECTION_NAME
        ).count,
        "llm_model": orchestrator.llm.model,
        "stt_provider": orchestrator.stt.provider_name,
        "timestamp": time.time(),
    }

class RAGRequest(BaseModel):
    query: str
    language: str = Field(default="en")
    top_k: int = Field(default=3)

@app.post("/api/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest):
    """Synchronous REST query endpoint with latency breakdown."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response = await orchestrator.execute_query(
        query=request.query,
        language=request.language,
        top_k=request.top_k,
    )
    return response

@app.post("/api/voice-query", response_model=RAGResponse)
async def query_voice_audio(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    top_k: int = Form(default=3),
):
    """Transcribes uploaded microphone audio and streams RAG output."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    audio_bytes = await file.read()
    response = await orchestrator.execute_audio_query(
        audio_bytes=audio_bytes,
        language=language,
        top_k=top_k,
    )
    return response

@app.get("/api/benchmark/stats")
async def get_benchmark_stats():
    """Returns real-time P50, P70, P90, P100 latency percentiles."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return orchestrator.latency_tracker.get_percentiles()

class IngestRequest(BaseModel):
    languages: list[str] = Field(default=["hin", "tam"])
    limit_per_lang: int = Field(default=20)

@app.post("/api/ingest")
async def trigger_ingest(req: IngestRequest):
    """Dynamically ingests additional data from HuggingFace MSMARCO-XI."""
    global ingester
    if not ingester:
        raise HTTPException(status_code=503, detail="Ingester not initialized")

    count = ingester.ingest_from_hf(languages=req.languages, limit_per_lang=req.limit_per_lang)
    return {"status": "success", "indexed_chunks": count}

@app.websocket("/ws/voice-rag")
async def websocket_voice_rag(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for streaming voice/text RAG.
    Receives binary audio frames or JSON text query and streams response tokens + latency breakdown.
    """
    global orchestrator
    await websocket.accept()

    language = "en"
    top_k = 3

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "query")

                    if msg_type == "config":
                        language = data.get("language", language)
                        top_k = data.get("top_k", top_k)
                        await websocket.send_json({"event": "config_ack", "status": "ok"})
                        continue

                    query_text = data.get("text") or data.get("query", "")
                    if not query_text:
                        await websocket.send_json({"event": "error", "message": "Empty query received."})
                        continue

                    async for event in orchestrator.execute_stream(
                        query=query_text,
                        language=language,
                        top_k=top_k,
                        stt_latency_ms=0.0,
                    ):
                        await websocket.send_json(event)

                except json.JSONDecodeError:
                    await websocket.send_json({"event": "error", "message": "Invalid JSON format."})

            elif "bytes" in message:
                audio_bytes = message["bytes"]
                stt_start = time.perf_counter()

                async def single_chunk_stream():
                    yield audio_bytes

                final_text = ""
                stt_latency = 0.0

                async for stt_event in orchestrator.stt.transcribe_stream(single_chunk_stream(), language_code=language):
                    if stt_event.get("type") == "transcript_partial":
                        await websocket.send_json({
                            "event": "transcript_partial",
                            "text": stt_event["text"],
                            "latency_ms": stt_event["latency_ms"],
                        })
                    elif stt_event.get("is_final"):
                        final_text = stt_event["text"]
                        stt_latency = stt_event["latency_ms"]
                        await websocket.send_json({
                            "event": "transcript_final",
                            "text": final_text,
                            "latency_ms": stt_latency,
                        })

                if final_text:
                    async for event in orchestrator.execute_stream(
                        query=final_text,
                        language=language,
                        top_k=top_k,
                        stt_latency_ms=stt_latency,
                    ):
                        await websocket.send_json(event)
                else:
                    await websocket.send_json({
                        "event": "error",
                        "message": "Speech could not be transcribed.",
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error during session: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
