"""
FastAPI Application & WebSocket Server for Voice-Enabled Sub-200ms RAG System.
Provides REST and real-time WebSocket endpoints for voice/text querying, latency tracking, and dataset management.
"""

import sys
import os
import time
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Set utf-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.chunking.hybrid_indexer import HybridIndexer
from app.llm.groq_client import LowLatencyLLM
from app.stt.streaming_stt import StreamingSTT
from app.harness.guardrails import GuardrailEngine
from app.harness.orchestrator import RAGOrchestrator, RAGRequest, RAGResponse
from dataset_ingest import MSMARCOXIngramIngester

# Global application state
orchestrator: Optional[RAGOrchestrator] = None
ingester: Optional[MSMARCOXIngramIngester] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, ingester
    print("\n[Server Startup] Initializing Voice-Enabled RAG Sub-200ms Pipeline...")
    start_init = time.perf_counter()

    # 1. Initialize Hybrid Indexer (In-memory Qdrant + FastEmbed + BM25)
    indexer = HybridIndexer()

    # 2. Ingest seed dataset into memory
    ingester = MSMARCOXIngramIngester(hybrid_indexer=indexer)
    ingester.ingest_curated_seed()

    # Ingest a small live partition from HF MSMARCO-XI if available
    try:
        ingester.ingest_from_hf(languages=["hin"], limit_per_lang=10)
    except Exception as e:
        print(f"[Server Startup] Warning: Could not download live HF partition ({e}). Using curated seed.")

    # 3. Initialize Orchestrator
    llm = LowLatencyLLM()
    stt = StreamingSTT()
    guardrails = GuardrailEngine()
    
    orchestrator = RAGOrchestrator(
        hybrid_indexer=indexer,
        llm_client=llm,
        stt_client=stt,
        guardrail_engine=guardrails,
    )

    init_ms = (time.perf_counter() - start_init) * 1000.0
    print(f"[Server Startup] Pipeline initialized and ready in {init_ms:.2f}ms!\n")

    yield

    print("[Server Shutdown] Cleaning up resources...")

app = FastAPI(
    title="Voice-Enabled Sub-200ms RAG System",
    description="Sub-200ms End-to-End Voice RAG pipeline for HH Goa 2026",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static folder exists
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main web dashboard if index.html exists, otherwise simple HTML."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Voice-Enabled Sub-200ms RAG API Running</h1><p>Access /docs for API documentation.</p>")

@app.get("/api/health")
async def health_check():
    """Health and status endpoint."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return {
        "status": "healthy",
        "vector_store_count": orchestrator.indexer.vector_store.count(),
        "llm_model": orchestrator.llm.model_name,
        "embedding_model": orchestrator.indexer.embedding_engine.model_name,
        "stt_provider": orchestrator.stt.provider_name,
        "timestamp": time.time(),
    }

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
            # Handle incoming message (can be text JSON or binary audio)
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

                    # Stream text query response
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

                # Stream audio to STT engine
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
                    # Stream RAG pipeline response
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
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Error during session: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
