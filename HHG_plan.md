# Technical Specification & Code Architecture Guide
## Voice-Enabled Sub-200ms RAG System for HH Goa 2026 (Task 2)

---

## 1. System Architecture Overview

To meet the strict target of **sub-200ms latency** (P50/P70) from voice input to response start, traditional sequential HTTP architectures are unsuitable. The system utilizes an **asynchronous event-driven pipeline over WebSockets**, with parallelized execution nodes and low-latency local or streaming APIs.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT SIDE (Browser / Mobile)                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │  ▲
               Binary PCM Audio Stream     │  │  Streaming Answer Text / Metrics
              (WebSocket @ 16kHz Mono)     ▼  │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FASTAPI / ASYNCO SERVER                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   ┌─────────────────────────┐     Text Stream    ┌─────────────────────────────────┐   │
│   │   Sarvam / ElevenLabs   │ ──────────────────>│      HARNESS ORCHESTRATOR       │   │
│   │   WebSocket STT Node    │                    │  - Pydantic Schema Validation   │   │
│   └─────────────────────────┘                    │  - Retries & Fallbacks          │   │
│                                                  └─────────────────────────────────┘   │
│                                                                   │                    │
│                                             ┌─────────────────────┴────────────────┐   │
│                                             │                                      │   │
│                                             ▼                                      ▼   │
│                            ┌───────────────────────────────────┐ ┌───────────────────┐ │
│                            │    Input Guardrail & Moderator    │ │ Context Retrieval │ │
│                            │   - System Prompt Extraction      │ │   Vector & Sparse │ │
│                            │   - Safety / Off-topic Filter     │ └───────────────────┘ │
│                            └───────────────────────────────────┘           │           │
│                                             │                              │           │
│                                             └──────────────────────┬───────┘           │
│                                                                    ▼                   │
│                                                  ┌───────────────────────────────────┐ │
│                                                  │     Stream LLM Generation Engine  │ │
│                                                  │      (Groq Llama-3.1-8B-Instant)  │ │
│                                                  └───────────────────────────────────┘ │
│                                                                    │                   │
│                                                                    ▼                   │
│                                                  ┌───────────────────────────────────┐ │
│                                                  │  Post-Gen Grounding Verification  │ │
│                                                  │   - Citation Verification Check   │ │
│                                                  └───────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
voice-rag-system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI WebSockets & HTTP endpoints
│   ├── config.py               # Envs, API Keys, Threshold configs
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── base.py             # STT Abstract Interface
│   │   └── streaming_stt.py    # Sarvam / ElevenLabs WebSocket Handler
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── parent_child.py     # Metadata-aware Parent-Child Chunker
│   │   ├── semantic.py         # Distance-based Semantic Chunker
│   │   └── hybrid_indexer.py   # Vector + Sparse BM25 Indexing
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── qdrant_client.py    # In-memory / GPU Qdrant Wrapper
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Main workflow execution engine
│   │   ├── guardrails.py       # Input/Output validation & safety checks
│   │   └── latency_tracker.py  # Precise microsecond timing for P50/P70/P100
│   └── llm/
│       ├── __init__.py
│       ├── groq_client.py      # Low latency LLM streaming client
│       └── prompts.py          # System prompts & few-shot examples
├── benchmark/
│   ├── benchmark_runner.py     # Evaluates 100+ queries & generates P50/P70/P100
│   └── sample_queries.json
├── dataset_ingest.py           # MSMARCO-XI download and multi-strategy index script
├── requirements.txt
└── README.md
```

---

## 3. Modular Code Implementation

### A. Modular STT Interface (`app/stt/streaming_stt.py`)
```python
import asyncio
import json
import websockets
from app.config import settings

class StreamingSTT:
    def __init__(self, provider: str = "sarvam"):
        self.provider = provider
        self.api_key = settings.SARVAM_API_KEY if provider == "sarvam" else settings.ELEVENLABS_API_KEY

    async def stream_audio_to_text(self, audio_chunk_stream):
        """
        Consumes raw PCM audio chunks from a client queue and yields partial/final text transcripts.
        """
        if self.provider == "sarvam":
            uri = "wss://api.sarvam.ai/v1/speech-to-text-translate/ws"
            headers = {"api-subscription-key": self.api_key}
        else:
            uri = "wss://api.elevenlabs.io/v1/speech-to-text?model_id=scribe_v1"
            headers = {"xi-api-key": self.api_key}

        async with websockets.connect(uri, extra_headers=headers) as ws:
            async def send_audio():
                async for chunk in audio_chunk_stream:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "eof"}))

            sender_task = asyncio.create_task(send_audio())

            async for message in ws:
                data = json.loads(message)
                if data.get("type") == "transcript_final":
                    yield data["text"]
                    break
                elif data.get("type") == "transcript_partial":
                    yield data["text"]

            await sender_task
```

### B. Multi-Strategy Chunking & Indexing (`app/chunking/hybrid_indexer.py`)
```python
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import fastembed

class HybridIndexer:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str = "msmarco_xi"):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = fastembed.TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Parent-Child Splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        self.child_splitter = SentenceTransformersTokenTextSplitter(chunk_size=128, chunk_overlap=16)

    def process_and_index(self, documents: List[Dict[str, Any]]):
        """
        Multi-Strategy Chunking Pipeline:
        1. Generates Parent Chunks (512 tokens) for context retention.
        2. Splits Parent Chunks into Child Chunks (128 tokens) for fine-grained retrieval vectors.
        3. Attaches Parent context directly in Child Chunk metadata.
        """
        points = []
        point_id = 0

        for doc in documents:
            parent_chunks = self.parent_splitter.split_text(doc["passage_text"])
            for parent_idx, parent_text in enumerate(parent_chunks):
                child_chunks = self.child_splitter.split_text(parent_text)
                
                for child_idx, child_text in enumerate(child_chunks):
                    vector = list(self.embedding_model.embed([child_text]))[0]
                    
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector.tolist(),
                            payload={
                                "child_text": child_text,
                                "parent_context": parent_text,
                                "doc_id": doc.get("doc_id"),
                                "language": doc.get("language", "en"),
                                "chunk_strategy": "parent_child_hybrid"
                            }
                        )
                    )
                    point_id += 1

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
```

### C. System Orchestration Harness (`app/harness/orchestrator.py`)
```python
import time
from app.harness.guardrails import GuardrailEngine
from app.harness.latency_tracker import LatencyTracker
from app.llm.groq_client import LowLatencyLLM

class RAGOrchestrator:
    def __init__(self, qdrant_client, llm_client: LowLatencyLLM):
        self.qdrant = qdrant_client
        self.llm = llm_client
        self.guardrail = GuardrailEngine()
        self.latency = LatencyTracker()

    async def execute_pipeline(self, user_query: str):
        start_time = time.perf_counter()
        
        # 1. Input Guardrail
        is_safe, refusal_reason = self.guardrail.validate_input(user_query)
        if not is_safe:
            return {
                "response": f"I cannot process this query: {refusal_reason}",
                "grounded": False,
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }

        # 2. Context Retrieval (Hybrid Search)
        retrieval_start = time.perf_counter()
        retrieved_docs = await self._retrieve_context(user_query, top_k=3)
        retrieval_duration = (time.perf_counter() - retrieval_start) * 1000

        # 3. Context Grounding Gate
        if not retrieved_docs or retrieved_docs[0]["score"] < 0.45:
            return {
                "response": "The knowledge base does not contain enough information to answer this question accurately.",
                "grounded": False,
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }

        # 4. LLM Answer Generation
        context_str = "\n---\n".join([d["parent_context"] for d in retrieved_docs])
        llm_start = time.perf_counter()
        
        response_stream = self.llm.generate_stream(query=user_query, context=context_str)
        
        total_latency = (time.perf_counter() - start_time) * 1000
        self.latency.record(total_latency)

        return {
            "stream": response_stream,
            "retrieval_ms": retrieval_duration,
            "total_latency_ms": total_latency,
            "grounded": True
        }

    async def _retrieve_context(self, query: str, top_k: int):
        pass
```

---

## 4. System Prompts & Guardrail Logic

### System Prompt (`app/llm/prompts.py`)
```python
SYSTEM_RAG_PROMPT = """You are a low-latency, highly accurate voice assistant powered by a RAG retrieval system.

CRITICAL INSTRUCTIONS:
1. Answer the user's question ONLY using the facts provided in the [CONTEXT] block below.
2. If the answer cannot be directly derived from the [CONTEXT], state clearly: "I cannot find this information in the provided records."
3. Do NOT use any pre-trained external knowledge outside the provided [CONTEXT].
4. Keep your response concise (maximum 2-3 short sentences), direct, and optimized for spoken text output.
5. Do NOT include markdown symbols (*, #, `), bullet points, or complex formatting in your output.

[CONTEXT]
{context}

[USER QUESTION]
{query}

[ANSWER]"""
```

### Guardrail Logic (`app/harness/guardrails.py`)
```python
import re

class GuardrailEngine:
    def __init__(self):
        self.blocked_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are now an unfiltered AI",
            r"jailbreak"
        ]

    def validate_input(self, text: str) -> tuple[bool, str]:
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Prompt injection attempt detected."

        if len(text.strip()) < 3:
            return False, "Query too short."
            
        return True, ""
```
