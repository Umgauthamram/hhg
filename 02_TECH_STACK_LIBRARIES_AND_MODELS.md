# Tech Stack, Libraries & Infrastructure Guide
## Dependencies, Dataset Strategy, and Deployment Blueprint

---

## 1. Complete Tech Stack Overview

| Layer | Recommended Choice | Rationale / Alternatives | Target Latency Overhead |
| :--- | :--- | :--- | :--- |
| **Speech-to-Text (STT)** | **Sarvam AI WebSocket** or **ElevenLabs Conversational API** | Sarvam specializes in Indic languages (MSMARCO-XI); ElevenLabs Scribe offers ultra-fast response. | ~50ms - 80ms |
| **Vector Database** | **Qdrant (In-Memory / Rust)** | Blazing fast HNSW index execution; low memory footprint. | ~5ms - 15ms |
| **Embedding Model** | **`BAAI/bge-small-en-v1.5`** or **`fastembed` (ONNX runtime)** | Running locally via ONNX CPU execution eliminates cloud network roundtrips. | ~15ms - 25ms |
| **LLM Inference** | **Groq Cloud API (`llama-3.1-8b-instant`)** | High Speed LPUs (400+ tokens/sec) ensure TTFT < 80ms. | ~50ms - 80ms |
| **Orchestrator/Server** | **FastAPI + Asyncio + WebSockets** | Non-blocking I/O event loop; handles concurrent stream processing natively. | ~2ms - 5ms |
| **Data Ingestion** | **Hugging Face `datasets` + PyArrow** | Native processing for `ai4bharat/MSMARCO-XI`. | N/A (Pre-computed) |

---

## 2. Dataset Strategy: Handling `ai4bharat/MSMARCO-XI`

### Dataset Structure
The dataset contains queries paired with context passages translated across multiple Indian languages alongside original English texts.

### Ingestion Strategy
1. **Language Ingestion Filtering:** Extract both English passages and Indian language translations (`hi`, `ta`, `te`, `bn`, etc.).
2. **Chunking Pipeline:**
   - **Strategy A (Parent-Child):** Create Parent blocks (512 tokens) stored as metadata payloads and Child blocks (128 tokens) converted into 384-dimension vector embeddings.
   - **Strategy B (Semantic Boundary):** Group sentences by distance thresholds using sentence-transformers to retain contextual cohesion.
3. **Hybrid Sparse-Dense Indexing:** Combine Qdrant dense vector search with in-memory BM25 sparse index (via `Tantivy` or `rank_bm25`) fused via Reciprocal Rank Fusion (RRF).

---

## 3. Library & Dependency Requirements

### `requirements.txt`
```text
# Web Framework & Async I/O
fastapi==0.110.0
uvicorn[standard]==0.28.0
websockets==12.0
pydantic==2.6.4

# STT & LLM API Clients
groq==0.4.2
elevenlabs==0.2.27
requests==2.31.0

# Vector Database & Fast Embedding Engine
qdrant-client==1.8.0
fastembed==0.2.6
sentence-transformers==2.5.1
rank-bm25==0.2.2

# Dataset & ML Utils
datasets==2.18.0
pyarrow==15.0.0
numpy==1.26.4
torch==2.2.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Benchmarking & Analytics
pandas==2.2.1
tabulate==0.9.0
```

---

## 4. Local Development vs. Production Setup

### Local Setup (Development & Profiling)
* **Embedding Model:** Local execution using `fastembed` with ONNX Runtime CPU execution provider (Zero network latency).
* **Vector Store:** Local In-Memory Qdrant (`QdrantClient(":memory:")`).
* **STT/LLM:** Cloud API Keys via `env` variables (`SARVAM_API_KEY`, `GROQ_API_KEY`).
* **Profiling Tool:** `time.perf_counter_ns()` wrappers embedded in the harness to print latency tables per node.

### Production Setup (Demo Deployment)
* **Hosting:** Deploy backend on **Vercel / Render / AWS EC2 (c6i.xlarge)** located in regions close to STT/LLM endpoints (e.g., `ap-south-1` Mumbai for low network ping).
* **Vector Store:** Self-hosted Qdrant instance or Qdrant Cloud Cluster.
* **Client Frontend:** Single-page Next.js / React application connected directly via WebSocket.

---

## 5. Latency Analytics & Benchmarking Script

Use the following benchmark module to calculate required P50 / P70 / P100 metrics across 100 test queries.

### `benchmark/benchmark_runner.py`
```python
import asyncio
import time
import numpy as np
import pandas as pd
from app.harness.orchestrator import RAGOrchestrator

async def run_benchmark(orchestrator: RAGOrchestrator, test_queries: list):
    latencies = []
    
    print(f"Starting benchmark across {len(test_queries)} queries...")
    for idx, query in enumerate(test_queries):
        start = time.perf_counter()
        result = await orchestrator.execute_pipeline(query)
        end = time.perf_counter()
        
        total_ms = (end - start) * 1000
        latencies.append(total_ms)
        print(f"Query {idx+1}/{len(test_queries)} | Latency: {total_ms:.2f}ms")

    # Compute percentiles
    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p100 = np.percentile(latencies, 100)

    metrics_df = pd.DataFrame([{
        "Metric": ["P50 (Median)", "P70", "P100 (Max Worst Case)"],
        "Latency (ms)": [f"{p50:.2f} ms", f"{p70:.2f} ms", f"{p100:.2f} ms"]
    }])
    
    print("\n=== PIPELINE LATENCY BENCHMARK RESULTS ===")
    print(metrics_df.to_string(index=False))
    return metrics_df

if __name__ == "__main__":
    pass
```
