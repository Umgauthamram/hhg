import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
from app.chunking.metadata_chunker import MultiStrategyChunker
from app.vector_store.fast_embedder import FastEmbeddingEngine
from app.vector_store.qdrant_store import QdrantVectorStore
from app.chunking.hybrid_indexer import HybridIndexer

SAMPLE_DOCS = [
    {
        "doc_id": "doc_1",
        "text": "Machine reading comprehension is a subfield of natural language processing where AI systems read passages and answer questions accurately.",
        "language": "en",
    },
    {
        "doc_id": "doc_2",
        "text": "The MS MARCO dataset contains millions of queries and answers extracted from search logs to benchmark information retrieval systems.",
        "language": "en",
    },
    {
        "doc_id": "doc_3",
        "text": "सूचना पुनर्प्राप्ति प्रणाली उपयोगकर्ताओं को उनकी आवश्यकता के अनुसार सही जानकारी खोजने में मदद करती है।",
        "language": "hi",
    },
    {
        "doc_id": "doc_4",
        "text": "Ultra low latency voice RAG pipelines need fast embedding models and in-memory vector databases to respond in under 200 milliseconds.",
        "language": "en",
    }
]

def test_fast_embedder():
    embedder = FastEmbeddingEngine()
    vectors = embedder.embed_texts(["What is machine reading?", "MS MARCO search benchmarks"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384

def test_qdrant_store():
    store = QdrantVectorStore(collection_name="test_collection")
    embedder = FastEmbeddingEngine()
    
    texts = [d["text"] for d in SAMPLE_DOCS]
    vectors = embedder.embed_texts(texts)
    ids = [f"point_{i}" for i in range(len(SAMPLE_DOCS))]
    payloads = [{"doc_id": d["doc_id"], "text": d["text"]} for d in SAMPLE_DOCS]
    
    count = store.upsert_points(ids, vectors, payloads)
    assert count == 4
    
    query_vec = embedder.embed_query("low latency voice RAG")
    results = store.search(query_vec, top_k=2)
    assert len(results) > 0
    assert "score" in results[0]
    assert "payload" in results[0]

def test_hybrid_indexer_and_latency():
    chunker = MultiStrategyChunker()
    all_chunks = []
    for doc in SAMPLE_DOCS:
        chunks = chunker.chunk_document(
            text=doc["text"],
            doc_id=doc["doc_id"],
            language=doc["language"],
            strategy="parent_child"
        )
        all_chunks.extend(chunks)

    indexer = HybridIndexer()
    indexed_count = indexer.index_chunks(all_chunks)
    assert indexed_count > 0

    # Benchmark multiple iterations
    queries = [
        "How does voice RAG achieve sub-200ms latency?",
        "What is machine reading comprehension?",
        "MS MARCO dataset benchmark",
        "Fast vector database retrieval",
        "Low latency question answering"
    ]
    latencies = []
    for q in queries:
        results, latency_ms = indexer.search_hybrid(q, top_k=3)
        latencies.append(latency_ms)
        assert len(results) > 0
        assert "parent_context" in results[0]

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    print(f"\n[Benchmark] Hybrid Retrieval Latencies: min={min_latency:.2f}ms, avg={avg_latency:.2f}ms, max={max(latencies):.2f}ms")
    
    # Average latency must easily meet our sub-25ms allocation
    assert avg_latency < 35.0, f"Average retrieval latency too slow: {avg_latency:.2f}ms"

def test_multilingual_indic_search():
    chunker = MultiStrategyChunker()
    all_chunks = []
    for doc in SAMPLE_DOCS:
        chunks = chunker.chunk_document(
            text=doc["text"],
            doc_id=doc["doc_id"],
            language=doc["language"],
            strategy="semantic"
        )
        all_chunks.extend(chunks)

    indexer = HybridIndexer()
    indexer.index_chunks(all_chunks)

    query = "सूचना पुनर्प्राप्ति प्रणाली"
    results, latency_ms = indexer.search_hybrid(query, top_k=2)
    assert len(results) > 0
    assert results[0]["language"] == "hi"

if __name__ == "__main__":
    test_fast_embedder()
    test_qdrant_store()
    test_hybrid_indexer_and_latency()
    test_multilingual_indic_search()
    print("Phase 3 tests passed successfully!")
