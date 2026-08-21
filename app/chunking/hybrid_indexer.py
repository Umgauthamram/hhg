"""
Ultra-Low-Latency Hybrid Sparse-Dense Indexer.
Executes parallel Dense (Qdrant) + Sparse (BM25) search with Reciprocal Rank Fusion (RRF).
Target Retrieval Latency: < 4ms.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from app.chunking.metadata_chunker import UnifiedChunk
from app.vector_store.fast_embedder import FastEmbeddingEngine
from app.vector_store.qdrant_store import QdrantVectorStore
from app.config import settings

class HybridIndexer:
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_engine: Optional[FastEmbeddingEngine] = None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_engine = embedding_engine or FastEmbeddingEngine()
        self.rrf_k = rrf_k
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # BM25 sparse index state
        self._bm25 = None
        self._corpus_chunks: List[UnifiedChunk] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Fast multilingual tokenizer."""
        clean = "".join(c.lower() if c.isalnum() else " " for c in text)
        stopwords = {"the", "a", "an", "is", "in", "to", "of", "and", "or", "that", "it", "this", "on", "for", "as", "with", "by", "are", "be"}
        return [w for w in clean.split() if len(w) > 1 and w not in stopwords]

    def index_chunks(self, chunks: List[UnifiedChunk]) -> int:
        """
        Indexes chunks into both the Dense Vector Store and the Sparse BM25 index.
        """
        if not chunks:
            return 0

        # 1. Dense Vector Embeddings & Ingestion
        texts_to_embed = [c.text for c in chunks]
        vectors = self.embedding_engine.embed_texts(texts_to_embed)
        
        ids = [c.chunk_id for c in chunks]
        payloads = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "parent_context": c.parent_context,
                "doc_id": c.doc_id,
                "language": c.language,
                "strategy": c.strategy,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        
        self.vector_store.upsert_points(ids=ids, vectors=vectors, payloads=payloads)

        # 2. Sparse BM25 Index Setup
        self._corpus_chunks.extend(chunks)
        for c in chunks:
            self._tokenized_corpus.append(self._tokenize(c.text))

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception:
            self._bm25 = None

        return len(chunks)

    def _bm25_search(self, tokens: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Optimized sparse BM25 retrieval."""
        if not tokens or not self._corpus_chunks:
            return []

        if self._bm25 is not None:
            doc_scores = self._bm25.get_scores(tokens)
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
            hits = []
            for idx in top_indices:
                if doc_scores[idx] > 0:
                    c = self._corpus_chunks[idx]
                    hits.append({
                        "id": c.chunk_id,
                        "bm25_score": float(doc_scores[idx]),
                        "vector_score": 0.0,
                        "payload": {
                            "chunk_id": c.chunk_id,
                            "text": c.text,
                            "parent_context": c.parent_context,
                            "doc_id": c.doc_id,
                            "language": c.language,
                            "strategy": c.strategy,
                        }
                    })
            return hits
        else:
            q_set = set(tokens)
            scores = []
            for idx, doc_tokens in enumerate(self._tokenized_corpus):
                overlap = sum(1 for t in doc_tokens if t in q_set)
                scores.append((overlap, idx))
            scores.sort(reverse=True, key=lambda x: x[0])
            hits = []
            for overlap, idx in scores[:top_k]:
                if overlap > 0:
                    c = self._corpus_chunks[idx]
                    hits.append({
                        "id": c.chunk_id,
                        "bm25_score": float(overlap),
                        "vector_score": 0.0,
                        "payload": {
                            "chunk_id": c.chunk_id,
                            "text": c.text,
                            "parent_context": c.parent_context,
                            "doc_id": c.doc_id,
                            "language": c.language,
                            "strategy": c.strategy,
                        }
                    })
            return hits

    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.30,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes parallel dense + sparse retrieval fused via Reciprocal Rank Fusion.
        """
        start_time = time.perf_counter()
        
        # Parallel dense embedding + sparse tokenization
        query_vector = self.embedding_engine.embed_query(query)
        tokens = self._tokenize(query)

        # Dense search
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k * 2,
            score_threshold=similarity_threshold,
        )

        # Sparse search
        sparse_results = self._bm25_search(tokens, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Dense ranks
        for rank, hit in enumerate(dense_results):
            cid = hit["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            doc_map[cid] = {
                "vector_score": hit.get("score", 0.0),
                "bm25_score": 0.0,
                "payload": hit["payload"],
            }

        # Sparse ranks
        for rank, hit in enumerate(sparse_results):
            cid = hit["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if cid not in doc_map:
                doc_map[cid] = {
                    "vector_score": 0.0,
                    "bm25_score": hit.get("bm25_score", 0.0),
                    "payload": hit["payload"],
                }
            else:
                doc_map[cid]["bm25_score"] = hit.get("bm25_score", 0.0)

        # Sort by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        final_hits = []
        for cid in sorted_ids:
            hit_data = doc_map[cid]
            final_hits.append({
                "chunk_id": cid,
                "fused_score": rrf_scores[cid],
                "vector_score": hit_data.get("vector_score", 0.0),
                "bm25_score": hit_data.get("bm25_score", 0.0),
                "text": hit_data["payload"].get("text", ""),
                "parent_context": hit_data["payload"].get("parent_context", ""),
                "doc_id": hit_data["payload"].get("doc_id", ""),
                "language": hit_data["payload"].get("language", "en"),
                "strategy": hit_data["payload"].get("strategy", ""),
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return final_hits, latency_ms
