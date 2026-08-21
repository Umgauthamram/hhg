"""
In-Memory Qdrant Vector Store wrapper optimized for sub-10ms retrieval.
Supports in-memory HNSW index configurations and rich payload search.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from app.config import settings

class QdrantVectorStore:
    def __init__(self, collection_name: str = None, vector_size: int = 384):
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.vector_size = vector_size
        self._client = None
        self._is_native = False
        self._fallback_vectors: List[np.ndarray] = []
        self._fallback_payloads: List[Dict[str, Any]] = []
        self._fallback_ids: List[str] = []
        
        self._init_qdrant()

    def _init_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance, HnswConfigDiff
            
            self._client = QdrantClient(":memory:")
            # Create or recreate collection with optimized HNSW index
            try:
                self._client.delete_collection(collection_name=self.collection_name)
            except Exception:
                pass

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                )
            )
            self._is_native = True
            print(f"[Qdrant] Initialized in-memory Qdrant collection: '{self.collection_name}'")
        except Exception as e:
            print(f"[Qdrant] Native Qdrant client unavailable ({e}), using optimized in-memory fallback vector index.")
            self._is_native = False

    def upsert_points(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> int:
        """Upserts vector points with metadata payloads."""
        if not ids:
            return 0

        if self._is_native:
            from qdrant_client.models import PointStruct
            points = [
                PointStruct(
                    id=int(abs(hash(ids[i]))) % (2**63 - 1),
                    vector=vectors[i],
                    payload={"point_id": ids[i], **payloads[i]}
                )
                for i in range(len(ids))
            ]
            self._client.upsert(collection_name=self.collection_name, points=points)
            return len(points)
        else:
            # Fallback memory store
            for i in range(len(ids)):
                v = np.array(vectors[i], dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                self._fallback_vectors.append(v)
                self._fallback_payloads.append({"point_id": ids[i], **payloads[i]})
                self._fallback_ids.append(ids[i])
            return len(ids)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Searches top-k similar vectors with similarity score."""
        if self._is_native:
            # Compatible with qdrant-client >= 1.10 (query_points) and legacy (search)
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
                points = response.points
            else:
                points = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )

            hits = []
            for res in points:
                hits.append({
                    "id": res.payload.get("point_id", str(res.id)),
                    "score": float(res.score),
                    "payload": res.payload,
                })
            return hits
        else:
            # High-speed numpy matrix dot-product fallback
            if not self._fallback_vectors:
                return []
            
            q = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            
            matrix = np.vstack(self._fallback_vectors)
            scores = np.dot(matrix, q)
            
            top_indices = np.argsort(scores)[::-1][:top_k]
            hits = []
            for idx in top_indices:
                score = float(scores[idx])
                if score >= score_threshold:
                    hits.append({
                        "id": self._fallback_ids[idx],
                        "score": score,
                        "payload": self._fallback_payloads[idx],
                    })
            return hits

    def count(self) -> int:
        """Returns the total number of indexed vectors."""
        if self._is_native:
            info = self._client.get_collection(collection_name=self.collection_name)
            return info.points_count or 0
        return len(self._fallback_vectors)
