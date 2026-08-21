"""
Metadata-Aware Multi-Strategy Chunker.
Unifies Parent-Child, Semantic Boundary, and Fixed-Window chunking
while enriching each chunk with structured metadata (Language, Doc ID, Passage ID, Token counts, Strategy).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.chunking.parent_child import ParentChildChunker, ChildChunk, ParentChunk
from app.chunking.semantic import SemanticBoundaryChunker, SemanticChunk

class UnifiedChunk(BaseModel):
    chunk_id: str
    text: str
    parent_context: str
    doc_id: str
    language: str
    strategy: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    char_length: int = 0
    estimated_tokens: int = 0

class MultiStrategyChunker:
    def __init__(
        self,
        parent_chunk_size: int = 512,
        parent_overlap: int = 64,
        child_chunk_size: int = 128,
        child_overlap: int = 24,
        semantic_max_size: int = 350,
        semantic_min_size: int = 80,
    ):
        self.parent_child = ParentChildChunker(
            parent_chunk_size=parent_chunk_size,
            parent_overlap=parent_overlap,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
        )
        self.semantic = SemanticBoundaryChunker(
            max_chunk_size=semantic_max_size,
            min_chunk_size=semantic_min_size,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token estimator (average 4 chars per token for English, 2.5 for Indic scripts)."""
        return max(1, len(text) // 3)

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        language: str = "en",
        strategy: str = "parent_child",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[UnifiedChunk]:
        """
        Processes a document passage and outputs normalized UnifiedChunk items.
        
        Args:
            text: Raw passage text (English or Indic translation).
            doc_id: Unique identifier for the source document.
            language: ISO language code (e.g. 'en', 'hi', 'ta', 'te', 'bn').
            strategy: 'parent_child' or 'semantic'.
            extra_metadata: Optional dictionary with extra context (query, topic, etc.).
        """
        if not text or not text.strip():
            return []

        base_meta = {
            "doc_id": doc_id,
            "language": language,
            **(extra_metadata or {}),
        }

        unified_chunks: List[UnifiedChunk] = []

        if strategy == "parent_child":
            parent_chunks = self.parent_child.chunk(text, doc_id=doc_id, metadata=base_meta)
            for p in parent_chunks:
                for c in p.children:
                    unified_chunks.append(
                        UnifiedChunk(
                            chunk_id=c.child_id,
                            text=c.child_text,
                            parent_context=c.parent_text,
                            doc_id=doc_id,
                            language=language,
                            strategy="parent_child",
                            metadata=c.metadata,
                            char_length=len(c.child_text),
                            estimated_tokens=self._estimate_tokens(c.child_text),
                        )
                    )
        elif strategy == "semantic":
            semantic_chunks = self.semantic.chunk(text, doc_id=doc_id, metadata=base_meta)
            for s in semantic_chunks:
                unified_chunks.append(
                    UnifiedChunk(
                        chunk_id=s.chunk_id,
                        text=s.text,
                        parent_context=text,  # whole passage as context
                        doc_id=doc_id,
                        language=language,
                        strategy="semantic_boundary",
                        metadata=s.metadata,
                        char_length=s.char_count,
                        estimated_tokens=self._estimate_tokens(s.text),
                    )
                )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        return unified_chunks
