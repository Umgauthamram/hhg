"""
Parent-Child Hierarchical Chunker.
Splits text into larger Parent context blocks (e.g., 512 chars/tokens)
and smaller Child search units (e.g., 128 chars/tokens with overlap).
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ChildChunk(BaseModel):
    child_id: str
    child_text: str
    parent_id: str
    parent_text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParentChunk(BaseModel):
    parent_id: str
    parent_text: str
    children: List[ChildChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParentChildChunker:
    def __init__(
        self,
        parent_chunk_size: int = 512,
        parent_overlap: int = 64,
        child_chunk_size: int = 128,
        child_overlap: int = 24,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[tuple[int, int, str]]:
        """Splits text into chunks with start_char, end_char, and chunk string."""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If not at the end, try to snap to nearest space to avoid breaking words
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space > start + (chunk_size // 2):
                    end = last_space

            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append((start, end, chunk_str))
            
            if end >= text_len:
                break
                
            start = max(start + 1, end - overlap)
            
        return chunks

    def chunk(self, text: str, doc_id: str = "doc_0", metadata: Dict[str, Any] = None) -> List[ParentChunk]:
        """
        Creates hierarchical Parent-Child chunks for a document.
        """
        if metadata is None:
            metadata = {}
            
        parent_segments = self._split_text(text, self.parent_chunk_size, self.parent_overlap)
        parents = []

        for p_idx, (p_start, p_end, p_text) in enumerate(parent_segments):
            parent_id = f"{doc_id}_p{p_idx}"
            parent_meta = {**metadata, "doc_id": doc_id, "parent_idx": p_idx, "strategy": "parent_child"}
            
            # Generate child chunks within this parent
            child_segments = self._split_text(p_text, self.child_chunk_size, self.child_overlap)
            children = []
            
            for c_idx, (c_start, c_end, c_text) in enumerate(child_segments):
                child_id = f"{parent_id}_c{c_idx}"
                child_meta = {
                    **parent_meta,
                    "child_idx": c_idx,
                    "parent_id": parent_id,
                }
                child = ChildChunk(
                    child_id=child_id,
                    child_text=c_text,
                    parent_id=parent_id,
                    parent_text=p_text,
                    start_char=p_start + c_start,
                    end_char=p_start + c_end,
                    metadata=child_meta,
                )
                children.append(child)
                
            parent_chunk = ParentChunk(
                parent_id=parent_id,
                parent_text=p_text,
                children=children,
                metadata=parent_meta,
            )
            parents.append(parent_chunk)

        return parents
