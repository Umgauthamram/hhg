"""
Semantic Boundary Chunker.
Splits text along linguistic and sentence boundaries, respecting both
English punctuation (., !, ?) and Indic sentence terminators (।, ॥, \n\n).
Groups cohesive sentences into contextually coherent units.
"""

import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class SemanticChunk(BaseModel):
    chunk_id: str
    text: str
    sentence_count: int
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SemanticBoundaryChunker:
    def __init__(
        self,
        max_chunk_size: int = 350,
        min_chunk_size: int = 80,
        sentence_overlap: int = 1,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_overlap = sentence_overlap
        
        # Regex matching sentence boundaries for English and Indic scripts
        # Includes Latin (.!?), Devanagari/Bengali danda (। ॥), Urdu/Arabic full stop (۔), and double newlines
        self.sentence_pattern = re.compile(
            r'(?<=[.!?।॥۔\n])\s+'
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Splits text into list of sentences respecting multilingual punctuation."""
        raw_sentences = self.sentence_pattern.split(text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def chunk(self, text: str, doc_id: str = "doc_0", metadata: Dict[str, Any] = None) -> List[SemanticChunk]:
        """
        Groups cohesive sentences into semantic chunks without severing thought units mid-sentence.
        """
        if metadata is None:
            metadata = {}
            
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[SemanticChunk] = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_len = len(sent)

            # If adding this sentence exceeds max_chunk_size and we have accumulated enough
            if current_len + sent_len > self.max_chunk_size and current_len >= self.min_chunk_size:
                chunk_text = " ".join(current_sentences)
                chunk_id = f"{doc_id}_sem_{chunk_idx}"
                chunk_meta = {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "strategy": "semantic_boundary",
                }
                chunks.append(
                    SemanticChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        sentence_count=len(current_sentences),
                        char_count=len(chunk_text),
                        metadata=chunk_meta,
                    )
                )
                chunk_idx += 1

                # Overlap: keep last `sentence_overlap` sentences
                if self.sentence_overlap > 0 and len(current_sentences) > self.sentence_overlap:
                    current_sentences = current_sentences[-self.sentence_overlap:]
                    current_len = sum(len(s) + 1 for s in current_sentences)
                else:
                    current_sentences = []
                    current_len = 0

            current_sentences.append(sent)
            current_len += sent_len + 1
            i += 1

        # Add remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = f"{doc_id}_sem_{chunk_idx}"
            chunk_meta = {
                **metadata,
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "strategy": "semantic_boundary",
            }
            chunks.append(
                SemanticChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    sentence_count=len(current_sentences),
                    char_count=len(chunk_text),
                    metadata=chunk_meta,
                )
            )

        return chunks
