"""
MSMARCO-XI Dataset Ingestion & Indexing Pipeline.
Streams and loads multilingual passages (Indic & English) from 'gauthamram/MSMARCO-XI',
applies multi-strategy chunking (Parent-Child & Semantic Boundary),
and indexes them into the in-memory Qdrant vector store and BM25 sparse index.
"""

import sys
import os
import argparse
from typing import List, Dict, Any, Optional
from app.chunking.metadata_chunker import MultiStrategyChunker, UnifiedChunk
from app.chunking.hybrid_indexer import HybridIndexer
from app.config import settings

# Force utf-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SUPPORTED_LANG_FILES = {
    "hin": "validation/hinval.parquet",
    "tam": "validation/tamval.parquet",
    "tel": "validation/telval.parquet",
    "ben": "validation/benval.parquet",
    "mar": "validation/marval.parquet",
    "guj": "validation/gujval.parquet",
}

class MSMARCOXIngramIngester:
    def __init__(self, hybrid_indexer: Optional[HybridIndexer] = None):
        self.indexer = hybrid_indexer or HybridIndexer()
        self.chunker = MultiStrategyChunker()
        self.ingested_samples: List[Dict[str, Any]] = []

    def ingest_from_hf(
        self,
        dataset_repo: str = "gauthamram/MSMARCO-XI",
        languages: List[str] = ["hin", "tam", "ben"],
        limit_per_lang: int = 50,
        include_unselected: bool = False,
    ) -> int:
        """
        Streams dataset parquet files from HuggingFace and indexes them into memory.
        """
        from datasets import load_dataset

        total_chunks = 0
        all_chunks: List[UnifiedChunk] = []

        for lang in languages:
            parquet_file = SUPPORTED_LANG_FILES.get(lang, f"validation/{lang}val.parquet")
            print(f"[Ingest] Loading {dataset_repo} ({parquet_file}) with limit={limit_per_lang}...")
            
            try:
                ds = load_dataset(
                    dataset_repo,
                    data_files={"val": parquet_file},
                    split=f"val[:{limit_per_lang}]"
                )
            except Exception as e:
                print(f"[Ingest] Warning: Failed to load {parquet_file}: {e}")
                continue

            for idx, row in enumerate(ds):
                query_id = str(row.get("query_id", f"{lang}_{idx}"))
                eng_query = row.get("Eng_Query", "")
                indic_query = row.get("query", "")
                eng_ans = row.get("Eng_Answer", "")
                indic_ans = row.get("Answer", "")
                
                passages_dict = row.get("passages", {})
                # Support both 'English_passages' and 'passage_text'
                eng_passages = passages_dict.get("English_passages") or passages_dict.get("passage_text", [])
                indic_passages = passages_dict.get("Translated_passages") or passages_dict.get("translated_passages", [])
                is_selected = passages_dict.get("is_selected", [])

                # Keep record of sample for test evaluation
                self.ingested_samples.append({
                    "query_id": query_id,
                    "eng_query": eng_query,
                    "indic_query": indic_query,
                    "eng_answer": eng_ans,
                    "indic_answer": indic_ans,
                    "language": lang,
                })

                for p_idx, eng_text in enumerate(eng_passages):
                    selected = is_selected[p_idx] if p_idx < len(is_selected) else 0
                    if not include_unselected and selected != 1 and p_idx > 1:
                        # Prioritize gold passages and top contexts
                        continue

                    # 1. Chunk English passage with Parent-Child strategy
                    doc_id_en = f"{query_id}_en_{p_idx}"
                    chunks_en = self.chunker.chunk_document(
                        text=eng_text,
                        doc_id=doc_id_en,
                        language="en",
                        strategy="parent_child",
                        extra_metadata={
                            "query_id": query_id,
                            "is_selected": selected,
                            "related_query": eng_query,
                        }
                    )
                    all_chunks.extend(chunks_en)

                    # 2. Chunk Indic translated passage with Semantic strategy
                    if p_idx < len(indic_passages):
                        indic_text = indic_passages[p_idx]
                        doc_id_indic = f"{query_id}_{lang}_{p_idx}"
                        chunks_indic = self.chunker.chunk_document(
                            text=indic_text,
                            doc_id=doc_id_indic,
                            language=lang,
                            strategy="semantic",
                            extra_metadata={
                                "query_id": query_id,
                                "is_selected": selected,
                                "related_query": indic_query,
                            }
                        )
                        all_chunks.extend(chunks_indic)

        print(f"[Ingest] Total multi-strategy chunks created: {len(all_chunks)}. Indexing into Qdrant & BM25...")
        indexed_count = self.indexer.index_chunks(all_chunks)
        print(f"[Ingest] Successfully indexed {indexed_count} chunks into memory.")
        return indexed_count

    def ingest_curated_seed(self) -> int:
        """
        Seeds essential diverse MSMARCO-XI topics for offline fallback/zero-network readiness.
        """
        seed_data = [
            {
                "doc_id": "seed_1",
                "text": "A corporation is a legal entity that is separate and distinct from its owners. Corporations enjoy most of the rights and responsibilities that individuals possess: they can enter contracts, loan and borrow money, sue and be sued, hire employees, own assets, and pay taxes.",
                "language": "en",
                "strategy": "parent_child",
                "extra_meta": {"query": "What is a corporation?"}
            },
            {
                "doc_id": "seed_1_hi",
                "text": "निगम एक कानूनी इकाई है जो अपने मालिकों से अलग और भिन्न होती है। निगमों को अधिकांश अधिकार और जिम्मेदारियां प्राप्त होती हैं जो व्यक्तियों के पास होती हैं। वे अनुबंध कर सकते हैं, ऋण ले सकते हैं और दे सकते हैं, मुकदमा कर सकते हैं और उन पर मुकदमा चलाया जा सकता है।",
                "language": "hin",
                "strategy": "semantic",
                "extra_meta": {"query": "निगम क्या है?"}
            },
            {
                "doc_id": "seed_2",
                "text": "Photosynthesis is the biological process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis in green plants, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.",
                "language": "en",
                "strategy": "parent_child",
                "extra_meta": {"query": "Explain photosynthesis process"}
            },
            {
                "doc_id": "seed_2_hi",
                "text": "प्रकाश संश्लेषण वह जैविक प्रक्रिया है जिसके द्वारा हरे पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलते हैं। इस प्रक्रिया में कार्बन डाइऑक्साइड और पानी का उपयोग करके ग्लूकोज और ऑक्सीजन का उत्पादन होता है।",
                "language": "hin",
                "strategy": "semantic",
                "extra_meta": {"query": "प्रकाश संश्लेषण क्या है?"}
            },
            {
                "doc_id": "seed_3",
                "text": "The Reserve Bank of India (RBI) is India's central bank and regulatory body responsible for the issue and supply of the Indian rupee and the regulation of the Indian banking system. It also manages the country's main payment systems and works to promote its economic development.",
                "language": "en",
                "strategy": "parent_child",
                "extra_meta": {"query": "What is the role of RBI?"}
            },
            {
                "doc_id": "seed_3_hi",
                "text": "भारतीय रिजर्व बैंक (RBI) भारत का केंद्रीय बैंक और नियामक निकाय है जो भारतीय रुपये के नियमन और आपूर्ति तथा बैंकिंग प्रणाली के संचालन के लिए जिम्मेदार है।",
                "language": "hin",
                "strategy": "semantic",
                "extra_meta": {"query": "आरबीआई की क्या भूमिका है?"}
            }
        ]

        all_chunks = []
        for item in seed_data:
            chunks = self.chunker.chunk_document(
                text=item["text"],
                doc_id=item["doc_id"],
                language=item["language"],
                strategy=item["strategy"],
                extra_metadata=item.get("extra_meta")
            )
            all_chunks.extend(chunks)

        indexed = self.indexer.index_chunks(all_chunks)
        print(f"[Ingest] Seeded {indexed} curated benchmark chunks.")
        return indexed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSMARCO-XI Ingestion")
    parser.add_argument("--limit", type=int, default=30, help="Number of queries per language")
    parser.add_argument("--langs", nargs="+", default=["hin", "tam"], help="Languages to ingest (e.g. hin tam ben)")
    parser.add_argument("--seed-only", action="store_true", help="Only load curated seed")
    args = parser.parse_args()

    ingester = MSMARCOXIngramIngester()
    if args.seed_only:
        ingester.ingest_curated_seed()
    else:
        ingester.ingest_from_hf(languages=args.langs, limit_per_lang=args.limit)
