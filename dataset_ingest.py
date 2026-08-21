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
                print(f"[Ingest] Warning: Could not stream from HF ({e}). Falling back.")
                continue

            for idx, row in enumerate(ds):
                doc_id = f"{lang}_{row.get('query_id', idx)}"
                passages_dict = row.get("passages", {})
                eng_passages = passages_dict.get("English_passages", [])
                indic_passages = passages_dict.get("Translated_passages", [])
                is_selected_flags = passages_dict.get("is_selected", [])

                query_text = row.get("Eng_Query") or row.get("query", "")
                answer_text = row.get("Eng_Answer") or row.get("Answer", "")

                for p_idx, eng_text in enumerate(eng_passages):
                    if not include_unselected and is_selected_flags and p_idx < len(is_selected_flags):
                        if not is_selected_flags[p_idx]:
                            continue

                    chunks = self.chunker.chunk_document(
                        text=eng_text,
                        doc_id=f"{doc_id}_eng_{p_idx}",
                        language="en",
                        strategy="parent_child",
                        extra_metadata={
                            "query": query_text,
                            "gold_answer": answer_text,
                            "source_lang": row.get("source_lang", "en"),
                            "is_selected": True
                        }
                    )
                    all_chunks.extend(chunks)

                for p_idx, indic_text in enumerate(indic_passages):
                    if not include_unselected and is_selected_flags and p_idx < len(is_selected_flags):
                        if not is_selected_flags[p_idx]:
                            continue

                    chunks = self.chunker.chunk_document(
                        text=indic_text,
                        doc_id=f"{doc_id}_{lang}_{p_idx}",
                        language=lang,
                        strategy="semantic",
                        extra_metadata={
                            "query": row.get("query", ""),
                            "source_lang": lang,
                            "is_selected": True
                        }
                    )
                    all_chunks.extend(chunks)

        if all_chunks:
            print(f"[Ingest] Total multi-strategy chunks created: {len(all_chunks)}. Indexing into Qdrant & BM25...")
            indexed_count = self.indexer.index_chunks(all_chunks)
            print(f"[Ingest] Successfully indexed {indexed_count} chunks into memory.")
            return indexed_count
        return 0

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
            },
            {
                "doc_id": "seed_phone",
                "text": "A mobile phone, cellular phone, or smartphone is a portable telephone that can make and receive calls over a radio frequency link while the user is moving within a telephone service area. Modern smartphones support a wide range of other services such as text messaging, multimedia messaging, email, Internet access, short-range wireless communications, business applications, video games, and digital photography.",
                "language": "en",
                "strategy": "parent_child",
                "extra_meta": {"query": "what is mobile phone"}
            },
            {
                "doc_id": "seed_phone_hi",
                "text": "मोबाइल फोन या सेलुलर फोन एक पोर्टेबल टेलीफोन है जो उपयोगकर्ता के टेलीफोन सेवा क्षेत्र के भीतर स्थानांतरित होने के दौरान रेडियो फ्रीक्वेंसी लिंक पर कॉल कर सकता है और प्राप्त कर सकता है।",
                "language": "hin",
                "strategy": "semantic",
                "extra_meta": {"query": "मोबाइल फोन क्या है?"}
            },
            {
                "doc_id": "seed_computer",
                "text": "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations automatically. Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks including data processing, simulation, and software execution.",
                "language": "en",
                "strategy": "parent_child",
                "extra_meta": {"query": "what is a computer"}
            },
            {
                "doc_id": "seed_internet",
                "text": "The Internet is the global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope.",
                "language": "en",
                "strategy": "parent_child",
                "extra_meta": {"query": "what is internet"}
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
