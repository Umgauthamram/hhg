import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Force utf-8 for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytest
from dataset_ingest import MSMARCOXIngramIngester
from app.chunking.hybrid_indexer import HybridIndexer

def test_seed_ingestion():
    indexer = HybridIndexer()
    ingester = MSMARCOXIngramIngester(hybrid_indexer=indexer)
    count = ingester.ingest_curated_seed()
    assert count > 0, "Seed ingestion should populate chunks"

    # Search for photosynthesis in English
    results_en, lat_en = indexer.search_hybrid("What does photosynthesis produce?", top_k=2)
    assert len(results_en) > 0
    assert "photosynthesis" in results_en[0]["parent_context"].lower() or "oxygen" in results_en[0]["parent_context"].lower()

    # Search for photosynthesis in Hindi
    results_hi, lat_hi = indexer.search_hybrid("प्रकाश संश्लेषण से क्या बनता है?", top_k=2)
    assert len(results_hi) > 0
    assert results_hi[0]["language"] == "hin"

def test_hf_msmarco_xi_ingestion():
    indexer = HybridIndexer()
    ingester = MSMARCOXIngramIngester(hybrid_indexer=indexer)
    
    # Ingest a small batch of 5 samples from Hindi validation split
    count = ingester.ingest_from_hf(
        dataset_repo="gauthamram/MSMARCO-XI",
        languages=["hin"],
        limit_per_lang=5
    )
    assert count > 0, "Should ingest chunks from MSMARCO-XI"
    assert len(ingester.ingested_samples) == 5

    # Retrieve using one of the ingested English queries
    sample = ingester.ingested_samples[0]
    test_query = sample["eng_query"]
    print(f"\n[Test] Querying ingested topic: '{test_query}'")
    
    results, lat_ms = indexer.search_hybrid(test_query, top_k=3)
    assert len(results) > 0
    print(f"[Test] Top result score: {results[0]['fused_score']:.4f} | Latency: {lat_ms:.2f}ms")
    assert "parent_context" in results[0]
    assert len(results[0]["parent_context"]) > 20

if __name__ == "__main__":
    test_seed_ingestion()
    test_hf_msmarco_xi_ingestion()
    print("\nPhase 4 tests passed successfully!")
