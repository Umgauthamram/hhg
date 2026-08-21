import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
from app.chunking.parent_child import ParentChildChunker
from app.chunking.semantic import SemanticBoundaryChunker
from app.chunking.metadata_chunker import MultiStrategyChunker

SAMPLE_ENGLISH_TEXT = (
    "The MS MARCO dataset is a large scale machine reading comprehension dataset. "
    "It consists of 1,010,916 questions and 8.8 million passages collected from Bing search results. "
    "The dataset is designed to benchmark deep learning models for question answering and retrieval tasks. "
    "In addition, translations into Indian languages make cross-lingual information retrieval possible. "
    "This system provides sub-200ms latency for voice-enabled querying."
)

SAMPLE_HINDI_TEXT = (
    "एमएस मार्को डेटासेट एक बड़ा मशीन रीडिंग कॉम्प्रिहेंशन डेटासेट है। "
    "इसमें बिंग खोज परिणामों से एकत्र किए गए 10 लाख से अधिक प्रश्न और 88 लाख परिच्छेद शामिल हैं। "
    "यह डेटासेट प्रश्न उत्तर और पुनर्प्राप्ति कार्यों के लिए मॉडल का मूल्यांकन करने के लिए डिज़ाइन किया गया है। "
    "भारतीय भाषाओं में अनुवाद क्रॉस-लिंगुअल सूचना पुनर्प्राप्ति को संभव बनाता है।"
)

def test_parent_child_chunker():
    chunker = ParentChildChunker(
        parent_chunk_size=150,
        parent_overlap=30,
        child_chunk_size=60,
        child_overlap=15
    )
    parents = chunker.chunk(SAMPLE_ENGLISH_TEXT, doc_id="doc_en_1", metadata={"language": "en"})
    
    assert len(parents) > 0, "Should generate at least one parent chunk"
    for p in parents:
        assert p.parent_id.startswith("doc_en_1_p")
        assert len(p.children) > 0, "Each parent should have child chunks"
        for c in p.children:
            assert c.parent_id == p.parent_id
            assert c.parent_text == p.parent_text
            assert len(c.child_text) > 0
            assert c.metadata["language"] == "en"

def test_semantic_boundary_chunker_english():
    chunker = SemanticBoundaryChunker(max_chunk_size=120, min_chunk_size=40, sentence_overlap=1)
    chunks = chunker.chunk(SAMPLE_ENGLISH_TEXT, doc_id="doc_en_2", metadata={"language": "en"})
    
    assert len(chunks) >= 2, "Should split into multiple semantic sentence groups"
    for chunk in chunks:
        assert chunk.sentence_count >= 1
        assert chunk.metadata["strategy"] == "semantic_boundary"
        assert chunk.metadata["language"] == "en"

def test_semantic_boundary_chunker_indic():
    chunker = SemanticBoundaryChunker(max_chunk_size=120, min_chunk_size=40, sentence_overlap=1)
    chunks = chunker.chunk(SAMPLE_HINDI_TEXT, doc_id="doc_hi_1", metadata={"language": "hi"})
    
    assert len(chunks) >= 2, "Should split Hindi danda-separated text into semantic groups"
    for chunk in chunks:
        assert chunk.metadata["language"] == "hi"
        assert "।" in chunk.text or len(chunk.text) > 0

def test_multi_strategy_chunker():
    multi_chunker = MultiStrategyChunker()
    
    # Test Parent-Child strategy
    pc_chunks = multi_chunker.chunk_document(
        text=SAMPLE_ENGLISH_TEXT,
        doc_id="doc_test_1",
        language="en",
        strategy="parent_child",
        extra_metadata={"domain": "science"}
    )
    assert len(pc_chunks) > 0
    assert pc_chunks[0].strategy == "parent_child"
    assert pc_chunks[0].metadata["domain"] == "science"
    assert pc_chunks[0].estimated_tokens > 0

    # Test Semantic strategy
    sem_chunks = multi_chunker.chunk_document(
        text=SAMPLE_HINDI_TEXT,
        doc_id="doc_test_2",
        language="hi",
        strategy="semantic",
        extra_metadata={"domain": "indic"}
    )
    assert len(sem_chunks) > 0
    assert sem_chunks[0].strategy == "semantic_boundary"
    assert sem_chunks[0].language == "hi"
    assert sem_chunks[0].metadata["domain"] == "indic"

def test_edge_cases():
    multi_chunker = MultiStrategyChunker()
    
    # Empty string
    empty_res = multi_chunker.chunk_document("", doc_id="empty_doc")
    assert empty_res == []

    # Single short sentence
    short_res = multi_chunker.chunk_document("Hello world.", doc_id="short_doc")
    assert len(short_res) == 1
    assert short_res[0].text == "Hello world."

if __name__ == "__main__":
    test_parent_child_chunker()
    test_semantic_boundary_chunker_english()
    test_semantic_boundary_chunker_indic()
    test_multi_strategy_chunker()
    test_edge_cases()
    print("Phase 2 tests passed successfully!")
