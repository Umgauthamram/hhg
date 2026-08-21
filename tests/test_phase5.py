import os
import sys
import asyncio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Force utf-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytest
from app.llm.groq_client import LowLatencyLLM
from app.config import settings

SAMPLE_CONTEXT = (
    "A corporation is a legal entity that is separate and distinct from its owners. "
    "Corporations enjoy most of the rights and responsibilities that individuals possess: "
    "they can enter contracts, loan and borrow money, sue and be sued, hire employees, own assets, and pay taxes."
)

SAMPLE_INDIC_CONTEXT = (
    "प्रकाश संश्लेषण वह जैविक प्रक्रिया है जिसके द्वारा हरे पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलते हैं। "
    "इस प्रक्रिया में कार्बन डाइऑक्साइड और पानी का उपयोग करके ग्लूकोज और ऑक्सीजन का उत्पादन होता है।"
)

@pytest.mark.asyncio
async def test_llm_streaming_and_ttft():
    llm = LowLatencyLLM()
    query = "What is a corporation?"
    
    tokens = []
    ttft_recorded = 0.0
    final_payload = None

    async for chunk in llm.generate_stream(query=query, context=SAMPLE_CONTEXT):
        if chunk.get("is_first"):
            ttft_recorded = chunk.get("ttft_ms", 0.0)
        if chunk.get("token"):
            tokens.append(chunk["token"])
        if chunk.get("is_done"):
            final_payload = chunk

    assert len(tokens) > 0, "Should receive streamed tokens"
    assert ttft_recorded > 0, "Should record positive TTFT"
    print(f"\n[Benchmark] LLM TTFT (Time-to-First-Token): {ttft_recorded:.2f}ms")
    print(f"[Benchmark] Total Generation Latency: {final_payload['total_ms']:.2f}ms")
    print(f"[Output] Streamed Answer: {final_payload['full_text']}")
    
    # Assert voice output formatting: clean spoken text
    assert "*" not in final_payload["full_text"], "Spoken output should not contain asterisks"
    assert "#" not in final_payload["full_text"], "Spoken output should not contain headers"
    assert "`" not in final_payload["full_text"], "Spoken output should not contain code ticks"
    assert len(final_payload["full_text"]) > 10

@pytest.mark.asyncio
async def test_llm_factual_accuracy():
    llm = LowLatencyLLM()
    query = "Can a corporation enter contracts and hire employees according to the text?"
    
    result = await llm.generate_complete(query=query, context=SAMPLE_CONTEXT)
    print(f"\n[Test Factual Accuracy] Answer: {result['text']}")
    assert "contract" in result["text"].lower() or "employee" in result["text"].lower() or "yes" in result["text"].lower()

@pytest.mark.asyncio
async def test_llm_indic_generation():
    llm = LowLatencyLLM()
    query = "प्रकाश संश्लेषण क्या है?"
    
    result = await llm.generate_complete(query=query, context=SAMPLE_INDIC_CONTEXT)
    print(f"\n[Test Indic] TTFT: {result['ttft_ms']:.2f}ms | Answer: {result['text']}")
    assert len(result["text"]) > 5

if __name__ == "__main__":
    asyncio.run(test_llm_streaming_and_ttft())
    asyncio.run(test_llm_factual_accuracy())
    asyncio.run(test_llm_indic_generation())
    print("\nPhase 5 tests passed successfully!")
