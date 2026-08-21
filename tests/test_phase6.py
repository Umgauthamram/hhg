import os
import sys
import asyncio
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
from app.stt.streaming_stt import StreamingSTT, get_stt_engine
from app.stt.mock_stt import MockSTT
from app.stt.sarvam_stt import SarvamSTT
from app.stt.elevenlabs_stt import ElevenLabsSTT

async def dummy_audio_generator(num_chunks: int = 4):
    """Simulates 16kHz PCM audio chunks streamed from client mic."""
    for i in range(num_chunks):
        # 1600 bytes = 50ms of 16kHz 16-bit mono audio
        yield b"\x00\x01" * 800

@pytest.mark.asyncio
async def test_mock_stt_streaming():
    stt = MockSTT(default_text="What is machine learning?", simulated_latency_ms=35.0)
    
    events = []
    async for event in stt.stream_audio_to_text(dummy_audio_generator(4)):
        events.append(event)

    assert len(events) >= 2, "Should yield partial and final transcripts"
    final_event = events[-1]
    assert final_event["is_final"] is True
    assert final_event["type"] == "transcript_final"
    assert final_event["text"] == "What is machine learning?"
    assert final_event["latency_ms"] > 0
    print(f"\n[Benchmark] Mock STT Final Latency: {final_event['latency_ms']:.2f}ms")
    assert final_event["latency_ms"] < 60.0, f"STT processing too slow: {final_event['latency_ms']:.2f}ms"

@pytest.mark.asyncio
async def test_sarvam_stt_initialization():
    stt = SarvamSTT()
    events = []
    async for event in stt.stream_audio_to_text(dummy_audio_generator(2), language_code="hi-IN"):
        events.append(event)

    assert len(events) >= 1
    assert events[-1]["is_final"] is True
    assert len(events[-1]["text"]) > 0
    print(f"[Test] Sarvam STT Result: {events[-1]['text']} (Lat: {events[-1]['latency_ms']:.2f}ms)")

@pytest.mark.asyncio
async def test_elevenlabs_stt_initialization():
    stt = ElevenLabsSTT()
    events = []
    async for event in stt.stream_audio_to_text(dummy_audio_generator(2), language_code="en"):
        events.append(event)

    assert len(events) >= 1
    assert events[-1]["is_final"] is True
    assert len(events[-1]["text"]) > 0
    print(f"[Test] ElevenLabs STT Result: {events[-1]['text']} (Lat: {events[-1]['latency_ms']:.2f}ms)")

@pytest.mark.asyncio
async def test_streaming_stt_factory():
    stt_mgr = StreamingSTT(provider="mock")
    events = []
    async for event in stt_mgr.transcribe_stream(dummy_audio_generator(3)):
        events.append(event)
    assert len(events) >= 1
    assert events[-1]["is_final"] is True

if __name__ == "__main__":
    asyncio.run(test_mock_stt_streaming())
    asyncio.run(test_sarvam_stt_initialization())
    asyncio.run(test_elevenlabs_stt_initialization())
    asyncio.run(test_streaming_stt_factory())
    print("\nPhase 6 tests passed successfully!")
