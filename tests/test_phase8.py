import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Force utf-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytest
from fastapi.testclient import TestClient
from app.main import app

def run_all_phase8_tests():
    # Use a single TestClient instance across all tests
    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/api/health")
        assert res.status_code == 200
        health = res.json()
        assert health["status"] == "healthy"
        assert health["vector_store_count"] > 0
        print(f"\n[Test Health] Status: {health['status']} | Vectors Indexed: {health['vector_store_count']}")

        # 2. REST Query
        payload = {
            "query": "What is a corporation?",
            "language": "en",
            "top_k": 3
        }
        res = client.post("/api/query", json=payload)
        assert res.status_code == 200
        query_data = res.json()
        assert query_data["is_safe"] is True
        assert query_data["is_grounded"] is True
        assert len(query_data["answer"]) > 10
        lat = query_data["latency"]
        print(f"\n[Test REST Query] Answer: {query_data['answer']}")
        print(f"[REST Latency] Retrieval: {lat['retrieval_ms']:.2f}ms | Total: {lat['total_pipeline_ms']:.2f}ms")

        # 3. Benchmark stats
        res = client.get("/api/benchmark/stats")
        assert res.status_code == 200
        stats = res.json()
        assert "P50" in stats
        assert "P70" in stats
        assert "P100" in stats
        print(f"\n[Test Stats] Sample count: {stats['sample_count']}")

        # 4. WebSocket Text Stream
        with client.websocket_connect("/ws/voice-rag") as ws:
            ws.send_json({"type": "config", "language": "en", "top_k": 3})
            ack = ws.receive_json()
            assert ack["event"] == "config_ack"

            ws.send_json({"type": "query", "text": "What is photosynthesis?"})
            
            received_tokens = []
            final_msg = None

            while True:
                msg = ws.receive_json()
                if msg["event"] == "token":
                    received_tokens.append(msg["token"])
                elif msg["event"] == "complete":
                    final_msg = msg
                    break

            assert len(received_tokens) > 0
            assert final_msg is not None
            assert final_msg["is_grounded"] is True
            print(f"\n[Test WS Stream] Full Answer: {final_msg['full_answer']}")
            print(f"[WS Latency] Total: {final_msg['latency']['total_pipeline_ms']:.2f}ms")

        # 5. WebSocket Binary Audio
        with client.websocket_connect("/ws/voice-rag") as ws:
            audio_bytes = b"\x00\x01" * 800
            ws.send_bytes(audio_bytes)

            events = []
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg["event"] in ["complete", "error", "grounding_rejection"]:
                    break

            assert len(events) >= 1
            print(f"\n[Test WS Audio] Final Event: {events[-1]['event']}")

    print("\nPhase 8 tests passed successfully!")

if __name__ == "__main__":
    run_all_phase8_tests()
