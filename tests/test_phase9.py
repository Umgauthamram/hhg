import os
import sys
import json
import asyncio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Force utf-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytest
from benchmark.benchmark_runner import BenchmarkRunner

@pytest.mark.asyncio
async def test_benchmark_runner_execution():
    runner = BenchmarkRunner()
    
    # Run benchmark over 10 sample queries
    results = await runner.run_benchmark(query_count=10, simulated_stt_ms=30.0)
    
    assert results is not None
    assert results["query_count"] == 10
    assert "metrics" in results
    
    metrics = results["metrics"]
    assert "total_e2e_ms" in metrics
    assert "retrieval_ms" in metrics
    assert "llm_ttft_ms" in metrics
    
    p50_total = metrics["total_e2e_ms"]["P50"]
    p70_total = metrics["total_e2e_ms"]["P70"]
    p100_total = metrics["total_e2e_ms"]["P100"]
    p50_retrieval = metrics["retrieval_ms"]["P50"]

    print(f"\n[Test Benchmark] P50 Total: {p50_total:.2f}ms | P70 Total: {p70_total:.2f}ms | P100: {p100_total:.2f}ms")
    print(f"[Test Benchmark] P50 Hybrid Retrieval: {p50_retrieval:.2f}ms")

    # Assert sub-35ms hybrid retrieval
    assert p50_retrieval < 35.0, f"P50 Retrieval too slow: {p50_retrieval:.2f}ms"
    
    # Assert benchmark results JSON file was written
    results_file = os.path.join(BASE_DIR, "benchmark", "benchmark_results.json")
    assert os.path.exists(results_file)
    with open(results_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        assert saved_data["query_count"] == 10

if __name__ == "__main__":
    asyncio.run(test_benchmark_runner_execution())
    print("\nPhase 9 tests passed successfully!")
