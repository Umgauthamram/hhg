"""
Latency Analytics & Statistical Benchmark Suite for Voice-Enabled Sub-200ms RAG.
Evaluates 100+ queries across multiple categories, computes P50, P70, P90, P100 percentiles,
and generates component latency breakdown tables.
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, tablefmt="grid"):
        header_str = " | ".join(headers)
        divider = "-" * len(header_str)
        body = "\n".join([" | ".join(str(cell) for cell in row) for row in rows])
        return f"{header_str}\n{divider}\n{body}"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Force utf-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.harness.orchestrator import RAGOrchestrator
from app.chunking.hybrid_indexer import HybridIndexer
from app.llm.groq_client import LowLatencyLLM
from app.stt.streaming_stt import StreamingSTT
from app.harness.guardrails import GuardrailEngine
from dataset_ingest import MSMARCOXIngramIngester

class BenchmarkRunner:
    def __init__(
        self,
        orchestrator: Optional[RAGOrchestrator] = None,
        queries_file: Optional[str] = None,
        ingest_hf: bool = False,
    ):
        self.queries_file = queries_file or os.path.join(BASE_DIR, "benchmark", "sample_queries.json")
        
        if orchestrator is None:
            print("[Benchmark] Initializing dedicated in-memory RAG pipeline...")
            indexer = HybridIndexer()
            ingester = MSMARCOXIngramIngester(hybrid_indexer=indexer)
            ingester.ingest_curated_seed()
            if ingest_hf:
                try:
                    ingester.ingest_from_hf(languages=["hin"], limit_per_lang=10)
                except Exception as e:
                    print(f"[Benchmark] Warning: Could not ingest from HF: {e}")

            self.orchestrator = RAGOrchestrator(
                hybrid_indexer=indexer,
                llm_client=LowLatencyLLM(),
                stt_client=StreamingSTT(),
                guardrail_engine=GuardrailEngine(),
            )
        else:
            self.orchestrator = orchestrator

    def load_queries(self, target_count: int = 100) -> List[Dict[str, Any]]:
        """Loads curated queries and multiplies to reach target evaluation count."""
        with open(self.queries_file, "r", encoding="utf-8") as f:
            base_queries = json.load(f)

        expanded = []
        repeat_factor = (target_count // len(base_queries)) + 1
        for i in range(repeat_factor):
            for q in base_queries:
                expanded.append({
                    **q,
                    "run_id": f"{q['id']}_iter_{i}"
                })
                if len(expanded) >= target_count:
                    break
            if len(expanded) >= target_count:
                break

        return expanded

    async def run_benchmark(self, query_count: int = 25, simulated_stt_ms: float = 30.0) -> Dict[str, Any]:
        """
        Executes end-to-end benchmark across query set and records granular metrics.
        """
        queries = self.load_queries(target_count=query_count)
        print(f"\n=======================================================")
        print(f"  STARTING VOICE RAG LATENCY BENCHMARK ({len(queries)} QUERIES)")
        print(f"  Target Latency Budget: < 200.00 ms (P50 / P70)")
        print(f"=======================================================\n")

        results = []
        category_counts = {}

        # Warm up pipeline
        _ = await self.orchestrator.execute_query("Warmup query")

        for idx, q_item in enumerate(queries):
            q_text = q_item["query"]
            lang = q_item.get("language", "en")
            cat = q_item.get("category", "general")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            start_t = time.perf_counter()
            response = await self.orchestrator.execute_query(query=q_text, language=lang)
            total_e2e_ms = (time.perf_counter() - start_t) * 1000.0 + simulated_stt_ms

            # Adjust breakdown with simulated STT latency
            response.latency.stt_ms = simulated_stt_ms
            response.latency.total_pipeline_ms = total_e2e_ms

            results.append({
                "run_id": q_item.get("run_id", f"q_{idx}"),
                "query": q_text,
                "category": cat,
                "language": lang,
                "is_safe": response.is_safe,
                "is_grounded": response.is_grounded,
                "stt_ms": response.latency.stt_ms,
                "guardrail_ms": response.latency.guardrail_ms,
                "retrieval_ms": response.latency.retrieval_ms,
                "llm_ttft_ms": response.latency.llm_ttft_ms,
                "llm_total_ms": response.latency.llm_total_ms,
                "total_e2e_ms": total_e2e_ms,
                "answer_snippet": response.answer[:60] + "..." if len(response.answer) > 60 else response.answer,
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == len(queries):
                print(f"  Processed {idx+1}/{len(queries)} queries | Last Total: {total_e2e_ms:.2f}ms | Retrieval: {response.latency.retrieval_ms:.2f}ms")

        # Compute statistical distributions
        df = pd.DataFrame(results)

        metrics_summary = {}
        for col in ["stt_ms", "guardrail_ms", "retrieval_ms", "llm_ttft_ms", "total_e2e_ms"]:
            vals = df[col].values
            metrics_summary[col] = {
                "P50": float(np.percentile(vals, 50)),
                "P70": float(np.percentile(vals, 70)),
                "P90": float(np.percentile(vals, 90)),
                "P100": float(np.percentile(vals, 100)),
                "Mean": float(np.mean(vals)),
                "Min": float(np.min(vals)),
                "Std": float(np.std(vals)),
            }

        # Build Formatted Table
        table_rows = [
            [
                "STT Transcription",
                f"{metrics_summary['stt_ms']['P50']:.2f} ms",
                f"{metrics_summary['stt_ms']['P70']:.2f} ms",
                f"{metrics_summary['stt_ms']['P90']:.2f} ms",
                f"{metrics_summary['stt_ms']['P100']:.2f} ms",
                f"{metrics_summary['stt_ms']['Mean']:.2f} ms",
            ],
            [
                "Input Guardrails & Security",
                f"{metrics_summary['guardrail_ms']['P50']:.2f} ms",
                f"{metrics_summary['guardrail_ms']['P70']:.2f} ms",
                f"{metrics_summary['guardrail_ms']['P90']:.2f} ms",
                f"{metrics_summary['guardrail_ms']['P100']:.2f} ms",
                f"{metrics_summary['guardrail_ms']['Mean']:.2f} ms",
            ],
            [
                "Hybrid Retrieval (Qdrant + BM25)",
                f"{metrics_summary['retrieval_ms']['P50']:.2f} ms",
                f"{metrics_summary['retrieval_ms']['P70']:.2f} ms",
                f"{metrics_summary['retrieval_ms']['P90']:.2f} ms",
                f"{metrics_summary['retrieval_ms']['P100']:.2f} ms",
                f"{metrics_summary['retrieval_ms']['Mean']:.2f} ms",
            ],
            [
                "LLM TTFT (Time-To-First-Token)",
                f"{metrics_summary['llm_ttft_ms']['P50']:.2f} ms",
                f"{metrics_summary['llm_ttft_ms']['P70']:.2f} ms",
                f"{metrics_summary['llm_ttft_ms']['P90']:.2f} ms",
                f"{metrics_summary['llm_ttft_ms']['P100']:.2f} ms",
                f"{metrics_summary['llm_ttft_ms']['Mean']:.2f} ms",
            ],
            [
                "TOTAL PIPELINE (End-to-End)",
                f"{metrics_summary['total_e2e_ms']['P50']:.2f} ms",
                f"{metrics_summary['total_e2e_ms']['P70']:.2f} ms",
                f"{metrics_summary['total_e2e_ms']['P90']:.2f} ms",
                f"{metrics_summary['total_e2e_ms']['P100']:.2f} ms",
                f"{metrics_summary['total_e2e_ms']['Mean']:.2f} ms",
            ],
        ]

        headers = ["Pipeline Component", "P50 (Median)", "P70", "P90", "P100 (Worst)", "Mean"]
        ascii_table = tabulate(table_rows, headers=headers, tablefmt="fancy_grid")

        print("\n" + ascii_table + "\n")

        p50_total = metrics_summary["total_e2e_ms"]["P50"]
        p70_total = metrics_summary["total_e2e_ms"]["P70"]
        p100_total = metrics_summary["total_e2e_ms"]["P100"]

        print(f"Summary Verification:")
        print(f"  • Total Evaluated Queries: {len(queries)}")
        print(f"  • Categories: {category_counts}")
        print(f"  • P50 Latency: {p50_total:.2f} ms")
        print(f"  • P70 Latency: {p70_total:.2f} ms")
        print(f"  • P100 Max Worst Case: {p100_total:.2f} ms")

        # Save results JSON
        output_path = os.path.join(BASE_DIR, "benchmark", "benchmark_results.json")
        benchmark_payload = {
            "timestamp": time.time(),
            "query_count": len(queries),
            "category_distribution": category_counts,
            "metrics": metrics_summary,
            "detailed_runs": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_payload, f, indent=2, ensure_ascii=False)
        print(f"  • Results saved to: {output_path}\n")

        return benchmark_payload

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG Latency Benchmark")
    parser.add_argument("--count", type=int, default=25, help="Number of queries to benchmark")
    parser.add_argument("--hf", action="store_true", help="Include online HF ingestion")
    args = parser.parse_args()

    runner = BenchmarkRunner(ingest_hf=args.hf)
    asyncio.run(runner.run_benchmark(query_count=args.count))
