"""
Microsecond-Precision Latency Tracking & Analytics Engine.
Records node-level latency metrics (STT, Guardrail, Retrieval, LLM TTFT, Total)
and computes statistical P50, P70, P90, and P100 distributions.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    guardrail_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)

class LatencyTracker:
    def __init__(self):
        self.records: List[LatencyBreakdown] = []

    def record(self, breakdown: LatencyBreakdown):
        """Appends a new latency measurement."""
        self.records.append(breakdown)

    def get_percentiles(self) -> Dict[str, Dict[str, float]]:
        """
        Computes P50 (Median), P70, P90, and P100 (Worst Case) metrics across all runs.
        """
        if not self.records:
            return {
                "P50": {"total_ms": 0.0, "retrieval_ms": 0.0, "llm_ttft_ms": 0.0, "stt_ms": 0.0},
                "P70": {"total_ms": 0.0, "retrieval_ms": 0.0, "llm_ttft_ms": 0.0, "stt_ms": 0.0},
                "P90": {"total_ms": 0.0, "retrieval_ms": 0.0, "llm_ttft_ms": 0.0, "stt_ms": 0.0},
                "P100": {"total_ms": 0.0, "retrieval_ms": 0.0, "llm_ttft_ms": 0.0, "stt_ms": 0.0},
            }

        totals = [r.total_pipeline_ms for r in self.records]
        retrievals = [r.retrieval_ms for r in self.records]
        ttfts = [r.llm_ttft_ms for r in self.records]
        stts = [r.stt_ms for r in self.records]

        return {
            "P50": {
                "total_ms": float(np.percentile(totals, 50)),
                "retrieval_ms": float(np.percentile(retrievals, 50)),
                "llm_ttft_ms": float(np.percentile(ttfts, 50)),
                "stt_ms": float(np.percentile(stts, 50)),
            },
            "P70": {
                "total_ms": float(np.percentile(totals, 70)),
                "retrieval_ms": float(np.percentile(retrievals, 70)),
                "llm_ttft_ms": float(np.percentile(ttfts, 70)),
                "stt_ms": float(np.percentile(stts, 70)),
            },
            "P90": {
                "total_ms": float(np.percentile(totals, 90)),
                "retrieval_ms": float(np.percentile(retrievals, 90)),
                "llm_ttft_ms": float(np.percentile(ttfts, 90)),
                "stt_ms": float(np.percentile(stts, 90)),
            },
            "P100": {
                "total_ms": float(np.percentile(totals, 100)),
                "retrieval_ms": float(np.percentile(retrievals, 100)),
                "llm_ttft_ms": float(np.percentile(ttfts, 100)),
                "stt_ms": float(np.percentile(stts, 100)),
            },
            "sample_count": len(self.records),
        }

    def clear(self):
        self.records.clear()
