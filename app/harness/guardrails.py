"""
Guardrail Engine for Voice-Enabled RAG.
Provides multi-layer security:
1. Input Guardrails: Prompt injection, jailbreak defense, input sanity.
2. Retrieval Grounding Gate: Confidence threshold evaluation (knowing when NOT to answer).
3. Output Verification: Post-generation citation and hallucination detection.
"""

import re
from typing import Tuple, List, Dict, Any

class GuardrailEngine:
    def __init__(self, min_confidence_score: float = 0.015):
        self.min_confidence_score = min_confidence_score
        
        # High-risk adversarial & prompt injection patterns
        self.injection_patterns = [
            re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
            re.compile(r"(reveal|print|show|output|leak)\s+(your\s+)?(system\s+prompt|instructions|rules)", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+(an?\s+unfiltered|in\s+dan\s+mode|jailbroken)", re.IGNORECASE),
            re.compile(r"override\s+(all\s+)?(security|safety|moderation)", re.IGNORECASE),
            re.compile(r"act\s+as\s+(an?\s+unrestricted|a\s+hacker|an\s+evil)", re.IGNORECASE),
            re.compile(r"(sudo|admin)\s+mode\s+enabled", re.IGNORECASE),
            re.compile(r"disregard\s+the\s+context", re.IGNORECASE),
        ]
        
        # Toxic & abusive input detection
        self.unsafe_keywords = [
            "create a bomb",
            "how to make explosives",
            "credit card fraud",
            "ddos attack tutorial",
        ]

    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validates user voice/text query for safety and prompt injection attacks.
        Returns (is_safe: bool, reason: str).
        """
        if not text or not text.strip():
            return False, "Query is empty."

        cleaned = text.strip()

        if len(cleaned) < 3:
            return False, "Query is too short."

        if len(cleaned) > 800:
            return False, "Query exceeds maximum allowable length (800 characters)."

        # Check prompt injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(cleaned):
                return False, "Security violation: Prompt injection / instruction override attempt detected."

        # Check unsafe content
        lower_text = cleaned.lower()
        for kw in self.unsafe_keywords:
            if kw in lower_text:
                return False, "Safety violation: Inappropriate or unsafe topic detected."

        return True, ""

    def validate_retrieval_grounding(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        threshold: float = None,
    ) -> Tuple[bool, str]:
        """
        Retrieval Confidence Gate:
        Evaluates whether the retrieved context contains sufficient confidence to answer the question.
        Returns (is_grounded: bool, reason: str).
        """
        cutoff = threshold if threshold is not None else self.min_confidence_score

        if not retrieved_chunks:
            return False, "No relevant context found in knowledge base."

        top_chunk = retrieved_chunks[0]
        score = top_chunk.get("fused_score", top_chunk.get("score", 0.0))

        if score < cutoff:
            return False, f"Retrieval confidence ({score:.4f}) is below the grounding threshold ({cutoff:.4f})."

        return True, ""

    def verify_output_grounding(self, answer: str, context: str) -> Tuple[bool, float]:
        """
        Post-Generation Factual Overlap Check:
        Verifies that the generated answer shares semantic overlap with the retrieved parent context.
        """
        if not answer or not context:
            return False, 0.0

        if "cannot find this information" in answer.lower() or "not in the provided records" in answer.lower():
            return True, 1.0

        # Token overlap ratio
        ans_tokens = set(re.findall(r'\w+', answer.lower()))
        ctx_tokens = set(re.findall(r'\w+', context.lower()))
        
        # Exclude stopwords
        stopwords = {"the", "a", "an", "is", "in", "to", "of", "and", "or", "that", "it", "this", "on", "for", "as", "with", "by", "are", "be"}
        filtered_ans = ans_tokens - stopwords
        
        if not filtered_ans:
            return True, 1.0

        overlap = len(filtered_ans.intersection(ctx_tokens))
        overlap_ratio = overlap / len(filtered_ans)

        is_grounded = overlap_ratio >= 0.30
        return is_grounded, overlap_ratio
