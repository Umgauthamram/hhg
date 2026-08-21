"""
Ultra-Compact, Low-Latency Voice RAG Prompts.
Streamlined for sub-50ms TTFT prefill speed on LPUs.
"""

VOICE_RAG_SYSTEM_PROMPT = """You are a sub-200ms voice AI. Answer the user's question directly in 1-2 spoken sentences using only the facts in the Context.
Rules:
1. Ground strictly on Context. Never hallucinate.
2. If Context has no answer, reply exactly: "I cannot find this information in the provided records."
3. No markdown, asterisks, or lists. Plain spoken words only.
"""

def format_voice_user_prompt(query: str, context: str) -> str:
    """Combines retrieved context and query with minimal token overhead."""
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

format_rag_prompt = format_voice_user_prompt
