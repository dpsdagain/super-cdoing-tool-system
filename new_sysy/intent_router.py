from __future__ import annotations
import re
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    CLOUDROUTER_MODELS,
    OLLAMA_BASE_URL,
    OLLAMA_MODELS,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    MAX_TOKENS,
    ANTHROPIC_CACHE_BETA_HEADER,
    ENABLE_PROMPT_CACHING,
    ENABLE_AUTO_SPECIALIST,
    MAX_CACHE_CHECKPOINTS,
    SEMANTIC_CACHE_THRESHOLD,
    SENTINEL_MAX_TOKENS,
    SENTINEL_TOKEN_THRESHOLD,
    SENTINEL_INTERVAL,
    TRUST_NATIVE_CACHE,
    PROVIDER_CACHE_PROFILES,
    ENABLE_HYBRID_SEARCH,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    USE_RERANKER,
    RERANK_MODEL,
    RERANK_TOP_K,
    RERANK_CANDIDATES,
    PINNED_RELEVANCE_THRESHOLD,
    STICKY_PINNED_CONTEXT,
    SPECIALIST_MAPPING,
    GHOST_HISTORY_WINDOW,
    GHOST_HISTORY_MAX,
    AI_RESPONSE_MAX_CHARS,
    GHOST_AI_CHARS,
    MAX_HISTORY_TOKENS,
    MAX_ZERO_CHUNK_CHARS,
    AGENT_ROUTER_MODEL,
    OLLAMA_PREFIX,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_CLOUD_PREFIX,
)
from estimator import ContextEstimator
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from backend import SQLiteFTS5BM25
import atexit as _atexit
import functools

import logging
import numpy as np
logger = logging.getLogger(__name__)

class VectorRouter:
    """
    Zero-latency decision engine using vector similarity to handle 
    classification and state management without LLM overhead.
    """
    def __init__(self):
        # We reuse the embedding model already loaded in backend.py
        pass

    def classify_intent(self, query: str, history: list[BaseMessage]) -> str:
        """
        Classify as NEW topic or FOLLOW-UP.
        Uses heuristic fast-paths first; falls through to local LLM only
        for ambiguous cases to avoid 200ms-2s latency on every query.
        """
        if not history:
            return "NEW"

        import re
        q = query.lower().strip()

        # Fast-path: pronouns/demonstratives strongly indicate follow-up
        if re.match(r"^(it|this|that|these|those|the same|above|previous|also|and |more )\b", q):
            return "FOLLOW-UP"
        # Fast-path: explicit new-topic signals
        if re.match(r"^(new topic|switch to|let's talk about|forget|start over)\b", q):
            return "NEW"

        try:
            llm = get_llm(model=AGENT_ROUTER_MODEL, temperature=0.0, streaming=False)
            
            history_text = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}" 
                for m in history[-2:]
            ])
            
            prompt = (
                f"Previous turns:\n{history_text}\n\n"
                f"Current query: '{query}'\n\n"
                "Output exactly 'FOLLOW-UP' if the user is referring to the current topic or files, "
                "or 'NEW' if they are asking about a different file or a fresh concept. "
                "Return ONLY the word."
            )
            response = llm.invoke(prompt)
            result = response.content.strip().upper()
            return "FOLLOW-UP" if "FOLLOW-UP" in result else "NEW"
        except Exception:
            # Ollama is down — heuristic fallback: if the current query
            # shares significant content words with the last user message,
            # it's likely a follow-up.  Defaulting to "NEW" here silently
            # destroys context union for every follow-up when Ollama is off.
            last_user = ""
            for m in reversed(history):
                if isinstance(m, HumanMessage):
                    last_user = m.content.lower()
                    break
            if last_user:
                _stop = {"the","is","a","an","in","of","to","for","and","or",
                         "how","does","what","it","this","that","can","do","i"}
                cur_words = set(q.split()) - _stop
                prev_words = set(last_user.split()) - _stop
                if cur_words and prev_words:
                    overlap = len(cur_words & prev_words) / max(len(cur_words), 1)
                    if overlap >= 0.3:
                        return "FOLLOW-UP"
            return "NEW"

    def detect_specialty(self, query: str) -> str:
        """
        Detect the best specialist for the query using robust regex word boundaries.
        Returns one of: ['CODE', 'REASONING', 'VISION', 'GENERAL']
        """
        import re
        q = query.lower()

        # 💻 Coding Specialist Triggers — checked BEFORE the short-question
        # fast-path so "What is the best way to implement a function in Python?"
        # correctly routes to CODE even though it starts with "what".
        code_triggers = [
            r"code", r"python", r"javascript", r"verilog", r"function", r"class",
            r"refactor", r"bug", r"debug", r"compile", r"script", r"hdl", r"rtl",
            r"implement", r"write a", r"api", r"library", r"sql", r"html",
            r"cpp", r"c\+\+", r"rust", r"golang"
        ]
        if any(re.search(rf"\b{t}\b", q) for t in code_triggers) or "```" in q:
            return "CODE"

        # Fast-path: short factual questions with no code signal → GENERAL
        if len(query) < 60 and re.match(r"^(what|where|who|when|which|is|does|can)\b", q):
            return "GENERAL"

        # 👁️ Vision Triggers
        vision_triggers = [r"image", r"plot", r"chart", r"diagram", r"vision", r"see this"]
        if any(re.search(rf"\b{t}\b", q) for t in vision_triggers):
            return "VISION"
            
        # 🧠 Reasoning / Math Triggers
        reasoning_triggers = [
            r"analyze", r"logic", r"math", r"derive", r"prove", r"step by step",
            r"complex", r"calculate", r"deepseek", r"reason", r"philosophy", 
            r"compare", r"architecture", r"design pattern", r"explain how"
        ]
        if any(re.search(rf"\b{t}\b", q) for t in reasoning_triggers):
            return "REASONING"
            
        # Default
        return "GENERAL"

    @staticmethod
    def _extractive_fallback(history: list[BaseMessage]) -> str:
        """
        Pure-Python extractive summary used when the local LLM (Ollama)
        is unavailable.  Keeps the first sentence of each recent human
        message to preserve topic continuity without any external calls.
        """
        bullets = []
        for m in history[-8:]:
            if not isinstance(m, HumanMessage):
                continue
            text = m.content.strip()
            # Take the first sentence (up to first period, question mark, or 120 chars)
            end = len(text)
            for ch in ".?!":
                idx = text.find(ch)
                if 0 < idx < end:
                    end = idx + 1
            snippet = text[:min(end, 120)].strip()
            if snippet:
                bullets.append(f"- {snippet}")
        return "\n".join(bullets[-3:]) if bullets else "No summary available."

    def summarize_state_fast(self, history: list[BaseMessage]) -> str:
        """
        Summarize conversation state.  Tries the local Ollama model first;
        falls back to a pure-Python extractive summary if Ollama is down
        so history compression is never silently skipped.
        """
        try:
            llm = get_llm(model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}", temperature=0.0, streaming=False)
            context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:500]}" for m in history[-6:]])
            prompt = (
                f"History:\n{context}\n\n"
                "Summarize the conversation state so far in exactly 3 dense bullet points. "
                "Focus on technical topics discussed. Reply ONLY with the bullet points."
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            logger.warning("Ollama unavailable for sentinel — using extractive fallback")
            return self._extractive_fallback(history)

_router_instance = None

_router_lock = threading.Lock()

def get_router():
    global _router_instance
    with _router_lock:
        if _router_instance is None:
            _router_instance = VectorRouter()
        return _router_instance

