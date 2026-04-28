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

from prompt_builder import CORE_INSTRUCTIONS
from llm_factory import is_cache_capable, get_cache_profile, format_message_content, get_llm
from search_engine import _get_pinned_embedding, get_reranker, hybrid_search, calculate_cosine_similarity, _sort_docs_deterministically
from cache_engine import SemanticCache, get_semantic_cache, reset_semantic_cache
from intent_router import VectorRouter, get_router

"""
rag_chain.py — Retrieval-Augmented Generation Query Pipeline.

Handles:
  • OpenRouter LLM configuration (free model by default)
  • ChromaDB retriever setup
  • LangChain retrieval chain construction
"""

logger = logging.getLogger(__name__)

_llm_cache = {}

_llm_cache_lock = threading.Lock()

MAX_CONTEXT_UNION = 15

_background_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sentinel")

_atexit.register(_background_executor.shutdown, wait=False)

_atexit.register(_rewrite_executor.shutdown, wait=False)

def _background_summarize(history: list[BaseMessage]):
    """Background task to update sentinel state without stalling the main stream."""
    try:
        router = get_router()
        new_state = router.summarize_state_fast(history)
        return new_state
    except Exception as e:
        logger.error(f"❌ Background summary failed: {e}")
        return None

def _normalize_query(query: str) -> str:
    """Normalize query for cache-hit robustness.

    Preserves ``+`` and ``#`` so language names (C++, C#, F#) don't
    collide after stripping.  Keeps ``_`` and ``.`` for code identifiers.
    """
    import re
    return re.sub(r'[^\w\s+#.]', '', query).lower().strip()

def _get_max_tokens(specialty: str | None, query: str) -> int:
    """Return output token budget scaled to query complexity.

    CODE and REASONING tasks, or verbose queries (>200 chars), likely need
    the full budget.  Short factual/general questions rarely exceed 1024 tokens,
    so reserving 4096 for them wastes provider quota.
    """
    if specialty in ("CODE", "REASONING") or len(query) > 200:
        return MAX_TOKENS
    return 1024

def _truncate_ai_in_history(history: list[BaseMessage]) -> list[BaseMessage]:
    """
    Cap AI response length in chat history to reduce token waste,
    while aggressively preserving code blocks so the LLM remembers
    the actual code it wrote.
    """
    import re
    truncated = []
    for msg in history:
        if isinstance(msg, AIMessage) and len(msg.content) > AI_RESPONSE_MAX_CHARS:
            code_blocks = re.findall(r"(```.*?```)", msg.content, flags=re.DOTALL)
            if code_blocks:
                gist = msg.content[:400]
                trimmed = f"{gist}\n... [prose truncated]\n\n" + "\n\n".join(code_blocks)
                if len(trimmed) > AI_RESPONSE_MAX_CHARS * 3:
                    trimmed = trimmed[:AI_RESPONSE_MAX_CHARS * 3] + "\n```\n... [code truncated]"
            else:
                trimmed = msg.content[:AI_RESPONSE_MAX_CHARS] + "\n... [truncated for context efficiency]"
            truncated.append(AIMessage(content=trimmed))
        else:
            truncated.append(msg)
    return truncated

def _est_tokens(msgs):
    """Rough-cut estimation using the centralized ContextEstimator."""
    return ContextEstimator.estimate_tokens(msgs)

def _content_len(content):
    """Rough-cut estimation using the centralized ContextEstimator."""
    return ContextEstimator.estimate_content_tokens(content)

def compress_chat_history(history: list[BaseMessage], sentinel_state: str) -> list[BaseMessage]:
    """
    Intelligently trim the chat history based on Sentinel Summaries
    or Ghost History logic for token budget preservation.
    """
    if sentinel_state and sentinel_state != "No summary generated yet.":
        # Sentinel summary available — keep last 4 messages for immediate context
        # but ALWAYS protect the final user+AI pair so the LLM sees the
        # most recent exchange even if the budget is tight.
        keep = 4  # last 2 user+AI pairs
        truncated_history = history[-keep:]
        truncated_history = _truncate_ai_in_history(truncated_history)
    elif len(history) <= GHOST_HISTORY_MAX + 2:
        # Short enough that ghosting isn't needed yet (off-by-one fix:
        # +2 ensures we don't jump to ghost mode when ghost_section would
        # be empty, e.g. 11 messages with WINDOW=10 gives [2:1] = empty).
        truncated_history = _truncate_ai_in_history(history)
    else:
        anchor = history[:2]
        # Keep the last GHOST_HISTORY_WINDOW messages intact, ghost the middle
        window = history[-GHOST_HISTORY_WINDOW:]
        ghost_section = history[2:-GHOST_HISTORY_WINDOW]
        ghosts = []
        for msg in ghost_section:
            if isinstance(msg, AIMessage):
                trimmed = msg.content[:GHOST_AI_CHARS]
                if len(msg.content) > GHOST_AI_CHARS:
                    trimmed += "\n... [truncated]"
                ghosts.append(AIMessage(content=trimmed))
            else:
                ghosts.append(msg)
        truncated_history = _truncate_ai_in_history(anchor + ghosts + window)

    # Hard token budget: drop oldest ghost messages until under budget,
    # but ALWAYS protect the last 2 messages (most recent exchange).
    while _est_tokens(truncated_history) > MAX_HISTORY_TOKENS and len(truncated_history) > 4:
        # Remove the 3rd message (first ghost after anchor pair),
        # never touch the last 2 (protected tail).
        if len(truncated_history) > 4:
            truncated_history.pop(2)
        else:
            break

    return truncated_history

def _prepare_history_with_cache(history: list[BaseMessage], model: str | None) -> list[BaseMessage]:
    """
    Prepare chat history with optional caching.

    For models with >4 breakpoints (Gemini), we add a cache marker to
    the second-to-last message (the most recent AI response) rather than
    the last message (user query that changes every turn).  This way the
    checkpoint is reusable across turns — only the new user message is
    uncached, not the entire history.
    For Claude (4 breakpoints), system blocks already consume the limit.
    """
    if not history:
        return history

    max_bp, _ = get_cache_profile(model)
    # If we have spare breakpoints (Gemini supports 8+), use one for history.
    # We use 4 for system blocks, so 5+ is the threshold.
    if max_bp > 4 and is_cache_capable(model) and ENABLE_PROMPT_CACHING:
        new_history = list(history)
        # Mark the second-to-last message (stable across turns) instead of
        # the last message (volatile user query) to avoid all-writes-no-reads.
        target_idx = -2 if len(new_history) >= 2 else -1
        target_msg = new_history[target_idx]
        if isinstance(target_msg.content, str):
            # CRITICAL: create a NEW message object instead of mutating in-place.
            # The history list shares objects with lc_history/full_history;
            # in-place mutation corrupts them (content: str → list) and garbles
            # the sentinel summary and token estimates downstream.
            cls = type(target_msg)  # HumanMessage or AIMessage
            new_msg = cls(content=[
                {
                    "type": "text",
                    "text": target_msg.content,
                    "cache_control": {"type": "ephemeral"}
                }
            ])
            new_history[target_idx] = new_msg
        return new_history

    return list(history)

class LocalReRanker:
    """
    Local Cross-Encoder "Critic" that re-scores retrieved chunks 
    to ensure surgical precision before the context is passed to the LLM.
    """
    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        if USE_RERANKER:
            self.model = self._get_cached_cross_encoder()

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_cached_cross_encoder():
        try:
            return CrossEncoder(RERANK_MODEL)
        except Exception as e:
            logger.error(f"❌ Re-ranker failed to load: {e}")
            return None

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Re-score and filter documents using the Cross-Encoder."""
        if not self.model or not documents:
            return documents[:top_k]

        # Prepare pairs for cross-encoding (Query, Chunk)
        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = self.model.predict(pairs)
            
            # Combine scores with docs and sort
            scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            
            # 🚀 Phase 5: Store the top score for telemetry
            self.last_top_score = float(scored_docs[0][0]) if scored_docs else 0.0
            
            # Log the top score for telemetry
            if scored_docs:
                logger.info(f"🎯 Top Re-rank Relevance Score: {scored_docs[0][0]:.4f}")
            
            return [doc for score, doc in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"❌ Re-ranking execution failed: {e}")
            return documents[:top_k]

def build_rag_chain(db: Chroma, model: str | None = None):
    """
    Build a retrieval chain with stable Full-Context Caching (Architecture A).
    """
    llm = get_llm(model=model)
    
    # 🚀 Professional Polish: Dynamic Retrieval Configuration
    # We build our retrievers inside the lambda to support the 
    # Pinned File exclusion filter.

    # Dual-Path Prompt Construction
    # Only Claude supports Anthropic-style cache_control blocks via OpenRouter.
    # All other models (Gemini/Qwen/DeepSeek/Ollama) get a clean string prompt.
    
    is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING
    
    max_bp, _ = get_cache_profile(model)

    if is_cc:
        # Dynamic Cache Blocks — Claude only
        # Ordered from most stable to most volatile.  We only attach
        # cache_control markers to the first ``max_bp`` blocks; the
        # rest are plain text (no wasted cache writes).
        # Stable order: Instructions > Pinned > Sentinel > RAG context.
        static_system_text = CORE_INSTRUCTIONS
        # Order: most-stable → most-volatile.  The first max_bp blocks
        # get cache_control markers, so placing the volatile sentinel and
        # new-discoveries at the end avoids invalidating the prefix cache
        # every turn.
        block_specs = [
            static_system_text,
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}",
            "STABLE RAG CONTEXT (DETERMINISTIC):\n{stable_context}",
            "CONVERSATION STATE:\n{sentinel_state}",
            "NEW RAG DISCOVERIES:\n{new_context}"
        ]
        system_blocks = []
        for idx, text in enumerate(block_specs):
            use_cache_marker = idx < max_bp  # only mark up to max_bp blocks
            formatted = format_message_content(text, model, use_cache=use_cache_marker)
            # format_message_content returns a list for cache-capable models
            if isinstance(formatted, list):
                system_blocks.append(formatted[0])
            else:
                # Plain string — wrap in the Anthropic text-block format
                # so the system_blocks list stays homogeneous.
                system_blocks.append({"type": "text", "text": formatted})

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    else:
        # Mirror the cache-path ordering: most-stable → most-volatile.
        # Even non-cache providers (DeepSeek, Qwen) do implicit prefix
        # caching, so putting volatile sentinel AFTER stable RAG context
        # preserves more of the prefix across turns.
        system_text = (
            f"{CORE_INSTRUCTIONS}\n\n"
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}\n\n"
            "STABLE RAG CONTEXT (DETERMINISTIC):\n{stable_context}\n\n"
            "CONVERSATION STATE:\n{sentinel_state}\n\n"
            "NEW RAG DISCOVERIES:\n{new_context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    # 🚀 Platinum Standard: Metadata-Aware LCEL Chain
    # We remove StrOutputParser to preserve the 'response_metadata' (for caching token counts)
    # inside the raw message chunks.
    # 🚀 Professional Polish: Instantiate Vector Router & Re-ranker
    router = get_router()
    reranker = get_reranker()

    question_answer_chain = prompt | llm
    _specialist_llm_cache: dict[str, object] = {}
    # Cooldown tracker: prevents sentinel from re-firing every turn once the
    # token threshold is crossed.  Stored as a mutable dict so the closure
    # can mutate it without a `nonlocal` declaration.
    _sentinel_cooldown: dict[str, int] = {"last_turn": 0}
    _sentinel_failures: dict[str, int] = {"count": 0}
    MAX_SENTINEL_FAILURES = 3

    def _on_sentinel_done(future):
        try:
            future.result()
            _sentinel_failures["count"] = 0 # Reset on success
        except Exception as e:
            _sentinel_failures["count"] += 1
            logger.error(f"Sentinel failure ({_sentinel_failures['count']}/{MAX_SENTINEL_FAILURES}): {e}")

    def _full_context_cache_chain(inputs: dict):
        """
        Unified chain with Agentic Routing, Hybrid Search,
        and Cross-Provider cache awareness.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        coll_name = inputs.get("collection_name", "default")
        
        # 🚀 Fix: Get last query and its embedding from inputs
        last_query = inputs.get("last_query")
        last_query_emb = inputs.get("last_query_embedding")
        force_retrieval = inputs.get("force_retrieval", False)
        
        # 🚀 PHASE 3: Semantic Cache Lookup (Pre-Everything)
        # Scope by collection + pinned fingerprint + model so an answer
        # grounded in file A / model X is never served for file B / model Y.
        sem_cache = get_semantic_cache()
        if not force_retrieval:
            cached_ans = sem_cache.lookup(user_input, threshold=SEMANTIC_CACHE_THRESHOLD,
                                          collection_scope=coll_name,
                                          pinned_content=pinned_content,
                                          model=model)
            if cached_ans:
                yield {"answer": cached_ans, "intent": "CACHE_HIT"}
                return

        # 🚀 ASYNC SENTINEL TRIGGER
        background_future = None

        # 1. Initialize all prompt template variables to prevent KeyError
        inputs["full_source_context"] = inputs.get("full_source_context", "None pinned.")
        inputs["sentinel_state"] = inputs.get("sentinel_state", "No summarized state available.")
        inputs["stable_context"] = inputs.get("stable_context", "None previously established.")
        inputs["new_context"] = inputs.get("new_context", "No new discoveries.")
        inputs["chat_history"] = inputs.get("chat_history", [])

        # 2. Context Awareness (Latency-Free)
        # Use the true global turn count from app.py — compressed history
        # is capped at 4 messages (≤2 turns), which permanently deadlocks
        # the sentinel trigger if we count from it.
        turn_count = inputs.get("global_turn_count",
                                sum(1 for m in history if isinstance(m, HumanMessage)))
        # Full uncompressed history for sentinel summarization
        full_history = inputs.get("full_history", history)

        # Token-aware sentinel trigger: estimate from FULL history so the
        # budget reflects the real conversation size, not the compressed window.
        estimated_history_tokens = sum(_content_len(m.content) for m in full_history) // 3
        # Fire when history exceeds the token budget AND at least SENTINEL_INTERVAL
        # turns have passed since the last sentinel run.  Without the cooldown,
        # once the threshold is crossed it fires every single turn.
        # 🛡️ Circuit Breaker: Stop retrying if sentinel is failing consistently.
        should_summarize = (
            turn_count > 0
            and _sentinel_failures["count"] < MAX_SENTINEL_FAILURES
            and estimated_history_tokens >= SENTINEL_TOKEN_THRESHOLD
            and (turn_count - _sentinel_cooldown["last_turn"]) >= SENTINEL_INTERVAL
        )

        # Calculate semantic similarity once
        current_similarity = 0.0
        current_emb = None
        user_input_norm = _normalize_query(user_input)
        
        if not force_retrieval:
            # Layer 0: Exact-match cache (zero compute for identical queries)
            if last_query and last_query_emb and user_input_norm == _normalize_query(last_query):
                current_emb = last_query_emb
            else:
                from backend import get_embedding_model
                current_emb = get_embedding_model().embed_query(user_input)
            
            if last_query_emb:
                current_similarity = calculate_cosine_similarity(current_emb, last_query_emb)

        # Semantic Intent Detection (Latency-Free)

        # Semantic hit drives two behaviours:
        # 1. For Claude (cache-capable): retrieval still happens so the
        #    provider cache can fire on the deterministic prefix.
        # 2. For all other models: retrieval is skipped on a semantic hit
        #    since there's no provider-side cache benefit from re-fetching.
        is_semantic_hit = (
            current_similarity >= SEMANTIC_CACHE_THRESHOLD
        )
        
        # 3. Pinned context passthrough with Relevance Gate
        pinned_eligible = False
        if pinned_content and pinned_content != "None pinned.":
            if STICKY_PINNED_CONTEXT:
                pinned_eligible = True
            elif current_emb:
                # Use only the prefix to avoid massive embedding calls just for gating
                pinned_emb = _get_pinned_embedding(pinned_content[:2000])
                pinned_sim = calculate_cosine_similarity(current_emb, pinned_emb)
                if pinned_sim >= PINNED_RELEVANCE_THRESHOLD:
                    pinned_eligible = True
            else:
                pinned_eligible = True

        inputs["full_source_context"] = pinned_content if pinned_eligible else "None pinned."

        # 4. Define Previous Context Union
        previous_union = inputs.get("cached_docs") or []

        # 5. 🤖 Zero-Latency Vector Routing & Specialist Detection ─────────
        # Use LLM-based classification for high-precision follow-up detection
        intent = router.classify_intent(user_input, history) if history else "NEW"
        
        # Phase 4: Specialist Detection
        enable_auto = inputs.get("auto_specialist", ENABLE_AUTO_SPECIALIST)
        specialty = router.detect_specialty(user_input) if enable_auto else "GENERAL"
        
        # 🚀 Fix: Only switch models if we find a REAL specialty.
        # If it's just a GENERAL query, stay on the user's manual selection.
        specialist_model = (
            SPECIALIST_MAPPING.get(specialty)
            if (enable_auto and specialty != "GENERAL")
            else None
        )

        # Guard: never route to a cloud specialist when the user's chosen model
        # is local (Ollama). E.g. VISION maps to Gemma on OpenRouter by default —
        # that would silently bypass the user's local-only intent.
        if (specialist_model
                and not specialist_model.startswith(OLLAMA_PREFIX)
                and model
                and model.startswith(OLLAMA_PREFIX)):
            specialist_model = None

        pinned_file = inputs.get("exclude_file")
        ext_filter = inputs.get("filter_extensions")

        # Hybrid search (ChromaDB + BM25)
        # When the model supports provider-side prefix caching (Claude/Gemini/DeepSeek),
        # always retrieve so the deterministic sort can maximise cache hits.
        # For all other models the provider cache doesn't help, so skip
        # retrieval on semantic cache hits to save compute.
        k_fetch = RERANK_CANDIDATES if USE_RERANKER else RETRIEVER_K

        # 🚀 Fix: Include DeepSeek/Qwen as cache-capable for prefix stability, 
        # even if they don't use explicit Anthropic-style markers.
        provider_has_cache = is_cache_capable(model) or any(
            p in (model or "").lower() for p in ["deepseek", "qwen", "mistral"]
        )
        
        trust_native_cache = inputs.get("trust_native_cache", True)
        skip_retrieval = (
            not (trust_native_cache and provider_has_cache)
            and is_semantic_hit
            and bool(previous_union)
            and not force_retrieval
        )

        search_query = user_input
        if ENABLE_HYBRID_SEARCH and intent == "FOLLOW-UP":
            try:
                # 🚀 Fix Problem 1: Build formatted history string
                history_text = "\n".join([
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                    for m in history[-2:]
                ])
                # 🚀 Fix Problem 2: Call Ollama with a 2-second timeout to prevent blocking
                def _do_rewrite():
                    llm_rewrite = get_llm(model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}", temperature=0.0, streaming=False)
                    # 🚀 Fix: Direct timeout on invoke() to prevent thread leakage
                    llm_rewrite = llm_rewrite.with_config({"timeout": 2.0})
                    prompt_rewrite = (
                        f"History:\n{history_text}\n\n"
                        f"Rewrite this query to be standalone: '{user_input}'\n"
                        "Return ONLY the rewritten query, no explanation."
                    )
                    raw = llm_rewrite.invoke(prompt_rewrite).content.strip()
                    # Strip common LLM preamble that pollutes BM25 search
                    for prefix in ("Sure,", "Here's", "The rewritten query is:", "Rewritten query:"):
                        if raw.lower().startswith(prefix.lower()):
                            raw = raw[len(prefix):].strip().strip('"').strip("'")
                    # If rewriter returned something way longer than the input
                    # it's probably an explanation, not a query — fall back.
                    if len(raw) > len(user_input) * 3:
                        return user_input
                    return raw

                future = _rewrite_executor.submit(_do_rewrite)
                search_query = future.result(timeout=2.0)
            except Exception as e:
                logger.warning(f"Ollama rewrite failed or timed out: {e}")
                search_query = user_input

        # ── Fix C1: Anchor Term Injection ──────────────────────────────
        # Extract code identifiers from the user query.
        # all_identifiers: any word with underscore or ALL_CAPS (code-like tokens)
        # anchor_terms: strictly ALL_CAPS constants (ENABLE_PROMPT_CACHING, etc.)
        #   — used for propagation detection and targeted retrieval
        # This prevents "how does" from triggering propagation on every query.
        all_identifiers = list(set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", user_input)))
        anchor_terms = [t for t in all_identifiers if t.isupper() and len(t) > 3]
        # Also include snake_case identifiers (contain underscore) for retrieval
        snake_case_ids = [t for t in all_identifiers if "_" in t and not t.isupper()]
        # Combined anchors for retrieval (ALL_CAPS + snake_case)
        retrieval_anchors = anchor_terms + snake_case_ids

        # Inject ALL_CAPS into search query so BM25 can find definition sites
        if anchor_terms:
            for term in anchor_terms:
                if term not in search_query:
                    search_query = f"{search_query} {term}"

        # ── Fix D: 3-Hop Propagation Detection ─────────────────────────
        # Only fires when the query contains ACTUAL code identifiers (ALL_CAPS
        # or snake_case), not just common English words that happen to be 4+ chars.
        # "changes" alone was too common ("what changes were made?") but
        # causality-tracing phrasings must still trigger propagation.
        # Using stem "chang" + context words avoids false positives while
        # catching "change when", "changes when", "changed by", etc.
        _PROPAGATION_KEYWORDS = (
            "propagat", "impact", "affect", "changes when", "change when",
            "changed by", "changed if",
            "trace", "alter", "flow", "step-by-step", "step by step",
            "how does", "ultimately",
        )
        is_propagation_query = bool(anchor_terms or snake_case_ids) and any(
            kw in user_input.lower() for kw in _PROPAGATION_KEYWORDS
        )

        # 🚀 Fix I: Aggregation query detection (L3)
        # Queries asking for completeness ("every place", "all functions") need
        # more results to survive reranking so scattered utility helpers aren't culled.
        _AGGREGATION_KEYWORDS = (
            "every ", "all places", "each place", "everywhere",
            "all functions", "list all", "every file", "all files",
            "all the places", "every place", "each function",
            "all methods", "every method", "scattered", "each file",
            "which files", "what files", "list down", "files in",
            "files you see", "list of files",
        )
        is_aggregation_query = any(
            kw in user_input.lower() for kw in _AGGREGATION_KEYWORDS
        )

        reranker_score = 0.0
        new_retrievals = []
        if not skip_retrieval and db:
            new_retrievals = hybrid_search(
                db, search_query,
                collection_name=coll_name,
                k=k_fetch,
                exclude_file=pinned_file,
                filter_extensions=ext_filter,
                query_embedding=current_emb if search_query == user_input else None,
            )

            # Fix D & E: extra retrieval passes for propagation queries.
            # All FTS5 and ChromaDB queries are batched (OR) to minimize
            # database round-trips instead of looping per anchor term.
            if is_propagation_query:
                with SQLiteFTS5BM25(coll_name) as fts:
                    top_anchors = retrieval_anchors[:3]

                    # 🚀 Fix E: Call-graph retrieval via FTS5 OR (L2)
                    try:
                        call_docs = fts.search_by_calls_batch(top_anchors, k=15)
                        if call_docs:
                            logger.info(f"📍 Fix E: Injected {len(call_docs)} chunks calling {top_anchors}")
                            new_retrievals.extend(call_docs)
                    except Exception as e:
                        logger.warning(f"Fix E FTS5 batch call retrieval failed: {e}")

                    # 🚀 Fix G: Constant-reference retrieval via FTS5 OR (L4)
                    try:
                        const_docs = fts.search_by_constants_batch(top_anchors, k=15)
                        if const_docs:
                            logger.info(f"📍 Fix G: Injected {len(const_docs)} chunks referencing constants {top_anchors}")
                            new_retrievals.extend(const_docs)
                    except Exception as e:
                        logger.warning(f"Fix G FTS5 batch constant retrieval failed: {e}")

                # 🚀 Fix J: Guaranteed anchor text retrieval via ChromaDB $or (L4)
                # BM25 misses config files (implicit AND + length penalty).
                # Vector search misses them (no semantic similarity).
                # Use ChromaDB where_document $or for exact substring matching —
                # guaranteed to find any chunk containing any anchor text.
                if db:
                    try:
                        if len(top_anchors) > 1:
                            where_doc = {"$or": [{"$contains": a} for a in top_anchors]}
                        else:
                            where_doc = {"$contains": top_anchors[0]}
                        text_docs = db.similarity_search(
                            user_input, k=10,
                            where_document=where_doc
                        )
                        if text_docs:
                            logger.info(f"📍 Fix J: Injected {len(text_docs)} chunks containing {top_anchors}")
                            new_retrievals.extend(text_docs)
                    except Exception as e:
                        logger.warning(f"Fix J text search failed: {e}")

                for anchor in retrieval_anchors[:2]:
                    for sub_q in (
                        f"function that reads {anchor}",
                        f"where {anchor} is used or evaluated",
                    ):
                        try:
                            extra = hybrid_search(
                                db, sub_q,
                                collection_name=coll_name,
                                k=6,
                                exclude_file=pinned_file,
                                filter_extensions=ext_filter,
                                query_embedding=None,
                            )
                            new_retrievals.extend(extra)
                        except Exception as e:
                            logger.warning(f"Sub-query retrieval failed for '{sub_q}': {e}")

                # Deduplicate the merged pool before reranking.
                seen_hashes_pre = set()
                deduped = []
                for d in new_retrievals:
                    h = d.metadata.get("content_hash") or d.page_content[:200]
                    if h not in seen_hashes_pre:
                        deduped.append(d)
                        seen_hashes_pre.add(h)
                new_retrievals = deduped
        elif skip_retrieval:
            # Semantic cache hit — reuse previous docs as the fresh set.
            new_retrievals = list(previous_union)

        # Save pre-rerank pool for Fix H post-reranker injection
        pre_rerank_pool = list(new_retrievals) if is_propagation_query else []

        # 3. 🎯 Local Re-ranking (Phase 3) ──────────────────────────────
        if USE_RERANKER and reranker and new_retrievals and not skip_retrieval:
            # Fix I: Widen reranker window for aggregation queries so scattered
            # utility helpers aren't culled from the top-k.
            effective_top_k = RERANK_TOP_K * 2 if is_aggregation_query else RERANK_TOP_K
            new_retrievals = reranker.rerank(
                search_query,
                new_retrievals,
                top_k=effective_top_k
            )
            # Capture the top relevance score for telemetry
            reranker_score = getattr(reranker, 'last_top_score', 0.0)

        # 🚀 Fix H: Post-reranker anchor injection (L4)
        # Force-include definition-site chunks the cross-encoder culled.
        # This ensures config constants and bridge functions survive reranking.
        if is_propagation_query and retrieval_anchors and pre_rerank_pool:
            reranked_hashes = {
                d.metadata.get("content_hash", d.page_content[:200])
                for d in new_retrievals
            }
            for anchor in retrieval_anchors[:3]:
                injected = 0
                for d in pre_rerank_pool:
                    if injected >= 2:
                        break
                    h = d.metadata.get("content_hash", d.page_content[:200])
                    if h in reranked_hashes:
                        continue
                    # Match definition sites (CONST = ...) or direct references
                    if f"{anchor}" in d.page_content:
                        ref_consts = d.metadata.get("references_constants", "")
                        is_zero = d.metadata.get("zero_chunk", False)
                        # Inject if: it's the definition file (zero-chunk with the constant)
                        # or it references the constant in its metadata
                        if is_zero or anchor in ref_consts:
                            new_retrievals.append(d)
                            reranked_hashes.add(h)
                            injected += 1
                            logger.info(f"📍 Fix H: Post-reranker injected chunk from {d.metadata.get('source', '?')} for '{anchor}'")

        # Intent-Aware Union Logic with Context Decay
        if intent == "FOLLOW-UP":
            # 🚀 Fix: Prevent "Knowledge Lock-in" by ensuring fresh retrievals 
            # always have priority. We calculate unique new docs first.
            seen_hashes = set()
            unique_new = []
            for d in new_retrievals:
                h = d.metadata.get("content_hash", d.page_content)
                if h not in seen_hashes:
                    unique_new.append(d)
                    seen_hashes.add(h)

            # Cap the new retrievals at MAX_CONTEXT_UNION
            unique_new = unique_new[:MAX_CONTEXT_UNION]
            
            # Rebuild seen_hashes based on the sliced unique_new to avoid dropping valid old docs
            seen_hashes = {d.metadata.get("content_hash", d.page_content) for d in unique_new}
            
            # Calculate how many slots are left for the older stable docs
            available_old_slots = MAX_CONTEXT_UNION - len(unique_new)

            # Eviction: keep old docs in the SAME ORDER they had in
            # previous_union so the established context block is
            # byte-stable across turns — critical for the provider
            # prefix cache (Anthropic / Gemini / DeepSeek).
            #
            # Earlier versions scored old docs by current-query keyword
            # overlap and re-sorted.  That changed membership AND order
            # whenever the user rephrased, destroying the cached prefix
            # on almost every turn.  Freshness priority is already
            # preserved by `unique_new` taking the first N slots; the
            # old docs just fill the tail in their original order.
            surviving_old = []
            for d in previous_union:
                if len(surviving_old) >= available_old_slots:
                    break
                h = d.metadata.get("content_hash", d.page_content)
                if h in seen_hashes:
                    continue  # already in unique_new
                surviving_old.append(d)
                seen_hashes.add(h)

            final_docs = surviving_old + unique_new
            protected_count = len(unique_new)
        else:
            final_docs = new_retrievals[:MAX_CONTEXT_UNION]
            protected_count = 0

        # Filter massive zero-chunks from retrieval results for all models.
        # Zero-chunks can be up to ZERO_CHUNK_THRESHOLD (100k chars / ~33k tokens) and destroy
        # signal-to-noise when surfaced via retrieval.  The pinned-file mechanism handles
        # deliberate full-file viewing; retrieved zero-chunks are almost never the right behaviour.
        final_docs = [
            d for d in final_docs
            if not (d.metadata.get("zero_chunk") and len(d.page_content) > MAX_ZERO_CHUNK_CHARS)
        ]

        # ── Context window budget enforcement ──────────────────────────
        # Estimate total prompt tokens and drop trailing RAG chunks until
        # we fit.  This prevents silent API failures on models with small
        # context windows (8K Ollama, 32K free-tier).
        # Specific patterns MUST appear before generic ones — the first
        # match wins, so "qwen2.5:3b" must precede "qwen", etc.
        _CONTEXT_BUDGETS = {
            "qwen2.5:3b": 6000, "llama3.2:1b": 4000,
            "ollama": 6000, "llama": 6000,
            "gemma": 28000, "gemini": 28000, "claude": 180000,
            "gpt-oss": 28000, "gpt": 120000,
            "deepseek": 60000, "qwen": 28000,
        }
        _budget = 28000  # default
        for _pattern, _limit in _CONTEXT_BUDGETS.items():
            if _pattern in (model or "").lower():
                _budget = _limit
                break
        # Estimate: system prompt + pinned + history + RAG + user query
        _sys_est = len(CORE_INSTRUCTIONS) // 3
        _pinned_est = _content_len(inputs.get("full_source_context", "")) // 3
        _hist_est = _est_tokens(history)
        _query_est = len(user_input) // 3
        _overhead = _sys_est + _pinned_est + _hist_est + _query_est + 500  # safety margin
        _rag_budget = _budget - _overhead
        # Drop chunks from the end (lowest relevance) until within budget.
        # Account for _format_docs overhead (~80 chars per chunk for
        # "SOURCE: ...\nCONTENT: " prefix).
        _FMT_OVERHEAD_PER_CHUNK = 80
        
        # 🚀 Fix Performance: Calculate total length once and decrement instead of re-summing in a loop (O(N) vs O(N^2))
        total_rag_chars = sum(len(d.page_content) + _FMT_OVERHEAD_PER_CHUNK for d in final_docs)
        
        while final_docs and (total_rag_chars // 3) > _rag_budget:
            # Pop from the tail end of surviving_old first, otherwise pop from unique_new
            if len(final_docs) > protected_count:
                idx = len(final_docs) - protected_count - 1
            else:
                idx = -1
                
            removed_doc = final_docs.pop(idx)
            total_rag_chars -= (len(removed_doc.page_content) + _FMT_OVERHEAD_PER_CHUNK)

        # 🚀 Split Context: Prefix cache hits on <established_context>, Relevance hits on <new_discoveries>
        stable_hashes = {
            d.metadata.get("content_hash", "")
            for d in previous_union
        } if previous_union and intent == "FOLLOW-UP" else None

        established_docs = []
        new_docs = []
        if stable_hashes:
            for d in final_docs:
                if d.metadata.get("content_hash", "") in stable_hashes:
                    established_docs.append(d)
                else:
                    new_docs.append(d)
        else:
            new_docs = final_docs

        # Only sort the established context deterministically to preserve identical byte-string
        established_docs = _sort_docs_deterministically(established_docs, stable_hashes=None)
        new_docs = _sort_docs_deterministically(new_docs, stable_hashes=None)

        def _format_docs(docs):
            return "\n\n".join([f"SOURCE: {d.metadata.get('source')}\nCONTENT: {d.page_content}" for d in docs]) if docs else ""

        stable_block = _format_docs(established_docs)
        new_block = _format_docs(new_docs)

        inputs["stable_context"] = f"<established_context>\n{stable_block}\n</established_context>" if stable_block else "None previously established."
        inputs["new_context"] = f"<new_discoveries>\n{new_block}\n</new_discoveries>" if new_block else "No new discoveries."
        inputs["context"] = established_docs + new_docs
        inputs["chat_history"] = _prepare_history_with_cache(history, model)
        
        # Dynamic Specialist Swap — cached LLM instances
        active_chain = question_answer_chain
        if enable_auto and specialist_model:
            current_m = getattr(
                active_chain.bound if hasattr(active_chain, "bound") else active_chain,
                "model_name", "",
            )
            if specialist_model != current_m:
                if specialist_model not in _specialist_llm_cache:
                    _specialist_llm_cache[specialist_model] = get_llm(
                        model=specialist_model, streaming=True
                    )
                active_chain = prompt | _specialist_llm_cache[specialist_model]

        # Dynamic output token budget — reduce for simple queries to free provider quota
        output_tokens = _get_max_tokens(specialty, user_input)
        if output_tokens != MAX_TOKENS:
            base_llm = (
                _specialist_llm_cache[specialist_model]
                if (enable_auto and specialist_model and specialist_model in _specialist_llm_cache)
                else llm
            )
            # ChatOllama uses 'num_predict' for output token budget; OpenAI/OpenRouter use 'max_tokens'.
            active_model_id = specialist_model or model or ""
            if active_model_id.startswith(OLLAMA_PREFIX):
                active_chain = prompt | base_llm.bind(num_predict=output_tokens)
            else:
                active_chain = prompt | base_llm.bind(max_tokens=output_tokens)
        
        # Ensure we always have an embedding to pass back for next turn.
        if current_emb is None:
            if is_semantic_hit and last_query_emb:
                current_emb = last_query_emb
            else:
                from backend import get_embedding_model
                current_emb = get_embedding_model().embed_query(user_input)

        # 🚀 ASYNC SENTINEL TRIGGER — uses full_history so the summary
        # covers the entire conversation, not just the compressed window.
        background_future = None
        if should_summarize and not inputs.get("sentinel_future_active"):
            _sentinel_cooldown["last_turn"] = turn_count
            # CRITICAL FIX: Pass a snapshot (shallow copy) to prevent thread race condition
            background_future = _background_executor.submit(_background_summarize, list(full_history))
            background_future.add_done_callback(_on_sentinel_done)

        yield {
            "context": inputs["context"], 
            "intent": intent, 
            "query_embedding": current_emb,
            "specialist_active": specialist_model if enable_auto else None,
            "top_relevance_score": reranker_score,
            "sentinel_future": background_future # Pass future to UI for persistence
        }
        
        full_answer = ""
        for chunk in active_chain.stream(inputs):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += content
            yield {"answer": content, "raw_chunk": chunk}
            
        # 🚀 PHASE 3: Update Semantic Cache with fresh generation
        # Tag with the full scope so future lookups can reject the entry
        # if the user re-pins, changes the file, or switches models.
        if not is_semantic_hit and len(full_answer) > 50:
            sem_cache.upsert(user_input, full_answer, collection_scope=coll_name,
                             pinned_content=pinned_content, model=model)

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)

