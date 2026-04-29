"""
rag_core.py — Retrieval-Augmented Generation Query Pipeline.

Handles:
  - OpenRouter LLM configuration
  - ChromaDB retriever setup
  - LangChain retrieval chain construction
"""
from __future__ import annotations
import re
import threading
import logging
import atexit as _atexit
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import (
    RETRIEVER_K,
    ENABLE_PROMPT_CACHING,
    ENABLE_AUTO_SPECIALIST,
    SEMANTIC_CACHE_THRESHOLD,
    SENTINEL_TOKEN_THRESHOLD,
    SENTINEL_INTERVAL,
    ENABLE_HYBRID_SEARCH,
    AGENT_ROUTER_MODEL,
    OLLAMA_PREFIX,
    USE_RERANKER,
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
)
from estimator import ContextEstimator
from backend import load_existing_chroma
from fts5_engine import SQLiteFTS5BM25
from cache_engine import get_semantic_cache, reset_semantic_cache
from intent_router import get_router
from prompt_builder import CORE_INSTRUCTIONS
from llm_factory import is_cache_capable, get_cache_profile, format_message_content, get_llm
from search_engine import _get_pinned_embedding, get_reranker, hybrid_search, calculate_cosine_similarity, _sort_docs_deterministically, _rewrite_executor

logger = logging.getLogger(__name__)

MAX_CONTEXT_UNION = 15
MAX_TOKENS = 4096

_background_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sentinel")
_atexit.register(_background_executor.shutdown, wait=False)

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

def _truncate_ai_in_history(history: list[BaseMessage]) -> list[BaseMessage]:

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
    return truncated_history

def _prepare_history_with_cache(history: list[BaseMessage], model: str | None) -> list[BaseMessage]:
    """
    Optimizes history for prefix caching by identifying the last
    stable turn and injecting a cache marker.
    """
    if not history: return []
    
    # Prefix caching strategy: Cache the entire history up to the last user message.
    # We find the index of the second to last HumanMessage.
    user_msg_indices = [i for i, m in enumerate(history) if isinstance(m, HumanMessage)]
    
    if len(user_msg_indices) >= 2:
        # We cache everything up to the previous user message (and the AI's response to it)
        target_idx = user_msg_indices[-1] - 1
        if target_idx >= 0:
            msg = history[target_idx]
            new_msg = AIMessage(
                content=format_message_content(msg.content, model, use_cache=True)
            )
            # Copy other attributes if necessary
            new_history = list(history)
            # 🚀 Handle complex content (list of blocks)
            # In LangChain, AIMessage content can be a list. 
            # If so, we ensure the last block has the cache marker.
            new_history[target_idx] = new_msg
        return new_history

    return list(history)

class ContextCacheChain:
    """
    Unified chain with Agentic Routing, Hybrid Search,
    and Cross-Provider cache awareness.
    """
    def __init__(self, db: Chroma, model: str | None, llm, prompt, router, reranker):
        self.db = db
        self.model = model
        self.llm = llm
        self.prompt = prompt
        self.router = router
        self.reranker = reranker
        self.question_answer_chain = prompt | llm
        self._specialist_llm_cache = {}
        self._sentinel_cooldown = {"last_turn": 0}
        self._sentinel_failures = {"count": 0}
        self.MAX_SENTINEL_FAILURES = 3

    def _on_sentinel_done(self, future):
        try:
            future.result()
            self._sentinel_failures["count"] = 0
        except Exception as e:
            self._sentinel_failures["count"] += 1
            logger.error(f"Sentinel failure ({self._sentinel_failures['count']}/{self.MAX_SENTINEL_FAILURES}): {e}")

    def _check_semantic_cache(self, user_input, coll_name, pinned_content, force_retrieval):
        sem_cache = get_semantic_cache()
        if not force_retrieval:
            cached_ans = sem_cache.lookup(user_input, threshold=SEMANTIC_CACHE_THRESHOLD,
                                          collection_scope=coll_name,
                                          pinned_content=pinned_content,
                                          model=self.model)
            if cached_ans:
                return cached_ans, sem_cache
        return None, sem_cache

    def _handle_sentinel(self, should_summarize, inputs, turn_count, full_history):
        background_future = None
        if should_summarize and not inputs.get("sentinel_future_active"):
            self._sentinel_cooldown["last_turn"] = turn_count
            background_future = _background_executor.submit(_background_summarize, list(full_history))
            background_future.add_done_callback(self._on_sentinel_done)
        return background_future

    def _prepare_context(self, final_docs, previous_union, intent):
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

        established_docs = _sort_docs_deterministically(established_docs, stable_hashes=None)
        new_docs = _sort_docs_deterministically(new_docs, stable_hashes=None)

        def _format_docs(docs):
            return "\n\n".join([f"SOURCE: {d.metadata.get('source')}\nCONTENT: {d.page_content}" for d in docs]) if docs else ""

        stable_block = _format_docs(established_docs)
        new_block = _format_docs(new_docs)

        stable_context_str = f"<established_context>\n{stable_block}\n</established_context>" if stable_block else "None previously established."
        new_context_str = f"<new_discoveries>\n{new_block}\n</new_discoveries>" if new_block else "No new discoveries."
        
        return established_docs, new_docs, stable_context_str, new_context_str

    def _execute_llm_stream(self, active_chain, inputs, is_semantic_hit, user_input, coll_name, pinned_content, sem_cache):
        full_answer = []
        for chunk in active_chain.stream(inputs):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer.append(content)
            yield {"answer": content, "raw_chunk": chunk}
            
        full_answer_str = "".join(full_answer)
        if not is_semantic_hit and len(full_answer_str) > 50:
            sem_cache.upsert(user_input, full_answer_str, collection_scope=coll_name,
                             pinned_content=pinned_content, model=self.model)

    def __call__(self, inputs: dict):
        """
        Unified chain with Agentic Routing, Hybrid Search,
        and Cross-Provider cache awareness.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        coll_name = inputs.get("collection_name", "default")
        
        # Fix: Get last query and its embedding from inputs
        last_query = inputs.get("last_query")
        last_query_emb = inputs.get("last_query_embedding")
        force_retrieval = inputs.get("force_retrieval", False)
        
        # PHASE 3: Semantic Cache Lookup (Pre-Everything)
        # Scope by collection + pinned fingerprint + self.model so an answer
        # grounded in file A / self.model X is never served for file B / self.model Y.
        cached_ans, sem_cache = self._check_semantic_cache(user_input, coll_name, pinned_content, force_retrieval)
        if cached_ans:
            yield {"answer": cached_ans, "intent": "CACHE_HIT"}
            return

        # ASYNC SENTINEL TRIGGER
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
        estimated_history_tokens = sum(ContextEstimator.estimate_content_tokens(m.content) for m in full_history) // 3
        # Fire when history exceeds the token budget AND at least SENTINEL_INTERVAL
        # turns have passed since the last sentinel run.  Without the cooldown,
        # once the threshold is crossed it fires every single turn.
        # Circuit Breaker: Stop retrying if sentinel is failing consistently.
        should_summarize = (
            turn_count > 0
            and self._sentinel_failures["count"] < self.MAX_SENTINEL_FAILURES
            and estimated_history_tokens >= SENTINEL_TOKEN_THRESHOLD
            and (turn_count - self._sentinel_cooldown["last_turn"]) >= SENTINEL_INTERVAL
        )

        # Calculate semantic similarity once
        # Reuse existing embedding if query hasn't changed
        current_emb = last_query_emb if last_query == user_input else None
        
        # 3. 🛡️ Knowledge Boundary (Phase 1) ──────────────────────────
        # Optimization: Only re-embed if we don't have a recent hit from app.py
        previous_union = inputs.get("context", [])
        
        # 4. 🧠 Strategic Drift Protection (Phase 2) ─────────────────────
        # If turnover is high, we might want to prioritize established context
        # to prevent the agent from wandering away from the original goal.
        intent = inputs.get("intent", "NEW")
        
        skip_retrieval = False
        is_semantic_hit = False
        
        # If intent is provided and is a follow-up, we still retrieve unless 
        # semantic cache told us not to.
        # BUT: For cloud models with prefix caching, we always want to provide
        # the full RAG context even on semantic hits to maintain the prefix.
        
        # 5. 🤖 Zero-Latency Vector Routing & Specialist Detection ─────────
        # Use LLM-based classification for high-precision follow-up detection
        intent = self.router.classify_intent(user_input, history) if history else "NEW"
        
        # Phase 4: Specialist Detection
        enable_auto = inputs.get("auto_specialist", ENABLE_AUTO_SPECIALIST)
        specialty = self.router.detect_specialty(user_input) if enable_auto else "GENERAL"
        
        # Fix: Only switch models if we find a REAL specialty.
        # If it's just a GENERAL query, stay on the user's manual selection.
        specialist_model = (
            SPECIALIST_MAPPING.get(specialty)
            if specialty != "GENERAL"
            else None
        )

        # Guard: never route to a cloud specialist when the user's chosen self.model
        # is local (Ollama). E.g. VISION maps to Gemma on OpenRouter by default —
        # that would silently bypass the user's local-only intent.
        if (specialist_model
                and not specialist_model.startswith(OLLAMA_PREFIX)
                and self.model
                and self.model.startswith(OLLAMA_PREFIX)):
            specialist_model = None

        pinned_file = inputs.get("exclude_file")
        ext_filter = inputs.get("filter_extensions")

        # Hybrid search (ChromaDB + BM25)
        # When the self.model supports provider-side prefix caching (Claude/Gemini/DeepSeek),
        # always retrieve so the deterministic sort can maximise cache hits.
        # For all other models the provider cache doesn't help, so skip
        # retrieval on semantic cache hits to save compute.
        k_fetch = RERANK_CANDIDATES if USE_RERANKER else RETRIEVER_K

        # Fix: Include DeepSeek/Qwen as cache-capable for prefix stability, 
        # even if they don't use explicit Anthropic-style markers.
        provider_has_cache = is_cache_capable(self.model) or any(
            p in (self.model or "").lower() for p in ["deepseek", "qwen", "mistral"]
        )
        
        trust_native_cache = inputs.get("trust_native_cache", True)
        if trust_native_cache and is_semantic_hit and not provider_has_cache:
            skip_retrieval = True

        # Discovery Layer (Hybrid Search)
        retrieval_anchors = []
        anchor_terms = []
        snake_case_ids = []
        
        search_query = user_input
        if ENABLE_HYBRID_SEARCH and intent == "FOLLOW-UP":
            try:
                # Fix Problem 1: Build formatted history string
                history_text = "\n".join([
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                    for m in history[-2:]
                ])
                # Fix Problem 2: Call Ollama with a 2-second timeout to prevent blocking
                def _do_rewrite():
                    self.llm_rewrite = get_llm(model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}", temperature=0.0, streaming=False)
                    # Fix: Direct timeout on invoke() to prevent thread leakage
                    self.llm_rewrite = self.llm_rewrite.with_config({"timeout": 2.0})
                    prompt_rewrite = (
                        f"History:\n{history_text}\n\n"
                        f"Rewrite this query to be standalone: '{user_input}'\n"
                        "Return ONLY the rewritten query, no explanation."
                    )
                    raw = self.llm_rewrite.invoke(prompt_rewrite).content.strip()
                    # Strip common LLM preamble that pollutes BM25 search
                    for prefix in ("Sure,", "Here's", "The rewritten query is:", "Rewritten query:"):
                        if raw.lower().startswith(prefix.lower()):
                            raw = raw[len(prefix):].strip()
                    return raw

                search_query = _rewrite_executor.submit(_do_rewrite).result(timeout=2.2)
                logger.info(f"🔄 Query Rewritten: '{user_input}' -> '{search_query}'")
            except Exception as e:
                logger.warning(f"Query rewrite failed or timed out: {e}")
                search_query = user_input

        # Analysis Layer: Extract technical anchors for deep propagation fixes
        # Look for CamelCase or snake_case or CAPS_ID or paths/extensions
        retrieval_anchors = re.findall(r'([a-zA-Z0-9_/.]+\.[a-z]{1,4}|[A-Z][a-z]+[A-Z][a-z]+|[a-z]+_[a-z_]+|[A-Z]{3,}_[A-Z0-9_]+)', user_input)
        
        # Also extract individual terms for cross-referencing
        anchor_terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', user_input)
        snake_case_ids = [t for t in anchor_terms if "_" in t]

        # ── Fix B: Deep-link context propagation ──────────────────────
        # If we detect a specific file or symbol, we should also retrieve
        # its dependents or siblings in the next tick.
        if retrieval_anchors:
            for anchor in retrieval_anchors:
                # Add to BM25 query to boost definition sites
                term = f'"{anchor}"'
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

        # Fix I: Aggregation query detection (L3)
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
        if not skip_retrieval and self.db:
            new_retrievals = hybrid_search(
                self.db, search_query,
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

                    # Fix E: Call-graph retrieval via FTS5 OR (L2)
                    try:
                        call_docs = fts.search_by_calls_batch(top_anchors, k=15)
                        if call_docs:
                            logger.info(f"📍 Fix E: Injected {len(call_docs)} chunks calling {top_anchors}")
                            new_retrievals.extend(call_docs)
                    except Exception as e:
                        logger.warning(f"Fix E FTS5 batch call retrieval failed: {e}")

                    # Fix G: Constant-reference retrieval via FTS5 OR (L4)
                    try:
                        const_docs = fts.search_by_constants_batch(top_anchors, k=15)
                        if const_docs:
                            logger.info(f"📍 Fix G: Injected {len(const_docs)} chunks referencing constants {top_anchors}")
                            new_retrievals.extend(const_docs)
                    except Exception as e:
                        logger.warning(f"Fix G FTS5 batch constant retrieval failed: {e}")

                # Fix J: Guaranteed anchor text retrieval via ChromaDB $or (L4)
                # BM25 misses config files (implicit AND + length penalty).
                # Vector search misses them (no semantic similarity).
                # Use ChromaDB where_document $or for exact substring matching —
                # guaranteed to find any chunk containing any anchor text.
                if self.db:
                    try:
                        if len(top_anchors) > 1:
                            where_doc = {"$or": [{"$contains": a} for a in top_anchors]}
                        else:
                            where_doc = {"$contains": top_anchors[0]}
                        text_docs = self.db.similarity_search(
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
                                self.db, sub_q,
                                collection_name=coll_name,
                                k=6,
                                exclude_file=pinned_file,
                                filter_extensions=ext_filter,
                            )
                            if extra:
                                logger.info(f"📍 Fix B: Injected {len(extra)} propagation chunks for '{anchor}'")
                                new_retrievals.extend(extra)
                        except Exception as e:
                            logger.warning(f"Fix B hybrid search failed: {e}")

        if skip_retrieval and not provider_has_cache:
            # Semantic cache hit — reuse previous docs as the fresh set.
            new_retrievals = list(previous_union)

        # Save pre-rerank pool for Fix H post-reranker injection
        pre_rerank_pool = list(new_retrievals) if is_propagation_query else []

        # 3. Local Re-ranking (Phase 3) ──────────────────────────────
        if USE_RERANKER and self.reranker and new_retrievals and not skip_retrieval:
            # Fix I: Widen self.reranker window for aggregation queries so scattered
            # utility helpers aren't culled from the top-k.
            effective_top_k = RERANK_TOP_K * 2 if is_aggregation_query else RERANK_TOP_K
            new_retrievals = self.reranker.rerank(
                search_query,
                new_retrievals,
                top_k=effective_top_k
            )
            # Capture the top relevance score for telemetry
            reranker_score = getattr(self.reranker, 'last_top_score', 0.0)

        # Fix H: Post-self.reranker anchor injection (L4)
        # Force-include definition-site chunks the cross-encoder culled.
        # This ensures config constants and bridge functions survive reranking.
        if is_propagation_query and retrieval_anchors and pre_rerank_pool:
            reranked_hashes = {d.metadata.get("content_hash", "") for d in new_retrievals}
            injected = 0
            for anchor in retrieval_anchors[:3]:
                for d in pre_rerank_pool:
                    h = d.metadata.get("content_hash", "")
                    if h not in reranked_hashes:
                        # Simple substring match in chunk to confirm anchor presence
                        if anchor in d.page_content:
                            new_retrievals.append(d)
                            reranked_hashes.add(h)
                            injected += 1
                            logger.info(f"📍 Fix H: Post-self.reranker injected chunk from {d.metadata.get('source', '?')} for '{anchor}'")

        # Intent-Aware Union Logic with Context Decay
        if intent == "FOLLOW-UP":
            # Fix: Prevent "Knowledge Lock-in" by ensuring fresh retrievals 
            # always have priority. We calculate unique new docs first.
            seen_hashes = set()
            unique_new = []
            for d in new_retrievals:
                h = d.metadata.get("content_hash", "")
                if h not in seen_hashes:
                    unique_new.append(d)
                    seen_hashes.add(h)
            
            # Context Decay: Reduce priority of older context over turns.
            # We keep a union of current and previous context, but capped.
            prev_docs = [d for d in previous_union if d.metadata.get("content_hash", "") not in seen_hashes]
            
            # Sort previous docs by their internal metadata if available (turn_retrieved)
            # or just take the most recent ones.
            final_docs = unique_new + prev_docs[:MAX_CONTEXT_UNION - len(unique_new)]
        else:
            final_docs = new_retrievals[:MAX_CONTEXT_UNION]

        # 🚀 Final Filter: Context Window Budgeting
        # Scale budget based on model family
        _CONTEXT_BUDGETS = {
            "claude-3-5": 160000,
            "claude-3": 120000,
            "gpt-4": 80000,
            "gemini": 250000,
            "deepseek": 40000,
            "qwen": 24000,
        }
        _budget = 28000  # default
        for _pattern, _limit in _CONTEXT_BUDGETS.items():
            if _pattern in (self.model or "").lower():
                _budget = _limit
                break
        # Estimate: system prompt + pinned + history + RAG + user query
        # RAG portion shouldn't exceed ~40% of total budget to leave room for history
        _rag_budget = int(_budget * 0.4)
        
        # Token density check: if total chars / 3 > budget, trim.
        # We trim from the END (oldest or least relevant after re-ranking).
        
        # Format overhead estimate (roughly 80 chars per chunk for 
        # "SOURCE: ...\nCONTENT: " prefix).
        _FMT_OVERHEAD_PER_CHUNK = 80
        
        # Fix Performance: Calculate total length once and decrement instead of re-summing in a loop (O(N) vs O(N^2))
        total_rag_chars = sum(len(d.page_content) + _FMT_OVERHEAD_PER_CHUNK for d in final_docs)
        
        while final_docs and (total_rag_chars // 3) > _rag_budget:
            # Drop the doc with the lowest relevance if re-ranked, or just the last one
            # If follow-up, unique_new are at the front, prev_docs at the back.
            # We pop from the back.
            idx = -1
            removed_doc = final_docs.pop(idx)
            total_rag_chars -= (len(removed_doc.page_content) + _FMT_OVERHEAD_PER_CHUNK)

        established_docs, new_docs, stable_block_str, new_block_str = self._prepare_context(final_docs, previous_union, intent)

        inputs["stable_context"] = stable_block_str
        inputs["new_context"] = new_block_str
        inputs["context"] = established_docs + new_docs
        inputs["chat_history"] = _prepare_history_with_cache(history, self.model)
        
        # Dynamic Specialist Swap — cached LLM instances
        active_chain = self.question_answer_chain
        if enable_auto and specialist_model:
            current_m = getattr(
                active_chain.bound if hasattr(active_chain, "bound") else active_chain,
                "model_name", "",
            )
            if specialist_model != current_m:
                if specialist_model not in self._specialist_llm_cache:
                    self._specialist_llm_cache[specialist_model] = get_llm(
                        model=specialist_model, streaming=True
                    )
                active_chain = self.prompt | self._specialist_llm_cache[specialist_model]

        # Dynamic output token budget — reduce for simple queries to free provider quota
        output_tokens = _get_max_tokens(specialty, user_input)
        if output_tokens != MAX_TOKENS:
            base_llm = (
                self._specialist_llm_cache[specialist_model]
                if (enable_auto and specialist_model and specialist_model in self._specialist_llm_cache)
                else self.llm
            )
            # ChatOllama uses 'num_predict' for output token budget; OpenAI/OpenRouter use 'max_tokens'.
            active_model_id = specialist_model or self.model or ""
            if active_model_id.startswith(OLLAMA_PREFIX):
                active_chain = self.prompt | base_llm.bind(num_predict=output_tokens)
            else:
                active_chain = self.prompt | base_llm.bind(max_tokens=output_tokens)
        
        # Ensure we always have an embedding to pass back for next turn.
        if current_emb is None:
            try:
                current_emb = sem_cache.embedding_model.embed_query(user_input)
            except Exception:
                from backend import get_embedding_model
                current_emb = get_embedding_model().embed_query(user_input)

        # ASYNC SENTINEL TRIGGER — uses full_history so the summary
        # covers the entire conversation, not just the compressed window.
        background_future = self._handle_sentinel(should_summarize, inputs, turn_count, full_history)

        yield {
            "context": inputs["context"], 
            "intent": intent, 
            "specialty": specialty, 
            "reranker_score": reranker_score,
            "query_embedding": current_emb,
            "sentinel_future": background_future # Pass future to UI for persistence
        }
        
        yield from self._execute_llm_stream(active_chain, inputs, is_semantic_hit, user_input, coll_name, pinned_content, sem_cache)


def build_rag_chain(db: Chroma, model: str | None = None):
    """
    Build a retrieval chain with stable Full-Context Caching (Architecture A).
    """
    llm = get_llm(model=model)
    
    is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING
    
    max_bp, _ = get_cache_profile(model)

    if is_cc:
        static_system_text = CORE_INSTRUCTIONS
        block_specs = [
            static_system_text,
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}",
            "STABLE RAG CONTEXT (DETERMINISTIC):\n{stable_context}",
            "CONVERSATION STATE:\n{sentinel_state}",
            "NEW RAG DISCOVERIES:\n{new_context}"
        ]
        system_blocks = []
        for idx, text in enumerate(block_specs):
            use_cache_marker = idx < max_bp
            formatted = format_message_content(text, model, use_cache=use_cache_marker)
            if isinstance(formatted, list):
                system_blocks.append(formatted[0])
            else:
                system_blocks.append({"type": "text", "text": formatted})

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    else:
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

    router = get_router()
    reranker = get_reranker()

    from langchain_core.runnables import RunnableLambda
    pipeline = ContextCacheChain(db, model, llm, prompt, router, reranker)
    return RunnableLambda(pipeline)
