"""
rag_core.py — Retrieval-Augmented Generation Query Pipeline.

Handles:
  - OpenRouter LLM configuration
  - ChromaDB retriever setup
  - LangChain retrieval chain construction
"""

# pylint: disable=unused-argument,too-many-instance-attributes,too-many-arguments,too-many-locals

from __future__ import annotations
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
    OLLAMA_PREFIX,
    USE_RERANKER,
    RERANK_TOP_K,
    RERANK_CANDIDATES,
    SPECIALIST_MAPPING,
    GHOST_HISTORY_WINDOW,
    GHOST_HISTORY_MAX,
    AI_RESPONSE_MAX_CHARS,
    GHOST_AI_CHARS,
)
from estimator import ContextEstimator
from cache_engine import get_semantic_cache
from intent_router import get_router
from prompt_builder import CORE_INSTRUCTIONS
from llm_factory import (
    is_cache_capable,
    get_cache_profile,
    format_message_content,
    get_llm,
)
from search_engine import get_reranker, hybrid_search, _sort_docs_deterministically

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
    except Exception:
        logger.exception("❌ Background summary failed:")
        return None


def _truncate_ai_in_history(history: list[BaseMessage]) -> list[BaseMessage]:
    """
    Cap AI response length in chat history to reduce token waste,
    while aggressively preserving code blocks.
    """
    import re

    truncated = []
    for msg in history:
        if isinstance(msg, AIMessage) and len(msg.content) > AI_RESPONSE_MAX_CHARS:
            code_blocks = re.findall(r"(```.*?```)", msg.content, flags=re.DOTALL)
            if code_blocks:
                gist = msg.content[:400]
                trimmed = f"{gist}\n... [prose truncated]\n\n" + "\n\n".join(
                    code_blocks
                )
                if len(trimmed) > AI_RESPONSE_MAX_CHARS * 3:
                    trimmed = (
                        trimmed[: AI_RESPONSE_MAX_CHARS * 3]
                        + "\n```\n... [code truncated]"
                    )
            else:
                trimmed = (
                    msg.content[:AI_RESPONSE_MAX_CHARS]
                    + "\n... [truncated for context efficiency]"
                )
            truncated.append(AIMessage(content=trimmed))
        else:
            truncated.append(msg)
    return truncated


def compress_chat_history(
    history: list[BaseMessage], sentinel_state: str
) -> list[BaseMessage]:
    """Intelligently trim the chat history based on Sentinel Summaries or Ghost History logic."""
    if sentinel_state and sentinel_state != "No summary generated yet.":
        keep = 4  # last 2 user+AI pairs
        truncated_history = history[-keep:]
        truncated_history = _truncate_ai_in_history(truncated_history)
    elif len(history) <= GHOST_HISTORY_MAX + 2:
        truncated_history = _truncate_ai_in_history(history)
    else:
        anchor = history[:2]
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


def _prepare_history_with_cache(
    history: list[BaseMessage], model: str | None
) -> list[BaseMessage]:
    """Optimizes history for prefix caching."""
    if not history:
        return []
    user_msg_indices = [i for i, m in enumerate(history) if isinstance(m, HumanMessage)]
    if len(user_msg_indices) >= 2:
        target_idx = user_msg_indices[-1] - 1
        if target_idx >= 0:
            msg = history[target_idx]
            new_msg = AIMessage(
                content=format_message_content(msg.content, model, use_cache=True)
            )
            new_history = list(history)
            new_history[target_idx] = new_msg
        return new_history
    return list(history)


class SpecialistRouter:
    """Handles specialist model selection and dynamic budgeting."""

    def __init__(self, main_model: str | None):
        self.main_model = main_model
        self.router = get_router()
        self._specialist_llm_cache = {}

    def get_max_tokens(self, specialty: str | None, query: str) -> int:
        """Return output token budget scaled to query complexity."""
        if specialty in ("CODE", "REASONING") or len(query) > 200:
            return MAX_TOKENS
        return 1024

    def route(
        self, user_input: str, history: list, enable_auto: bool
    ) -> tuple[str, str, str | None]:
        """Detect intent and specialty, returning (intent, specialty, specialist_model)."""
        intent = self.router.classify_intent(user_input, history) if history else "NEW"
        specialty = (
            self.router.detect_specialty(user_input) if enable_auto else "GENERAL"
        )

        specialist_model = (
            SPECIALIST_MAPPING.get(specialty) if specialty != "GENERAL" else None
        )

        # Guard: Local model should not route to cloud specialist
        if (
            specialist_model
            and not specialist_model.startswith(OLLAMA_PREFIX)
            and self.main_model
            and self.main_model.startswith(OLLAMA_PREFIX)
        ):
            specialist_model = None

        return intent, specialty, specialist_model

    def get_active_llm(self, specialist_model: str | None, base_llm, prompt):
        """Get the LLM instance for the current turn, potentially swapping for a specialist."""
        if specialist_model:
            if specialist_model not in self._specialist_llm_cache:
                self._specialist_llm_cache[specialist_model] = get_llm(
                    model=specialist_model, streaming=True
                )
            return self._specialist_llm_cache[specialist_model]
        return base_llm


class ContextBudgeter:
    """Manages the context window and RAG document union/truncation."""

    def __init__(self, model: str | None):
        self.model = model

    def get_budget(self) -> int:
        _CONTEXT_BUDGETS = {
            "claude-3-5": 160000,
            "claude-3": 120000,
            "gpt-4": 80000,
            "gemini": 250000,
            "deepseek": 40000,
            "qwen": 24000,
        }
        _budget = 28000
        for p, limit in _CONTEXT_BUDGETS.items():
            if p in (self.model or "").lower():
                _budget = limit
                break
        return int(_budget * 0.4)

    def process_union(
        self, new_docs: list[Document], previous_union: list[Document], intent: str
    ) -> list[Document]:
        """Crops and prioritizes the union of fresh and existing RAG context."""
        if intent == "FOLLOW-UP":
            seen_hashes = {d.metadata.get("content_hash", "") for d in new_docs}
            unique_new = []
            for d in new_docs:
                if d.metadata.get("content_hash", "") not in seen_hashes:
                    unique_new.append(d)
            prev_docs = [
                d
                for d in previous_union
                if d.metadata.get("content_hash", "") not in seen_hashes
            ]
            final_docs = unique_new + prev_docs[: MAX_CONTEXT_UNION - len(unique_new)]
        else:
            final_docs = new_docs[:MAX_CONTEXT_UNION]

        # Truncation loop
        budget = self.get_budget()
        while (
            final_docs
            and ContextEstimator.estimate_content_tokens(
                [d.page_content for d in final_docs]
            )
            > budget
        ):
            final_docs.pop(-1)
        return final_docs


class ContextCacheChain:
    """Decomposed RAG Orchestration Engine."""

    def __init__(self, db: Chroma, model: str | None, llm, prompt, router, reranker):
        self.db = db
        self.model = model
        self.llm = llm
        self.prompt = prompt
        self.reranker = reranker
        self.question_answer_chain = prompt | llm

        self.specialist_router = SpecialistRouter(model)
        self.budgeter = ContextBudgeter(model)

        self._sentinel_cooldown = {"last_turn": 0}
        self._sentinel_failures = {"count": 0}
        self.MAX_SENTINEL_FAILURES = 3

    def _on_sentinel_done(self, future):
        try:
            future.result()
            self._sentinel_failures["count"] = 0
        except Exception:
            self._sentinel_failures["count"] += 1
            logger.exception("Sentinel failure (%d):", self._sentinel_failures["count"])

    def _prepare_context_strings(self, final_docs, previous_union, intent):
        stable_hashes = (
            {d.metadata.get("content_hash", "") for d in previous_union}
            if previous_union and intent == "FOLLOW-UP"
            else None
        )
        established_docs, new_docs = [], []
        if stable_hashes:
            for d in final_docs:
                if d.metadata.get("content_hash", "") in stable_hashes:
                    established_docs.append(d)
                else:
                    new_docs.append(d)
        else:
            new_docs = final_docs

        def _format(docs):
            return (
                "\n\n".join(
                    [
                        f"SOURCE: {d.metadata.get('source')}\nCONTENT: {d.page_content}"
                        for d in docs
                    ]
                )
                if docs
                else ""
            )

        return (
            established_docs + new_docs,
            f"<established_context>\n{_format(_sort_docs_deterministically(established_docs))}\n</established_context>",
            f"<new_discoveries>\n{_format(_sort_docs_deterministically(new_docs))}\n</new_discoveries>",
        )

    def __call__(self, inputs: dict):
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        coll_name = inputs.get("collection_name", "default")
        previous_union = inputs.get("context", [])

        # 1. Semantic Cache Lookup
        sem_cache = get_semantic_cache()
        if not inputs.get("force_retrieval", False):
            cached = sem_cache.lookup(
                user_input,
                threshold=SEMANTIC_CACHE_THRESHOLD,
                collection_scope=coll_name,
                pinned_content=pinned_content,
                model=self.model,
            )
            if cached:
                yield {"answer": cached, "intent": "CACHE_HIT"}
                return

        # 2. Intent & Specialist Routing
        enable_auto = inputs.get("auto_specialist", ENABLE_AUTO_SPECIALIST)
        intent, specialty, specialist_model = self.specialist_router.route(
            user_input, history, enable_auto
        )

        # 3. Discovery & Retrieval (Decomposed to search_engine)
        k_fetch = RERANK_CANDIDATES if USE_RERANKER else RETRIEVER_K
        new_retrievals = hybrid_search(
            self.db,
            user_input,
            collection_name=coll_name,
            k=k_fetch,
            exclude_file=inputs.get("exclude_file"),
            filter_extensions=inputs.get("filter_extensions"),
            query_embedding=(
                inputs.get("last_query_embedding")
                if inputs.get("last_query") == user_input
                else None
            ),
            history=history,
            intent=intent,
        )

        # 4. Reranking & Budgeting
        if USE_RERANKER and self.reranker and new_retrievals:
            # Re-rank window adjustment logic is now in hybrid_search intent analysis
            new_retrievals = self.reranker.rerank(
                user_input, new_retrievals, top_k=RERANK_TOP_K
            )

        final_docs = self.budgeter.process_union(new_retrievals, previous_union, intent)
        full_context_docs, stable_str, new_str = self._prepare_context_strings(
            final_docs, previous_union, intent
        )

        # 5. Execution State Preparation
        inputs.update(
            {
                "full_source_context": pinned_content or "None pinned.",
                "stable_context": stable_str,
                "new_context": new_str,
                "context": full_context_docs,
                "chat_history": _prepare_history_with_cache(history, self.model),
            }
        )

        # 6. Specialist Model Execution
        active_llm = self.specialist_router.get_active_llm(
            specialist_model, self.llm, self.prompt
        )
        output_tokens = self.specialist_router.get_max_tokens(specialty, user_input)

        if output_tokens != MAX_TOKENS:
            active_llm = (
                active_llm.bind(num_predict=output_tokens)
                if (specialist_model or self.model or "").startswith(OLLAMA_PREFIX)
                else active_llm.bind(max_tokens=output_tokens)
            )

        active_chain = self.prompt | active_llm

        # 7. Sentinel Summarization (Background)
        full_history = inputs.get("full_history", history)
        est_tokens = ContextEstimator.estimate_tokens(full_history)
        turn_count = inputs.get(
            "global_turn_count", sum(1 for m in history if isinstance(m, HumanMessage))
        )

        if (
            turn_count > 0
            and self._sentinel_failures["count"] < self.MAX_SENTINEL_FAILURES
            and est_tokens >= SENTINEL_TOKEN_THRESHOLD
            and (turn_count - self._sentinel_cooldown["last_turn"]) >= SENTINEL_INTERVAL
        ):
            self._sentinel_cooldown["last_turn"] = turn_count
            fut = _background_executor.submit(_background_summarize, list(full_history))
            fut.add_done_callback(self._on_sentinel_done)
            inputs["sentinel_future"] = fut

        # 8. Streaming LLM Invocation
        yield {"context": full_context_docs, "intent": intent, "specialty": specialty}
        full_answer = []
        for chunk in active_chain.stream(inputs):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer.append(content)
            yield {"answer": content, "raw_chunk": chunk}

        if len("".join(full_answer)) > 50:
            sem_cache.upsert(
                user_input,
                "".join(full_answer),
                collection_scope=coll_name,
                pinned_content=pinned_content,
                model=self.model,
            )


def build_rag_chain(db: Chroma, model: str | None = None):
    llm = get_llm(model=model)
    is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING
    max_bp, _ = get_cache_profile(model)

    if is_cc:
        block_specs = [
            CORE_INSTRUCTIONS,
            "FULL SOURCE CONTEXT:\n{full_source_context}",
            "STABLE RAG CONTEXT:\n{stable_context}",
            "CONVERSATION STATE:\n{sentinel_state}",
            "NEW RAG DISCOVERIES:\n{new_context}",
        ]
        system_blocks = [
            {
                "type": "text",
                "text": format_message_content(t, model, use_cache=(i < max_bp)),
            }
            for i, t in enumerate(block_specs)
        ]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_blocks),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    else:
        system_text = f"{CORE_INSTRUCTIONS}\n\nPINNED:\n{{full_source_context}}\n\nSTABLE:\n{{stable_context}}\n\nSTATE:\n{{sentinel_state}}\n\nNEW:\n{{new_context}}"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(
        ContextCacheChain(db, model, llm, prompt, get_router(), get_reranker())
    )
