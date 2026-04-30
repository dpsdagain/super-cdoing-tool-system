"""
search_engine.py — Hybrid Search (BM25 + Vector) and Cross-Encoder Re-ranking.
"""

# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations
import re
import threading
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
import atexit as _atexit
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from backend import SQLiteFTS5BM25
from config import (
    ENABLE_HYBRID_SEARCH,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    USE_RERANKER,
    RERANK_MODEL,
)

logger = logging.getLogger(__name__)

_rewrite_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rewrite")
_atexit.register(_rewrite_executor.shutdown, wait=False)


def _analyze_query_intent(query: str):
    """Extract technical anchors and detect propagation/aggregation intents."""
    # Analysis Layer: Extract technical anchors for deep propagation fixes
    # Look for CamelCase or snake_case or CAPS_ID or paths/extensions
    retrieval_anchors = re.findall(
        r"([a-zA-Z0-9_/.]+\.[a-z]{1,4}|[A-Z][a-z]+[A-Z][a-z]+|[a-z]+_[a-z_]+|[A-Z]{3,}_[A-Z0-9_]+)",
        query,
    )

    # Also extract individual terms for cross-referencing
    anchor_terms = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", query)
    snake_case_ids = [t for t in anchor_terms if "_" in t]

    # ── Fix D: 3-Hop Propagation Detection ─────────────────────────
    _PROPAGATION_KEYWORDS = (
        "propagat",
        "impact",
        "affect",
        "changes when",
        "change when",
        "changed by",
        "changed if",
        "trace",
        "alter",
        "flow",
        "step-by-step",
        "step by step",
        "how does",
        "ultimately",
    )
    is_propagation_query = bool(anchor_terms or snake_case_ids) and any(
        kw in query.lower() for kw in _PROPAGATION_KEYWORDS
    )

    # Fix I: Aggregation query detection (L3)
    _AGGREGATION_KEYWORDS = (
        "every ",
        "all places",
        "each place",
        "everywhere",
        "all functions",
        "list all",
        "every file",
        "all files",
        "all the places",
        "every place",
        "each function",
        "all methods",
        "every method",
        "scattered",
        "each file",
        "which files",
        "what files",
        "list down",
        "files in",
        "files you see",
        "list of files",
    )
    is_aggregation_query = any(kw in query.lower() for kw in _AGGREGATION_KEYWORDS)

    return {
        "anchors": retrieval_anchors,
        "is_propagation": is_propagation_query,
        "is_aggregation": is_aggregation_query,
    }


def _rewrite_query(query: str, history: list | None, intent: str) -> str:
    """Phase 0: Query Rewriting (Contextual Awareness)."""
    from llm_factory import get_llm
    from config import OLLAMA_PREFIX, AGENT_ROUTER_MODEL
    from langchain_core.messages import HumanMessage

    if not history or intent != "FOLLOW-UP":
        return query

    try:
        history_text = "\n".join(
            [
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in history[-2:]
            ]
        )

        def _do_rewrite():
            llm_rewrite = get_llm(
                model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}",
                temperature=0.0,
                streaming=False,
            )
            llm_rewrite = llm_rewrite.with_config({"timeout": 2.0})
            prompt_rewrite = (
                f"History:\n{history_text}\n\n"
                f"Rewrite this query to be standalone: '{query}'\n"
                "Return ONLY the rewritten query, no explanation."
            )
            raw = llm_rewrite.invoke(prompt_rewrite).content.strip()
            for prefix in (
                "Sure,",
                "Here's",
                "The rewritten query is:",
                "Rewritten query:",
            ):
                if raw.lower().startswith(prefix.lower()):
                    raw = raw[len(prefix) :].strip()
            return raw

        search_query = _rewrite_executor.submit(_do_rewrite).result(timeout=2.2)
        logger.info(f"🔄 Query Rewritten: '{query}' -> '{search_query}'")
        return search_query
    except Exception:
        logger.exception("Query rewrite failed or timed out:")
        return query


def _apply_retrieval_fixes(
    new_retrievals: list,
    db: Chroma,
    search_query: str,
    collection_name: str,
    anchors: list,
    exclude_file: str | None,
    filter_extensions: list | None,
) -> None:
    """Phase 2: Advanced Retrieval Fixes (B, D, E, G, J)"""
    with SQLiteFTS5BM25(collection_name) as fts:
        top_anchors = anchors[:3]

        # Fix E: Call-graph retrieval
        try:
            call_docs = fts.search_by_calls_batch(top_anchors, k=15)
            if call_docs:
                logger.info(
                    f"📍 Fix E: Injected {len(call_docs)} chunks calling {top_anchors}"
                )
                new_retrievals.extend(call_docs)
        except Exception:
            logger.exception("Fix E FTS5 batch call retrieval failed:")

        # Fix G: Constant-reference retrieval
        try:
            const_docs = fts.search_by_constants_batch(top_anchors, k=15)
            if const_docs:
                logger.info(
                    f"📍 Fix G: Injected {len(const_docs)} chunks referencing constants {top_anchors}"
                )
                new_retrievals.extend(const_docs)
        except Exception:
            logger.exception("Fix G FTS5 batch constant retrieval failed:")

    # Fix J: Guaranteed anchor text retrieval via ChromaDB $or
    try:
        if len(top_anchors) > 1:
            where_doc = {"$or": [{"$contains": a} for a in top_anchors]}
        else:
            where_doc = {"$contains": top_anchors[0]}
        text_docs = db.similarity_search(search_query, k=10, where_document=where_doc)
        if text_docs:
            logger.info(
                f"📍 Fix J: Injected {len(text_docs)} chunks containing {top_anchors}"
            )
            new_retrievals.extend(text_docs)
    except Exception:
        logger.exception("Fix J text search failed:")

    # Deep propagation (Fix B)
    for anchor in anchors[:2]:
        for sub_q in (f"function that reads {anchor}", f"where {anchor} is used"):
            try:
                # Recursive call but without history/intent to avoid infinite loop
                extra = hybrid_search(
                    db,
                    sub_q,
                    collection_name=collection_name,
                    k=6,
                    exclude_file=exclude_file,
                    filter_extensions=filter_extensions,
                )
                if extra:
                    new_retrievals.extend(extra)
            except Exception:
                pass


def hybrid_search(
    db: Chroma,
    query: str,
    collection_name: str = "default",
    k: int = 10,
    exclude_file: str | None = None,
    filter_extensions: list[str] | None = None,
    query_embedding: list[float] | None = None,
    history: list | None = None,
    intent: str = "NEW",
) -> list[Document]:
    """
    Perform Hybrid Search (BM25 + Vector) with Reciprocal Rank Fusion (RRF)
    and advanced architectural fixes (Call-graph, Constants, Anchor Boosting).
    """

    # ── Phase 0: Query Rewriting (Contextual Awareness) ──────────────
    search_query = _rewrite_query(query, history, intent)

    # ── Phase 1: Intent & Anchor Analysis ──────────────────────────
    analysis = _analyze_query_intent(search_query)
    anchors = analysis["anchors"]
    is_propagation = analysis["is_propagation"]
    is_aggregation = analysis["is_aggregation"]

    # Fix B: Anchor Boosting
    if anchors:
        for anchor in anchors:
            term = f'"{anchor}"'
            if term not in search_query:
                search_query = f"{search_query} {term}"

    if not ENABLE_HYBRID_SEARCH:
        # Fallback to standard vector search.
        chroma_filter = None
        if exclude_file:
            chroma_filter = {"source": {"$ne": exclude_file}}
        if query_embedding:
            return db.similarity_search_by_vector(
                query_embedding, k=k, filter=chroma_filter
            )
        return db.similarity_search(search_query, k=k, filter=chroma_filter)

    # Build a ChromaDB metadata filter from the caller's exclusion criteria
    _conditions: list[dict] = []
    if exclude_file:
        _conditions.append({"source": {"$ne": exclude_file}})
    if filter_extensions:
        _conditions.append({"file_extension": {"$in": filter_extensions}})
    if len(_conditions) == 0:
        _chroma_filter = None
    elif len(_conditions) == 1:
        _chroma_filter = _conditions[0]
    else:
        _chroma_filter = {"$and": _conditions}

    # 1. Vector Search (Semantic)
    k_fetch = k * 3
    if query_embedding:
        vector_docs = db.similarity_search_by_vector(
            query_embedding, k=k_fetch, filter=_chroma_filter
        )
    else:
        vector_docs = db.similarity_search(
            search_query, k=k_fetch, filter=_chroma_filter
        )

    # 2. BM25 Keyword Search (SQLite FTS5)
    with SQLiteFTS5BM25(collection_name) as fts:
        bm25_docs = fts.search(search_query, k=k_fetch)

    # Apply remaining excludes
    if exclude_file or filter_extensions:
        bm25_docs = [
            d
            for d in bm25_docs
            if not (exclude_file and d.metadata.get("source") == exclude_file)
            and not (
                filter_extensions
                and d.metadata.get("file_extension") not in filter_extensions
            )
        ]

    new_retrievals = list(vector_docs) + list(bm25_docs)

    # ── Phase 2: Advanced Retrieval Fixes (B, D, E, G, J) ───────────
    if is_propagation:
        _apply_retrieval_fixes(
            new_retrievals,
            db,
            search_query,
            collection_name,
            anchors,
            exclude_file,
            filter_extensions,
        )

    # 3. Reciprocal Rank Fusion (RRF)
    RRF_K = 60
    scores = {}
    doc_map = {}

    def _rank_docs(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            doc_id = (
                doc.metadata.get("source"),
                doc.metadata.get("chunk_index", 0),
                doc.metadata.get("content_hash", doc.page_content[:64]),
            )
            score = (1.0 / (RRF_K + rank + 1)) * weight
            scores[doc_id] = scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc

    _rank_docs(vector_docs, weight=VECTOR_WEIGHT)
    _rank_docs(bm25_docs, weight=BM25_WEIGHT)
    # Give injected docs a baseline rank if they weren't in the top-k
    _rank_docs(
        [
            d
            for d in new_retrievals
            if (
                d.metadata.get("source"),
                d.metadata.get("chunk_index", 0),
                d.metadata.get("content_hash", d.page_content[:64]),
            )
            not in scores
        ],
        weight=0.5,
    )

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Fix I: Widen top-k for aggregation queries
    effective_k = k * 2 if is_aggregation else k
    rrf_results = [doc_map[did] for did in sorted_ids[:effective_k]]

    return rrf_results


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
        except Exception:
            logger.exception("Re-ranker failed to load:")
            return None

    def rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        """Re-score and filter documents using the Cross-Encoder."""
        if not self.model or not documents:
            return documents[:top_k]

        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = self.model.predict(pairs)
            scored_docs = sorted(
                zip(scores, documents), key=lambda x: x[0], reverse=True
            )
            self.last_top_score = float(scored_docs[0][0]) if scored_docs else 0.0

            if scored_docs:
                logger.info(f"Top Re-rank Relevance Score: {scored_docs[0][0]:.4f}")

            return [doc for score, doc in scored_docs[:top_k]]
        except Exception:
            logger.exception("Re-ranking execution failed:")
            return documents[:top_k]


_reranker_lock = threading.Lock()


def get_reranker():
    """Lazily initialize the LocalReRanker singleton."""
    if not USE_RERANKER:
        return None

    if not hasattr(get_reranker, "instance"):
        with _reranker_lock:
            if not hasattr(get_reranker, "instance"):
                try:
                    get_reranker.instance = LocalReRanker()
                    logger.info("Reranker initialized (LocalReRanker)")
                except Exception:
                    logger.exception("Reranker init failed:")
                    return None
    return get_reranker.instance


def _sort_docs_deterministically(
    docs: list[Document],
    stable_hashes: set[str] | None = None,
) -> list[Document]:
    """
    Prefix-Preserving Deterministic Sort.
    """

    def _sort_key(d):
        is_new = (
            0
            if (stable_hashes and d.metadata.get("content_hash", "") in stable_hashes)
            else (1 if stable_hashes else 0)
        )
        return (
            is_new,
            str(d.metadata.get("source", "")),
            int(d.metadata.get("chunk_index", 0)),
            str(d.metadata.get("content_hash", "")),
            str(d.page_content),
        )

    return sorted(docs, key=_sort_key)
