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

_rewrite_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rewrite")

def _get_pinned_embedding(pinned_prefix: str) -> list[float]:
    """Return a cached embedding for the pinned content prefix. Thread-safe."""
    from backend import get_embedding_model
    return get_embedding_model().embed_query(pinned_prefix)

def hybrid_search(
    db: Chroma,
    query: str,
    collection_name: str = "default",
    k: int = 10,
    exclude_file: str | None = None,
    filter_extensions: list[str] | None = None,
    query_embedding: list[float] | None = None,
) -> list[Document]:
    """
    Perform Hybrid Search (BM25 + Vector) with Reciprocal Rank Fusion (RRF).

    If *query_embedding* is provided, it is reused for the vector search
    via ``similarity_search_by_vector``, avoiding a redundant embedding
    inference that ChromaDB would otherwise perform internally.
    """
    if not ENABLE_HYBRID_SEARCH:
        # Fallback to standard vector search.
        # fetch_k is an MMR-only parameter and is not supported by
        # similarity_search / similarity_search_by_vector — omit it.
        chroma_filter = None
        if exclude_file:
            chroma_filter = {"source": {"$ne": exclude_file}}
        if query_embedding:
            return db.similarity_search_by_vector(query_embedding, k=k, filter=chroma_filter)
        return db.similarity_search(query, k=k, filter=chroma_filter)

    # Build a ChromaDB metadata filter from the caller's exclusion criteria so
    # the vector store never fetches docs that will be thrown away post-fetch.
    # BM25 has no filter API — _rank_docs() still handles that side.
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

    # 1. 🔍 Vector Search (Semantic)
    # We fetch a larger candidate pool for RRF to merge.
    # Reuse pre-computed embedding when available to avoid double-embedding.
    if query_embedding:
        vector_docs = db.similarity_search_by_vector(query_embedding, k=k*3, filter=_chroma_filter)
    else:
        vector_docs = db.similarity_search(query, k=k*3, filter=_chroma_filter)
    
    # 2. 🔍 BM25 Keyword Search (Transitioned to SQLite FTS5)
    with SQLiteFTS5BM25(collection_name) as fts:
        # Note: SQLite search internalizes metadata filtering for better performance
        bm25_docs = fts.search(query, k=k*3)
    
    # Apply remaining excludes that are not yet in FTS SQL query
    if exclude_file or filter_extensions:
        bm25_docs = [
            d for d in bm25_docs
            if not (exclude_file and d.metadata.get("source") == exclude_file)
            and not (filter_extensions and d.metadata.get("file_extension") not in filter_extensions)
        ]

    # 3. 🧪 Reciprocal Rank Fusion (RRF)
    # RRF Score(d) = sum(1 / (k + rank))
    RRF_K = 60
    scores = {} # {doc_id: score}
    doc_map = {} # {doc_id: doc_object}
    
    def _rank_docs(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            if exclude_file and doc.metadata.get("source") == exclude_file:
                continue
            if filter_extensions and doc.metadata.get("file_extension") not in filter_extensions:
                continue
                
            # 🚀 Fix: Use content excerpt to prevent collisions on zero-chunk docs with missing hashes
            doc_id = (
                doc.metadata.get("source"),
                doc.metadata.get("chunk_index", 0),
                doc.metadata.get("content_hash", doc.page_content[:64])
            )
            score = (1.0 / (RRF_K + rank + 1)) * weight
            scores[doc_id] = scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc

    _rank_docs(vector_docs, weight=VECTOR_WEIGHT)
    _rank_docs(bm25_docs, weight=BM25_WEIGHT)
    
    # Sort by merged RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    rrf_results = [doc_map[did] for did in sorted_ids[:k]]

    # Cross-encoder re-ranking is handled by LocalReRanker.rerank() in the
    # caller — applying it here as well would score the same docs twice with
    # the same model for zero quality gain.

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

_reranker_instance = None
_reranker_lock = threading.Lock()

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None and USE_RERANKER:
        with _reranker_lock:
            if _reranker_instance is None:  # double-check after lock
                try:
                    _reranker_instance = LocalReRanker()
                    logger.info("✅ Reranker initialized (LocalReRanker)")
                except Exception as e:
                    logger.error(f"❌ Reranker init failed: {e}")
    return _reranker_instance

def _sort_docs_deterministically(
    docs: list[Document],
    stable_hashes: set[str] | None = None,
) -> list[Document]:
    """
    Prefix-Preserving Deterministic Sort.

    When *stable_hashes* is provided (the content hashes of docs that were
    already in the prompt on the previous turn), those docs sort FIRST
    (``_is_new=0``), preserving the exact byte prefix that the provider
    cache (Anthropic/Gemini) already stored.  New docs sort AFTER
    (``_is_new=1``) so they append to the end of the block without
    breaking the cached prefix.

    Within each group the order is fully deterministic:
    ``(source, chunk_index, content_hash, page_content)``.
    """
    def _sort_key(d):
        is_new = 0 if (
            stable_hashes
            and d.metadata.get("content_hash", "") in stable_hashes
        ) else (1 if stable_hashes else 0)
        return (
            is_new,
            str(d.metadata.get("source", "")),
            int(d.metadata.get("chunk_index", 0)),
            str(d.metadata.get("content_hash", "")),
            str(d.page_content),
        )
    return sorted(docs, key=_sort_key)

def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

