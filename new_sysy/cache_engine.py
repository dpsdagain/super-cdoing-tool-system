"""
cache_engine.py — Semantic Cache using ChromaDB.
Bypasses RAG + LLM for repeat queries with high similarity.
"""
from __future__ import annotations
import hashlib
import threading
import logging
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Persistent Query/Response cache using ChromaDB.
    Bypasses RAG and LLM for repeat queries.
    """
    def __init__(self, collection_name: str = "semantic_cache"):
        from backend import get_embedding_model
        from config import CHROMA_DB_DIR
        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embedding_model(),
            collection_name=collection_name,
        )

    @staticmethod
    def _pinned_fingerprint(pinned_content: str | None) -> str:
        """Short stable hash of the pinned context.

        Using the content (not path) means "file A pinned then edited" and
        "file A pinned fresh" produce different fingerprints, so the cache
        won't serve pre-edit answers for post-edit content.
        Empty / None collapses to the sentinel "_none_" so unpinned turns
        share a cache bucket.
        """
        if not pinned_content:
            return "_none_"
        return hashlib.md5(pinned_content.encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def _model_fingerprint(model: str | None) -> str:
        """Normalize a model id so provider/size changes bust the cache.
        Different models phrase answers differently — don't cross-serve.
        """
        return (model or "_default_").strip().lower()

    def lookup(self, query: str, threshold: float = 0.95,
               collection_scope: str | None = None,
               pinned_content: str | None = None,
               model: str | None = None) -> str | None:
        """Find a cached answer if similarity exceeds threshold.

        Scope dimensions (all enforced):
          - collection_scope: which knowledge base the answer was grounded in
          - pinned_fp:        fingerprint of the pinned file content at time of answer
          - model_fp:         which model produced the answer

        Any mismatch is treated as a miss — safer to regenerate than to
        cross-serve an answer grounded in a different file / model.
        """
        results = self.db.similarity_search_with_relevance_scores(query, k=3)
        pinned_fp = self._pinned_fingerprint(pinned_content)
        model_fp = self._model_fingerprint(model)
        for doc, score in results:
            if score < threshold:
                break
            md = doc.metadata
            if collection_scope and md.get("source_collection") != collection_scope:
                continue
            # Only reject on a pinned-fp mismatch when BOTH sides declare one.
            cached_pinned = md.get("pinned_fp")
            if cached_pinned and cached_pinned != pinned_fp:
                continue
            cached_model = md.get("model_fp")
            if cached_model and cached_model != model_fp:
                continue
            logger.info(f"⚡ Semantic Cache Hit (Score: {score:.4f})")
            return md.get("answer")
        return None

    def upsert(self, query: str, answer: str, collection_scope: str | None = None,
               pinned_content: str | None = None, model: str | None = None):
        """Save successful generation to cache, tagged with the full scope."""
        meta = {
            "answer": answer,
            "type": "cached_response",
            "pinned_fp": self._pinned_fingerprint(pinned_content),
            "model_fp": self._model_fingerprint(model),
        }
        if collection_scope:
            meta["source_collection"] = collection_scope
        self.db.add_texts(
            texts=[query],
            metadatas=[meta]
        )

_semantic_cache_instance = None
_semantic_cache_lock = threading.Lock()

def get_semantic_cache():
    global _semantic_cache_instance
    with _semantic_cache_lock:
        if _semantic_cache_instance is None:
            _semantic_cache_instance = SemanticCache()
        return _semantic_cache_instance

def reset_semantic_cache():
    """Drop the cached SemanticCache singleton so the next call rebuilds it."""
    global _semantic_cache_instance
    with _semantic_cache_lock:
        _semantic_cache_instance = None
