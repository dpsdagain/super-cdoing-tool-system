"""
embeddings.py — Singleton Embedding Model Provider.
Provides the shared HuggingFaceEmbeddings instance used by all RAG components.
"""

import threading
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_embedding_model: HuggingFaceEmbeddings | None = None
_embed_lock = threading.Lock()


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the singleton embedding model, downloading on first call (thread-safe)."""
    global _embedding_model
    with _embed_lock:
        if _embedding_model is None:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _embedding_model
