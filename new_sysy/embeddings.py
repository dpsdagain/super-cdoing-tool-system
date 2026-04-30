"""
embeddings.py — Singleton Embedding Model Provider.
Provides the shared HuggingFaceEmbeddings instance used by all RAG components.
"""

import threading
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_embed_lock = threading.Lock()


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the singleton embedding model, downloading on first call (thread-safe)."""
    if not hasattr(get_embedding_model, "instance"):
        with _embed_lock:
            if not hasattr(get_embedding_model, "instance"):
                get_embedding_model.instance = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
    return get_embedding_model.instance
