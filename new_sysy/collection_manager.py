from __future__ import annotations
import fnmatch
import hashlib
import logging
import os
import tempfile
import pickle
import threading
import sqlite3
import json
import ast as _ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO, Callable, Any
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tree_sitter_languages
from tree_sitter import Node
from config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CODE_CHUNK_SIZE,
    PDF_CHUNK_SIZE,
    CHROMA_DB_DIR,
    CODE_EXTENSIONS,
    EXCLUDED_FILE_PATTERNS,
    ZERO_CHUNK_THRESHOLD,
)

import logging
logger = logging.getLogger(__name__)

def load_existing_chroma(collection_name: str = "default") -> Chroma | None:
    """
    Load a previously-persisted ChromaDB collection.
    Returns None if the collection doesn't exist on disk or is empty.
    """
    if not os.path.isdir(CHROMA_DB_DIR):
        return None

    # 🚀 Platinum Safety: Verify the collection exists in the client's registry 
    # before attempting to instantiate the LangChain wrapper.
    try:
        available = list_collections()
        if collection_name not in available:
            logger.warning("Collection '%s' not found in registry.", collection_name)
            return None
    except Exception:
        # If client initialization fails, we can't load anything
        return None

    embedding = get_embedding_model()
    try:
        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding,
            collection_name=collection_name,
        )
        # Verify non-empty
        count = db._collection.count()
        if count == 0:
            return None
        return db
    except Exception as e:
        logger.error("Failed to load existing collection '%s': %s", collection_name, e)
        return None

def delete_collection(collection_name: str) -> bool:
    """Delete a ChromaDB collection and its associated BM25 index. Returns True if deleted."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return False
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        return False
    # 🚀 Platinum Fix: Also delete the SQLite FTS5 index
    fts_path = os.path.join(CHROMA_DB_DIR, f"{collection_name}_fts5.db")
    if os.path.exists(fts_path):
        try:
            os.remove(fts_path)
        except Exception as e:
            logger.warning("Could not delete FTS5 index for '%s': %s", collection_name, e)

    with _bm25_lock:
        _bm25_cache.pop(collection_name, None)
    invalidate_collection_info_cache(collection_name)
    return True

