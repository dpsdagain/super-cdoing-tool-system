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

