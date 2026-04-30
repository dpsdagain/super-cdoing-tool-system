"""
backend.py - Knowledge Base Ingestion & Retrieval Backend Facade
This module has been decomposed into specialized components.
Please see embeddings.py, chunkers.py, fts5_engine.py, collection_manager.py, and ingestion.py.
"""

from embeddings import get_embedding_model
from fts5_engine import SQLiteFTS5BM25
from collection_manager import load_existing_chroma
from chunkers import get_text_splitter

__all__ = [
    "get_embedding_model",
    "SQLiteFTS5BM25",
    "load_existing_chroma",
    "get_text_splitter",
]
