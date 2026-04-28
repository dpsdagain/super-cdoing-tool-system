"""
collection_manager.py — ChromaDB Collection Lifecycle Manager.
Handles loading, deleting, and listing collections.
"""
import os
import logging
from langchain_chroma import Chroma
from config import CHROMA_DB_DIR
from embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def load_existing_chroma(collection_name: str = "default") -> Chroma | None:
    """
    Load a previously-persisted ChromaDB collection.
    Returns None if the collection doesn't exist on disk or is empty.
    """
    if not os.path.isdir(CHROMA_DB_DIR):
        return None

    try:
        available = list_collections()
        if collection_name not in available:
            logger.warning("Collection '%s' not found in registry.", collection_name)
            return None
    except Exception:
        return None

    embedding = get_embedding_model()
    try:
        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding,
            collection_name=collection_name,
        )
        count = db._collection.count()
        if count == 0:
            return None
        return db
    except Exception as e:
        logger.error("Failed to load existing collection '%s': %s", collection_name, e)
        return None


def _get_chroma_client():
    """Return a persistent ChromaDB client."""
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


def list_collections() -> list[str]:
    """List all collection names in the ChromaDB directory."""
    client = _get_chroma_client()
    return [c.name for c in client.list_collections()]


def delete_collection(collection_name: str) -> bool:
    """Delete a ChromaDB collection and its associated BM25 index. Returns True if deleted."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return False
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        return False
    # Also delete the SQLite FTS5 index
    fts_path = os.path.join(CHROMA_DB_DIR, f"{collection_name}_fts5.db")
    if os.path.exists(fts_path):
        try:
            os.remove(fts_path)
        except Exception as e:
            logger.warning("Could not delete FTS5 index for '%s': %s", collection_name, e)
    return True


def invalidate_collection_info_cache(collection_name: str) -> None:
    """Placeholder for cache invalidation (used by other modules)."""
    pass
