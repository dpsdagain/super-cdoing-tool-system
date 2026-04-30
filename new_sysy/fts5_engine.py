"""
fts5_engine.py — On-disk Full Text Search engine using SQLite FTS5.
Replaces the RAM-heavy rank_bm25 with incremental, disk-based BM25 indexing.
"""

from __future__ import annotations
import hashlib
import logging
import os
import threading
import sqlite3
import json
from langchain_core.documents import Document
from config import CHROMA_DB_DIR

logger = logging.getLogger(__name__)

# Thread-local storage for SQLite connections (one per thread)
_sqlite_connections = threading.local()

# Global lock for FTS5 write serialization
_FTS5_GLOBAL_LOCK = threading.Lock()


def _content_hash(doc: Document) -> str:
    """Return a SHA-256 hex digest of a document's page_content only."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


class SQLiteFTS5BM25:
    """
    On-disk Full Text Search engine replacing RAM-heavy rank_bm25.
    Uses SQLite's FTS5 extension which is built into standard Python.
    Reuses a thread-local connection to avoid open/close overhead and handle concurrency.
    """

    def __init__(self, collection_name: str):
        self.db_path = os.path.join(CHROMA_DB_DIR, f"{collection_name}_fts5.db")
        self._lock = _FTS5_GLOBAL_LOCK
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local reusable connection."""
        if not hasattr(_sqlite_connections, "conn") or _sqlite_connections.conn is None:
            # Use high timeout and write-ahead logging (WAL) for better concurrency
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _sqlite_connections.conn = conn
        return _sqlite_connections.conn

    def _init_db(self):
        conn = self._get_conn()
        # Enable FTS5 and create table with indexed symbols (calls + constants)
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(content, source_name, metadata_json UNINDEXED, calls, constants)"
            )
        except sqlite3.OperationalError:
            conn.execute("DROP TABLE IF EXISTS docs_fts")
            conn.execute(
                "CREATE VIRTUAL TABLE docs_fts USING fts5(content, source_name, metadata_json UNINDEXED, calls, constants)"
            )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hashes (content_hash TEXT PRIMARY KEY)"
        )
        try:
            conn.execute("ALTER TABLE hashes ADD COLUMN rowid_ref INTEGER")
        except sqlite3.OperationalError:
            pass
        conn.commit()

    def add_documents(self, documents: list[Document]):
        """Incremental on-disk indexing."""
        conn = self._get_conn()
        try:
            for doc in documents:
                h = doc.metadata.get("content_hash", _content_hash(doc))
                # Skip if already indexed
                res = conn.execute(
                    "SELECT 1 FROM hashes WHERE content_hash = ?", (h,)
                ).fetchone()
                if res:
                    continue

                meta_json = json.dumps(doc.metadata)
                source_name = os.path.basename(doc.metadata.get("source", "")).lower()
                calls = doc.metadata.get("calls_functions", "")
                constants = doc.metadata.get("references_constants", "")
                cursor = conn.execute(
                    "INSERT INTO docs_fts(content, source_name, metadata_json, calls, constants) VALUES (?, ?, ?, ?, ?)",
                    (doc.page_content, source_name, meta_json, calls, constants),
                )
                rowid = cursor.lastrowid
                conn.execute(
                    "INSERT INTO hashes(content_hash, rowid_ref) VALUES (?, ?)",
                    (h, rowid),
                )
            conn.commit()
        except (sqlite3.Error, json.JSONDecodeError, OSError) as e:
            logger.exception("FTS5: Failed to add documents")
            conn.rollback()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # We no longer close the connection here to keep it persistent for the thread
        pass

    def close(self):
        """Explicitly close the connection (thread-local cleanup handled manually if needed)."""
        if (
            hasattr(_sqlite_connections, "conn")
            and _sqlite_connections.conn is not None
        ):
            try:
                _sqlite_connections.conn.close()
                _sqlite_connections.conn = None
            except sqlite3.Error:
                pass

    def search(self, query: str, k: int = 10) -> list[Document]:
        """Fast keyword search via SQLite FTS5."""
        if not os.path.exists(self.db_path):
            return []

        # Clean query for FTS5 (strip special chars that break FTS5 grammar)
        # Preserve underscores and dots — critical for code identifiers
        clean_query = "".join(
            c if c.isalnum() or c.isspace() or c in "_.+#" else " " for c in query
        )
        clean_query = clean_query.strip()
        if not clean_query:
            return []

        with self._lock:
            conn = self._get_conn()
            try:
                # Use BM25 scoring via FTS5 'rank'
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
                    (clean_query, k),
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback for empty or invalid queries
                return []
        docs = []
        for content, meta_raw in rows:
            docs.append(Document(page_content=content, metadata=json.loads(meta_raw)))
        return docs

    def search_by_call(self, func_name: str, k: int = 10) -> list[Document]:
        """Directly query the 'calls' index for specific function call sites."""
        if not os.path.exists(self.db_path):
            return []
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE calls MATCH ? LIMIT ?",
                    (func_name, k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def search_by_calls_batch(self, terms: list[str], k: int = 10) -> list[Document]:
        """Batch query: find chunks calling ANY of the given function names (FTS5 OR)."""
        if not terms or not os.path.exists(self.db_path):
            return []
        or_query = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE calls MATCH ? LIMIT ?",
                    (or_query, k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def search_by_constant(self, const_name: str, k: int = 10) -> list[Document]:
        """Directly query the 'constants' index for chunks referencing a specific constant."""
        if not os.path.exists(self.db_path):
            return []
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE constants MATCH ? LIMIT ?",
                    (const_name, k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def search_by_constants_batch(
        self, terms: list[str], k: int = 10
    ) -> list[Document]:
        """Batch query: find chunks referencing ANY of the given constants (FTS5 OR)."""
        if not terms or not os.path.exists(self.db_path):
            return []
        or_query = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE constants MATCH ? LIMIT ?",
                    (or_query, k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def delete_by_hashes(self, hashes: set[str]):
        """Remove stale entries from both FTS5 and the hash dedup table.

        FTS5 virtual tables don't support DELETE with arbitrary WHERE clauses
        on content columns.  We use the ``metadata_json`` (which stores the
        content_hash) to locate matching rowids, then delete by rowid.
        """
        if not hashes or not os.path.exists(self.db_path):
            return
        with self._lock:
            conn = self._get_conn()
            # Identify rowids using the optimized hashes mapping table
            hash_list = list(hashes)
            stale_rowids = []
            for i in range(0, len(hash_list), 900):
                batch = hash_list[i : i + 900]
                placeholders = ",".join("?" * len(batch))
                cursor = conn.execute(
                    f"SELECT rowid_ref FROM hashes WHERE content_hash IN ({placeholders})",
                    batch,
                )
                stale_rowids.extend(
                    [r[0] for r in cursor.fetchall() if r[0] is not None]
                )
            if stale_rowids:
                for i in range(0, len(stale_rowids), 900):
                    batch = stale_rowids[i : i + 900]
                    placeholders = ",".join("?" * len(batch))
                    conn.execute(
                        f"DELETE FROM docs_fts WHERE rowid IN ({placeholders})", batch
                    )
                hash_list = list(hashes)
                for i in range(0, len(hash_list), 900):
                    batch = hash_list[i : i + 900]
                    hph = ",".join("?" * len(batch))
                    conn.execute(
                        f"DELETE FROM hashes WHERE content_hash IN ({hph})", batch
                    )
                conn.commit()
                logger.info("FTS5: Deleted %d stale entries.", len(stale_rowids))
