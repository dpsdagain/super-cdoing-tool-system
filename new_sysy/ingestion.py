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

from embeddings import get_embedding_model
from chunkers import ASTChunker, RegexHDLChunker, get_text_splitter
from fts5_engine import SQLiteFTS5BM25
from collection_manager import load_existing_chroma, invalidate_collection_info_cache

"""
backend.py — Data Ingestion Engine.

Handles:
  • PDF loading and chunking
  • Codebase / directory loading and chunking (with exclusions)
  • Embedding via local HuggingFace BGE model
  • Persistent storage in ChromaDB
"""

def _is_excluded(filepath: str) -> bool:
    """Return True if the file matches any exclusion pattern."""
    name = os.path.basename(filepath)
    full = filepath.replace("\\", "/")
    for pattern in EXCLUDED_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full, pattern):
            return True
    return False

def load_and_chunk_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF from *file_path* and split it into overlapping text chunks.

    This helper uses ``PyPDFLoader`` from LangChain to read the PDF and then
    applies the library‑wide text splitter (with the PDF‑specific chunk size).

    The function also implements *zero‑chunk* handling: if the combined text
    length of the PDF is below :data:`ZERO_CHUNK_THRESHOLD`, a single
    ``Document`` containing the whole content is returned and flagged with
    ``zero_chunk=True``.  Otherwise, the text is split into overlapping chunks
    and each chunk is enriched with ``chunk_index`` and a stable
    ``content_hash`` for deterministic caching.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A list of :class:`langchain.docstore.document.Document` objects, each
        representing a chunk of the PDF (or a single merged document when
        zero‑chunking is applied).
    """
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    
    # ── Zero Chunking (Phase 3 Upgrade) ──────────────────────────────
    total_content = "\n".join([d.page_content for d in raw_docs])
    if len(total_content) < ZERO_CHUNK_THRESHOLD:
        # Create a single merged document
        merged_doc = Document(
            page_content=total_content,
            metadata={
                "source": file_path,
                "zero_chunk": True,
                "chunk_index": 0,
                "content_hash": hashlib.sha256(total_content.encode("utf-8")).hexdigest()
            }
        )
        return [merged_doc]

    splitter = _get_splitter(chunk_size_override=PDF_CHUNK_SIZE)
    chunks = splitter.split_documents(raw_docs)
    
    # Enrich metadata for cache-stable sorting
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["content_hash"] = _content_hash(chunk)
    return chunks

def load_and_chunk_pdf_upload(uploaded_file: BinaryIO, filename: str) -> list[Document]:
    """
    Accept a Streamlit UploadedFile, write it to a temp file,
    ingest it, then clean up.
    """
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        return load_and_chunk_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

def _collect_code_files(directory: str) -> list[str]:
    """
    Walk *directory* and return absolute paths of code files
    whose extension is in CODE_EXTENSIONS and that are NOT excluded.
    Also picks up extensionless files named 'Dockerfile'.
    """
    paths: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Prune heavy directories early
        dirs[:] = [
            d for d in dirs
            if d not in {"node_modules", "venv", ".venv", "__pycache__",
                         ".git", "chroma_db", ".tox", ".mypy_cache"}
        ]
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()

            # Extensionless special files
            if fname in ("Dockerfile", "Makefile", "Jenkinsfile", ".dockerignore"):
                if not _is_excluded(fpath):
                    paths.append(fpath)
                continue

            if ext in CODE_EXTENSIONS and not _is_excluded(fpath):
                paths.append(fpath)
    return paths

def load_and_chunk_codebase(
    directory: str,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[Document]:
    """
    Recursively load a code directory, applying language-aware
    chunking per file and attaching source metadata.

    *on_progress(current, total, filename)* is called after each file.
    """
    all_chunks: list[Document] = []
    file_paths = _collect_code_files(directory)
    total = len(file_paths)

    for idx, fpath in enumerate(file_paths):
        if on_progress:
            on_progress(idx + 1, total, os.path.basename(fpath))
        ext = Path(fpath).suffix.lower()
        try:
            # 🚀 Reverting to stable autodetect now that 'chardet' is installed in the venv
            loader = TextLoader(fpath, autodetect_encoding=True)
            raw_docs = loader.load()
            if not raw_docs:
                # Fallback to UTF-8 if autodetect fails to find content
                loader = TextLoader(fpath, encoding="utf-8")
                raw_docs = loader.load()
            if not raw_docs:
                continue
            content = raw_docs[0].page_content
        except Exception:
            try:
                # Secondary Fallback: Force UTF-8 if autodetect crashes on certain characters
                loader = TextLoader(fpath, encoding="utf-8")
                raw_docs = loader.load()
                if not raw_docs:
                    continue
                content = raw_docs[0].page_content
            except Exception as exc:
                logger.warning("Skipping %s: %s", fpath, exc)
                continue

        # ── Zero Chunking (Phase 3 Upgrade) ──────────────────────────────
        if len(content) < ZERO_CHUNK_THRESHOLD:
            # Extract call-graph and constant-reference metadata even for
            # zero-chunks so FTS5 search_by_calls_batch / search_by_constants_batch
            # can find them.  Without this, config.py (the most common zero-chunk)
            # is invisible to Fix E and Fix G.
            _zc_chunker = CodeASTChunker()
            _zc_calls = " ".join(_zc_chunker._extract_called_functions(content))
            _zc_consts = " ".join(_zc_chunker._extract_referenced_constants(content))
            chunk = Document(
                page_content=content,
                metadata={
                    "source": fpath,
                    "source_type": "code",
                    "file_extension": ext,
                    "zero_chunk": True,
                    "chunk_index": 0,
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "calls_functions": _zc_calls,
                    "references_constants": _zc_consts,
                }
            )
            all_chunks.append(chunk)
            continue

        # ── AST-Aware Chunking (New Upgrade) ──────────────────────────────
        ast_chunker = CodeASTChunker(chunk_size=CODE_CHUNK_SIZE)
        ast_chunks = ast_chunker.chunk_file(content, fpath, ext)
        
        if ast_chunks:
            # Enrich metadata and add to total
            for i, chunk in enumerate(ast_chunks):
                chunk.metadata["source_type"] = "code"
                chunk.metadata["file_extension"] = ext
                chunk.metadata["chunk_index"] = i
                chunk.metadata["content_hash"] = _content_hash(chunk)
            all_chunks.extend(ast_chunks)
            continue

        # Fallback to legacy splitter if AST returns nothing
        splitter = _get_splitter(ext, chunk_size_override=CODE_CHUNK_SIZE)
        chunks = splitter.split_documents(raw_docs)

        # Enrich metadata for citation and cache-stable sorting
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_type"] = "code"
            chunk.metadata["file_extension"] = ext
            chunk.metadata["chunk_index"] = i
            chunk.metadata["content_hash"] = _content_hash(chunk)
        all_chunks.extend(chunks)

    return all_chunks

def _content_hash(doc: Document) -> str:
    """Return a SHA-256 hex digest of a document's page_content only."""
    content_str = doc.page_content
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

def ingest_into_chroma(
    documents: list[Document],
    collection_name: str = "default",
) -> tuple[Chroma, int]:
    """
    Embed *documents* and upsert them into a persistent ChromaDB collection,
    skipping duplicates based on content hash.

    Returns (Chroma instance, number of new documents added).
    """
    if not documents:
        raise ValueError("No documents to ingest — check your file/folder path.")

    # Stamp content_hash only on documents that don't already have one.
    # load_and_chunk_codebase and load_and_chunk_pdf set this during chunking,
    # so recomputing it here is redundant for the normal ingestion path.
    for doc in documents:
        if "content_hash" not in doc.metadata:
            doc.metadata["content_hash"] = _content_hash(doc)

    embedding = get_embedding_model()

    # Try loading existing collection to deduplicate
    existing_db = load_existing_chroma(collection_name)
    if existing_db is not None:
        # Delete stale chunks from modified files before adding new ones.
        # Without this, re-ingesting a changed file leaves BOTH old and new
        # versions in the DB, feeding contradictory context to the LLM.
        #
        # Two-phase metadata scan (bounded RAM, no unfiltered full-dump):
        #   Phase A — targeted server-side filter for stale rows only:
        #             where={"source": {"$in": [...]}}. Gives us both
        #             stale_ids and stale_hashes in one call. Previously
        #             this did a full collection get() then python-side
        #             filtering — OOM risk on large corpora.
        #   Phase B — paged scan to build existing_hashes for dedup.
        #             We still must see every hash, but we page in chunks
        #             of PAGE so peak RAM is bounded regardless of
        #             collection size.
        incoming_sources = {d.metadata.get("source") for d in documents if d.metadata.get("source")}

        # Phase A: fetch only stale rows via server-side $in filter.
        stale_ids: list[str] = []
        stale_hashes: set[str] = set()
        if incoming_sources:
            src_list = list(incoming_sources)
            try:
                # Chroma supports {"$in": [...]} directly; a single-key
                # dict also works for a single source.
                if len(src_list) == 1:
                    where = {"source": src_list[0]}
                else:
                    where = {"source": {"$in": src_list}}
                stale_data = existing_db.get(where=where, include=["metadatas"])
                stale_ids = list(stale_data.get("ids", []))
                for meta in stale_data.get("metadatas", []) or []:
                    if meta and "content_hash" in meta:
                        stale_hashes.add(meta["content_hash"])
            except Exception as e:
                # Filter variants differ between Chroma versions; fall back
                # to a paged scan rather than crash.
                logger.warning("Chroma where-filter failed (%s); falling back to paged scan.", e)
                stale_ids = []
                stale_hashes = set()
                PAGE = 10000
                offset = 0
                while True:
                    batch = existing_db.get(limit=PAGE, offset=offset, include=["metadatas"])
                    ids = batch.get("ids", []) or []
                    metas = batch.get("metadatas", []) or []
                    if not ids:
                        break
                    for did, meta in zip(ids, metas):
                        if meta and meta.get("source") in incoming_sources:
                            stale_ids.append(did)
                            if "content_hash" in meta:
                                stale_hashes.add(meta["content_hash"])
                    if len(ids) < PAGE:
                        break
                    offset += PAGE

        if stale_ids:
            # Use the public delete() wrapper (avoids reaching into
            # ._collection, which is version-fragile).
            try:
                existing_db.delete(ids=stale_ids)
            except AttributeError:
                existing_db._collection.delete(ids=stale_ids)
            # Purge matching FTS5 entries so BM25 doesn't return stale content
            if stale_hashes:
                fts = SQLiteFTS5BM25(collection_name)
                fts.delete_by_hashes(stale_hashes)
            logger.info("🗑️ Ingestion: Deleted %d stale chunks from %d re-ingested files.", len(stale_ids), len(incoming_sources))

        # Phase B: paged scan to collect existing (non-stale) hashes for dedup.
        # Bounded peak RAM by PAGE rows regardless of collection size.
        existing_hashes: set[str] = set()
        stale_id_set = set(stale_ids)
        PAGE = 10000
        offset = 0
        while True:
            batch = existing_db.get(limit=PAGE, offset=offset, include=["metadatas"])
            ids = batch.get("ids", []) or []
            metas = batch.get("metadatas", []) or []
            if not ids:
                break
            for did, meta in zip(ids, metas):
                if not meta:
                    continue
                if did in stale_id_set:
                    continue  # about-to-delete; don't block new versions
                h = meta.get("content_hash")
                if h:
                    existing_hashes.add(h)
            if len(ids) < PAGE:
                break
            offset += PAGE

        new_docs = [d for d in documents if d.metadata["content_hash"] not in existing_hashes]
        if not new_docs:
            logger.info("Ingestion: All %d chunks are already in the database. 100%% De-duplicated.", len(documents))
            return existing_db, 0

        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            logger.info("Ingestion: Adding %d new chunks. (Skipped %d duplicates)", len(new_docs), skipped)

        existing_db.add_documents(new_docs)
        # Keep BM25 in sync with ChromaDB — must update here too,
        # not just on the fresh-collection path below.
        _update_bm25_index(new_docs, collection_name)
        invalidate_collection_info_cache(collection_name)
        return existing_db, len(new_docs)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name,
    )

    # Advanced Upgrade: Build Hybrid BM25 Index
    _update_bm25_index(documents, collection_name)
    invalidate_collection_info_cache(collection_name)

    return db, len(documents)

def _update_bm25_index(new_docs: list[Document], collection_name: str):
    """
    Transitioned to SQLite FTS5 for incremental, disk-based indexing.
    """
    fts = SQLiteFTS5BM25(collection_name)
    fts.add_documents(new_docs)

class AsyncIngestionTask:
    """
    Run ingestion (loading, chunking, and embedding) in a background thread
    so the Streamlit UI stays responsive.
    """

    def __init__(self, target_path: str, collection_name: str = "default", is_pdf: bool = False):
        self.target_path = target_path
        self.collection_name = collection_name
        self.is_pdf = is_pdf
        self.progress: float = 0.0          # 0.0 to 1.0
        self.status: str = "pending"        # pending | running | done | error
        self.current_step: str = ""         # "Collecting files...", "Embedding chunks..."
        self.result: tuple | None = None    # (Chroma, added_count) on success
        self.error: str = ""
        self._thread: threading.Thread | None = None

    def start(self):
        """Launch the ingestion process in a background thread."""
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            # 1. Loading Phase
            self.current_step = f"Loading {'PDF' if self.is_pdf else 'codebase'}..."
            self.progress = 0.1
            
            if self.is_pdf:
                # Route through load_and_chunk_pdf so zero-chunking,
                # content_hash, and chunk_index metadata are all set
                # consistently — same as the sync ingestion path.
                chunks = load_and_chunk_pdf(self.target_path)
            else:
                # Codebase ingestion with file-by-file progress
                def _update_progress(curr, tot, name):
                    self.current_step = f"Collecting codebase: {name}"
                    # Loading phase covers 0.1 to 0.4 progress
                    self.progress = 0.1 + (curr / tot) * 0.3

                chunks = load_and_chunk_codebase(self.target_path, on_progress=_update_progress)

            if not chunks:
                self.error = "No relevant content found to ingest."
                self.status = "error"
                return

            # 2. Ingestion Phase
            self.current_step = f"Embedding {len(chunks)} chunks into ChromaDB..."
            self.progress = 0.5
            
            # Note: ChromaDB ingestion is synchronous but embedding happens here
            db, added = ingest_into_chroma(chunks, self.collection_name)
            
            self.progress = 1.0
            self.current_step = f"Successfully ingested {added} chunks!"
            self.result = (db, added)
            self.status = "done"
        except Exception as e:
            self.error = f"Ingestion failed: {str(e)}"
            self.status = "error"

    @property
    def is_done(self) -> bool:
        return self.status in ("done", "error")

def summarize_document_for_pin(file_path: str, max_chars: int = 3000) -> str:
    """
    Context Summarization (Phase 2b): Create a compact summary of
    a large file for pinning instead of the full content.

    Uses extractive summarization (no LLM call):
      - For code: extracts function/class signatures + docstrings
      - For text: extracts first N characters with paragraph boundaries
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return ""

    if not content:
        return ""

    ext = Path(file_path).suffix.lower()

    # Code files: extract signatures
    if ext in (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"):
        return _extract_code_signatures(content, max_chars)

    # Text/PDF/markup: extract leading paragraphs
    paragraphs = content.split("\n\n")
    summary = ""
    for para in paragraphs:
        if len(summary) + len(para) > max_chars:
            break
        summary += para + "\n\n"
    return summary.strip() if summary else content[:max_chars]

def _extract_code_signatures(content: str, max_chars: int) -> str:
    """Extract function/class definitions and docstrings from code."""
    import re
    lines = content.split("\n")
    signatures = []
    total_len = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match common definition patterns
        if (stripped.startswith(("def ", "class ", "function ", "func ",
                                 "export ", "public ", "private ", "async def "))
                or re.match(r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", stripped)):
            # Include the signature line
            signatures.append(line)
            total_len += len(line)
            # Include docstring/comment on next line if present
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith(('"""', "'''", "//", "/*", "#", "*")):
                    signatures.append(lines[i + 1])
                    total_len += len(lines[i + 1])
            if total_len > max_chars:
                break

    if not signatures:
        return content[:max_chars]

    return "\n".join(signatures)

