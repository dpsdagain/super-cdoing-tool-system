"""
file_tools.py — File I/O, Search, and Edit Tools.
Provides code_search, file_read, file_edit, file_write, grep, glob, brief, and symbol_search.
"""

# pylint: disable=too-many-locals,too-many-nested-blocks

import os
import logging
import re
import fnmatch
import glob
from typing import List, Optional
from pydantic import BaseModel, Field
from config import WORKSPACE_ROOT, RETRIEVER_K, RERANK_TOP_K, USE_RERANKER
from tool_registry import register_tool, validate_path

logger = logging.getLogger(__name__)


def _validate_path(path: str) -> str:
    return validate_path(path)


# ═══════════════════════════════════════════════════════════════════════════
#  Pydantic Input Schemas
# ═══════════════════════════════════════════════════════════════════════════


class CodeSearchInput(BaseModel):
    query: str = Field(
        description="The natural language query or keywords to search for in the codebase."
    )
    collection_name: str = Field(
        default="default", description="The name of the collection to search within."
    )
    k: int = Field(
        default=RETRIEVER_K, description="Number of initial documents to retrieve."
    )


class FileReadInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read.")
    start_line: Optional[int] = Field(
        default=None, description="The 1-based line number to start reading from."
    )
    end_line: Optional[int] = Field(
        default=None, description="The 1-based line number to end reading at."
    )


class FileEditInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to edit.")
    old_string: str = Field(description="The exact literal text to replace.")
    new_string: str = Field(description="The text to replace old_string with.")


class FileWriteInput(BaseModel):
    file_path: str = Field(
        description="The absolute path to the file to create or overwrite."
    )
    content: str = Field(description="The full content to write to the file.")


class GlobInput(BaseModel):
    pattern: str = Field(
        description='The glob pattern to search for (e.g., "**/*.py").'
    )


class SymbolSearchInput(BaseModel):
    symbol: str = Field(
        description="The name of the class, function, or variable to find the definition of."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Tool Functions (lazy imports to avoid circular deps at module load)
# ═══════════════════════════════════════════════════════════════════════════


@register_tool(
    name="code_search",
    description="Search the codebase for relevant functions, classes, or logic using keywords or natural language.",
    input_schema=CodeSearchInput,
    is_read_only=True,
)
def code_search(
    query: str, collection_name: str = "default", k: int = RETRIEVER_K
) -> str:
    """Search the codebase using hybrid search (Vector + BM25)."""
    from backend import load_existing_chroma
    from search_engine import hybrid_search, get_reranker

    db = load_existing_chroma(collection_name)
    if not db:
        return f"Error: Collection '{collection_name}' not found or is empty."
    docs = hybrid_search(db, query, collection_name=collection_name, k=k)
    if USE_RERANKER:
        reranker = get_reranker()
        if reranker:
            docs = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    if not docs:
        return "No relevant code snippets found."
    formatted_results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        formatted_results.append(
            f"--- Result {i + 1} ({source}) ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted_results)


@register_tool(
    name="file_read",
    description="Read the content of a file. Supports line ranges for large files.",
    input_schema=FileReadInput,
    is_read_only=True,
)
def file_read(
    file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
) -> str:
    """Read a file's content, optionally within a line range."""
    try:
        file_path = _validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if start_line is not None or end_line is not None:
            start = start_line - 1 if start_line else 0
            end = end_line if end_line else len(lines)
            content = "".join(lines[start:end])
            return (
                f"--- Content of {file_path} (Lines {start + 1}-{end}) ---\n{content}"
            )
        else:
            content = "".join(lines)
            return f"--- Content of {file_path} ---\n{content}"
    except (OSError, IOError, UnicodeError) as e:
        return f"Error reading file: {str(e)}"


@register_tool(
    name="file_edit",
    description="Edit a file by replacing an exact string with a new string. This is safer than overwriting the whole file.",
    input_schema=FileEditInput,
    is_read_only=False,
)
def file_edit(file_path: str, old_string: str, new_string: str) -> str:
    """Surgically replace old_string with new_string in a file (F-05 Parity)."""
    from edit_utils import FuzzyMatcher

    try:
        file_path = _validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        actual_old = FuzzyMatcher.find_actual_string(content, old_string)
        if not actual_old:
            return f"Error: Could not find match for 'old_string' in {file_path}. Fuzzy matching also failed."
        occurrences = content.count(actual_old)
        if occurrences > 1:
            return f"Error: Found {occurrences} fuzzy occurrences. Please provide more context."
        new_content = content.replace(actual_old, new_string)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return (
            f"Successfully edited {file_path}. Fuzzy match succeeded on detected block."
        )
    except (OSError, IOError, UnicodeError) as e:
        return f"Error editing file: {str(e)}"


@register_tool(
    name="file_write",
    description="Create a new file or completely overwrite an existing file with new content.",
    input_schema=FileWriteInput,
    is_read_only=False,
)
def file_write(file_path: str, content: str) -> str:
    """Create or overwrite a file with the provided content."""
    try:
        file_path = _validate_path(file_path)
    except PermissionError as e:
        return str(e)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}."
    except (OSError, IOError) as e:
        return f"Error writing file: {str(e)}"


@register_tool(
    name="glob",
    description="Search for files using glob patterns. Returns a list of relative paths.",
    input_schema=GlobInput,
    is_read_only=True,
)
def glob_tool(pattern: str) -> str:
    """Find files matching a glob pattern."""
    try:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return f"No files found matching pattern: {pattern}"
        safe_matches = []
        for m in matches:
            try:
                safe_matches.append(_validate_path(m))
            except PermissionError:
                continue
        if not safe_matches:
            return "No files found within the allowed workspace."
        return "Found files:\n" + "\n".join(safe_matches)
    except OSError as e:
        return f"Error executing glob: {str(e)}"


@register_tool(
    name="symbol_search",
    description="Search for a specific code symbol (class or function) definition across the codebase.",
    input_schema=SymbolSearchInput,
    is_read_only=True,
)
def symbol_search(symbol: str) -> str:
    """Find the definition of a class or function across the codebase (Python-native)."""
    import re
    import fnmatch

    patterns = [
        f"def {symbol}\\b",
        f"class {symbol}\\b",
        f"function {symbol}\\b",
        f"const {symbol}\\s*=",
        f"let {symbol}\\s*=",
        f"var {symbol}\\s*=",
    ]
    compiled_patterns = [re.compile(p) for p in patterns]
    matches = []
    root_dir = str(WORKSPACE_ROOT)
    for root, _, files in os.walk(root_dir):
        if any(
            (
                fnmatch.fnmatch(root, f"*{exc}*")
                for exc in ["__pycache__", "venv", ".git", "chroma_db"]
            )
        ):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if any((p.search(line) for p in compiled_patterns)):
                            matches.append(f"{rel_path}:{i}:{line.strip()}")
            except (OSError, IOError, UnicodeError):
                continue
    if not matches:
        return f"Definition for '{symbol}' not found. Try a regular grep_search."
    return "Potential Definitions:\n\n" + "\n".join(matches[:100])
