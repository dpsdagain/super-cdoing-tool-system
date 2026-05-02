"""
file_tools.py — File I/O, Search, and Edit Tools.
Provides code_search, file_read, file_edit, file_write, grep, glob, brief, and symbol_search.
"""

# pylint: disable=too-many-locals,too-many-nested-blocks

import os
import logging
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices
from config import WORKSPACE_ROOT, RETRIEVER_K, RERANK_TOP_K, USE_RERANKER
from tool_registry import register_tool, validate_path

logger = logging.getLogger(__name__)

# Directories that are never useful to walk for source-code search.
# Centralised so glob/symbol_search/future tools share one definition.
EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        "venv",
        ".venv",
        ".git",
        "chroma_db",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
    }
)


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
    file_path: str = Field(validation_alias=AliasChoices('file_path', 'path'), description="The absolute path to the file to read.")
    start_line: Optional[int] = Field(
        default=None, description="The 1-based line number to start reading from."
    )
    end_line: Optional[int] = Field(
        default=None, description="The 1-based line number to end reading at."
    )


class FileEditInput(BaseModel):
    file_path: str = Field(validation_alias=AliasChoices('file_path', 'path'), description="The absolute path to the file to edit.")
    old_string: str = Field(description="The exact literal text to replace.")
    new_string: str = Field(description="The text to replace old_string with.")


class FileWriteInput(BaseModel):
    file_path: str = Field(
        validation_alias=AliasChoices('file_path', 'path'),
        description="The absolute path to the file to create or overwrite."
    )
    content: str = Field(description="The full content to write to the file.")


class GlobInput(BaseModel):
    pattern: str = Field(
        description='The glob pattern to search for (e.g., "**/*.py").'
    )


class ListDirectoryInput(BaseModel):
    directory: str = Field(
        default=".", description="The relative or absolute path to the directory to list."
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
    """Read a file's content, optionally within a 1-based, inclusive line range."""
    # Fail Fast: validate range arguments before doing any I/O.
    if start_line is not None and start_line < 1:
        return "Error: start_line must be >= 1."
    if end_line is not None and end_line < 1:
        return "Error: end_line must be >= 1."
    if (
        start_line is not None
        and end_line is not None
        and end_line < start_line
    ):
        return "Error: end_line must be >= start_line."

    try:
        file_path = _validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if start_line is None and end_line is None:
            content = "".join(lines)
            return f"--- Content of {file_path} ---\n{content}"

        # Convert from 1-based inclusive to Python slice indices, clamped to file length.
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else len(lines)
        end_idx = min(end_idx, len(lines))
        content = "".join(lines[start_idx:end_idx])
        return (
            f"--- Content of {file_path} (Lines {start_idx + 1}-{end_idx}) ---\n{content}"
        )
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

    # Fail Fast: a no-op edit means the agent is confused about its own state.
    # Reporting "success" would let it loop indefinitely thinking it changed something.
    if old_string == new_string:
        return (
            "Error: old_string and new_string are identical — nothing to change. "
            "This usually means the agent has already applied this edit or is "
            "looking at stale content."
        )

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
        if new_content == content:
            # Defensive: fuzzy match returned a string that, after substitution,
            # produced no change. Treat as a hard failure so the agent re-plans.
            return (
                f"Error: Replacement in {file_path} produced no change. "
                "Re-read the file and provide a fresh old_string."
            )
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}."
    except (OSError, IOError) as e:
        return f"Error writing file: {str(e)}"


@register_tool(
    name="glob",
    description="Search for files using glob patterns relative to the workspace root (e.g. '**/*.py'). Absolute patterns are rejected.",
    input_schema=GlobInput,
    is_read_only=True,
)
def glob_tool(pattern: str) -> str:
    """Find files matching a glob pattern, anchored under WORKSPACE_ROOT.

    Fail Fast: absolute patterns are rejected up-front rather than being walked
    and silently filtered later, so the tool never enumerates paths outside
    the workspace.
    """
    if not pattern:
        return "Error: glob pattern must be non-empty."
    if Path(pattern).is_absolute() or pattern.startswith(("/", "\\")):
        return (
            "Error: glob pattern must be relative to the workspace root. "
            "Got an absolute path."
        )

    root = Path(WORKSPACE_ROOT).resolve()
    try:
        candidates = list(root.glob(pattern))
    except (OSError, ValueError) as e:
        return f"Error executing glob: {str(e)}"

    safe_matches = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            safe_matches.append(_validate_path(str(candidate)))
        except PermissionError:
            continue

    if not safe_matches:
        return f"No files found matching pattern: {pattern}"
    return "Found files:\n" + "\n".join(safe_matches)


@register_tool(
    name="list_directory",
    description="List the contents of a directory. Returns a list of files and subdirectories.",
    input_schema=ListDirectoryInput,
    is_read_only=True,
)
def list_directory(directory: str = ".") -> str:
    """List the contents of a directory."""
    try:
        # Resolve and validate path
        abs_path = _validate_path(directory)
    except PermissionError as e:
        return str(e)

    if not os.path.isdir(abs_path):
        return f"Error: '{directory}' is not a directory or does not exist."

    try:
        entries = os.listdir(abs_path)
        if not entries:
            return f"Directory '{directory}' is empty."

        result = []
        for entry in sorted(entries):
            entry_path = os.path.join(abs_path, entry)
            if os.path.isdir(entry_path):
                result.append(f"[DIR] {entry}")
            else:
                size = os.path.getsize(entry_path)
                result.append(f"{entry} ({size} bytes)")

        return f"Directory listing for {directory}:\n" + "\n".join(result)
    except OSError as e:
        return f"Error listing directory: {str(e)}"


SYMBOL_SEARCH_RESULT_LIMIT = 100


@register_tool(
    name="symbol_search",
    description="Search for a specific code symbol (class or function) definition across the codebase.",
    input_schema=SymbolSearchInput,
    is_read_only=True,
)
def symbol_search(symbol: str) -> str:
    """Find the definition of a class or function across the codebase.

    Prunes well-known build/cache directories (see EXCLUDED_DIRS) so os.walk
    does not descend into them — without pruning, the previous implementation
    walked the entire tree and only filtered after the fact.
    """
    if not symbol:
        return "Error: symbol must be non-empty."

    escaped = re.escape(symbol)
    patterns = [
        rf"\bdef {escaped}\b",
        rf"\bclass {escaped}\b",
        rf"\bfunction {escaped}\b",
        rf"\bconst {escaped}\s*=",
        rf"\blet {escaped}\s*=",
        rf"\bvar {escaped}\s*=",
    ]
    compiled_patterns = [re.compile(p) for p in patterns]

    matches: List[str] = []
    root_dir = str(WORKSPACE_ROOT)
    for current_root, dirs, files in os.walk(root_dir):
        # Prune in-place so os.walk stops descending into excluded dirs.
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            file_path = os.path.join(current_root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if any(p.search(line) for p in compiled_patterns):
                            matches.append(f"{rel_path}:{line_no}:{line.strip()}")
                            if len(matches) >= SYMBOL_SEARCH_RESULT_LIMIT:
                                return _format_symbol_matches(matches)
            except (OSError, IOError, UnicodeError):
                continue

    if not matches:
        return f"Definition for '{symbol}' not found. Try a regular grep_search."
    return _format_symbol_matches(matches)


def _format_symbol_matches(matches: List[str]) -> str:
    return "Potential Definitions:\n\n" + "\n".join(matches)
