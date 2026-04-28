import os
import threading
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from tool_registry import tool, validate_path, current_engine
from config import WORKSPACE_ROOT
logger = logging.getLogger(__name__)



from config import RETRIEVER_K, RERANK_TOP_K, USE_RERANKER
from pydantic import BaseModel, Field
from backend import load_existing_chroma, SQLiteFTS5BM25
from rag_core import hybrid_search
from search_engine import get_reranker


class CodeSearchInput(BaseModel):
    query: str = Field(description='The natural language query or keywords to search for in the codebase.')
    collection_name: str = Field(default='default', description='The name of the collection to search within.')
    k: int = Field(default=RETRIEVER_K, description='Number of initial documents to retrieve.')

class FileReadInput(BaseModel):
    file_path: str = Field(description='The absolute path to the file to read.')
    start_line: Optional[int] = Field(default=None, description='The 1-based line number to start reading from.')
    end_line: Optional[int] = Field(default=None, description='The 1-based line number to end reading at.')

class FileEditInput(BaseModel):
    file_path: str = Field(description='The absolute path to the file to edit.')
    old_string: str = Field(description='The exact literal text to replace.')
    new_string: str = Field(description='The text to replace old_string with.')

class MultiFileEditInput(BaseModel):
    file_path: str = Field(description='The absolute path to the file to edit.')
    replacements: List[Replacement] = Field(description='A list of replacement pairs.')

class FileWriteInput(BaseModel):
    file_path: str = Field(description='The absolute path to the file to create or overwrite.')
    content: str = Field(description='The full content to write to the file.')

class SymbolSearchInput(BaseModel):
    symbol: str = Field(description='The name of the class, function, or variable to find the definition of.')

def grep_tool(pattern: str, include_pattern: Optional[str]=None, exclude_pattern: Optional[str]=None, case_sensitive: bool=False) -> str:
    """Search for a pattern across the codebase using Python-native regex for platform consistency."""
    import re
    import fnmatch
    flags = re.IGNORECASE if not case_sensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f'Invalid regex pattern: {str(e)}'
    matches = []
    root_dir = str(WORKSPACE_ROOT)
    for root, _, files in os.walk(root_dir):
        if any((fnmatch.fnmatch(root, f'*{exc}*') for exc in ['__pycache__', 'venv', '.git', 'chroma_db'])):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            if include_pattern and (not fnmatch.fnmatch(file, include_pattern)):
                continue
            if exclude_pattern and fnmatch.fnmatch(file, exclude_pattern):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f'{rel_path}:{i}:{line.strip()}')
            except Exception:
                continue
    if not matches:
        return f'No matches found for pattern: {pattern}'
    return '\n'.join(matches[:500])

def multi_file_edit(file_path: str, replacements: List[Replacement]) -> str:
    """Apply multiple surgical replacements to a single file in one go."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        original_content = content
        diffs = []
        for rep in replacements:
            old_str = rep.old_string
            new_str = rep.new_string
            if old_str not in content:
                return f"Error: Could not find exact match for '{old_str}' in {file_path}."
            occurrences = content.count(old_str)
            if occurrences > 1:
                return f"Error: Found {occurrences} occurrences of '{old_str}'. Please provide more context."
            content = content.replace(old_str, new_str)
            diffs.append(f'[DIFF_START]{old_str}[DIFF_DIVIDER]{new_str}[DIFF_END]')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f'Successfully applied {len(replacements)} edits to {file_path}.\n' + '\n'.join(diffs)
    except Exception as e:
        return f'Error in multi_file_edit: {str(e)}'

def code_search(query: str, collection_name: str='default', k: int=RETRIEVER_K) -> str:
    """Search the codebase using hybrid search (Vector + BM25)."""
    db = load_existing_chroma(collection_name)
    if not db:
        return f"Error: Collection '{collection_name}' not found or is empty."
    docs = hybrid_search(db, query, collection_name=collection_name, k=k)
    if USE_RERANKER:
        reranker = get_reranker()
        if reranker:
            docs = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    if not docs:
        return 'No relevant code snippets found.'
    formatted_results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        formatted_results.append(f'--- Result {i + 1} ({source}) ---\n{doc.page_content}')
    return '\n\n'.join(formatted_results)

def file_read(file_path: str, start_line: Optional[int]=None, end_line: Optional[int]=None) -> str:
    """Read a file's content, optionally within a line range."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if start_line is not None or end_line is not None:
            start = start_line - 1 if start_line else 0
            end = end_line if end_line else len(lines)
            content = ''.join(lines[start:end])
            return f'--- Content of {file_path} (Lines {start + 1}-{end}) ---\n{content}'
        else:
            content = ''.join(lines)
            return f'--- Content of {file_path} ---\n{content}'
    except Exception as e:
        return f'Error reading file: {str(e)}'

def file_edit(file_path: str, old_string: str, new_string: str) -> str:
    """Surgically replace old_string with new_string in a file (F-05 Parity)."""
    from edit_utils import FuzzyMatcher
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        actual_old = FuzzyMatcher.find_actual_string(content, old_string)
        if not actual_old:
            return f"Error: Could not find match for 'old_string' in {file_path}. Fuzzy matching also failed."
        occurrences = content.count(actual_old)
        if occurrences > 1:
            return f'Error: Found {occurrences} fuzzy occurrences. Please provide more context.'
        new_content = content.replace(actual_old, new_string)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f'Successfully edited {file_path}. Fuzzy match succeeded on detected block.'
    except Exception as e:
        return f'Error editing file: {str(e)}'

def file_write(file_path: str, content: str) -> str:
    """Create or overwrite a file with the provided content."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f'Successfully wrote to {file_path}.'
    except Exception as e:
        return f'Error writing file: {str(e)}'

def glob_tool(pattern: str) -> str:
    """Find files matching a glob pattern."""
    import glob
    try:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return f'No files found matching pattern: {pattern}'
        safe_matches = []
        for m in matches:
            try:
                safe_matches.append(validate_path(m))
            except PermissionError:
                continue
        if not safe_matches:
            return 'No files found within the allowed workspace.'
        return 'Found files:\n' + '\n'.join(safe_matches)
    except Exception as e:
        return f'Error executing glob: {str(e)}'

def brief_tool(file_path: str) -> str:
    """Provide a brief outline of a file to save context."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        if file_path.endswith('.py'):
            import ast
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            outline = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    outline.append(f'class {node.name}:')
                    for sub in node.body:
                        if isinstance(sub, ast.FunctionDef):
                            outline.append(f'    def {sub.name}(...)')
                elif isinstance(node, ast.FunctionDef):
                    outline.append(f'def {node.name}(...)')
            if outline:
                return f'Outline of {file_path}:\n' + '\n'.join(outline) + '\n\nUse file_read to see full implementation.'
            return f'File {file_path} contains no top-level classes or functions.'
        else:
            size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                head = ''.join([next(f) for _ in range(20)])
            return f'File {file_path} ({size} bytes). First 20 lines:\n{head}\n... Use file_read to see more.'
    except Exception as e:
        return f'Error generating brief: {str(e)}'

def symbol_search(symbol: str) -> str:
    """Find the definition of a class or function across the codebase (Python-native)."""
    import re
    import fnmatch
    patterns = [f'def {symbol}\\b', f'class {symbol}\\b', f'function {symbol}\\b', f'const {symbol}\\s*=', f'let {symbol}\\s*=', f'var {symbol}\\s*=']
    compiled_patterns = [re.compile(p) for p in patterns]
    matches = []
    root_dir = str(WORKSPACE_ROOT)
    for root, _, files in os.walk(root_dir):
        if any((fnmatch.fnmatch(root, f'*{exc}*') for exc in ['__pycache__', 'venv', '.git', 'chroma_db'])):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if any((p.search(line) for p in compiled_patterns)):
                            matches.append(f'{rel_path}:{i}:{line.strip()}')
            except Exception:
                continue
    if not matches:
        return f"Definition for '{symbol}' not found. Try a regular grep_search."
    return 'Potential Definitions:\n\n' + '\n'.join(matches[:100])