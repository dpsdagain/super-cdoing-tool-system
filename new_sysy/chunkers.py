import ast
import logging
from typing import Any, List, Optional, Set
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)

logger = logging.getLogger(__name__)

class ASTChunker:
    """
    A language-aware chunker that uses AST parsing for Python to extract
    meaningful units like functions and classes.
    """
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def _extract_called_functions(self, content: str) -> Set[str]:
        """Extract called function names from Python content using AST."""
        try:
            tree = ast.parse(content)
            calls = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            return calls
        except Exception:
            return set()

    def _extract_referenced_constants(self, content: str) -> Set[str]:
        """Extract referenced constants (UPPER_CASE names) from Python content."""
        try:
            tree = ast.parse(content)
            consts = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id.isupper():
                    consts.add(node.id)
            return consts
        except Exception:
            return set()

    def chunk_file(self, content: str, fpath: str, ext: str) -> List[Document]:
        """
        Chunk a file based on its extension. Currently specialized for Python.
        """
        if ext.lower() == ".py":
            return self._chunk_python(content, fpath)
        
        # Fallback for non-python code
        splitter = get_text_splitter(ext, chunk_size_override=self.chunk_size)
        return splitter.create_documents([content], metadatas=[{"source": fpath}])

    def _chunk_python(self, content: str, fpath: str) -> List[Document]:
        """Specific chunking logic for Python files using AST."""
        try:
            tree = ast.parse(content)
            chunks = []
            lines = content.splitlines()

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", start_line + 1)
                    node_content = "\n".join(lines[start_line:end_line])
                    
                    if len(node_content) > self.chunk_size:
                        # Further split if a single function/class is too large
                        sub_splitter = get_text_splitter(".py", chunk_size_override=self.chunk_size)
                        sub_chunks = sub_splitter.create_documents([node_content])
                        for i, sc in enumerate(sub_chunks):
                            sc.metadata.update({
                                "source": fpath,
                                "type": type(node).__name__,
                                "name": node.name,
                                "sub_chunk": i
                            })
                            chunks.append(sc)
                    else:
                        chunks.append(Document(
                            page_content=node_content,
                            metadata={
                                "source": fpath,
                                "type": type(node).__name__,
                                "name": node.name,
                                "calls_functions": " ".join(self._extract_called_functions(node_content)),
                                "references_constants": " ".join(self._extract_referenced_constants(node_content)),
                            }
                        ))
            
            if not chunks:
                # If no top-level functions/classes found, split normally
                return self.chunk_file(content, fpath, ".txt")
            
            return chunks
        except Exception as e:
            logger.warning("AST parsing failed for %s: %s. Falling back to text splitting.", fpath, e)
            splitter = get_text_splitter(".py", chunk_size_override=self.chunk_size)
            return splitter.create_documents([content], metadatas=[{"source": fpath}])

def get_text_splitter(ext: Optional[str] = None, chunk_size_override: int = 1000):
    """
    Factory function for LangChain text splitters, optionally language-aware.
    """
    overlap = 100
    
    if ext:
        ext = ext.lower()
        if ext == ".py":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=chunk_size_override, chunk_overlap=overlap
            )
        elif ext in (".js", ".ts"):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.JS, chunk_size=chunk_size_override, chunk_overlap=overlap
            )
        elif ext == ".java":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.JAVA, chunk_size=chunk_size_override, chunk_overlap=overlap
            )
        elif ext in (".cpp", ".c", ".h", ".hpp"):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP, chunk_size=chunk_size_override, chunk_overlap=overlap
            )
        elif ext == ".go":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.GO, chunk_size=chunk_size_override, chunk_overlap=overlap
            )
        elif ext == ".rs":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.RUST, chunk_size=chunk_size_override, chunk_overlap=overlap
            )
    
    # Default text splitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_override,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
