"""
rag_chain.py - RAG Orchestration Facade
This module has been decomposed into specialized components.
Please see search_engine.py, cache_engine.py, intent_router.py, llm_factory.py, and prompt_builder.py.
"""

from prompt_builder import CORE_INSTRUCTIONS
from llm_factory import get_llm
from search_engine import hybrid_search, get_reranker
from rag_core import build_rag_chain

__all__ = [
    "CORE_INSTRUCTIONS",
    "get_llm",
    "hybrid_search",
    "get_reranker",
    "build_rag_chain"
]
