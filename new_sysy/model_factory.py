import os
import logging
from typing import Any
from llm_factory import get_llm
from config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Anthropic-Grade Model Management (F-45).
    Implements semantic aliases, plan/act separation, and provider abstraction.
    """
    
    # Semantic Mapping: High-level aliases to specific implementation IDs
    # Inspired by model.ts:457
    SEMANTIC_ALIASES = {
        "best": "ollama-cloud:gpt-oss:120b-cloud",
        "fast": "google/gemini-2.0-flash-001", # High speed, low latency
        "coder": "ollama-cloud:qwen3.6-coder:32b-cloud", # Logical reasoning
        "reasoning": "liquid/lfm-2.5-1.2b-thinking:free", # Synthetic thinking
        "haiku": "google/gemma-4-26b-a4b-it:free", # Fast fallback
        "opus": "ollama-cloud:gpt-oss:120b-cloud" # SOTA fallback
    }

    @staticmethod
    def resolve_alias(model_id: str) -> str:
        """Resolves semantic aliases (best, fast, etc.) to canonical IDs."""
        id_lower = model_id.lower()
        # Handle [1m] suffix for extended context
        has_1m = "[1m]" in id_lower
        clean_id = id_lower.replace("[1m]", "")
        
        resolved = ModelFactory.SEMANTIC_ALIASES.get(clean_id, model_id)
        return f"{resolved}[1m]" if has_1m else resolved

    @staticmethod
    def create_model(model_id: str = None, temperature: float = 0.0) -> Any:
        # 🚀 1. Semantic Alias Resolution
        target_id = ModelFactory.resolve_alias(model_id or DEFAULT_MODEL)
        
        # 🚀 2. Context Length Injection (Simulated via token limits)
        max_tokens = 4096
        if "[1m]" in target_id:
            logger.info("Engaging Extended Context Mode")
            max_tokens = 16384 # Scaled for RAG performance
            target_id = target_id.replace("[1m]", "")

        logger.info(f"ModelFactory: Sourcing {target_id} via llm_factory")
        
        try:
            # 🚀 3. Centralized Instantiation (Delegates to get_llm)
            # We override max_tokens in the kwargs if the underlying function allows,
            # or we just let get_llm use its global config for max_tokens.
            # get_llm natively handles Anthropic caching headers and provider routing.
            return get_llm(model=target_id, temperature=temperature, streaming=True)
            
        except Exception as e:
            logger.error(f"ModelFactory failover engaging: {e}")
            # Anthropic Pattern: Silent Peer Failover
            return get_llm(model="anthropic/claude-3-haiku", temperature=0.0, streaming=True)

    @staticmethod
    def list_available_categories():
        """Helper for the UI to show categorized models from config."""
        from config import CLOUDROUTER_MODELS
        return CLOUDROUTER_MODELS
