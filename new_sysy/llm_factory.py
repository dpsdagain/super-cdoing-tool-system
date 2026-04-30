"""
llm_factory.py — LLM Model Factory and Cache-Capability Utilities.
Provides get_llm(), ModelFactory, and cache-profile helpers.
"""

from __future__ import annotations
import threading
import logging
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    LLM_TEMPERATURE,
    MAX_TOKENS,
    ANTHROPIC_CACHE_BETA_HEADER,
    ENABLE_PROMPT_CACHING,
    MAX_CACHE_CHECKPOINTS,
    PROVIDER_CACHE_PROFILES,
    OLLAMA_BASE_URL,
    OLLAMA_PREFIX,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_CLOUD_PREFIX,
)

logger = logging.getLogger(__name__)

_llm_cache_lock = threading.Lock()


def is_cache_capable(model: str | None) -> bool:
    """
    Check if the model/provider supports Anthropic-style prompt caching blocks.

    In 2026, this includes Claude, Gemini 2+, Gemma 4 (Google), and GLM 5.
    OpenAI and DeepSeek use implicit prefix caching (no markers needed).
    """
    if not model or model.startswith(OLLAMA_PREFIX):
        return False

    m_lower = model.lower()
    # Broaden detection for SOTA models that favor explicit markers
    cache_brands = ["claude", "gemini", "gemma", "glm-5"]
    return any(brand in m_lower for brand in cache_brands)


def get_cache_profile(model: str | None) -> tuple[int, int]:
    """
    Cross-Provider Cache Router.
    Returns (max_checkpoints, min_tokens_for_cache) for the given model.
    Different providers have different cache economics:
      - Claude: 4 breakpoints, 1024 token minimum
      - Gemini: more breakpoints allowed, 1028 token minimum
      - DeepSeek/Qwen: similar to Claude
    Falls back to global defaults if model is unknown.
    """
    if not model:
        return (MAX_CACHE_CHECKPOINTS, 1024)
    m_lower = model.lower()
    for pattern, profile in PROVIDER_CACHE_PROFILES.items():
        if pattern in m_lower:
            return profile
    return (MAX_CACHE_CHECKPOINTS, 1024)


def format_message_content(
    text: str, model: str | None, use_cache: bool = False
) -> str | list[dict]:
    """
    Return content as a plain string for non-cache models,
    or a block-list for cache-capable ones.
    """
    # If caching is globally disabled or model can't handle it,
    # always return a plain string.
    if not ENABLE_PROMPT_CACHING or not use_cache or not is_cache_capable(model):
        return text

    # Return Anthropic-style block format with cache markers
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool = True,
    max_tokens: int | None = None,
):
    """
    Return a chat model instance.

    If *model* is ``OLLAMA_SENTINEL`` the function returns a local
    ``ChatOllama``; otherwise it returns a ``ChatOpenAI`` pointed at
    OpenRouter.
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    final_max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS

    # Cache Check (thread-safe)
    cache_key = (model, temp, streaming, final_max_tokens)
    if not hasattr(get_llm, "cache"):
        with _llm_cache_lock:
            if not hasattr(get_llm, "cache"):
                get_llm.cache = {}

    with _llm_cache_lock:
        if cache_key in get_llm.cache:
            return get_llm.cache[cache_key]

    def _cache_and_return(llm_obj):
        with _llm_cache_lock:
            get_llm.cache[cache_key] = llm_obj
        return llm_obj

    # ── Local Ollama path ──────────────────────────────────────────────
    if model and model.startswith(OLLAMA_PREFIX):
        ollama_model_name = model[len(OLLAMA_PREFIX) :]
        return _cache_and_return(
            ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=ollama_model_name,
                temperature=temp,
                num_predict=final_max_tokens,
            )
        )

    # ── Ollama Cloud path ──────────────────────────────────────────────
    if model and model.startswith(OLLAMA_CLOUD_PREFIX):
        cloud_model_name = model[len(OLLAMA_CLOUD_PREFIX) :]
        if not OLLAMA_CLOUD_API_KEY:
            raise ValueError(
                "OLLAMA_CLOUD_API_KEY is not set. " "Add it to your .env file."
            )
        return _cache_and_return(
            ChatOpenAI(
                base_url=OLLAMA_CLOUD_BASE_URL,
                api_key=OLLAMA_CLOUD_API_KEY,
                model=cloud_model_name,
                temperature=temp,
                streaming=streaming,
                max_tokens=final_max_tokens,
            )
        )

    # ── OpenRouter path ────────────────────────────────────────────────
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-..."
        )
    # Professional Polish: Conditional Header Safety
    # Only send Anthropic-specific headers when using a Claude model
    default_headers = {
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Private AI Knowledge Base",
    }

    current_model = model or DEFAULT_MODEL
    if "claude" in current_model.lower():
        default_headers["anthropic-beta"] = ANTHROPIC_CACHE_BETA_HEADER

    return _cache_and_return(
        ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=current_model,
            temperature=temp,
            streaming=streaming,
            max_tokens=MAX_TOKENS,
            default_headers=default_headers,
            # Enable usage in stream for telemetry visibility
            model_kwargs={"stream_options": {"include_usage": True}},
        )
    )


class ModelFactory:
    SEMANTIC_ALIASES = {
        "best": "ollama-cloud:gpt-oss:120b-cloud",
        "fast": "google/gemini-2.0-flash-001",
        "coder": "ollama-cloud:qwen3.6-coder:32b-cloud",
        "reasoning": "liquid/lfm-2.5-1.2b-thinking:free",
        "haiku": "google/gemma-4-26b-a4b-it:free",
        "opus": "ollama-cloud:gpt-oss:120b-cloud",
    }

    @staticmethod
    def resolve_alias(model_id: str) -> str:
        id_lower = model_id.lower()
        has_1m = "[1m]" in id_lower
        clean_id = id_lower.replace("[1m]", "")

        resolved = ModelFactory.SEMANTIC_ALIASES.get(clean_id, model_id)
        return f"{resolved}[1m]" if has_1m else resolved

    @staticmethod
    def create_model(model_id: str = None, temperature: float = 0.0):
        target_id = ModelFactory.resolve_alias(model_id or DEFAULT_MODEL)

        max_tokens = 4096
        if "[1m]" in target_id:
            logger.info("Engaging Extended Context Mode")
            max_tokens = 16384
            target_id = target_id.replace("[1m]", "")

        logger.info(f"ModelFactory: Sourcing {target_id} via llm_factory")

        try:
            return get_llm(
                model=target_id,
                temperature=temperature,
                streaming=True,
                max_tokens=max_tokens,
            )
        except Exception:
            logger.exception("ModelFactory failover engaging:")
            return get_llm(
                model="anthropic/claude-3-haiku",
                temperature=0.0,
                streaming=True,
                max_tokens=max_tokens,
            )

    @staticmethod
    def list_available_categories():
        from config import CLOUDROUTER_MODELS

        return CLOUDROUTER_MODELS
