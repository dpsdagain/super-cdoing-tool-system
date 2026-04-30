import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Context Limits (utils/context.ts:149)
MODEL_LIMITS = {
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "default": 128_000,
}


class ContextEstimator:
    """
    Advanced Pre-Flight Estimator.
    Predicts context overflows before they reach the API.
    """

    @staticmethod
    def estimate_content_tokens(content: Any) -> int:
        """
        Estimates the number of tokens in a piece of content (string or list).
        Heuristic: 1 token approx 3.8 characters.
        """
        if not content:
            return 0
        if isinstance(content, str):
            return int(len(content) / 3.8)
        elif isinstance(content, list):
            # Handles multi-modal/tool blocks by converting to string representation
            return int(len(str(content)) / 3.8)
        return int(len(str(content)) / 3.8)

    @classmethod
    def estimate_tokens(cls, messages: List[Any]) -> int:
        """
        Performs a 'Rough-Cut' token estimation for a list of messages.
        """
        total_tokens = 0
        for msg in messages:
            content = getattr(msg, "content", "")
            total_tokens += cls.estimate_content_tokens(content)
        return total_tokens

    @staticmethod
    def check_flight_safety(model: str, messages: List[Any]) -> Dict[str, Any]:
        """
        Audits context against model limits (context.ts:118).
        """
        limit = MODEL_LIMITS.get(model, MODEL_LIMITS["default"])
        estimated = ContextEstimator.estimate_tokens(messages)
        usage_pct = (estimated / limit) * 100

        status = "SAFE"
        if usage_pct > 95:
            status = "CRITICAL"  # Must compact
        elif usage_pct > 85:
            status = "WARNING"  # Warn user

        return {
            "estimated_tokens": estimated,
            "limit": limit,
            "usage_pct": round(usage_pct, 2),
            "status": status,
        }
