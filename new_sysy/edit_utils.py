import re
from typing import Optional


class FuzzyMatcher:
    """
    Advanced Fuzzy String Matcher.
    Ported from FileEditTool/utils.ts.

    Ensures file edits succeed even if the LLM makes minor
    formatting/typographic errors.
    """

    # String Matcher Constants (utils.ts:21-24)
    CURLY_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"'}

    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Ported from utils.ts:31."""
        for curly, straight in FuzzyMatcher.CURLY_QUOTES.items():
            text = text.replace(curly, straight)
        return text

    @staticmethod
    def strip_trailing_whitespace(text: str) -> str:
        """Ported from utils.ts:44."""
        lines = text.splitlines(keepends=True)
        return "".join([re.sub(r"[ \t]+$", "", line) for line in lines])

    @classmethod
    def find_actual_string(cls, content: str, search: str) -> Optional[str]:
        """
        Ported from utils.ts:73.
        Tries exact match, then normalized match.
        """
        # 1. Exact try
        if search in content:
            return search

        # 2. Regex-based approximation (Whitespace handling)
        import re
        parts = re.split(r'\s+', search)
        escaped_parts = [re.escape(p) for p in parts]
        regex_pattern = r'\s+'.join(escaped_parts)
        
        match = re.search(regex_pattern, content)
        if match:
            return match.group(0)

        return None
