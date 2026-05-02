import json
import logging
from typing import List, Dict, Any
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a strategic orchestrator for an AI agent.
Based on the conversation history, decide the next logical step.

ACTIONS:
1. "tool": Use to perform a specific action (read/write/edit files, run commands, grep search). This is your primary way of interacting with the codebase. CRITICAL: If the user asks to read, analyze, or rate a file, you MUST return "tool".
2. "final": Use ONLY if you have completed the user's request using tools, or if it is a casual non-coding question (like math or greetings). Do NOT hallucinate file contents.

Respond ONLY with JSON:
{
    "action": "tool" | "final",
    "input": "Instruction for tool use, or final response",
    "reason": "Brief justification"
}"""

HISTORY_PREVIEW_LIMIT = 150
HISTORY_WINDOW = 5


class StrategicPlanner:
    """Strategic Decision Layer with Strategy Evolution (Self-Awareness)."""

    def __init__(self, llm):
        self.llm = llm

    def call_planner(
        self, messages: List[Any], turn: int, action_history: List[str]
    ) -> Dict[str, Any]:
        """Strategic Decision Layer with Strategy Evolution (Self-Awareness)."""
        history_summary = "".join(
            f"{self._role_label(m)}: {self._preview(m)}\n"
            for m in messages[-HISTORY_WINDOW:]
        )
        action_hist_text = (
            "\n".join(action_history[-HISTORY_WINDOW:])
            if action_history
            else "No previous actions."
        )

        latest_user_request = self._latest_user_request(messages)

        prompt = f"""[AGENT_STATUS]
Current Turn: {turn}

Recent History:
{history_summary}

Recent Actions Taken:
{action_hist_text}

USER REQUEST: {latest_user_request}
[/AGENT_STATUS]

Based on this status, determine the next action."""

        # Note: we deliberately do NOT swallow LLM/network errors here — they
        # should surface to the caller's outer handler. Only JSON parse failures
        # fall back to a safe default.
        response = self.llm.invoke(
            [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=prompt)]
        )

        try:
            return self._parse_decision(response.content)
        except json.JSONDecodeError:
            logger.warning("Planner JSON parse failed; defaulting to 'tool'.")
            return {
                "action": "tool",
                "input": "",
                "reason": "planner json fallback",
            }

    @staticmethod
    def _role_label(message: Any) -> str:
        if isinstance(message, HumanMessage):
            return "User"
        if isinstance(message, AIMessage):
            return "Assistant"
        if isinstance(message, ToolMessage):
            return "Tool"
        return "System"

    @staticmethod
    def _preview(message: Any, limit: int = HISTORY_PREVIEW_LIMIT) -> str:
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        if len(content) > limit:
            return content[:limit] + "..."
        return content

    @staticmethod
    def _latest_user_request(messages: List[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                content = message.content
                return content if isinstance(content, str) else str(content)
        return "None"

    @staticmethod
    def _parse_decision(raw_content: Any) -> Dict[str, Any]:
        """Extract a JSON decision from a possibly-fenced LLM response.

        Raises json.JSONDecodeError on failure so callers can decide policy.
        """
        content = (raw_content if isinstance(raw_content, str) else str(raw_content)).strip()

        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif not content.startswith("{"):
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]

        return json.loads(content)
