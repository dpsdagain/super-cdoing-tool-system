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
1. "tool": Use to perform a specific action (read/write/edit files, run commands, grep search). This is your primary way of interacting with the codebase.
2. "final": Use ONLY if you have completed the user's request or have a definitive answer.

Respond ONLY with JSON:
{
    "action": "tool" | "final",
    "input": "Instruction for tool use, or final response",
    "reason": "Brief justification"
}"""


class StrategicPlanner:
    """Strategic Decision Layer with Strategy Evolution (Self-Awareness)."""

    def __init__(self, llm):
        self.llm = llm

    def call_planner(
        self, messages: List[Any], turn: int, action_history: List[str]
    ) -> Dict[str, Any]:
        """Strategic Decision Layer with Strategy Evolution (Self-Awareness)."""
        try:
            # Construct a summary of recent history for the planner
            history_summary = ""
            for m in messages[-5:]:
                type_name = (
                    "User"
                    if isinstance(m, HumanMessage)
                    else (
                        "Assistant"
                        if isinstance(m, AIMessage)
                        else "Tool" if isinstance(m, ToolMessage) else "System"
                    )
                )
                content = m.content if isinstance(m.content, str) else str(m.content)
                history_summary += f"{type_name}: {content[:150]}...\n"

            # Action history awareness
            action_hist_text = (
                "\n".join(action_history[-5:])
                if action_history
                else "No previous actions."
            )

            # Find the user request (usually the second message after the system prompt)
            user_request = "None"
            for m in messages:
                if isinstance(m, HumanMessage):
                    user_request = m.content
                    break

            prompt = f"""[AGENT_STATUS]
Current Turn: {turn}

Recent History:
{history_summary}

Recent Actions Taken:
{action_hist_text}

USER REQUEST: {user_request}
[/AGENT_STATUS]

Based on this status, determine the next action."""

            # Use the specialized Planner Model for strategy overrides
            response = self.llm.invoke(
                [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=prompt)]
            )

            # Cleanup JSON (handling cases where LLM adds Markdown blocks)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("{") and "}" in content:
                pass
            else:
                # Basic heuristic extraction if JSON is buried
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    content = content[start : end + 1]

            return json.loads(content)
        except (json.JSONDecodeError, Exception):
            logger.exception("Planner failed. Defaulting to 'tool'.")
            return {"action": "tool", "input": "", "reason": "Planner error fallback"}
