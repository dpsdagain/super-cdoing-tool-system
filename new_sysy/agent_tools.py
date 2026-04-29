from tool_registry import register_tool
"""
agent_tools.py — Sub-agent Delegation Tool.
"""
import logging
from typing import List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentDelegateInput(BaseModel):
    task: str = Field(description='The specific task or goal for the sub-agent to achieve.')
    context_files: List[str] = Field(default=[], description='List of file paths the sub-agent should focus on.')
    current_depth: int = Field(default=0, description='Internal field to track recursion.', exclude=True)


@register_tool(name="agent_delegate", description="Spawn a sub-agent to handle a complex sub-task. Returns the agent's final report.", input_schema=AgentDelegateInput, is_read_only=False, requires_engine=True)
def agent_delegate(task: str, context_files: List[str] = [], _engine_instance=None) -> str:
    """
    Spawn a sub-agent to handle a specific delegated task.
    Orchestrated by the Multi-Agent Coordinator.
    """
    if _engine_instance is None:
        return 'Error: Coordinator not initialized.'
    context_summary = f"Focus files: {', '.join(context_files)}" if context_files else 'General repository context.'
    try:
        return _engine_instance.coordinator.delegate(task, context_summary)
    except Exception as e:
        return f'Error in agent delegation: {str(e)}'