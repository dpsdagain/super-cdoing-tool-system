import os
import threading
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from tool_registry import tool, validate_path, current_engine
from config import WORKSPACE_ROOT
logger = logging.getLogger(__name__)



class AgentDelegateInput(BaseModel):
    task: str = Field(description='The specific task or goal for the sub-agent to achieve.')
    context_files: List[str] = Field(default=[], description='List of file paths the sub-agent should focus on.')

class AgentDelegateInput(BaseModel):
    task: str = Field(description='The specific task or goal for the sub-agent to achieve.')
    context_files: List[str] = Field(default=[], description='List of file paths the sub-agent should focus on.')
    current_depth: int = Field(default=0, description='Internal field to track recursion.', exclude=True)

def agent_delegate(task: str, context_files: List[str]=[]) -> str:
    """
    Spawn a sub-agent to handle a specific delegated task.
    Orchestrated by the Multi-Agent Coordinator (Coordinator.ts).
    """
    if current_engine.instance is None:
        return 'Error: Coordinator not initialized.'
    context_summary = f"Focus files: {', '.join(context_files)}" if context_files else 'General repository context.'
    try:
        return current_engine.instance.coordinator.delegate(task, context_summary)
    except Exception as e:
        return f'Error in agent delegation: {str(e)}'