"""
tools.py - Agentic Engine Tool Facade
This module has been decomposed. Please see *_tools.py modules for implementations.
"""
# pylint: disable=unused-import

from tool_registry import (
    validate_path,
    ToolMetadata,
    _UNSORTED_TOOLS,
    build_available_tools,
    register_tool
)

from bash_tool import cleanup_active_processes, bash_tool
from file_tools import (
    code_search, file_read, file_write, file_edit, multi_file_edit,
    grep_tool, glob_tool, brief_tool, symbol_search
)
from git_manager import git_status, git_diff, git_commit, git_root, git_log
from web_utils import web_search, web_fetch, read_url
from agent_tools import agent_delegate
from misc_tools import (
    linter_tool, memory_tool, ask_user, arch_visualizer, undercover_mode,
    task_budget, update_plan, set_status, undo_last_edit, notebook_edit,
    system_doctor, cost_report, switch_model
)

AVAILABLE_TOOLS = build_available_tools()

# Export everything needed by query_engine.py and others
__all__ = [
    "AVAILABLE_TOOLS", 
    "cleanup_active_processes",
    "validate_path"
]
