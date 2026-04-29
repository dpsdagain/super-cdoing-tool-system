"""
tool_registry.py — Centralized Tool Factory & Registry.
All tools in the system must be registered through this module.
"""
import os
import threading
import logging
from typing import Any, Dict
from pathlib import Path
from pydantic import BaseModel
from config import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Explicit Imports (replacing wildcard imports)
# ═══════════════════════════════════════════════════════════════════════════
from file_tools import (
    code_search, file_read, file_edit, file_write, multi_file_edit,
    grep_tool, glob_tool, brief_tool, symbol_search,
    CodeSearchInput, FileReadInput, FileEditInput, FileWriteInput,
    MultiFileEditInput, GrepInput, GlobInput, BriefInput, SymbolSearchInput,
    UndoInput,
)
from bash_tool import bash_tool, BashInput
from web_tools import (
    web_search, web_fetch, read_url,
    WebSearchInput, WebFetchInput,
)
from git_tools import (
    git_status, git_diff, git_commit, git_root, git_log,
    GitStatusInput, GitDiffInput, GitCommitInput, GitLogInput,
)
from misc_tools import (
    switch_model, update_plan, set_status, cost_report, system_doctor,
    notebook_edit, undo_last_edit, linter_tool, memory_tool,
    arch_visualizer, undercover_mode, task_budget, ask_user,
    SwitchModelInput, UpdatePlanInput, SetStatusInput, NotebookEditInput,
    AskUserInput, ArchVisualizerInput, TaskBudgetInput,
    LinterInput, MemoryInput, UndercoverInput, CostInput, DoctorInput,
)
from agent_tools import agent_delegate, AgentDelegateInput


# ═══════════════════════════════════════════════════════════════════════════
#  Core Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

class ToolMetadata(BaseModel):
    name: str
    description: str
    input_schema: Any
    func: Any
    is_read_only: bool = False
    is_destructive: bool = False
    is_concurrency_safe: bool = False


def build_tool(
    name: str, 
    description: str, 
    input_schema: Any, 
    func: Any, 
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
) -> Dict[str, Any]:
    """
    Centralized tool factory that enforces security defaults.
    Every tool in the system must pass through this gate.
    """
    return {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "func": func,
        "is_read_only": is_read_only,
        "is_destructive": is_destructive,
        "is_concurrency_safe": is_concurrency_safe,
    }


class EngineContext(threading.local):
    def __init__(self):
        self.instance = None

current_engine = EngineContext()


FORBIDDEN_PATTERNS = [".env", ".git", "id_rsa", "id_ed25519", "credentials", ".ssh", ".aws", ".config"]

def validate_path(path: str) -> str:
    """Ensure the path is within the WORKSPACE_ROOT boundary and doesn't target sensitive files."""
    try:
        target = Path(path).resolve()
        root = Path(WORKSPACE_ROOT).resolve()

        if not target.is_relative_to(root):
            raise PermissionError(f"Access Denied: Path '{target}' is outside the allowed workspace '{root}'.")

        if any(p in FORBIDDEN_PATTERNS for p in target.parts):
            raise PermissionError(f"Access Denied: Path '{target}' contains forbidden components.")

        return str(target)
    except Exception as e:
        if isinstance(e, PermissionError):
            raise e
        raise PermissionError(f"Access Denied: Could not validate path '{path}'.")

# ═══════════════════════════════════════════════════════════════════════════
#  Tool Registry
# ═══════════════════════════════════════════════════════════════════════════

_UNSORTED_TOOLS = {
    "code_search": build_tool(
        name="code_search",
        func=code_search,
        input_schema=CodeSearchInput,
        description="Search the codebase for relevant functions, classes, or logic using keywords or natural language.",
        is_read_only=True
    ),
    "file_read": build_tool(
        name="file_read",
        func=file_read,
        input_schema=FileReadInput,
        description="Read the content of a file. Supports line ranges for large files.",
        is_read_only=True
    ),
    "file_edit": build_tool(
        name="file_edit",
        func=file_edit,
        input_schema=FileEditInput,
        description="Edit a file by replacing an exact string with a new string. This is safer than overwriting the whole file.",
        is_read_only=False
    ),
    "file_write": build_tool(
        name="file_write",
        func=file_write,
        input_schema=FileWriteInput,
        description="Create a new file or completely overwrite an existing file with new content.",
        is_read_only=False
    ),
    "multi_file_edit": build_tool(
        name="multi_file_edit",
        func=multi_file_edit,
        input_schema=MultiFileEditInput,
        description="Apply multiple surgical replacements to a single file in one go. Much more efficient than multiple file_edit calls.",
        is_read_only=False
    ),
    "grep_search": build_tool(
        name="grep_search",
        func=grep_tool,
        input_schema=GrepInput,
        description="Search for a pattern across the codebase using regex. Returns file paths and matching lines.",
        is_read_only=True
    ),
    "bash": build_tool(
        name="bash",
        func=bash_tool,
        input_schema=BashInput,
        description="Execute a shell command. Use this for running tests, build scripts, or git commands.",
        is_read_only=False
    ),
    "glob": build_tool(
        name="glob",
        func=glob_tool,
        input_schema=GlobInput,
        description="Search for files using glob patterns (e.g., '**/*.py').",
        is_read_only=True
    ),
    "brief": build_tool(
        name="brief",
        func=brief_tool,
        input_schema=BriefInput,
        description="Provide a brief outline (classes and functions) of a file to save context.",
        is_read_only=True
    ),
    "linter": build_tool(
        name="linter",
        func=linter_tool,
        input_schema=LinterInput,
        description="Run a linter on a specific file to check for syntax errors or formatting issues.",
        is_read_only=True
    ),
    "memory": build_tool(
        name="memory",
        func=memory_tool,
        input_schema=MemoryInput,
        description="Save a learned fact, architecture decision, or user preference into long-term memory (MEMORY.md).",
        is_read_only=False
    ),
    "ask_user": build_tool(
        name="ask_user",
        func=ask_user,
        input_schema=AskUserInput,
        description="Pause agent execution to ask the human user a clarifying question.",
        is_read_only=True,
        is_concurrency_safe=True
    ),
    "web_search": build_tool(
        name="web_search",
        func=web_search,
        input_schema=WebSearchInput,
        description="Search the internet for documentation, libraries, or coding solutions.",
        is_read_only=True
    ),
    "web_fetch": build_tool(
        name="web_fetch",
        func=web_fetch,
        input_schema=WebFetchInput,
        description="Fetch and read the text content of a specific URL.",
        is_read_only=True
    ),
    "symbol_search": build_tool(
        name="symbol_search",
        func=symbol_search,
        input_schema=SymbolSearchInput,
        description="Find the definition of a class or function across the codebase.",
        is_read_only=True
    ),
    "agent_delegate": build_tool(
        name="agent_delegate",
        func=agent_delegate,
        input_schema=AgentDelegateInput,
        description="Delegate a specific task to a sub-agent with its own context.",
        is_read_only=False
    ),
    "arch_visualizer": build_tool(
        name="arch_visualizer",
        func=arch_visualizer,
        input_schema=ArchVisualizerInput,
        description="Generate a high-level architecture overview of classes and methods in Mermaid format.",
        is_read_only=True
    ),
    "undercover_mode": build_tool(
        name="undercover_mode",
        func=undercover_mode,
        input_schema=UndercoverInput,
        description="Strip AI identifiers and local filesystem paths from a text string.",
        is_read_only=True
    ),
    "task_budget": build_tool(
        name="task_budget",
        func=task_budget,
        input_schema=TaskBudgetInput,
        description="Set a maximum token limit for the current task to control costs.",
        is_read_only=False
    ),
    "update_plan": build_tool(
        name="update_plan",
        func=update_plan,
        input_schema=UpdatePlanInput,
        description="Synthetic Tool: Update your internal master plan. Use this to track progress, rejected ideas, and next steps.",
        is_read_only=False
    ),
    "set_status": build_tool(
        name="set_status",
        func=set_status,
        input_schema=SetStatusInput,
        description="Sets the current activity status for the TUI.",
        is_read_only=False
    ),
    "git_status": build_tool(
        name="git_status",
        func=git_status,
        input_schema=GitStatusInput,
        description="Get porcelain git status.",
        is_read_only=True
    ),
    "git_diff": build_tool(
        name="git_diff",
        func=git_diff,
        input_schema=GitDiffInput,
        description="Get git diff for the repo or a file.",
        is_read_only=True
    ),
    "git_commit": build_tool(
        name="git_commit",
        func=git_commit,
        input_schema=GitCommitInput,
        description="Commit staged changes.",
        is_read_only=False
    ),
    "git_log": build_tool(
        name="git_log",
        func=git_log,
        input_schema=GitLogInput,
        description="View recent commit history.",
        is_read_only=True
    ),
    "switch_model": build_tool(
        name="switch_model",
        func=switch_model,
        input_schema=SwitchModelInput,
        description="Switch the active LLM brain. Ideal for scaling intelligence up or down.",
        is_read_only=True
    ),
    "undo_last_edit": build_tool(
        name="undo_last_edit",
        func=undo_last_edit,
        input_schema=UndoInput,
        description="Roll back file changes made in a specific turn. Use the tool_use_id of the turn to revert.",
        is_read_only=False
    ),
    "notebook_edit": build_tool(
        name="notebook_edit",
        func=notebook_edit,
        input_schema=NotebookEditInput,
        description="Surgically edit, insert, or delete Jupyter Notebook (.ipynb) cells.",
        is_read_only=False
    ),
    "system_doctor": build_tool(
        name="system_doctor",
        func=system_doctor,
        input_schema=DoctorInput,
        description="Audit system health (binaries, network, workspace toxicity). Run if tools are failing.",
        is_read_only=True
    ),
    "read_url": build_tool(
        name="read_url",
        func=read_url,
        input_schema=WebFetchInput,
        description="Fetch and distill web documentation or articles into Markdown.",
        is_read_only=True
    ),
    "cost_report": build_tool(
        name="cost_report",
        func=cost_report,
        input_schema=CostInput,
        description="Show the current session's token usage and USD cost report.",
        is_read_only=True
    ),
}

# ═══════════════════════════════════════════════════════════════════════════
#  Partitioned Tool Sets
# ═══════════════════════════════════════════════════════════════════════════

CORE_TOOL_NAMES = {"file_read", "file_edit", "file_write", "multi_file_edit", "bash", "glob", "code_search", "grep_search"}

core_partition = sorted([k for k in _UNSORTED_TOOLS if k in CORE_TOOL_NAMES])
plugin_partition = sorted([k for k in _UNSORTED_TOOLS if k not in CORE_TOOL_NAMES])

# Populate AVAILABLE_TOOLS from both partitions (core first, then plugins)
AVAILABLE_TOOLS = {}
for name in core_partition + plugin_partition:
    AVAILABLE_TOOLS[name] = _UNSORTED_TOOLS[name]
