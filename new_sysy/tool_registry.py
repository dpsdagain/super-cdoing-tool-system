"""
tool_registry.py — Centralized Tool Factory & Registry.
All tools in the system must be registered through this module.
"""
import os
import logging
import contextvars
from typing import Any, Dict, Callable
from pathlib import Path
from pydantic import BaseModel
from config import WORKSPACE_ROOT, FORBIDDEN_PATTERNS

logger = logging.getLogger(__name__)


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
    requires_engine: bool = False

_UNSORTED_TOOLS = {}

def register_tool(
    name: str, 
    description: str, 
    input_schema: Any, 
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False,
    requires_engine: bool = False,
) -> Callable:
    """
    Centralized tool decorator that enforces security defaults.
    Every tool in the system must pass through this gate.
    """
    def decorator(func: Callable) -> Callable:
        _UNSORTED_TOOLS[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "func": func,
            "is_read_only": is_read_only,
            "is_destructive": is_destructive,
            "is_concurrency_safe": is_concurrency_safe,
            "requires_engine": requires_engine,
        }
        return func
    return decorator


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


def build_available_tools() -> Dict[str, Any]:
    CORE_TOOL_NAMES = {"file_read", "file_edit", "file_write", "multi_file_edit", "bash", "glob", "code_search", "grep_search"}
    core_partition = sorted([k for k in _UNSORTED_TOOLS if k in CORE_TOOL_NAMES])
    plugin_partition = sorted([k for k in _UNSORTED_TOOLS if k not in CORE_TOOL_NAMES])

    AVAILABLE_TOOLS = {}
    for name in core_partition + plugin_partition:
        AVAILABLE_TOOLS[name] = _UNSORTED_TOOLS[name]
    return AVAILABLE_TOOLS
