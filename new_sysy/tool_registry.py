"""
tool_registry.py — Centralized Tool Factory & Registry.
All tools in the system must be registered through this module.
"""

# pylint: disable=too-many-arguments

import logging
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
    """Ensure the path is within the WORKSPACE_ROOT boundary, is not a symlink,
    and doesn't target sensitive files.

    Only OSError is translated into PermissionError — other exceptions
    (TypeError, ValueError) are real bugs and propagate so they can be fixed.
    """
    raw = Path(path)

    # Reject symlinks before resolve(). Path.resolve() *follows* the link, so
    # a workspace-internal symlink pointing outside the workspace would fail
    # the boundary check — but a symlink whose target also lives inside the
    # workspace would silently allow writes that traverse the link, which is
    # rarely what the agent intends.
    #
    # We branch outside the try/except: PermissionError inherits from OSError,
    # so wrapping the raise inside `except OSError` would double-wrap the
    # message.
    try:
        is_link = raw.is_symlink()
    except OSError as e:
        raise PermissionError(
            f"Access Denied: Could not stat path '{path}': {e}"
        ) from e

    if is_link:
        raise PermissionError(
            f"Access Denied: '{path}' is a symbolic link; refusing to follow."
        )

    try:
        target = raw.resolve()
        root = Path(WORKSPACE_ROOT).resolve()
    except OSError as e:
        raise PermissionError(
            f"Access Denied: Could not resolve path '{path}': {e}"
        ) from e

    if not target.is_relative_to(root):
        raise PermissionError(
            f"Access Denied: Path '{target}' is outside the allowed workspace '{root}'."
        )

    if any(p in FORBIDDEN_PATTERNS for p in target.parts):
        raise PermissionError(
            f"Access Denied: Path '{target}' contains forbidden components."
        )

    return str(target)


def build_available_tools() -> Dict[str, Any]:
    CORE_TOOL_NAMES = {
        "file_read",
        "file_edit",
        "file_write",
        "bash",
        "glob",
        "code_search",
        "list_directory",
    }
    core_partition = sorted([k for k in _UNSORTED_TOOLS if k in CORE_TOOL_NAMES])
    plugin_partition = sorted([k for k in _UNSORTED_TOOLS if k not in CORE_TOOL_NAMES])

    AVAILABLE_TOOLS = {}
    for name in core_partition + plugin_partition:
        AVAILABLE_TOOLS[name] = _UNSORTED_TOOLS[name]
    return AVAILABLE_TOOLS
