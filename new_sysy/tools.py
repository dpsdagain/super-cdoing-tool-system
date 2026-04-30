"""
tools.py - Agentic Engine Tool Facade
This module has been decomposed. Please see *_tools.py modules for implementations.
"""

# pylint: disable=unused-import

from tool_registry import validate_path, build_available_tools

from bash_tool import cleanup_active_processes
import file_tools
import git_manager
import misc_tools
import web_utils

AVAILABLE_TOOLS = build_available_tools()

# Export everything needed by query_engine.py and others
__all__ = ["AVAILABLE_TOOLS", "cleanup_active_processes", "validate_path"]
