import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PermissionDecision:
    """Result of a permission check. Used by query_engine.py to gate tool execution."""
    behavior: str   # "allow", "ask", or "deny"
    reason: str = ""

# Permission modes
# AUTO: Execute without asking
# ASK: Prompt user for approval
# DENY: Always reject

import os
from pathlib import Path
from config import WORKSPACE_ROOT

class PermissionManager:
    def __init__(self, mode: str = "ASK", max_delegation_depth: int = 2):
        self.mode = mode
        self.max_delegation_depth = max_delegation_depth
        # Commands that are always safe (mostly read-only)
        self.safe_bash_commands = ["ls", "pwd", "git status", "git diff", "grep", "find", "cat", "dir", "type", "git log", "git branch"]
        # Commands that are always dangerous
        self.dangerous_bash_commands = ["rm -rf", "sudo", "mkfs", "dd", "format", "del /s", "rd /s"]
        # Forbidden path components (folders or files)
        self.forbidden_patterns = [".env", ".git", "id_rsa", "id_ed25519", "credentials", ".ssh", ".aws", ".config"]

    def validate_path(self, target_path: str) -> bool:
        """
        The Path Sentinel: Ensures the path is within the WORKSPACE_ROOT
        and doesn't target sensitive files or directories.
        """
        try:
            from tool_registry import validate_path
            validate_path(target_path)
            return True
        except Exception:
            return False

    def check_permission(self, tool_name: str, tool_args: Dict[str, Any], current_depth: int = 0) -> PermissionDecision:
        """
        Check if the tool execution is allowed.
        Returns a PermissionDecision with .behavior ('allow', 'ask', 'deny') and .reason.
        """
        # RECURSION GUARD: Prevent sub-agent death spirals
        if tool_name == "agent_delegate":
            if current_depth >= self.max_delegation_depth:
                logger.warning(f"SECURITY: Blocked agent_delegate at depth {current_depth}")
                return PermissionDecision(behavior="deny", reason=f"Maximum delegation depth ({current_depth}) reached.")

        # --- ENHANCED PATH SENTINEL ---

        # Intercept and validate ANY argument that might contain a path
        path_keys = ["file_path", "directory", "path", "context_files"]
        for key in path_keys:
            if key in tool_args:
                value = tool_args[key]
                # Handle lists (like context_files)
                if isinstance(value, list):
                    for p in value:
                        if not self.validate_path(p):
                            logger.warning(f"SECURITY: Path Sentinel blocked access to item in {key}: {p}")
                            return PermissionDecision(behavior="deny", reason=f"Path '{p}' is outside workspace or targets forbidden patterns.")
                elif not self.validate_path(value):
                    logger.warning(f"SECURITY: Path Sentinel blocked access to {key}: {value}")
                    return PermissionDecision(behavior="deny", reason=f"Path '{value}' is outside workspace or targets forbidden patterns.")

        if self.mode == "DENY":
            return PermissionDecision(behavior="deny", reason="Permission mode is DENY.")
        
        if self.mode == "AUTO":
            # Even in AUTO mode, we should never allow shell injection
            if tool_name == "bash":
                command = tool_args.get("command", "")
                if any(op in command for op in ["&&", ";", "|", "\n", "`", "$("]):
                    logger.warning(f"SECURITY: Blocked shell injection attempt in AUTO mode: {command}")
                    return PermissionDecision(behavior="deny", reason="Shell injection detected.")
            return PermissionDecision(behavior="allow")
            
        # In ASK mode, we perform some heuristic checks to see if we can auto-approve
        if tool_name == "bash":
            command = tool_args.get("command", "").lower()
            
            # BASH JAIL: Block attempts to escape workspace or use forbidden paths
            # 1. Block shell chaining operators
            if any(op in command for op in ["&&", ";", "|", "\n", "`", "$("]):
                return PermissionDecision(behavior="deny", reason="Shell metacharacters detected.") 
                
            if ".." in command or any(f in command for f in self.forbidden_patterns):
                return PermissionDecision(behavior="deny", reason="Forbidden path pattern detected.") 
                
            if any(cmd in command for cmd in self.dangerous_bash_commands):
                return PermissionDecision(behavior="deny", reason=f"Dangerous command detected.") 
            
            # Auto-approve safe commands and non-destructive git commands
            if any(command.startswith(cmd) for cmd in self.safe_bash_commands):
                return PermissionDecision(behavior="allow")
                
        if tool_name in ["code_search", "file_read", "file_write"]:
            return PermissionDecision(behavior="allow")  # Auto-approved ONLY IF validate_path passed
            
        return PermissionDecision(behavior="ask", reason=f"Tool '{tool_name}' requires user approval.")  # Default to ASK
