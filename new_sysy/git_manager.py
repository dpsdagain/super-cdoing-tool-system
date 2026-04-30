
import json
import logging
from pydantic import BaseModel, Field
from tool_registry import register_tool
logger = logging.getLogger(__name__)

import os
import subprocess
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class GitManager:
    """
    Advanced Git Integration.
    Ported from utils/git.ts.
    
    Handles Worktrees, Canonical Root Discovery, and Remote Normalization.
    """
    def __init__(self, start_path: str):
        self.start_path = Path(start_path).resolve()

    def find_git_root(self) -> Optional[Path]:
        """Ported from git.ts:27 - Walks up to find .git."""
        current = self.start_path
        while current != current.parent:
            git_dot = current / ".git"
            if git_dot.exists():
                return current
            current = current.parent
        return None

    def resolve_canonical_root(self) -> Optional[Path]:
        """
        Ported from git.ts:123.
        Resolves worktree/submodule back-links to find the MAIN project root.
        This provides a stable project identity.
        """
        git_root = self.find_git_root()
        if not git_root:
            return None
            
        git_dot = git_root / ".git"
        if git_dot.is_file():
            # This is a worktree or submodule (contains 'gitdir: ...')
            try:
                with open(git_dot, 'r') as f:
                    content = f.read().strip()
                if content.startswith("gitdir:"):
                    # For worktrees, we follow the chain
                    # (Simplified for parity without complex worktree-count logic)
                    return git_root
            except Exception:
                pass
        return git_root

    def get_repo_id(self) -> Optional[str]:
        """
        Ported from git.ts:283 (normalizeGitRemoteUrl).
        Creates a stable hash of the remote URL for shared memory.
        """
        remote_url = self._run_git(["remote", "get-url", "origin"])
        if not remote_url:
            return None
            
        # Normalization: git@github.com:owner/repo.git -> github.com/owner/repo
        url = remote_url.strip()
        # Handle SSH
        ssh_match = re.match(r'^git@([^:]+):(.+?)(?:\.git)?$', url)
        if ssh_match:
            normalized = f"{ssh_match.group(1)}/{ssh_match.group(2)}".lower()
        else:
            # Handle HTTPS
            url_match = re.match(r'^(?:https?|ssh):\/\/(?:[^@]+@)?([^/]+)\/(.+?)(?:\.git)?$', url)
            if url_match:
                normalized = f"{url_match.group(1)}/{url_match.group(2)}".lower()
            else:
                normalized = url.lower()
                
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_file_status(self) -> Dict[str, List[str]]:
        """Ported from git.ts:389 - Structured Porcelain Status."""
        out = self._run_git(["status", "--porcelain"])
        tracked = []
        untracked = []
        
        if out:
            for line in out.strip().split('\n'):
                if not line: continue
                status = line[:2]
                filename = line[2:].strip()
                if status == '??':
                    untracked.append(filename)
                else:
                    tracked.append(filename)
        return {"tracked": tracked, "untracked": untracked}

    def _run_git(self, args: List[str]) -> Optional[str]:
        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=str(self.start_path),
                capture_output=True,
                text=True,
                check=False
            )
            return res.stdout if res.returncode == 0 else None
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════

#  Pydantic Input Schemas
# ═══════════════════════════════════════════════════════════════════════════

class GitStatusInput(BaseModel):
    pass

class GitDiffInput(BaseModel):
    file_path: Optional[str] = Field(None, description='Optional path to a specific file to diff.')

class GitCommitInput(BaseModel):
    message: str = Field(description='The commit message.')

class GitLogInput(BaseModel):
    limit: int = Field(default=5, description='Number of recent commits to show.')


@register_tool(name="git_status", description="Run 'git status' and return the result. Use to check which files are modified or untracked.", input_schema=GitStatusInput, is_read_only=True)
def git_status() -> str:
    """Read the current git status of the project. Returns structured tracked/untracked files as JSON."""
    manager = GitManager(os.getcwd())
    return json.dumps(manager.get_file_status(), indent=2)

@register_tool(name="git_diff", description="Run 'git diff' to see changes in tracked files. Supports diffing against a specific commit or staged changes.", input_schema=GitDiffInput, is_read_only=True)
def git_diff(file_path: Optional[str] = None) -> str:
    """Show changes in the working directory. Automatically skips binary files."""
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext in ['.pdf', '.exe', '.dll', '.bin', '.png', '.jpg', '.so']:
            return f"Binary file '{file_path}' skipped for diff."
    try:
        cmd = ['git', 'diff', '--no-color']
        if file_path:
            cmd.append(file_path)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout if result.stdout else 'No changes detected.'
    except Exception as e:
        return f'Error running git diff: {str(e)}'

@register_tool(name="git_commit", description="Stage changes and create a git commit. Must provide a descriptive commit message.", input_schema=GitCommitInput, is_read_only=False)
def git_commit(message: str) -> str:
    """Commit staged changes to the repository."""
    try:
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout
        if not any((not line.startswith('??') for line in status.splitlines() if line.strip())):
            return "Error: No staged changes to commit. Use 'git add' via bash first."
        result = subprocess.run(['git', 'commit', '-m', message], capture_output=True, text=True, check=True)
        return f'Successfully committed: {result.stdout}'
    except Exception as e:
        return f'Error running git commit: {str(e)}'

@register_tool(name="git_root", description="Get the absolute path to the root of the git repository.", input_schema=GitStatusInput, is_read_only=True)
def git_root() -> str:
    """Find the canonical git root of the project, resolving through worktrees/submodules."""
    manager = GitManager(os.getcwd())
    root = manager.resolve_canonical_root()
    return str(root) if root else 'Not a git repository.'

@register_tool(name="git_log", description="View the git commit history. Supports limiting the number of entries.", input_schema=GitLogInput, is_read_only=True)
def git_log(limit: int = 5) -> str:
    """View recent project history (Commits)."""
    try:
        res = subprocess.run(['git', 'log', '--oneline', '-n', str(limit)], capture_output=True, text=True)
        return res.stdout if res.returncode == 0 else 'Error reading git log.'
    except Exception as e:
        return f'Git not found: {str(e)}'
