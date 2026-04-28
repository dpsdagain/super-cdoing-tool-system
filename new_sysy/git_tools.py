"""
git_tools.py — Git Integration Tools.
git_status, git_diff, git_commit, git_root, git_log.
"""
import os
import subprocess
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GitStatusInput(BaseModel):
    pass

class GitDiffInput(BaseModel):
    file_path: Optional[str] = Field(None, description='Optional path to a specific file to diff.')

class GitCommitInput(BaseModel):
    message: str = Field(description='The commit message.')

class GitLogInput(BaseModel):
    limit: int = Field(default=5, description='Number of recent commits to show.')


def git_status() -> str:
    """Read the current git status of the project. Returns structured tracked/untracked files as JSON."""
    from git_manager import GitManager
    import json
    manager = GitManager(os.getcwd())
    return json.dumps(manager.get_file_status(), indent=2)

def git_diff(file_path: Optional[str] = None) -> str:
    """Show changes in the working directory. Automatically skips binary files."""
    from pathlib import Path
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

def git_root() -> str:
    """Find the canonical git root of the project, resolving through worktrees/submodules."""
    from git_manager import GitManager
    manager = GitManager(os.getcwd())
    root = manager.resolve_canonical_root()
    return str(root) if root else 'Not a git repository.'

def git_log(limit: int = 5) -> str:
    """View recent project history (Commits)."""
    try:
        res = subprocess.run(['git', 'log', '--oneline', '-n', str(limit)], capture_output=True, text=True)
        return res.stdout if res.returncode == 0 else 'Error reading git log.'
    except Exception as e:
        return f'Git not found: {str(e)}'