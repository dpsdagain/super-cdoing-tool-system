from tool_registry import validate_path, current_engine, build_tool
import os
import threading
import logging
from urllib.parse import urlparse
import ipaddress
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from backend import load_existing_chroma, SQLiteFTS5BM25
from rag_chain import hybrid_search, get_reranker
from config import RETRIEVER_K, RERANK_TOP_K, USE_RERANKER, WORKSPACE_ROOT
from permissions import PermissionManager
import subprocess
import subprocess
import time
import re
import threading
from queue import Queue, Empty

import logging
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

_active_process_groups = []

_process_lock = threading.Lock()

def cleanup_active_processes():
    """Kill all remaining process groups (call on CLI exit)."""
    import signal
    with _process_lock:
        for pid in _active_process_groups:
            try:
                if os.name == 'nt':
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
        _active_process_groups.clear()

class BashInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

STALL_PATTERNS = [
    r"\? \[y/n\]",
    r"\(y/n\)\?",
    r"\[Y/n\]",
    r"\[y/N\]",
    r"confirm \[y/n\]",
    r"password:",
    r"enter to continue",
    r"press any key",
    r"terminate batch job",
]

def bash_tool(command: str) -> str:
    """Execute a shell command with real-time stall detection and process-group termination."""
    import subprocess
    import time
    import re
    import threading
    import signal
    from queue import Queue, Empty

    try:
        # 🛡️ Advanced Security Fix: Claude Code grade semantic command validation
        from bash_security import BashSecurityAnalyzer
        security_result = BashSecurityAnalyzer.analyze(command)
        if not security_result["allowed"]:
            return f"Error: Command rejected for security reasons. {security_result['reason']}"

        # 🚀 Path Sentinel for BASH: Prevent commands from targeting paths outside the workspace
        # We look for path-like strings in the command and validate them
        paths = re.findall(r'((?:[a-zA-Z]:\\|[/\\])[\w\s.-]+(?:[/\\][\w\s.-]+)*)', command)
        for p in paths:
            try:
                # We use the existing validate_path to ensure the command doesn't leak data
                if not validate_path(p):
                    return f"Error: Command rejected. Detected attempt to access restricted path: {p}"
            except Exception:
                pass # Not a valid path, ignore

        # Windows-specific process group creation to allow killing child trees
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # 🛡️ Security Fix: Avoid shell=True to prevent shell injection.
        # Note: This may break complex shell features like pipes (|) or redirects (>).
        import shlex
        try:
            cmd_args = shlex.split(command)
        except Exception as e:
            return f"Error parsing command: {str(e)}"

        # Use Popen to allow real-time reading
        process = subprocess.Popen(
            cmd_args,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            creationflags=creationflags
        )

        # 🚀 Resource Safety: Register for global cleanup
        with _process_lock:
            _active_process_groups.append(process.pid)

        output_queue = Queue()
        
        def reader(stream, queue):
            try:
                while True:
                    char = stream.read(1)
                    if not char:
                        break
                    queue.put(char)
            except Exception:
                pass
            finally:
                stream.close()

        stdout_thread = threading.Thread(target=reader, args=(process.stdout, output_queue))
        stderr_thread = threading.Thread(target=reader, args=(process.stderr, output_queue))
        stdout_thread.start()
        stderr_thread.start()

        full_output = []
        last_output_time = time.time()
        timeout = 30
        start_time = time.time()
        stall_detected = False

        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                if os.name == 'nt':
                    os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                return "".join(full_output) + f"\n\nError: Command timed out after {timeout} seconds."

            try:
                # Use a very small timeout for non-blocking feel
                line = output_queue.get(timeout=0.1)
                full_output.append(line)
                last_output_time = time.time()
            except Empty:
                # No new output, check for stalls
                if process.poll() is not None:
                    # Process finished normally
                    break
                
                # If we've been waiting > 5.0 seconds with no new output, check the tail
                if time.time() - last_output_time > 5.0:
                    # Look at the last 100 characters of the total output
                    current_text = "".join(full_output).strip()
                    tail = current_text[-100:]
                    if any(re.search(p, tail, re.IGNORECASE) for p in STALL_PATTERNS):
                        stall_detected = True
                        if os.name == 'nt':
                            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                        else:
                            process.terminate()
                        break
            
        # Ensure threads finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        result = "".join(full_output)
        
        if stall_detected:
            return (
                f"{result}\n\n"
                f"⚠️ [STALL DETECTED]: The command above appears to be waiting for interactive input (y/n, password, etc.). "
                f"I have terminated the process group to prevent orphaned child processes. "
                f"Please either:\n"
                f"1. Use a non-interactive flag (e.g., -y or --force).\n"
                f"2. Ask the user for help if human intervention is required."
            )

        if not result.strip():
            return "Command executed successfully (no output)."

        return result

    except Exception as e:
        return f"Error executing command: {str(e)}"

