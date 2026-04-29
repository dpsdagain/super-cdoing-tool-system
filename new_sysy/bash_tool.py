"""
bash_tool.py — Shell Command Execution with Stall Detection.
"""
import os
import subprocess
import time
import re
import signal
import shlex
import threading
import logging
from queue import Queue, Empty
from pydantic import BaseModel, Field
from tool_registry import register_tool, validate_path
from bash_security import BashSecurityAnalyzer

logger = logging.getLogger(__name__)


_active_process_groups = []
_process_lock = threading.Lock()


def cleanup_active_processes():
    """Kill all remaining process groups (call on CLI exit)."""
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


@register_tool(name="bash", description="Execute a shell command. Use this for running tests, build scripts, or git commands.", input_schema=BashInput, is_read_only=False)
def bash_tool(command: str) -> str:
    """Execute a shell command with real-time stall detection and process-group termination."""

    try:
        # Security: semantic command validation
        security_result = BashSecurityAnalyzer.analyze(command)
        if not security_result["allowed"]:
            return f"Error: Command rejected for security reasons. {security_result['reason']}"

        # Path Sentinel: Prevent commands from targeting paths outside the workspace
        paths = re.findall(r'((?:[a-zA-Z]:\\|[/\\])[\w\s.-]+(?:[/\\][\w\s.-]+)*)', command)
        for p in paths:
            try:
                validate_path(p)
            except PermissionError as e:
                return f"Error: Command rejected. {str(e)}"
            except Exception:
                pass  # Not a valid path, ignore

        # Windows-specific process group creation
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            cmd_args = shlex.split(command)
        except Exception as e:
            return f"Error parsing command: {str(e)}"

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
            if time.time() - start_time > timeout:
                if os.name == 'nt':
                    os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                return "".join(full_output) + f"\n\nError: Command timed out after {timeout} seconds."

            try:
                line = output_queue.get(timeout=0.1)
                full_output.append(line)
                last_output_time = time.time()
            except Empty:
                if process.poll() is not None:
                    break
                
                if time.time() - last_output_time > 5.0:
                    current_text = "".join(full_output).strip()
                    tail = current_text[-100:]
                    if any(re.search(p, tail, re.IGNORECASE) for p in STALL_PATTERNS):
                        stall_detected = True
                        if os.name == 'nt':
                            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                        else:
                            process.terminate()
                        break
            
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        result = "".join(full_output)
        
        if stall_detected:
            return (
                f"{result}\n\n"
                f"[STALL DETECTED]: The command appears to be waiting for interactive input. "
                f"Process group terminated. Use a non-interactive flag (e.g., -y or --force) "
                f"or ask the user for help."
            )

        if not result.strip():
            return "Command executed successfully (no output)."

        return result

    except Exception as e:
        return f"Error executing command: {str(e)}"
