import os
import threading
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from tool_registry import tool, validate_path, current_engine
from config import WORKSPACE_ROOT
logger = logging.getLogger(__name__)



class SwitchModelInput(BaseModel):
    model_id: str = Field(..., description="The ID of the model to switch to (e.g., 'ollama-cloud:gpt-oss:120b-cloud').")

class UpdatePlanInput(BaseModel):
    plan: str = Field(description='The updated step-by-step plan for the current task.')

class SetStatusInput(BaseModel):
    status: str = Field(description="Brief status message for the UI (e.g. 'Analyzing index...').")

class NotebookEditInput(BaseModel):
    file_path: str = Field(description='Path to the .ipynb file.')
    cell_id: str = Field(description='UUID or virtual ID (cell-0, cell-1) of the cell.')
    new_source: str = Field(description='New content for the cell.')
    edit_mode: str = Field(default='replace', description='replace, insert, or delete.')
    cell_type: str = Field(default='code', description='code or markdown.')

class AskUserInput(BaseModel):
    question: str = Field(description='The question to ask the user.')

class ArchVisualizerInput(BaseModel):
    directory: str = Field(default='.', description='The directory to visualize.')

class TaskBudgetInput(BaseModel):
    max_tokens: int = Field(description='The maximum number of tokens allowed for this task.')

def switch_model(model_id: str) -> str:
    """Reboots the agent with a new LLM engine. All conversation context is preserved."""
    return f'[MODEL_SWITCHED] {model_id}'

def update_plan(plan: str) -> str:
    """Synthetic Tool: Update your internal master plan. Use this to track progress, rejected ideas, and next steps."""
    return f'[PLAN_UPDATED] {plan}'

def set_status(status: str) -> str:
    """Sets the current activity status for the TUI (e.g. 'Analyzing index...')."""
    return f'[STATUS_UPDATED] {status}'

@tool
def cost_report() -> str:
    """Generate a high-precision session cost report (F-18 Parity). Shows token usage and USD cost."""
    try:
        if not current_engine.instance or not current_engine.instance.usage_tracker:
            return 'Error: Usage Tracker not initialized.'
        return current_engine.instance.usage_tracker.get_report()
    except Exception as e:
        return f'Error generating cost report: {str(e)}'

@tool
def system_doctor() -> str:
    """Perform a full environmental diagnostic check (F-44 Parity). Audits binaries, network, and workspace toxicity."""
    from doctor import SystemDoctor
    import json
    try:
        report = SystemDoctor.audit()
        return f'--- System Health Report ---\n{json.dumps(report, indent=2)}'
    except Exception as e:
        return f'Error running diagnostics: {str(e)}'

@tool
def notebook_edit(file_path: str, cell_id: str, new_source: str, edit_mode: str='replace', cell_type: str='code') -> str:
    """Surgically edit a Jupyter Notebook cell (F-12 Parity). Resets execution state on modified cells."""
    from notebook_utils import NotebookMutator
    try:
        file_path = validate_path(file_path)
    except Exception as e:
        return str(e)
    return NotebookMutator.edit(file_path, cell_id, new_source, edit_mode, cell_type)

def undo_last_edit(message_id: str) -> str:
    """Roll back file changes made in a specific turn (F-28 Parity)."""
    try:
        if not current_engine.instance or not current_engine.instance.history_manager:
            return 'Error: History Manager not initialized.'
        reverted = current_engine.instance.history_manager.rollback(message_id)
        if not reverted:
            return f'No changes found to undo for turn ID: {message_id}'
        return f'Successfully reverted changes for {len(reverted)} files. Turn ID: {message_id}'
    except Exception as e:
        return f'Error performing undo: {str(e)}'

def linter_tool(file_path: str) -> str:
    """Run a basic linter check on a file."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if file_path.endswith('.py'):
        try:
            result = subprocess.run(['flake8', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return 'No linting errors found.'
            return result.stdout
        except FileNotFoundError:
            return 'Linter (flake8) not installed. Use bash tool to run a specific linter.'
    return 'Linter tool currently only supports Python (.py) files natively. Use bash for others.'

def memory_tool(fact: str) -> str:
    """Save a memory or preference to a persistent MEMORY.md file."""
    try:
        memory_path = os.path.join(str(WORKSPACE_ROOT), 'MEMORY.md')
        with open(memory_path, 'a', encoding='utf-8') as f:
            f.write(f'- {fact}\n')
        return f'Memory saved successfully to {memory_path}.'
    except Exception as e:
        return f'Error saving memory: {str(e)}'

def arch_visualizer(directory: str='.') -> str:
    """Generate a high-level architecture overview in Mermaid format."""
    try:
        directory = validate_path(directory)
    except PermissionError as e:
        return str(e)
    import ast
    mermaid = ['classDiagram']
    for root, _, files in os.walk(directory):
        if any((exc in root for exc in ['__pycache__', 'venv', '.git'])):
            continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        node = ast.parse(f.read())
                    for sub in node.body:
                        if isinstance(sub, ast.ClassDef):
                            mermaid.append(f'    class {sub.name} {{')
                            for item in sub.body:
                                if isinstance(item, ast.FunctionDef):
                                    mermaid.append(f'        +{item.name}()')
                            mermaid.append('    }')
                except Exception:
                    continue
    if len(mermaid) == 1:
        return 'No classes found to visualize.'
    return 'Architecture Diagram (Mermaid):\n\n```mermaid\n' + '\n'.join(mermaid) + '\n```'

def undercover_mode(text: str) -> str:
    """Strip AI identifiers and local paths from text for professional output."""
    import re
    markers = ['As an AI.*?model,', 'I am Claude', 'Anthropic', 'Assistant', "I don't have feelings"]
    cleaned = text
    for marker in markers:
        cleaned = re.sub(marker, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub('[a-zA-Z]:\\\\[\\\\\\w\\s.-]+', '[REDACTED_PATH]', cleaned)
    cleaned = re.sub('/(?:[\\w.-]+/)+[\\w.-]+', '[REDACTED_PATH]', cleaned)
    return cleaned.strip()

def task_budget(max_tokens: int) -> str:
    """Set or check a token budget for the current task."""
    try:
        with open('budget_config.json', 'w') as f:
            import json
            json.dump({'max_tokens': max_tokens}, f)
        return f'Budget set to {max_tokens} tokens. Agent will now monitor usage against this limit.'
    except Exception as e:
        return f'Error setting budget: {str(e)}'

def ask_user(question: str) -> str:
    """Pause execution and ask the human user a question."""
    return f'[INTERRUPT_REQUIRED] The agent needs human input: {question}'