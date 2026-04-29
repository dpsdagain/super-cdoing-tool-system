import os
import logging
import traceback
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

class ToolDispatcher:
    """Handles tool execution, permission gating, and result offloading."""
    def __init__(self, engine):
        self.engine = engine # Dependency on engine for current state and managers

    def execute_tool(self, name: str, args: Dict[str, Any], tool_id: str = "unknown") -> str:
        """Execute a tool by name with provided arguments."""
        if name not in AVAILABLE_TOOLS:
            return f"Error: Tool '{name}' not found."

        tool_info = AVAILABLE_TOOLS[name]
        try:
            # Phase 6: Dependency Injection
            if tool_info.get("requires_engine", False):
                args["_engine_instance"] = self.engine

            # Validate args against pydantic schema
            validated_args = tool_info["input_schema"](**args)
            result = tool_info["func"](**validated_args.model_dump())
            
            # Handle Synthetic State Interception
            if isinstance(result, str):
                if result.startswith("[PLAN_UPDATED]"):
                    self.engine.current_plan = result.replace("[PLAN_UPDATED] ", "")
                elif result.startswith("[STATUS_UPDATED]"):
                    self.engine.current_status = result.replace("[STATUS_UPDATED] ", "")

            # SYSTEM 2: Result Budget Gate (Archiving)
            offloaded_result = self.engine.context_manager.result_archive.offload_if_large(name, str(result))

            return offloaded_result
        except Exception as e:
            # Fix Reliability: Structured error feedback for the agent
            error_details = traceback.format_exc() if "DEBUG" in os.environ else str(e)
            return f"[TOOL_FAILURE] Tool '{name}' failed with error: {str(e)}. Please analyze the error and correct your arguments or approach."

    def process_tool_calls(self, tool_calls, state, session_id):
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "unknown")
            
            # 🛡️ F-14: Permission Guardian Gate
            perm_decision = self.engine.permission_manager.check_permission(tool_name, tool_args)
            if perm_decision.behavior == "deny":
                yield {"type": "status", "content": f"Blocked: {perm_decision.reason}"}
                result = f"Error: Permission denied. {perm_decision.reason}"
            elif perm_decision.behavior == "ask":
                if self.engine.permission_callback:
                    is_allowed = self.engine.permission_callback(tool_name, tool_args)
                    if not is_allowed:
                        yield {"type": "status", "content": "Blocked by User"}
                        result = "Error: Permission denied by user."
                        perm_decision.behavior = "deny"
                    else:
                        perm_decision.behavior = "allow"
                else:
                    yield {"type": "status", "content": "Blocked: Permission callback missing."}
                    result = "Error: Permission denied (no callback provided)."
                    perm_decision.behavior = "deny"

            if perm_decision.behavior == "allow":
                # ⏪ F-28: Automatic Pre-Edit Checkpointing
                # Phase 7: Expanded checkpointing list
                EDIT_TOOLS = ["file_write", "file_edit", "multi_file_edit", "notebook_edit", "memory_tool", "task_budget"]
                if tool_name in EDIT_TOOLS:
                    # For tools that might not have file_path in args (like memory_tool), we use canonical paths
                    target_path = tool_args.get("file_path", "")
                    if tool_name == "memory_tool": target_path = "MEMORY.md"
                    if tool_name == "task_budget": target_path = "budget_config.json"
                    
                    if target_path:
                        self.engine.history_manager.track_edit(target_path, tool_id)
                
                # HOOK: Pre-Tool Execution
                for hook in self.engine.hooks:
                    modified_args = hook.on_tool_call(tool_name, tool_args)
                    if modified_args is not None:
                        tool_args = modified_args

                yield {"type": "status", "content": f"Action: Running {tool_name}..."}
                result = self.execute_tool(tool_name, tool_args, tool_id=tool_id)
            
            # HOOK: Post-Tool Result
            for hook in self.engine.hooks:
                modified_result = hook.on_tool_result(tool_name, result)
                if modified_result is not None:
                    result = modified_result

            if isinstance(result, str) and result.startswith("[INTERRUPT_REQUIRED]"):
                yield {
                    "type": "interrupt", 
                    "content": result, 
                    "tool_name": tool_name, 
                    "tool_id": tool_id,
                    "messages": state.messages
                }
                return
            
            tool_msg = ToolMessage(content=str(result), tool_call_id=tool_id)
            state.messages.append(tool_msg)
            self.engine.journal.save_session(session_id, [tool_msg], append_only=True)
            yield {"type": "tool_result", "tool": tool_name, "result": result}
