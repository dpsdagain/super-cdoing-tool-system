# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals,too-many-branches,too-many-statements
import logging
import os
from typing import List, Any, Optional
from dataclasses import dataclass, field
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from config import (
    DEFAULT_MODEL,
    LLM_TEMPERATURE,
    SESSION_DIR,
    ENABLE_PROMPT_CACHING,
    ANTHROPIC_CACHE_BETA_HEADER,
)
from context_manager import ContextManager
from context_rules import ContextRules
from llm_factory import ModelFactory
from permissions import PermissionManager
from history_manager import HistoryManager
from dream_engine import DreamEngine
from estimator import ContextEstimator
from usage_tracker import UsageTracker
from rag_chain import CORE_INSTRUCTIONS as RAG_SYSTEM_PROMPT
from utils_messages import normalize_messages

from tools import AVAILABLE_TOOLS, cleanup_active_processes
from coordinator import Coordinator
from session_journal import SessionJournal
from planner import StrategicPlanner
from tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous AI software engineer operating on a Windows (win32) system. You have access to a set of tools to research, read, and edit code, as well as execute shell commands.

Your workflow:
1. Research: Use 'code_search' to find relevant code snippets.
2. Analyze: Use 'file_read' to examine the full content of relevant files.
3. Act: Use 'file_write' to create new files or 'file_edit' to make surgical changes.
4. Verify: Use 'bash' to run tests/commands.

Shell Environment (Windows):
- Use 'dir' instead of 'ls' if 'ls' is not available.
- Use 'type' instead of 'cat' if 'cat' is not available.
- For creating files, PREFER the 'file_write' tool over bash redirects.

Constraints:
- Always check your changes by running tests if available.
- Be surgical with 'file_edit'. Only replace the minimal necessary string.
- If you get stuck, explain why and ask for clarification.
- Do not assume a file exists without searching for it first.

You are operating in a local environment. Be careful with destructive commands."""


class QueryState:
    """Tracks lifecycle of a single user turn."""

    def __init__(self, messages: List[Any], max_turns: int = 15):
        self.messages = messages
        self.turn_count = 1
        self.max_turns = max_turns
        self.recovery_count = 0
        self.has_attempted_reactive_compact = False
        self.executed_actions: List[str] = []  # Detect repeating loops
        self.is_terminal = False


@dataclass
class EngineConfig:
    session_id: str
    model_id: str
    temperature: float
    permission_mode: str
    delegation_depth: int
    root_dir: str
    session_dir: str


@dataclass
class EngineState:
    current_plan: str = "No plan defined yet."
    current_status: str = "Initializing..."
    permission_callback: Optional[Any] = None
    hooks: List[Any] = field(default_factory=list)


class QueryEngine:
    """The Deconstructed High-Fidelity Agentic Engine Loop."""

    def __init__(
        self,
        session_id: str = "default",
        model_id: str = DEFAULT_MODEL,
        temperature: float = LLM_TEMPERATURE,
        permission_mode: str = "ASK",
        delegation_depth: int = 0,
        context_manager=None,
        permission_manager=None,
        history_manager=None,
        dream_engine=None,
    ):
        self.config = EngineConfig(
            session_id=session_id,
            model_id=model_id,
            temperature=temperature,
            permission_mode=permission_mode,
            delegation_depth=delegation_depth,
            root_dir=os.getcwd(),
            session_dir=SESSION_DIR,
        )
        self.state = EngineState()

        # Build extra headers for prompt caching if enabled
        self.extra_headers = {}
        if ENABLE_PROMPT_CACHING:
            self.extra_headers["anthropic-beta"] = ANTHROPIC_CACHE_BETA_HEADER

        self.llm = ModelFactory.create_model(
            self.config.model_id, temperature=self.config.temperature
        )

        # Isolated Strategy Planner
        self.planner = StrategicPlanner(
            ModelFactory.create_model(self.config.model_id, temperature=0.0)
        )

        # Subordinate Managers
        self.permission_manager = permission_manager or PermissionManager(
            mode=permission_mode
        )
        self.context_manager = context_manager or ContextManager()
        self.context_rules = ContextRules(self.config.root_dir)
        self.journal = SessionJournal(self.config.session_dir)
        self.history_manager = history_manager or HistoryManager(
            self.config.session_dir
        )
        self.usage_tracker = UsageTracker()
        self.dream_engine = dream_engine or DreamEngine(self.config.root_dir)
        self.coordinator = Coordinator(self)
        self.dispatcher = ToolDispatcher(self)

        # Initialize RAG Chain only when an indexed collection is available.
        from backend import load_existing_chroma
        from rag_chain import build_rag_chain

        try:
            self.rag_db = load_existing_chroma("default")
            self.rag_chain = (
                build_rag_chain(self.rag_db, model=self.config.model_id)
                if self.rag_db
                else None
            )
        except Exception:
            logger.exception("RAG initialization failed:")
            self.rag_db = None
            self.rag_chain = None

        # Pre-bind tools
        self.tools_metadata = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["input_schema"].model_json_schema(),
                },
            }
            for name, info in AVAILABLE_TOOLS.items()
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools_metadata)

    def process_query(
        self, query: str, session_id: str = "default"
    ) -> tuple[str, List[Any]]:
        """Non-streaming version of process_query_stream."""
        final_answer = ""
        last_messages = []
        for event in self.process_query_stream(query, session_id):
            if event["type"] == "chunk":
                final_answer += event["content"]
            elif event["type"] == "done":
                last_messages = event["messages"]
        return final_answer, last_messages

    def _apply_pre_query_grooming(self, messages: List[Any], plan: str) -> List[Any]:
        """Surgically prunes context BEFORE every turn."""
        memory_instructions = self.context_rules.get_instructions()
        messages = self.context_manager.compact(messages, plan=plan)

        system_instructions = (
            f"{RAG_SYSTEM_PROMPT}\n\n[MASTER_PLAN]\n{plan}\n\n{memory_instructions}"
        )
        if messages and isinstance(messages[0], SystemMessage):
            messages[0].content = system_instructions
        else:
            messages.insert(0, SystemMessage(content=system_instructions))
        return messages

    def process_query_stream(
        self,
        query: str,
        session_id: str = "default",
        messages: Optional[List[Any]] = None,
        max_turns: int = 15,
    ):
        """Advanced Agentic Loop (F-01)."""
        if messages is None or len(messages) == 0:
            import sys
            os_name = "Windows" if sys.platform == "win32" else "Linux/Unix"
            dynamic_prompt = SYSTEM_PROMPT.replace("Windows (win32)", os_name)
            sys_content = f"{dynamic_prompt}\n\n[MASTER_PLAN_SCRATCHPAD]\n{self.state.current_plan}\n[/MASTER_PLAN_SCRATCHPAD]"
            messages = [SystemMessage(content=sys_content)]

        if query:
            user_msg = HumanMessage(content=query)
            messages.append(user_msg)
            self.journal.save_session(session_id, [user_msg], append_only=True)

        state = QueryState(messages, max_turns)

        while not state.is_terminal:
            if state.turn_count > state.max_turns:
                yield {
                    "type": "status",
                    "content": "Reached max turn budget. Stopping.",
                }
                state.is_terminal = True
                break

            state.messages = self._apply_pre_query_grooming(
                state.messages, self.state.current_plan
            )
            history_length_before_tick = len(state.messages)

            try:
                for event in self._execute_tick(state, session_id):
                    yield event
            except Exception as e:
                cleanup_active_processes()
                orphans = state.messages[history_length_before_tick:]
                if orphans:
                    logger.warning(
                        f"Tombstone Action: Purging {len(orphans)} orphaned messages."
                    )
                    for msg in orphans:
                        yield {
                            "type": "tombstone",
                            "content": f"Discarding orphaned message: {type(msg).__name__}",
                        }
                    state.messages = state.messages[:history_length_before_tick]

                if isinstance(e, KeyboardInterrupt):
                    raise e
                logger.exception("Turn failure recovered:")

            state.turn_count += 1
        self.dream_engine.reflect_and_learn(state.messages, self.config.model_id)

    def _execute_tick(self, state: QueryState, session_id: str):
        """One iteration of the agent's logic engine."""
        messages = normalize_messages(state.messages)
        safety = ContextEstimator.check_flight_safety(self.config.model_id, messages)
        if safety["status"] == "CRITICAL":
            yield {
                "type": "status",
                "content": f"Context Critical ({safety['usage_pct']}%). Compacting...",
            }
            state.messages = self.context_manager.compact(state.messages)
            messages = normalize_messages(state.messages)
        elif safety["status"] == "WARNING":
            yield {
                "type": "status",
                "content": f"Warning: Context is {safety['usage_pct']}% full.",
            }

        # 1. Decision Layer
        planner_decision = self.planner.call_planner(
            messages, state.turn_count, state.executed_actions
        )
        action = planner_decision.get("action", "tool")
        action_input = planner_decision.get("input", "")

        action_sig = f"action:{action} input:{action_input[:40]}"
        if state.executed_actions.count(action_sig) >= 2:
            action = "tool"
        state.executed_actions.append(action_sig)

        # 2. Action Execution
        if action == "rag":
            yield {
                "type": "status",
                "content": f"RAG: Exploring knowledge for '{action_input}'...",
            }
            if self.rag_chain is None:
                yield {
                    "type": "status",
                    "content": "RAG unavailable: collection 'default' is empty or missing.",
                }
                action = "tool"
                state.messages.append(
                    SystemMessage(
                        content="[RAG_UNAVAILABLE] Collection 'default' is empty or missing. Use file/list/search tools instead."
                    )
                )
                return
            try:
                rag_result = ""
                for event in self.rag_chain.stream(
                    {"input": action_input, "chat_history": messages}
                ):
                    if isinstance(event, dict) and "answer" in event:
                        rag_result += event["answer"]

                memories = self.dream_engine.get_memories()
                rag_msg = SystemMessage(
                    content=f"[RAG_CONTEXT]\n{rag_result}\n\n{memories}\n[/RAG_CONTEXT]"
                )
                state.messages.append(rag_msg)
                self.journal.save_session(session_id, [rag_msg], append_only=True)
            except Exception:
                logger.exception("RAG failed:")

        elif action == "tool":
            yield {
                "type": "status",
                "content": f"Thinking: {planner_decision.get('reason', 'Processing...')}",
            }
            try:
                response = self.llm_with_tools.invoke(messages)
                state.messages.append(response)
                self.journal.save_session(session_id, [response], append_only=True)

                if not response.tool_calls:
                    state.is_terminal = True
                    action = "final"
                else:
                    yield from self.dispatcher.process_tool_calls(
                        response.tool_calls, state, session_id
                    )
            except Exception as e:
                if (
                    "context_length_exceeded" in str(e).lower()
                    and not state.has_attempted_reactive_compact
                ):
                    yield {
                        "type": "status",
                        "content": "Context Overloaded. Attempting Recovery...",
                    }
                    state.messages = self.context_manager.compact(
                        state.messages, plan=self.state.current_plan
                    )
                    state.has_attempted_reactive_compact = True
                    state.turn_count -= 1
                    return
                raise e

            if action == "tool" and not state.is_terminal:
                yield {
                    "type": "status",
                    "content": f"Action: {planner_decision.get('reason', 'Executing Tool...')}",
                }
                nudge = SystemMessage(
                    content="If external action is required, you MUST call a tool now."
                )
                full_response = None
                for chunk in self.llm_with_tools.stream(state.messages + [nudge]):
                    if full_response is None:
                        full_response = chunk
                    else:
                        full_response += chunk
                    if chunk.content:
                        yield {"type": "chunk", "content": chunk.content}

                state.messages.append(full_response)
                self.journal.save_session(session_id, [full_response], append_only=True)

                if (
                    hasattr(full_response, "usage_metadata")
                    and full_response.usage_metadata
                ):
                    u = full_response.usage_metadata
                    self.usage_tracker.track(
                        self.config.model_id,
                        {
                            "input_tokens": u.get("input_tokens", 0),
                            "output_tokens": u.get("output_tokens", 0),
                            "cache_read_input_tokens": u.get(
                                "cache_read_input_tokens", 0
                            ),
                            "cache_creation_input_tokens": u.get(
                                "cache_creation_input_tokens", 0
                            ),
                        },
                    )

                if not full_response.tool_calls:
                    if state.turn_count > 1:
                        action = "final"
                    else:
                        return

                yield from self.dispatcher.process_tool_calls(
                    full_response.tool_calls, state, session_id
                )

        if action == "final":
            final_prompt = """You are finishing the task. Provide a clear final answer and summarize any changes."""
            response = self.llm.invoke(
                state.messages + [SystemMessage(content=final_prompt)]
            )
            for hook in self.state.hooks:
                hook.on_turn_end(response.content)
            state.is_terminal = True
            yield {"type": "chunk", "content": response.content}
            yield {"type": "done", "messages": state.messages}
            return

        yield {"type": "error", "content": f"Safety limit reached."}
