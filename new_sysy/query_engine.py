# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals,too-many-branches,too-many-statements
import logging
import os
from collections import deque
from typing import Any, Deque, List, Optional
from dataclasses import dataclass, field
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from config import (
    DEFAULT_MODEL,
    LLM_TEMPERATURE,
    SESSION_DIR,
)

# Bound on QueryState.executed_actions so it doesn't grow unbounded
# across long sessions — only the recent window is needed for loop detection.
ACTION_HISTORY_MAX = 64
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
1. Research: Use 'list_directory' to see available files or 'code_search' to find relevant code snippets.
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
        # Bounded so .count() stays O(window) and memory doesn't grow forever.
        self.executed_actions: Deque[str] = deque(maxlen=ACTION_HISTORY_MAX)
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

    def reset_session(self, session_id: str) -> None:
        """Clear plan/scratchpad state and write an empty journal for `session_id`.

        Called by the CLI's `/clear` command. Per-turn state lives on QueryState
        and is rebuilt on the next process_query_stream call, so there is no
        per-turn state to clear here.
        """
        self.state.current_plan = "No plan defined yet."
        self.state.current_status = "Initializing..."
        self.journal.save_session(session_id, [], append_only=False)

    def _track_usage(self, response: Any) -> None:
        """Record token usage for `response` if usage_metadata is present."""
        usage_metadata = getattr(response, "usage_metadata", None)
        if not usage_metadata:
            return
        self.usage_tracker.track(
            self.config.model_id,
            {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "cache_read_input_tokens": usage_metadata.get(
                    "cache_read_input_tokens", 0
                ),
                "cache_creation_input_tokens": usage_metadata.get(
                    "cache_creation_input_tokens", 0
                ),
            },
        )

    def _stream_with_tools(self, messages: List[Any]):
        """Stream a tool-bound LLM response, yielding chunk events.

        Yields {"type": "chunk", ...} events as they arrive and finally
        {"type": "_aggregate", "response": <AIMessage>} so the caller can
        attach the assembled message to history.
        """
        full_response = None
        for chunk in self.llm_with_tools.stream(messages):
            full_response = chunk if full_response is None else full_response + chunk
            if chunk.content:
                yield {"type": "chunk", "content": chunk.content}
        yield {"type": "_aggregate", "response": full_response}

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
        """Surgically prunes context BEFORE every turn.

        Replaces the leading SystemMessage with a fresh instance rather than
        mutating its `content` in place — other holders of the original
        message (the journal, callers' history) should observe the value at
        the time of journaling, not the latest groomed version.
        """
        memory_instructions = self.context_rules.get_instructions()
        messages = self.context_manager.compact(messages, plan=plan)

        system_instructions = (
            f"{RAG_SYSTEM_PROMPT}\n\n[MASTER_PLAN]\n{plan}\n\n{memory_instructions}"
        )
        fresh_system = SystemMessage(content=system_instructions)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = fresh_system
        else:
            messages.insert(0, fresh_system)
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
            except KeyboardInterrupt:
                cleanup_active_processes()
                raise
            except Exception:
                cleanup_active_processes()
                orphans = state.messages[history_length_before_tick:]
                if orphans:
                    logger.warning(
                        "Tombstone Action: Purging %d orphaned messages.",
                        len(orphans),
                    )
                    for msg in orphans:
                        yield {
                            "type": "tombstone",
                            "content": f"Discarding orphaned message: {type(msg).__name__}",
                        }
                    state.messages = state.messages[:history_length_before_tick]
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
            messages, state.turn_count, list(state.executed_actions)
        )
        action = planner_decision.get("action", "tool")
        # Coerce defensively — planners sometimes emit non-string `input`.
        action_input = str(planner_decision.get("input", "") or "")

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
                response = None
                for event in self._stream_with_tools(messages):
                    if event["type"] == "_aggregate":
                        response = event["response"]
                    else:
                        yield event
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
                raise

            state.messages.append(response)
            self.journal.save_session(session_id, [response], append_only=True)
            self._track_usage(response)

            if not response.tool_calls:
                # Direct answer — terminate the turn. Content was already streamed
                # as chunks above, so we only emit the terminal "done" event.
                for hook in self.state.hooks:
                    hook.on_turn_end(response.content)
                state.is_terminal = True
                yield {"type": "done", "messages": state.messages}
                return

            yield from self.dispatcher.process_tool_calls(
                response.tool_calls, state, session_id
            )
            # Tool results are now in state.messages. The outer while-loop runs
            # the next tick, which re-invokes the planner + LLM with the new
            # context to either call more tools or answer.
            return

        if action == "final":
            # The Planner decided no tools are needed (e.g. casual math question).
            response = self.llm.invoke(state.messages)
            state.messages.append(response)
            self.journal.save_session(session_id, [response], append_only=True)
            self._track_usage(response)
            for hook in self.state.hooks:
                hook.on_turn_end(response.content)
            state.is_terminal = True
            yield {"type": "chunk", "content": response.content}
            yield {"type": "done", "messages": state.messages}
            return

        yield {
            "type": "error",
            "content": f"Unknown planner action: {action!r}.",
        }
