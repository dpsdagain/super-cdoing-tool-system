# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-branches,too-many-statements
import sys
import time

# FAST-PATH DISPATCHER — handle the trivial flags before touching heavy imports.
if len(sys.argv) > 1:
    fast_cmd = sys.argv[1]
    if fast_cmd in ["--version", "-v"]:
        print("Antigravity Agent v2.4.1 (Core)")
        sys.exit(0)
    elif fast_cmd == "--help":
        print(
            "Usage: antigravity [COMMAND] [OPTIONS]\n\nCommands:\n  doctor   Audit system health\n  cost     View session costs\n  clear    Reset session\n\nOptions:\n  --model  Specify LLM model\n  --v      Show version"
        )
        sys.exit(0)

import logging
import argparse
import os
import threading

# Lazy-loading pointers
QueryEngine = None
cleanup_active_processes = None

# Idempotency guard: process-wide, not per-instance — atexit registration must
# only happen once even if multiple AgentCLI instances are constructed
# (e.g. by tests).
_atexit_registered = False


def get_console():
    """Lazily initialize the rich console."""
    if not hasattr(get_console, "instance"):
        from rich.console import Console as RichConsole
        from rich.theme import Theme

        theme = Theme(
            {
                "info": "dim cyan",
                "warning": "magenta",
                "danger": "bold red",
                "user": "bold green",
                "agent": "bold blue",
                "status": "italic yellow",
                "tool": "bold cyan",
                "diff.add": "green",
                "diff.remove": "red",
                "model": "dim blue",
                "pill.think": "bold white on #4e4e4e",
                "pill.tool": "bold white on #005f5f",
            }
        )
        get_console.instance = RichConsole(theme=theme)
    return get_console.instance


# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.ERROR)

# Constants for UI
DOT = "●"
INDENT = "  "
PIPE = "│ "
AGENT_ICON = "🤖"
USER_ICON = "👤"
TOOL_ICON = "🛠️"
THINK_ICON = "🧠"


class AgentCLI:
    def __init__(self, session_id: str, model_id: str, permission_mode: str = "ASK"):
        self.session_id = session_id
        self.model_id = model_id
        self.permission_mode = permission_mode
        self.engine = None
        self.is_ready = False
        self.boot_error = None
        self.history = []

        # UI components are loaded only when AgentCLI is instantiated
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.styles import Style as PtStyle
        from prompt_toolkit.shortcuts import CompleteStyle

        # Start Background Hydration immediately
        self._boot_thread = threading.Thread(target=self._boot_engine, daemon=True)
        self._boot_thread.start()

        # Instant UI Setup
        os.makedirs(".sessions", exist_ok=True)
        self.prompt_session = PromptSession(
            history=FileHistory(os.path.join(".sessions", f"{session_id}_history.txt")),
            complete_style=CompleteStyle.MULTI_COLUMN,
        )
        self.completer = WordCompleter(
            ["/model", "/session", "/help", "exit", "quit", "clear"], ignore_case=True
        )

        self.pt_style = PtStyle.from_dict(
            {
                "prompt": "#00ff00 bold",
            }
        )

    def _boot_engine(self):
        """Heavy lifting happens here in the background."""
        global QueryEngine, cleanup_active_processes, _atexit_registered
        try:
            if QueryEngine is None:
                from query_engine import QueryEngine
            if cleanup_active_processes is None:
                from tools import cleanup_active_processes

            if not _atexit_registered:
                import atexit

                atexit.register(cleanup_active_processes)
                _atexit_registered = True

            self.engine = QueryEngine(
                model_id=self.model_id, permission_mode=self.permission_mode
            )
            self.engine.state.permission_callback = self.permission_callback
            self.history = self.engine.journal.load_session(self.session_id)
            self.is_ready = True
        except Exception as e:
            # We don't crash the UI thread, just log the error
            import traceback

            self.boot_error = f"{str(e)}\n{traceback.format_exc()}"
            logging.exception("Engine Boot Failed:")

    def _consume_stream_events(self, query, console):
        """Drive one pass of the engine stream, rendering events to the console.

        Returns the interrupt event dict if the engine asked for human input;
        returns None when the stream finishes normally. The caller is
        responsible for prompting the user and resuming.
        """
        active_status = None
        has_printed_pipe = False

        for event in self.engine.process_query_stream(
            query, session_id=self.session_id, messages=self.history
        ):
            event_type = event["type"]

            if event_type == "status":
                content = event["content"]
                icon = TOOL_ICON if "Executing" in content else THINK_ICON
                status_msg = f"{INDENT}[pill.think] {icon} {content} [/]"
                if active_status:
                    active_status.update(status_msg)
                else:
                    active_status = console.status(status_msg)
                    active_status.start()

            elif event_type == "chunk":
                if active_status:
                    active_status.stop()
                    active_status = None
                if not has_printed_pipe:
                    console.print(f"{PIPE}", end="")
                    has_printed_pipe = True
                content = event["content"]
                console.print(content.replace("\n", f"\n{PIPE}"), end="")

            elif event_type == "tombstone":
                if active_status:
                    active_status.stop()
                    active_status = None
                console.print(f"\n{INDENT}[warning] {event['content']} [/]")

            elif event_type == "interrupt":
                if active_status:
                    active_status.stop()
                return event

            elif event_type == "done":
                if active_status:
                    active_status.stop()
                    active_status = None
                self.history = event["messages"]
                console.print("\n")

            elif event_type == "error":
                if active_status:
                    active_status.stop()
                    active_status = None
                console.print(
                    f"\n{INDENT}[danger]Error: {event['content']}[/danger]"
                )

        return None

    def permission_callback(self, tool_name: str, tool_args: dict) -> bool:
        console = get_console()
        from rich.table import Table
        from rich.box import SIMPLE
        from rich.align import Align

        console.print("\n")
        console.print(
            f"{INDENT}[pill.tool] 🔑 PERMISSION [/] [bold white]Agent wants to use {tool_name}[/]"
        )
        table = Table(
            border_style="warning", box=SIMPLE, expand=False, show_header=False
        )
        for k, v in tool_args.items():
            table.add_row(f"[tool]{k}[/]", str(v))
        console.print(Align.left(table, pad=True))
        answer = (
            console.input(f"{INDENT}[bold yellow]Allow? (y/n) [y]: [/]").strip().lower()
        )
        return answer in ["", "y", "yes"]

    def print_help(self):
        console = get_console()
        from rich.table import Table
        from rich.box import ROUNDED
        from rich.align import Align

        table = Table(title="Interactive Commands", border_style="info", box=ROUNDED)
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        table.add_row("/model <ID>", "Switch to a different LLM model")
        table.add_row("/session", "Show current session info")
        table.add_row("/help", "Show this help menu")
        table.add_row("clear", "Clear the terminal screen")
        table.add_row("exit | quit", "Save and exit the session")
        console.print(Align.left(table, pad=True))

    def run(self):
        console = get_console()
        from rich.panel import Panel
        from rich.box import ROUNDED
        from rich.text import Text as RichText
        from prompt_toolkit.formatted_text import HTML

        console.print("\n")
        console.print(
            Panel(
                RichText.from_markup(
                    f"Session: [bold cyan]{self.session_id}[/]\nModel: [bold blue]{self.model_id}[/]\nHydration: [bold yellow]Background Tick...[/]"
                ),
                title=f"[bold agent]{AGENT_ICON} Autonomous Engineering Agent[/]",
                border_style="agent",
                box=ROUNDED,
                padding=(1, 2),
                expand=False,
            )
        )
        console.print(
            f"{INDENT}[info]Type '/help' for commands. Greedy UI Boot active.[/info]\n"
        )

        while True:
            try:
                query = self.prompt_session.prompt(
                    HTML(f"<b><ansigreen>{USER_ICON} User</ansigreen></b> > "),
                    completer=self.completer,
                    style=self.pt_style,
                ).strip()

                if not query:
                    continue
                if query.lower() in ["exit", "quit"]:
                    break
                if query.lower() in ["clear", "/clear"]:
                    console.clear()
                    if self.is_ready and self.engine:
                        self.history = []
                        self.engine.reset_session(self.session_id)
                        console.print(
                            f"{INDENT}[status]✨ Agent memory completely wiped.[/status]\n"
                        )
                    continue

                if query.startswith("/model "):
                    new_model = query.replace("/model ", "").strip()
                    with console.status(
                        f"{INDENT}[status]Switching to {new_model}...[/status]"
                    ):
                        while not self.is_ready:
                            time.sleep(0.1)  # Wait if still booting
                        self.engine = QueryEngine(
                            model_id=new_model,
                            permission_mode=self.engine.permission_manager.mode,
                        )
                        self.engine.state.permission_callback = self.permission_callback
                        self.model_id = new_model
                    console.print(
                        f"{INDENT}✅ Model switched to [bold blue]{self.model_id}[/]"
                    )
                    continue

                # --- ENSURE ENGINE IS READY ---
                if not self.is_ready:
                    if self.boot_error:
                        console.print(
                            f"\n{INDENT}[danger]CRITICAL: Engine failed to boot.[/danger]"
                        )
                        console.print(f"{INDENT}[dim]{self.boot_error}[/dim]\n")
                        break  # Exit the loop and end the session

                    with console.status(
                        f"{INDENT}[status]Warming up query engine...[/status]"
                    ):
                        while not self.is_ready and not self.boot_error:
                            time.sleep(0.1)

                        if self.boot_error:
                            console.print(
                                f"\n{INDENT}[danger]CRITICAL: Engine failed to boot.[/danger]"
                            )
                            console.print(f"{INDENT}[dim]{self.boot_error}[/dim]\n")
                            break  # Exit the loop and end the session

                # --- AGENT TURN (iterative; one iteration per interrupt) ---
                from rich.panel import Panel
                from rich.box import ROUNDED
                from rich.align import Align
                from rich.rule import Rule
                from langchain_core.messages import ToolMessage

                console.print(Rule(style="agent"))
                console.print(
                    f"\n{DOT} [agent]Agent[/agent] [model]({self.model_id})[/model]\n"
                )

                pending_query = query
                while True:
                    interrupted_event = self._consume_stream_events(
                        pending_query, console
                    )
                    if interrupted_event is None:
                        break

                    question = interrupted_event["content"].replace(
                        "[INTERRUPT_REQUIRED] The agent needs human input: ", ""
                    )
                    console.print("\n")
                    console.print(
                        Align.left(
                            Panel(
                                f"[bold yellow]🙋 Question:[/bold yellow] {question}",
                                border_style="warning",
                                box=ROUNDED,
                                width=min(console.width - 4, 100),
                            ),
                            pad=True,
                        )
                    )

                    answer = self.prompt_session.prompt(
                        HTML(
                            f"<b><ansiyellow>{USER_ICON} Answer</ansiyellow></b> > "
                        ),
                        style=self.pt_style,
                    ).strip()

                    self.history.append(
                        ToolMessage(
                            content=f"User replied: {answer}",
                            tool_call_id=interrupted_event["tool_id"],
                        )
                    )
                    console.print(
                        f"\n{DOT} [agent]Agent[/agent] [dim](Resuming...)[/dim]\n"
                    )
                    # Resume with no new query — the next stream is driven by
                    # the appended ToolMessage in self.history.
                    pending_query = None

                # --- POST-TURN RENDERS ---
                self.engine.journal.save_session(self.session_id, self.history)
                console.print(Rule(style="dim"))

            except (KeyboardInterrupt, EOFError):
                console.print(
                    f"\n{INDENT}[warning]Exiting and saving session...[/warning]"
                )
                if self.engine:
                    self.engine.journal.save_session(self.session_id, self.history)
                break
            except Exception as e:
                logging.exception("Unexpected error in CLI loop:")
                console.print(f"\n{INDENT}[danger]Unexpected Error: {str(e)}[/danger]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    cli = AgentCLI(
        args.session or "default_session",
        args.model or "ollama-cloud:gpt-oss:120b-cloud",
    )
    cli.run()
