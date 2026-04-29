import os
import logging
from typing import List, Any
from tool_registry import current_engine

logger = logging.getLogger(__name__)

class WorkerAgent:
    """
    Isolated Sub-Agent (Worker) with its own context window.
    Designed for surgical sub-tasks (Research, Testing, Linter fixes).
    """
    def __init__(self, delegation_depth: int, main_model: str, permission_mode: str = "ASK"):
        self.depth = delegation_depth
        self.model = main_model
        # Workers inherit the permission mode from their coordinator
        from query_engine import QueryEngine
        self.engine = QueryEngine(
            session_id=f"worker_{os.getpid()}_{self.depth}",
            model_id=self.model,
            permission_mode=permission_mode,
            delegation_depth=self.depth
        )

    def solve(self, task: str) -> str:
        """Executes the sub-task and returns the final report."""
        logger.info(f"Worker (Depth {self.depth}) starting task: {task[:50]}...")
        
        # Guard: Workers run in the same process/thread by default.
        # We must temporarily swap the current_engine.instance so tools
        # called by the worker reference the correct engine instance.
        previous_instance = current_engine.instance
        current_engine.instance = self.engine
        try:
            answer, _ = self.engine.process_query(task)
            return answer
        finally:
            current_engine.instance = previous_instance

class Coordinator:
    """
    Multi-Agent Manager (Coordinator).
    Orchestrates workers, parallelizes tasks, and prevents Strategic Drift.
    """
    def __init__(self, engine: Any):
        self.manager_engine = engine # The main QueryEngine session
        self.max_workers = 3
        self.active_workers: List[WorkerAgent] = []

    def delegate(self, task: str, context_summary: str = "") -> str:
        """
        Spawns an isolated Worker to solve a sub-problem.
        (Mirrors Coordinator.ts:delegation logic)
        """
        if self.manager_engine.delegation_depth >= 3:
            return "Error: Maximum delegation depth reached."
        
        worker = WorkerAgent(
            delegation_depth=self.manager_engine.delegation_depth + 1,
            main_model=self.manager_engine.model_id,
            permission_mode=self.manager_engine.permission_manager.mode
        )
        
        # Hydrate the worker with relevant context from the manager
        full_task = f"ROLE: Assistant Worker. CONTEXT: {context_summary}\nTASK: {task}"
        result = worker.solve(full_task)
        return f"--- WORKER REPORT (Depth {worker.depth}) ---\n{result}\n--- END REPORT ---"