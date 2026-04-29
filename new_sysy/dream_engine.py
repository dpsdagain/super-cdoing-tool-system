import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DreamEngine:
    """
    Advanced Reflection & Memory Engine.
    Processes session transcripts to extract 'Golden Rules' for the project.
    """
    
    def __init__(self, root_dir: str):
        self.memory_dir = os.path.join(root_dir, ".claude", "memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.memory_file = os.path.join(self.memory_dir, "lessons_learned.md")

    def reflect_and_learn(self, transcript: List[Any], engine_model: str) -> str:
        """
        Simulates the background 'AutoDream' process (autoDream.ts:319).
        In a production system, this would be a hidden LLM call.
        """
        # Logic: Extract tool failures and user corrections from transcript
        lessons = []
        for turn in transcript:
            # Handle both dicts and LangChain objects
            msg_type = ""
            msg_content = ""
            
            if isinstance(turn, dict):
                msg_type = turn.get("type", "")
                msg_content = str(turn.get("content", ""))
            elif hasattr(turn, "type"):
                msg_type = turn.type
                msg_content = str(turn.content)
            
            if (msg_type == "tool" or msg_type == "tool_result") and "Error" in msg_content:
                lessons.append(f"- Avoided error: {msg_content[:100]}...")
            
            # Simple simulation: Extracting patterns from user feedback
            if (msg_type == "human" or msg_type == "user") and ("no" in msg_content.lower() or "wrong" in msg_content.lower()):
                lessons.append(f"- User Correction Observed: '{msg_content[:100]}'")

        if not lessons:
            return "No new patterns discovered this turn."

        # Persist to Memory Vault
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(f"\n### Reflection ({engine_model})\n")
            f.write("\n".join(lessons) + "\n")
        
        return f"AutoDream: Extracted {len(lessons)} project-specific lessons."

    def get_memories(self) -> str:
        """Loads 'Dreamed' lessons as a context injection."""
        if not os.path.exists(self.memory_file):
            return ""
        
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                content = f.read()
            return f"\n--- PROJECT MEMORIES (Lessons from previous dreams) ---\n{content}\n"
        except Exception:
            return ""
