import os
import json
import logging
from typing import List, Any
from langchain_core.messages import (
    messages_from_dict,
    message_to_dict
)

logger = logging.getLogger(__name__)

class SessionJournal:
    """Live Journaling: Persist history turns line-by-line for crash resiliency."""
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)

    def save_session(self, session_id: str, messages: List[Any], append_only: bool = False):
        """Persist history turns line-by-line for crash resiliency."""
        file_path = os.path.join(self.session_dir, f"{session_id}.jsonl")
        mode = "a" if append_only else "w"
        
        with open(file_path, mode, encoding="utf-8") as f:
            for m in messages:
                serialized = message_to_dict(m)
                f.write(json.dumps(serialized) + "\n")
        logger.debug(f"Journaled {len(messages)} turns to {file_path}")

    def load_session(self, session_id: str) -> List[Any]:
        """Load conversation history from JSONL (with legacy .json migration)."""
        file_path = os.path.join(self.session_dir, f"{session_id}.jsonl")
        old_json_path = os.path.join(self.session_dir, f"{session_id}.json")

        if os.path.exists(old_json_path) and not os.path.exists(file_path):
            try:
                with open(old_json_path, "r", encoding="utf-8") as f:
                    messages = messages_from_dict(json.load(f))
                    self.save_session(session_id, messages)
                    return messages
            except Exception as e:
                logger.error(f"Migration failed for {session_id}: {e}")

        if not os.path.exists(file_path):
            return []

        messages = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        messages.append(messages_from_dict([json.loads(line)])[0])
                    except Exception as e:
                        logger.error(f"Failed to parse journal line: {e}")
        return messages

    def list_sessions(self) -> List[str]:
        """List available session IDs."""
        if not os.path.exists(self.session_dir):
            return []
        # Support both .json and .jsonl in listing
        sessions = set()
        for f in os.listdir(self.session_dir):
            if f.endswith(".json"): sessions.add(f.replace(".json", ""))
            elif f.endswith(".jsonl"): sessions.add(f.replace(".jsonl", ""))
        return sorted(list(sessions))
