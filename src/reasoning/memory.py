import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages both Short-Term (Contextual) and Long-Term (Experience) memory.
    """
    def __init__(self, memory_file="data/long_term_memory.json"):
        self.memory_file = memory_file
        self.short_term_history = []
        self.long_term_store = self._load_ltm()

    def _load_ltm(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
                return {}
        return {}

    def _save_ltm(self):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w") as f:
                json.dump(self.long_term_store, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def add_short_term(self, step_data):
        """
        Adds a step to working memory.
        step_data: dict containing 'analysis', 'action', 'timestamp'
        """
        self.short_term_history.append(step_data)

    def get_short_term_context(self, limit=5):
        """
        Returns a formatted string of the last N steps.
        """
        if not self.short_term_history:
            return "No previous actions in this session."
            
        context = []
        for i, step in enumerate(self.short_term_history[-limit:]):
            action = step.get("action", {})
            act_type = action.get("type")
            target = action.get("target_description", "unknown")
            context.append(f"Step {i+1}: Performed '{act_type}' on '{target}'.")
            
        return "\n".join(context)

    def get_last_action_coordinates(self):
        """Returns (x, y) of the last click, or None."""
        if not self.short_term_history:
            return None
        
        last_action = self.short_term_history[-1].get("action", {})
        if last_action.get("type") == "click":
            return last_action.get("coordinates")
        return None

    def retrieve_experience(self, goal):
        """
        Retrieves past successful strategies for a similar goal.
        """
        # Simple exact matching for now. Semantic search would be better for complex apps.
        return self.long_term_store.get(goal)

    def save_successful_run(self, goal):
        """
        Saves the current short-term history as a successful experience for this goal.
        """
        if not self.short_term_history:
            return
            
        summary = [
            f"{s['action'].get('type')} on {s['action'].get('target_description')}" 
            for s in self.short_term_history
        ]
        
        self.long_term_store[goal] = {
            "last_updated": str(datetime.now()),
            "steps_summary": summary,
            "full_history": self.short_term_history
        }
        self._save_ltm()
        logger.info(f"Saved experience for goal: '{goal}'")
