import json
import os
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages both Short-Term (Contextual) and Long-Term (Experience) memory.
    Now enhanced with Semantic Search.
    """
    def __init__(self, memory_file="data/long_term_memory.json"):
        self.memory_file = memory_file
        self.short_term_history = []
        self.long_term_store = self._load_ltm()
        self.embedding_model = None
        self.last_action_coordinates = None # Initialize to None

    def clear_memory(self):
        """Clears the short-term memory (contextual history) and last action coordinates."""
        self.short_term_history = []
        self.last_action_coordinates = None 
        logger.info("MemoryManager: Short-term memory and last action cleared.")

    def _load_model(self):
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model for Semantic Memory...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None

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
        step_data: dict containing 'analysis', 'action', 'timestamp', 'expected_outcome'
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
            outcome = step.get("expected_outcome", "None")
            status = step.get("verification", "unverified")
            
            context.append(f"Step {i+1}: Performed '{act_type}' on '{target}'. Status: {status}. Expecting: {outcome}")
            
        return "\n".join(context)

    def get_last_action_coordinates(self):
        """Returns (x, y) of the last click, or None."""
        if not self.short_term_history:
            return None
        
        last_action = self.short_term_history[-1].get("action", {})
        if last_action.get("type") == "click":
            return last_action.get("coordinates")
        return None

    def _cosine_similarity(self, vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def retrieve_experience(self, goal, threshold=0.6):
        """
        Retrieves past successful strategies for a similar goal using Semantic Search.
        """
        self._load_model()
        if not self.embedding_model or not self.long_term_store:
            return None

        try:
            query_embedding = self.embedding_model.encode(goal)
            
            best_goal = None
            best_score = -1.0
            
            for stored_goal, data in self.long_term_store.items():
                # Check if we have a pre-computed embedding, else compute (and maybe save later)
                stored_emb = data.get("embedding")
                if not stored_emb:
                    stored_emb = self.embedding_model.encode(stored_goal).tolist()
                    # Cache it back to memory for next time
                    self.long_term_store[stored_goal]["embedding"] = stored_emb
                    
                score = self._cosine_similarity(query_embedding, stored_emb)
                
                if score > best_score:
                    best_score = score
                    best_goal = stored_goal
            
            logger.info(f"Memory Retrieval: Best match '{best_goal}' with score {best_score:.2f}")
            
            if best_score >= threshold:
                return self.long_term_store[best_goal]
                
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            
        return None

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
        
        # Compute embedding if model is loaded
        embedding = []
        if self.embedding_model:
            embedding = self.embedding_model.encode(goal).tolist()
        else:
            # Try loading it just for this save
            self._load_model()
            if self.embedding_model:
                embedding = self.embedding_model.encode(goal).tolist()

        self.long_term_store[goal] = {
            "last_updated": str(datetime.now()),
            "steps_summary": summary,
            "full_history": self.short_term_history,
            "embedding": embedding
        }
        self._save_ltm()
        logger.info(f"Saved experience for goal: '{goal}'")
