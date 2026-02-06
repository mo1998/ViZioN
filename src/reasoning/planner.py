from src.config import Config
from src.understanding.parser import WorldParser
import logging

logger = logging.getLogger(__name__)

class Planner:
    """
    The 'Reasoner' & 'Planner'.
    """
    def __init__(self, perception_module):
        self.eyes = perception_module
        self.memory = [] # Short-term memory of steps

    def plan_next_step(self, image_path, user_goal):
        """
        Analyzes the image and user goal to produce the next action.
        """
        # 1. Construct the prompt
        # We ask the VLM to perform both perception (detect) and reasoning (decide) in one go 
        # for efficiency, or we could split it. Let's do a Chain-of-Thought prompt.
        
        prompt = f"""
User Goal: "{user_goal}"

Analyze the provided image (screenshot) to determine the next step.
1. Identify UI elements relevant to the goal.
2. Reason about the current state and what needs to happen next.
3. Output the single next action.

Output STRICTLY in this JSON format:
```json
{{
  "analysis": "Reasoning about state and next step...",
  "relevant_elements": [
    {{ "id": 1, "description": "Element Name", "bbox": [x1, y1, x2, y2] }}
  ],
  "next_action": {{
    "type": "click" | "type" | "wait" | "finish",
    "target_id": 1, 
    "target_description": "Description of target",
    "coordinates": [x, y],
    "text_content": "text to type if applicable"
  }}
}}
```
- Use "type": "click" for clicking elements.
- Use "type": "type" for entering text.
- Use "type": "wait" if the system is processing.
- ONLY use "type": "finish" if the goal is COMPLETELY achieved and no further actions are needed.
"""
        
        # 2. See and Reason
        logger.info(f"Planning step for goal: {user_goal}")
        raw_response = self.eyes.see(image_path, prompt)
        
        # 3. Parse
        plan_data = WorldParser.parse_json(raw_response)
        
        if not plan_data:
            logger.error("Failed to parse plan from VLM response.")
            return {"type": "error", "raw_response": raw_response}
            
        # 4. Update Memory
        self.memory.append({
            "goal": user_goal,
            "plan": plan_data
        })
        
        return plan_data
