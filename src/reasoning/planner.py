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

Analyze the provided image (screenshot).
1. Identify the UI elements relevant to the goal.
2. Reason about the current state.
3. Determine the next single action to take.

Output your response strictly in the following JSON format:
```json
{{
  "analysis": "Brief reasoning here...",
  "relevant_elements": [
    {{ "id": 1, "description": "Login Button", "bbox": [x1, y1, x2, y2] }}
  ],
  "next_action": {{
    "type": "click", 
    "target_id": 1,
    "target_description": "Login Button",
    "coordinates": [x_center, y_center] 
  }}
}}
```
If the goal is achieved, set "type" to "finish".
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
