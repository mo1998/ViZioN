from src.config import Config
from src.understanding.parser import WorldParser
from src.reasoning.memory import MemoryManager
from src.utils.vision import VisionUtils
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class Planner:
    """
    The 'Reasoner' & 'Planner'.
    """
    def __init__(self, perception_module):
        self.eyes = perception_module
        self.memory_manager = MemoryManager()

    def plan_next_step(self, image_source, user_goal):
        """
        Analyzes the image and user goal to produce the next action.
        """
        # 0. Pre-process Image with Visual Memory
        # Load image if it's a path, otherwise use object
        if isinstance(image_source, str):
            image = Image.open(image_source)
        else:
            image = image_source
            
        last_coords = self.memory_manager.get_last_action_coordinates()
        if last_coords:
            # Mark the last action location for the VLM
            image = VisionUtils.mark_action(image, last_coords)

        # 1. Retrieve Context
        short_term_context = self.memory_manager.get_short_term_context()
        experience = self.memory_manager.retrieve_experience(user_goal)
        
        hint_text = ""
        if experience:
            steps_summary = ", ".join(experience.get("steps_summary", [])[:5])
            hint_text = f"HINT (Memory): You have achieved this before by: {steps_summary}..."

        # 2. Construct the prompt
        prompt = f"""
User Goal: "{user_goal}"

Context (Recent History):
{short_term_context}

{hint_text}

Analyze the provided image (screenshot) to determine the next step.
The RED CROSS (if present) indicates where you clicked in the previous step.

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
        
        # 3. See and Reason
        logger.info(f"Planning step for goal: {user_goal}")
        raw_response = self.eyes.see(image, prompt)
        
        # 4. Parse
        plan_data = WorldParser.parse_json(raw_response)
        
        if not plan_data:
            logger.error("Failed to parse plan from VLM response.")
            return {"type": "error", "raw_response": raw_response}
            
        # 5. Update Memory
        step_record = {
            "timestamp": str(logging.Formatter().converter(logging.time.time())),
            "analysis": plan_data.get("analysis"),
            "action": plan_data.get("next_action")
        }
        self.memory_manager.add_short_term(step_record)
        
        # Save "Experience" if finished successfully
        if plan_data.get("next_action", {}).get("type") == "finish":
            self.memory_manager.save_successful_run(user_goal)
        
        return plan_data
