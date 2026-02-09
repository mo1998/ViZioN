from src.config import Config
from src.understanding.parser import WorldParser
from src.reasoning.memory import MemoryManager
from src.utils.vision import VisionUtils
from PIL import Image
import logging
import re

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
2. Reason about the current state. Consider if the goal has already been achieved based on the visual evidence and the Recent History.
3. If the goal is achieved, the next action should be "finish".
4. If not, determine the next optimal step.
5. Define the EXPECTED VISUAL OUTCOME of the action (what should change?).

Output STRICTLY in this JSON format:
```json
{{
  "analysis": "Reasoning about current state and how to achieve the user goal. Mention if previous actions seem to have failed or if we are repeating steps.",
  "next_action": {{
    "type": "click" | "type" | "scroll_up" | "scroll_down" | "wait" | "finish" | "error",
    "target_description": "Brief description of the element to interact with",
    "text_content": "Text to type (only if type action)",
    "bbox": [x1, y1, x2, y2]
  }},
  "relevant_elements": [
    {{ 
      "id": 1, 
      "description": "Element Name", 
      "type": "button" | "link" | "text_input" | "checkbox" | "radio" | "dropdown" | "area" | "icon" | "image" | "generic",
      "text": "visible text content (if any)",
      "bbox": [x1, y1, x2, y2] 
    }}
  ],
  "expected_outcome": "Description of what should happen after the next action"
}}
```
"""
        
        # 3. See and Reason
        logger.info(f"Planning step for goal: {user_goal}")
        raw_response = self.eyes.see(image, prompt)
        
        # 4. Parse
        plan_data = WorldParser.parse_json(raw_response)
        
        if not plan_data:
            logger.error("Failed to parse plan from VLM response.")
            return {"type": "error", "raw_response": raw_response}
            
        # --- Planner's Decision Logic ---
        # 1. Trust the VLM's suggested action first if it's well-formed
        vlm_action = plan_data.get("next_action", {})
        if vlm_action and vlm_action.get("type") in ["click", "type", "scroll_up", "scroll_down", "wait", "finish"]:
            next_action = vlm_action
            if "bbox" in next_action:
                next_action["coordinates"] = VisionUtils.get_center_coords(next_action["bbox"])
            
            # Match target_description back to an element ID if possible
            if "target_id" not in next_action:
                for elem in plan_data.get("relevant_elements", []):
                    if elem.get("description") == next_action.get("target_description"):
                        next_action["target_id"] = elem.get("id")
                        break

            logger.info(f"Using VLM suggested action: {next_action['type']} on '{next_action.get('target_description')}'")
        else:
            # 2. Fallback to heuristic logic if VLM didn't provide a clear action
            next_action = {"type": "wait", "target_description": "No clear action identified yet"}
            relevant_elements = plan_data.get("relevant_elements", [])
            user_goal_lower = user_goal.lower()
            
            best_match = None
            best_score = -1
            
            for element in relevant_elements:
                element_text = f"{element.get('description', '')} {element.get('text', '')}".lower()
                element_type = element.get('type', 'generic').lower()
                
                score = 0
                # Keywords indicating interaction
                interact_keywords = ["click", "open", "go to", "navigate", "search", "type", "enter", "fill"]
                if any(k in user_goal_lower for k in interact_keywords):
                    # Base score for naturally interactive types
                    if element_type in ["button", "link", "menu_item", "icon", "text_input"]:
                        score += 3
                    elif element_type in ["area", "image"]:
                        score += 1
                    
                    # Check for direct keyword matches
                    for keyword in user_goal_lower.split():
                        if len(keyword) > 2 and keyword in element_text:
                            # Generic elements get lower matching weight to avoid false positives
                            score += 1 if element_type == "generic" else 2

                if score > best_score:
                    best_score = score
                    best_match = element
            
            # Only proceed if we have a reasonably confident match
            if best_match and best_score >= 3:
                action_type = "click"
                # Improved type detection: "search" often implies typing
                type_keywords = ["type", "enter", "fill", "search"]
                if any(k in user_goal_lower for k in type_keywords) and best_match.get("type") == "text_input":
                    action_type = "type"

                next_action = {
                    "type": action_type,
                    "target_id": best_match.get("id"),
                    "target_description": best_match.get("description"),
                    "coordinates": VisionUtils.get_center_coords(best_match.get("bbox", [0,0,0,0]))
                }
                if action_type == "type":
                    # Extract what to type/search for
                    text_to_type = user_goal
                    # More robust extraction: find text between quotes or after keywords
                    quote_match = re.search(r"['\"](.*?)['\"]", user_goal)
                    if quote_match:
                        text_to_type = quote_match.group(1)
                    else:
                        for k in ["type ", "enter ", "fill ", "search for ", "search "]:
                            if k in user_goal_lower:
                                text_to_type = user_goal_lower.split(k, 1)[-1].strip()
                                # Strip common "into the..." suffix
                                text_to_type = re.split(r"\s+into\s+", text_to_type, flags=re.IGNORECASE)[0]
                                break
                    next_action["text_content"] = text_to_type
                
                logger.info(f"Heuristic selected action: {next_action['type']} on '{next_action['target_description']}'")
            else:
                logger.warning("Planner could not find a confident action based on goal. Defaulting to 'wait'.")

        # 3. Loop Detection & Anti-Stuck Logic
        last_context = self.memory_manager.short_term_history[-3:] if self.memory_manager.short_term_history else []
        if len(last_context) >= 2:
            identical_count = 0
            for prev_step in reversed(last_context):
                prev_action = prev_step.get("action", {})
                if (prev_action.get("type") == next_action.get("type") and 
                    prev_action.get("target_description") == next_action.get("target_description")):
                    identical_count += 1
                else:
                    break
            
            if identical_count >= 2:
                logger.warning(f"Loop detected! Action '{next_action['type']}' on '{next_action['target_description']}' repeated {identical_count} times.")
                
                # Special handling for "once" in goal
                if " once" in user_goal.lower() or user_goal.lower().endswith(" once"):
                    logger.info("Goal specifies 'once' and action was already attempted. Finishing.")
                    next_action = {"type": "finish", "target_description": "Goal 'once' likely achieved."}
                
                # If we were clicking, maybe we should try to type or wait?
                elif next_action["type"] == "click" and "search" in user_goal.lower():
                     logger.info("Attempting to switch from 'click' to 'type' to break loop.")
                     next_action["type"] = "type"
                     # Re-extract text content if missing
                     if "text_content" not in next_action:
                         text_to_type = user_goal
                         for k in ["search for ", "search "]:
                            if k in user_goal.lower():
                                text_to_type = user_goal.lower().split(k, 1)[-1].strip()
                                break
                         next_action["text_content"] = text_to_type
                else:
                    # If still stuck, maybe we need to wait or it's an error
                    logger.info("Stuck in a loop. Defaulting to 'wait' for 1 step.")
                    next_action = {"type": "wait", "target_description": "Loop detected, waiting..."}

        plan_data["next_action"] = next_action
        # --- End Planner's Decision Logic ---
            
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