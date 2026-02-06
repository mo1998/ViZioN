import json
import re
import logging

logger = logging.getLogger(__name__)

class WorldParser:
    """
    The 'Parser'. Converts raw text signals from the VLM into structured data.
    """
    
    @staticmethod
    def parse_json(text_output):
        """
        Attempts to extract and parse a JSON object from the model's text output.
        Handles code blocks like ```json ... ```.
        """
        try:
            # Try to find JSON block
            match = re.search(r"```json\s*(.*?)```", text_output, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Fallback: try to find the first '{' and last '}'
                start = text_output.find("{")
                end = text_output.rfind("}")
                if start != -1 and end != -1:
                    json_str = text_output[start : end + 1]
                else:
                    logger.warning("No JSON structure found in output.")
                    return None

            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            return None

    @staticmethod
    def struct_ui_graph(detection_result):
        """
        Converts a list of detected items (e.g. from VLM bounding box output) 
        into a simplified UI Graph.
        
        Args:
            detection_result (list or dict): The parsed JSON from the VLM.
            
        Returns:
            dict: Structured UI Graph with 'elements', 'relationships', etc.
        """
        # This assumes the VLM was asked to output a list of elements.
        # Structure: { "elements": [{ "id": 1, "type": "button", "bbox": [...], "text": "Submit" }] }
        
        if not detection_result:
            return {"elements": [], "error": "No data"}
            
        # Basic validation/normalization could happen here
        return detection_result
