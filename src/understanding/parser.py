import json
import re
import logging

logger = logging.getLogger(__name__)

class WorldParser:
    """
    The 'Parser'. Converts raw text signals from the VLM into structured data.
    """
    
    @staticmethod
    def _repair_json(json_str):
        """
        Attempts simple repairs on malformed JSON strings.
        """
        # 1. Remove trailing commas (e.g., {"a": 1,})
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        # 2. Fix unquoted keys (simple alphanumeric keys only)
        # json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        return json_str

    @staticmethod
    def parse_json(text_output):
        """
        Attempts to extract and parse a JSON object from the model's text output.
        Handles code blocks like ```json ... ```.
        """
        json_str = ""
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
            
            # Attempt generic parse
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Attempt repair
                logger.info("JSON parse failed. Attempting repair...")
                json_str_repaired = WorldParser._repair_json(json_str)
                data = json.loads(json_str_repaired)

            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON after repair: {e}")
            logger.debug(f"Faulty JSON: {json_str}")
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
