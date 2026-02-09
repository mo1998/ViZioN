import logging
import time

logger = logging.getLogger(__name__)

class ActionExecutor:
    def execute(self, action_plan):
        raise NotImplementedError

class MockExecutor(ActionExecutor):
    """
    Logs actions instead of executing them. Useful for testing/dry-run.
    """
    def execute(self, action_plan):
        action = action_plan.get("next_action", {})
        act_type = action.get("type")
        
        if act_type == "finish":
            logger.info("Task Completed successfully.")
            return True
            
        target = action.get("target_description", "unknown")
        coords = action.get("coordinates", [0, 0])
        
        logger.info(f"MOCK EXECUTION: Performing '{act_type}' on '{target}' at {coords}")
        return True

class RemoteExecutor(ActionExecutor):
    """
    Used when the agent logic runs on a server. 
    It doesn't execute anything locally, just logs that it's relaying to a client.
    """
    def execute(self, action_plan):
        action = action_plan.get("next_action", {})
        logger.info(f"REMOTE ACTION PREPARED: {action.get('type')} on {action.get('target_description')}")
        return True

class DesktopExecutor(ActionExecutor):
    """
    Executes actions using PyAutoGUI.
    """
    def __init__(self):
        # We check availability lazily to avoid import-time crashes
        self._pyautogui = None

    def _get_pyautogui(self):
        if self._pyautogui is None:
            try:
                import pyautogui
                self._pyautogui = pyautogui
                self._pyautogui.FAILSAFE = True
            except Exception as e:
                logger.error(f"Failed to import pyautogui: {e}")
                return None
        return self._pyautogui

    def execute(self, action_plan):
        pag = self._get_pyautogui()
        if not pag:
            logger.error("Cannot execute: pyautogui is missing or display is unavailable.")
            return False

        action = action_plan.get("next_action", {})
        act_type = action.get("type")
        coords = action.get("coordinates") # [x, y]

        if act_type == "click":
            if coords:
                pag.click(x=coords[0], y=coords[1])
                logger.info(f"Clicked at {coords}")
            else:
                logger.warning("No coordinates for click.")
        
        elif act_type == "type":
            text = action.get("text_content", "")
            if coords:
                pag.click(x=coords[0], y=coords[1])
                time.sleep(0.1) # Small delay to ensure focus
            pag.write(text)
            pag.press('enter')
            logger.info(f"Typed '{text}' at {coords} and pressed enter.")
            
        elif act_type == "finish":
            logger.info("Task finished.")
            
        else:
            logger.warning(f"Unknown action type: {act_type}")
            
        return True