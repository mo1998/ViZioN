import logging
import time
try:
    import pyautogui
except (ImportError, KeyError, Exception):
    pyautogui = None
from PIL import Image

from src.perception.eyes import VisualPerception
from src.reasoning.planner import Planner
from src.action.hands import MockExecutor, DesktopExecutor
from src.utils.vision import VisionUtils
from src.utils.safety import SafetyMonitor
from src.understanding.parser import WorldParser

logger = logging.getLogger(__name__)

class VisualAgent:
    def __init__(self, mode="mock", use_ocr=False):
        logger.info("Initializing ViZioN Agent...")
        self.mode = mode
        
        # 1. Initialize Eyes (Perception System)
        self.eyes = VisualPerception(use_ocr=use_ocr)
        
        # 2. Initialize Brain
        self.planner = Planner(self.eyes)
        
        # 3. Initialize Hands (Lazy loading handled inside executors or here)
        if mode == "desktop":
            self.executor = DesktopExecutor()
        else:
            self.executor = MockExecutor()
            
        # 4. Safety
        self.safety = SafetyMonitor()
        if mode == "desktop":
            self.safety.start()

        logger.info(f"Agent initialized in {mode} mode.")

    def _capture_screen(self):
        """Captures the current screen and returns a PIL Image."""
        if self.mode == "mock":
            return None 
        
        # Lazy import to avoid import-time crashes on headless systems
        try:
            import pyautogui
            return pyautogui.screenshot()
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None

    def _verify_outcome(self, current_image, previous_expectation):
        """
        Asks the VLM if the 'previous_expectation' has been met in 'current_image'.
        """
        if not previous_expectation or not current_image:
            return True

        prompt = f"""
You are a Verification Agent.
Expected Outcome: "{previous_expectation}"

Analyze the image. Has this outcome been achieved? 
Reply strictly JSON: {{ "verified": true }} or {{ "verified": false, "reason": "why not" }}
"""
        try:
            # We use the perception module directly for this check
            response = self.eyes.see(current_image, prompt)
            data = WorldParser.parse_json(response)
            if data and data.get("verified") is True:
                logger.info(f"✅ Outcome Verified: {previous_expectation}")
                return True
            else:
                reason = data.get("reason") if data else "Unknown"
                logger.warning(f"❌ Verification Failed: Expected '{previous_expectation}'. Reason: {reason}")
                return False
        except Exception as e:
            logger.error(f"Verification process error: {e}")
            return True # Fail open to avoid getting stuck if verification breaks

    def run(self, goal, initial_image_path=None, max_steps=10, interval=2.0):
        """
        Main loop: Capture -> Plan -> Act -> Verify -> Repeat
        """
        logger.info(f"Starting execution loop for goal: '{goal}' (Max steps: {max_steps})")
        
        last_image = None
        last_expected_outcome = None
        
        for step in range(1, max_steps + 1):
            # 0. Safety Check
            if self.safety.should_stop:
                logger.warning("🛑 Execution stopped by Safety Monitor.")
                break
                
            logger.info(f"--- Step {step}/{max_steps} ---")
            
            # 1. Acquire Image
            current_image = None
            if self.mode == "desktop":
                current_image = self._capture_screen()
            elif initial_image_path:
                if isinstance(initial_image_path, str):
                    current_image = Image.open(initial_image_path)
                else:
                    current_image = initial_image_path
            
            if current_image is None:
                logger.error("No image source available. Aborting.")
                break
                
            # 1b. Verification (Did the LAST step work?)
            if step > 1 and last_expected_outcome:
                 # In a real loop, we might want to capture a fresh image *after* the wait interval
                 # But current_image here is fresh enough
                 verified = self._verify_outcome(current_image, last_expected_outcome)
                 if not verified:
                     logger.warning("Previous action failed verification. Retrying logic might be needed.")
                     # In advanced versions, we'd inject this failure into the next prompt.
                     # For now, we log it and proceed, relying on memory to see the same state again.

            # 1c. Smart Polling (Performance Optimization)
            if last_image:
                similarity = VisionUtils.compute_similarity(last_image, current_image)
                logger.info(f"Screen Similarity: {similarity:.4f}")
                
                if similarity > 0.99:
                    logger.info("Screen unchanged. Skipping VLM inference (Waiting).")
                    if self.mode == "desktop":
                        time.sleep(interval)
                    continue

            # 2. Run Single Step
            plan = self.run_step(current_image, goal)
            last_image = current_image
            
            # 3. Check for Termination
            if not plan:
                logger.warning("No plan generated. Stopping.")
                break
            
            # Save expectation for next loop
            last_expected_outcome = plan.get("expected_outcome")
            
            action = plan.get("next_action", {})
            if action.get("type") == "finish":
                logger.info("Goal achieved according to planner. Stopping.")
                break
            
            # Wait before next step in desktop mode
            if self.mode == "desktop":
                time.sleep(interval)

        if self.mode == "desktop":
            self.safety.stop()

    def run_step(self, image_source, goal):
        """
        Executes a single step of the loop: See -> Think -> Act
        """
        logger.info("--- Start Step ---")
        
        # Optional: In the future, we will call self.eyes.perceive_scene(image_source) first
        # and pass the SceneGraph to the planner.
        
        # 1. Plan (See + Think)
        # Currently, planner calls eyes.see() internally.
        plan = self.planner.plan_next_step(image_source, goal)
        
        if not plan:
            logger.error("Planning failed.")
            return
            
        # 2. Act
        self.executor.execute(plan)
        
        logger.info("--- End Step ---")
        return plan