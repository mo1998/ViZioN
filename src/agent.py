import logging
import time
from PIL import Image

from src.perception.eyes import VisualPerception
from src.reasoning.planner import Planner
from src.action.hands import MockExecutor, DesktopExecutor
from src.utils.vision import VisionUtils

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

    def run(self, goal, initial_image_path=None, max_steps=10, interval=2.0):
        """
        Main loop: Capture -> Plan -> Act -> Repeat
        """
        logger.info(f"Starting execution loop for goal: '{goal}' (Max steps: {max_steps})")
        
        last_image = None
        
        for step in range(1, max_steps + 1):
            logger.info(f"--- Step {step}/{max_steps} ---")
            
            # 1. Acquire Image
            current_image = None
            if self.mode == "desktop":
                current_image = self._capture_screen()
                # Optional: Save debug screenshot if needed
                # if current_image: current_image.save(f"debug_step_{step}.png")
            elif initial_image_path:
                if isinstance(initial_image_path, str):
                    current_image = Image.open(initial_image_path)
                else:
                    current_image = initial_image_path
            
            if current_image is None:
                logger.error("No image source available. Aborting.")
                break
                
            # 1b. Smart Polling (Performance Optimization)
            # If screen hasn't changed significantly, skip heavy VLM inference.
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
                
            action = plan.get("next_action", {})
            if action.get("type") == "finish":
                logger.info("Goal achieved according to planner. Stopping.")
                break
            
            # Wait before next step in desktop mode
            if self.mode == "desktop":
                time.sleep(interval)

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