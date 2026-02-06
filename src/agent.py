import logging
from src.perception.eyes import VisualPerception
from src.reasoning.planner import Planner
from src.action.hands import MockExecutor, DesktopExecutor

logger = logging.getLogger(__name__)

class VisualAgent:
    def __init__(self, mode="mock", use_ocr=False):
        logger.info("Initializing ViZioN Agent...")
        
        # 1. Initialize Eyes (Perception System)
        # We can enable optional structural detectors here
        self.eyes = VisualPerception(use_ocr=use_ocr)
        
        # 2. Initialize Brain
        self.planner = Planner(self.eyes)
        
        # 3. Initialize Hands
        if mode == "desktop":
            self.executor = DesktopExecutor()
        else:
            self.executor = MockExecutor()
            
        logger.info(f"Agent initialized in {mode} mode.")

    def run_step(self, image_path, goal):
        """
        Executes a single step of the loop: See -> Think -> Act
        """
        logger.info("--- Start Step ---")
        
        # Optional: In the future, we will call self.eyes.perceive_scene(image_path) first
        # and pass the SceneGraph to the planner.
        
        # 1. Plan (See + Think)
        # Currently, planner calls eyes.see() internally.
        plan = self.planner.plan_next_step(image_path, goal)
        
        if not plan:
            logger.error("Planning failed.")
            return
            
        # 2. Act
        self.executor.execute(plan)
        
        logger.info("--- End Step ---")
        return plan