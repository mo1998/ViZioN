from PIL import Image
import logging
from src.config import Config
from src.perception.vlm import VLMDetector
from src.perception.ocr import PaddleOCRDetector
from src.perception.layout import MockLayoutDetector
from src.perception.schema import UISceneGraph, UIElement, BoundingBox

logger = logging.getLogger(__name__)

class VisualPerception:
    """
    The Orchestrator (formerly 'Eyes').
    Coordinated VLM, OCR, and Layout analysis to produce a ground truth.
    """
    def __init__(self, use_ocr=False, use_layout=False):
        logger.info("Initializing Perception System...")
        
        # 1. Semantic Core (VLM) - Always active
        self.vlm = VLMDetector()
        
        # 2. Structural Modules (Optional)
        self.ocr = PaddleOCRDetector() if use_ocr else None
        self.layout = MockLayoutDetector() if use_layout else None
        
        self.use_ocr = use_ocr
        self.use_layout = use_layout

    def see(self, image_source, prompt_text="Describe this image in detail.") -> str:
        """
        Legacy/Direct VLM access for reasoning.
        Returns raw text string.
        """
        if isinstance(image_source, str):
            image = Image.open(image_source)
        else:
            image = image_source
            
        return self.vlm.analyze(image, prompt_text)

    def perceive_scene(self, image_source) -> UISceneGraph:
        """
        Full Perception Pipeline.
        Returns a Structured UI Graph.
        """
        if isinstance(image_source, str):
            image = Image.open(image_source)
        else:
            image = image_source
            
        scene = UISceneGraph(resolution=image.size)
        
        # 1. Run Structural Detectors (if enabled)
        structural_elements = []
        if self.use_ocr:
            ocr_results = self.ocr.detect_text(image)
            for item in ocr_results:
                bbox = BoundingBox(*item['bbox'])
                elem = UIElement(
                    id=f"ocr_{len(structural_elements)}",
                    type="text",
                    bbox=bbox,
                    text=item['text'],
                    confidence=item['confidence'],
                    source="ocr"
                )
                structural_elements.append(elem)
                scene.add_element(elem)
        
        # 2. Run VLM for High-Level Understanding & Grounding
        # We can inject OCR results into the VLM prompt to help it!
        
        context_prompt = ""
        if structural_elements:
             context_prompt = f"Found text elements: {[e.text for e in structural_elements[:20]]}... "

        # For now, we mainly use the VLM to fill in the semantic gaps or just return reasoning.
        # Ideally, we would ask the VLM to 'verify' or 'classify' these elements.
        
        return scene