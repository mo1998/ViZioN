from typing import List, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class OCRDetector:
    """
    Abstract interface for OCR.
    """
    def detect_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        raise NotImplementedError

class PaddleOCRDetector(OCRDetector):
    """
    Wrapper for PaddleOCR.
    """
    def __init__(self, lang='en'):
        self.ocr = None
        self.lang = lang
        self._init_ocr()

    def _init_ocr(self):
        try:
            # Conditional import to avoid hard dependency if not installed
            from paddleocr import PaddleOCR
            # use_angle_cls=True helps with rotated text
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            logger.info("PaddleOCR initialized.")
        except ImportError:
            logger.warning("paddleocr not installed. OCR will be unavailable.")
            self.ocr = None

    def detect_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not self.ocr:
            return []
        
        import numpy as np
        # PaddleOCR expects numpy array
        img_array = np.array(image)
        result = self.ocr.ocr(img_array, cls=True)
        
        # Result structure: [[[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], (text, confidence)], ...]
        # We need to flatten and standardize
        
        detected_texts = []
        if result and result[0]:
            for line in result[0]:
                coords = line[0] # List of 4 points
                text_info = line[1] # (text, score)
                
                # Simple bbox from 4 points (min/max)
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                
                detected_texts.append({
                    "text": text_info[0],
                    "confidence": text_info[1],
                    "bbox": [x1, y1, x2, y2]
                })
        
        return detected_texts
