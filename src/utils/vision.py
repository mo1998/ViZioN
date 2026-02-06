import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw
import logging

logger = logging.getLogger(__name__)

class VisionUtils:
    @staticmethod
    def pil_to_cv2(pil_image):
        """Converts a PIL Image to an OpenCV BGR image."""
        if pil_image is None:
            return None
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def compute_similarity(img1_pil, img2_pil):
        """
        Computes Structural Similarity Index (SSIM) between two PIL images.
        Returns a float between -1.0 and 1.0 (1.0 = identical).
        """
        if img1_pil is None or img2_pil is None:
            return 0.0
            
        # Resize to match specific dimensions if needed, or ensure they match
        # For simplicity, we assume they are screenshots of the same resolution.
        if img1_pil.size != img2_pil.size:
            # Resize img2 to match img1
            img2_pil = img2_pil.resize(img1_pil.size)

        # Convert to grayscale for SSIM
        img1_gray = np.array(img1_pil.convert('L'))
        img2_gray = np.array(img2_pil.convert('L'))

        try:
            score, _ = ssim(img1_gray, img2_gray, full=True)
            return score
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0.0

    @staticmethod
    def match_template(screen_pil, template_pil, threshold=0.8):
        """
        Finds the template in the screen using Template Matching.
        Returns (x, y, w, h) or None.
        """
        if screen_pil is None or template_pil is None:
            return None
            
        screen_cv = VisionUtils.pil_to_cv2(screen_pil)
        template_cv = VisionUtils.pil_to_cv2(template_pil)
        
        # Convert to Gray
        screen_gray = cv2.cvtColor(screen_cv, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_cv, cv2.COLOR_BGR2GRAY)
        
        # Match
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            h, w = template_gray.shape
            x, y = max_loc
            return (x, y, w, h)
        
        return None

    @staticmethod
    def mark_action(image_pil, coords, color="red", size=10):
        """
        Draws a visual marker (cross) at the specified coordinates to represent memory.
        Returns a new PIL image with the marker.
        """
        if not image_pil or not coords:
            return image_pil
            
        marked_image = image_pil.copy()
        draw = ImageDraw.Draw(marked_image)
        x, y = coords
        
        # Draw 'X'
        draw.line((x - size, y - size, x + size, y + size), fill=color, width=3)
        draw.line((x - size, y + size, x + size, y - size), fill=color, width=3)
        
        return marked_image
