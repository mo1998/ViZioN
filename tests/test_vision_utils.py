import unittest
from PIL import Image
import numpy as np
from src.utils.vision import VisionUtils

class TestVisionUtils(unittest.TestCase):
    def test_compute_similarity_identical(self):
        img = Image.new('RGB', (100, 100), color='red')
        score = VisionUtils.compute_similarity(img, img)
        self.assertGreater(score, 0.99)

    def test_compute_similarity_different(self):
        img1 = Image.new('RGB', (100, 100), color='white')
        img2 = Image.new('RGB', (100, 100), color='black')
        score = VisionUtils.compute_similarity(img1, img2)
        # Even total opposite solid colors might have some SSIM, but it should be low.
        self.assertLess(score, 0.5)

    def test_mark_action(self):
        img = Image.new('RGB', (100, 100), color='white')
        coords = [50, 50]
        marked = VisionUtils.mark_action(img, coords)
        
        # Simple check that the image changed using numpy
        orig_arr = np.array(img)
        marked_arr = np.array(marked)
        self.assertFalse(np.array_equal(orig_arr, marked_arr))

if __name__ == '__main__':
    unittest.main()
