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
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')
        score = VisionUtils.compute_similarity(img1, img2)
        self.assertLess(score, 0.1)

    def test_mark_action(self):
        img = Image.new('RGB', (100, 100), color='white')
        coords = [50, 50]
        marked = VisionUtils.mark_action(img, coords)
        
        # Simple check that the image changed
        self.assertNotEqual(list(img.getdata()), list(marked.getdata()))

if __name__ == '__main__':
    unittest.main()
