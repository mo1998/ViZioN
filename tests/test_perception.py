import unittest
from unittest.mock import MagicMock, patch
from src.perception.eyes import VisualPerception
from src.perception.schema import UISceneGraph

class TestPerceptionSystem(unittest.TestCase):
    
    @patch('src.perception.vlm.Qwen2_5_VLForConditionalGeneration')
    @patch('src.perception.vlm.AutoProcessor')
    def test_initialization(self, mock_processor, mock_model):
        # Mocking the heavy VLM loading
        eyes = VisualPerception(use_ocr=False)
        self.assertIsNotNone(eyes.vlm)
        self.assertIsNone(eyes.ocr)

    @patch('src.perception.vlm.Qwen2_5_VLForConditionalGeneration')
    @patch('src.perception.vlm.AutoProcessor')
    @patch('src.perception.ocr.PaddleOCRDetector._init_ocr') # Mock the init instead of the external lib directly if needed
    def test_ocr_initialization(self, mock_init, mock_processor, mock_model):
        # Mocking initialization
        eyes = VisualPerception(use_ocr=True)
        self.assertIsNotNone(eyes.ocr)

if __name__ == '__main__':
    unittest.main()
