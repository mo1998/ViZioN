import pytest
from src.perception.eyes import VisualPerception

def test_initialization(mocker):
    # Mocking the heavy VLM loading using pytest-mock
    mocker.patch('src.perception.vlm.AutoModelForImageTextToText')
    mocker.patch('src.perception.vlm.AutoProcessor')
    
    eyes = VisualPerception(use_ocr=False)
    assert eyes.vlm is not None
    assert eyes.ocr is None

def test_ocr_initialization(mocker):
    # Mocking initialization
    mocker.patch('src.perception.vlm.AutoModelForImageTextToText')
    mocker.patch('src.perception.vlm.AutoProcessor')
    mocker.patch('src.perception.ocr.PaddleOCRDetector._init_ocr')
    
    eyes = VisualPerception(use_ocr=True)
    assert eyes.ocr is not None