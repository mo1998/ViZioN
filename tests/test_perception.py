import pytest
import requests
from unittest.mock import MagicMock, patch
from src.perception.eyes import VisualPerception
from src.perception.vlm import VLMDetector
from src.config import Config
from PIL import Image
import io
import base64

@pytest.fixture
def mock_vlm_response():
    """Fixture to provide a mock response from the vLLM server."""
    return {
        "choices": [
            {
                "message": {
                    "content": "A detailed description of the image."
                }
            }
        ]
    }

def test_vlm_detector_analyze_success(mock_vlm_response):
    """Test VLMDetector's analyze method with a successful response."""
    detector = VLMDetector()
    mock_image = Image.new('RGB', (100, 100))
    prompt = "Describe this image."

    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_vlm_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = detector.analyze(mock_image, prompt)

        assert result == "A detailed description of the image."
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Verify URL
        assert args[0] == Config.VLLM_URL
        
        # Verify payload content
        payload = kwargs['json']
        assert payload['model'] == Config.VLLM_MODEL_ID
        assert len(payload['messages']) == 1
        assert payload['messages'][0]['role'] == 'user'
        content = payload['messages'][0]['content']
        assert len(content) == 2
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == prompt
        assert content[1]['type'] == 'image_url'
        assert 'data:image/png;base64,' in content[1]['image_url']['url']

def test_vlm_detector_analyze_http_error():
    """Test VLMDetector's analyze method with an HTTP error."""
    detector = VLMDetector()
    mock_image = Image.new('RGB', (100, 100))
    prompt = "Describe this image."

    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("HTTP Error")
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.RequestException, match="HTTP Error"):
            detector.analyze(mock_image, prompt)

def test_vlm_detector_analyze_key_error():
    """Test VLMDetector's analyze method with an unexpected response format."""
    detector = VLMDetector()
    mock_image = Image.new('RGB', (100, 100))
    prompt = "Describe this image."

    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"bad_key": "bad_value"} # Malformed response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with pytest.raises(KeyError):
            detector.analyze(mock_image, prompt)

def test_initialization_visual_perception_with_vlm_detector():
    """Test VisualPerception initialization now that VLMDetector doesn't load model."""
    # We don't need to mock VLM loading directly anymore, as VLMDetector now makes HTTP requests
    # We can mock the VLMDetector's analyze method if we want to isolate VisualPerception further,
    # but for init, just ensuring it's created is enough.
    with patch('src.perception.ocr.PaddleOCRDetector._init_ocr'): # Only mock OCR if needed
        eyes = VisualPerception(use_ocr=False)
        assert eyes.vlm is not None
        assert isinstance(eyes.vlm, VLMDetector)
        assert eyes.ocr is None

def test_ocr_initialization_visual_perception(mocker):
    # Mocking OCR initialization
    mocker.patch('src.perception.ocr.PaddleOCRDetector._init_ocr')
    
    eyes = VisualPerception(use_ocr=True)
    assert eyes.ocr is not None