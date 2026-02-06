import pytest
from PIL import Image
from src.agent import VisualAgent

@pytest.fixture
def mock_agent(mocker):
    # Prevent heavy model loading
    mocker.patch('src.perception.vlm.AutoModelForImageTextToText')
    mocker.patch('src.perception.vlm.AutoProcessor')
    return VisualAgent(mode="mock")

def test_agent_initialization(mock_agent):
    assert mock_agent.mode == "mock"
    assert mock_agent.safety is not None

def test_verify_outcome_positive(mock_agent, mocker):
    # Mock VLM response to say "verified: true"
    mocker.patch.object(mock_agent.eyes, 'see', return_value='```json {"verified": true} ```')
    
    img = Image.new('RGB', (10, 10))
    result = mock_agent._verify_outcome(img, "Modal closed")
    assert result is True

def test_verify_outcome_negative(mock_agent, mocker):
    # Mock VLM response to say "verified: false"
    mocker.patch.object(mock_agent.eyes, 'see', return_value='```json {"verified": false, "reason": "still open"} ```')
    
    img = Image.new('RGB', (10, 10))
    result = mock_agent._verify_outcome(img, "Modal closed")
    assert result is False

def test_headless_safe_capture(mock_agent):
    # In mock mode, capture_screen should return None safely
    assert mock_agent._capture_screen() is None
