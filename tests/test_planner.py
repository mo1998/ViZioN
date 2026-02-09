import pytest
from unittest.mock import Mock
from PIL import Image
from src.reasoning.planner import Planner
from src.understanding.parser import WorldParser # To ensure parse_json is used if needed
from src.utils.vision import VisionUtils

# Mock the VisionUtils.get_center_coords as it's a utility function
@pytest.fixture(autouse=True)
def mock_vision_utils_get_center_coords(mocker):
    mocker.patch('src.utils.vision.VisionUtils.get_center_coords', side_effect=lambda bbox: [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

@pytest.fixture
def mock_perception_module(mocker):
    """Mocks the perception module (eyes) to return controlled VLM responses."""
    mock_eyes = Mock()
    # Ensure parse_json is called on the VLM's raw string response
    mocker.patch.object(WorldParser, 'parse_json', side_effect=lambda x: eval(x)) # Simple eval for testing json string
    return mock_eyes

@pytest.fixture
def planner_instance(mock_perception_module):
    """Provides a Planner instance with a mocked perception module."""
    return Planner(mock_perception_module)

def test_planner_selects_click_action_for_button(planner_instance, mock_perception_module):
    """Tests if the planner correctly selects a click action for a button."""
    vlm_response_json = """
{
  "analysis": "Identified a login button.",
  "relevant_elements": [
    {
      "id": 1,
      "description": "Login Button",
      "type": "button",
      "text": "Login",
      "value": "",
      "bbox": [100, 100, 200, 150]
    }
  ],
  "expected_outcome": "User logs in"
}
"""
    mock_perception_module.see.return_value = vlm_response_json
    
    goal = "Click the login button"
    image = Image.new('RGB', (1000, 1000)) # Dummy image

    plan = planner_instance.plan_next_step(image, goal)

    assert plan is not None
    assert plan["next_action"]["type"] == "click"
    assert plan["next_action"]["target_id"] == 1
    assert plan["next_action"]["target_description"] == "Login Button"
    assert plan["next_action"]["coordinates"] == [150.0, 125.0]

def test_planner_selects_type_action_for_text_input(planner_instance, mock_perception_module):
    """Tests if the planner correctly selects a type action for a text input."""
    vlm_response_json = """
{
  "analysis": "Identified a username input field.",
  "relevant_elements": [
    {
      "id": 1,
      "description": "Username Input",
      "type": "text_input",
      "text": "",
      "value": "",
      "bbox": [50, 50, 250, 100]
    }
  ],
  "expected_outcome": "Username entered"
}
"""
    mock_perception_module.see.return_value = vlm_response_json
    
    goal = "Type 'myusername' into the username field"
    image = Image.new('RGB', (1000, 1000))

    plan = planner_instance.plan_next_step(image, goal)

    assert plan is not None
    assert plan["next_action"]["type"] == "type"
    assert plan["next_action"]["target_id"] == 1
    assert plan["next_action"]["target_description"] == "Username Input"
    assert plan["next_action"]["text_content"] == "myusername"

def test_planner_prioritizes_best_match(planner_instance, mock_perception_module):
    """Tests if the planner prioritizes the best matching element."""
    vlm_response_json = """
{
  "analysis": "Identified multiple buttons.",
  "relevant_elements": [
    {
      "id": 1,
      "description": "Save Button",
      "type": "button",
      "text": "Save",
      "value": "",
      "bbox": [10, 10, 50, 30]
    },
    {
      "id": 2,
      "description": "Cancel Button",
      "type": "button",
      "text": "Cancel",
      "value": "",
      "bbox": [60, 10, 100, 30]
    },
    {
      "id": 3,
      "description": "Save Changes",
      "type": "link",
      "text": "Save Changes",
      "value": "",
      "bbox": [110, 10, 150, 30]
    }
  ],
  "expected_outcome": "Changes saved"
}
"""
    mock_perception_module.see.return_value = vlm_response_json
    
    goal = "Click Save Changes"
    image = Image.new('RGB', (1000, 1000))

    plan = planner_instance.plan_next_step(image, goal)

    assert plan is not None
    assert plan["next_action"]["type"] == "click"
    assert plan["next_action"]["target_id"] == 3
    assert plan["next_action"]["target_description"] == "Save Changes"

def test_planner_defaults_to_wait_if_no_clear_match(planner_instance, mock_perception_module):
    """Tests if the planner defaults to 'wait' when no clear action matches the goal."""
    vlm_response_json = """
{
  "analysis": "Identified some generic text.",
  "relevant_elements": [
    {
      "id": 1,
      "description": "Some random text",
      "type": "generic",
      "text": "Hello World",
      "value": "",
      "bbox": [10, 10, 50, 30]
    }
  ],
  "expected_outcome": "Nothing specific"
}
"""
    mock_perception_module.see.return_value = vlm_response_json
    
    goal = "Find the purple elephant" # No matching element
    image = Image.new('RGB', (1000, 1000))

    plan = planner_instance.plan_next_step(image, goal)

    assert plan is not None
    assert plan["next_action"]["type"] == "wait"
    assert "No clear action identified yet" in plan["next_action"]["target_description"]

def test_planner_extracts_fill_text(planner_instance, mock_perception_module):
    """Tests extraction of text for 'fill' action."""
    vlm_response_json = """
{
  "analysis": "Identified an email input field.",
  "relevant_elements": [
    {
      "id": 1,
      "description": "Email Input",
      "type": "text_input",
      "text": "",
      "value": "",
      "bbox": [50, 50, 250, 100]
    }
  ],
  "expected_outcome": "Email entered"
}
"""
    mock_perception_module.see.return_value = vlm_response_json
    
    goal = "Fill 'test@example.com' into the email input"
    image = Image.new('RGB', (1000, 1000))

    plan = planner_instance.plan_next_step(image, goal)

    assert plan is not None
    assert plan["next_action"]["type"] == "type"
    assert plan["next_action"]["text_content"] == "test@example.com"

def test_planner_no_action_for_non_interactive_elements_with_click_goal(planner_instance, mock_perception_module):
    """Tests that a generic element is not clicked if goal is 'click' but element is not clickable."""
    vlm_response_json = """
{
  "analysis": "Identified a generic title.",
  "relevant_elements": [
    {
      "id": 1,
      "description": "Welcome Title",
      "type": "generic",
      "text": "Welcome",
      "value": "",
      "bbox": [100, 100, 200, 150]
    }
  ],
  "expected_outcome": "Title seen"
}
"""
    mock_perception_module.see.return_value = vlm_response_json
    
    goal = "Click Welcome Title"
    image = Image.new('RGB', (1000, 1000))

    plan = planner_instance.plan_next_step(image, goal)

    assert plan is not None
    # Expect wait because "generic" is not in clickable types without strong keyword match
    assert plan["next_action"]["type"] == "wait"
    assert "No clear action identified yet" in plan["next_action"]["target_description"]

