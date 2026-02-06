import pytest
from src.understanding.parser import WorldParser

def test_parse_json_markdown():
    text = """
    Here is the analysis:
    ```json
    {
        "analysis": "Found a button.",
        "next_action": {"type": "click"}
    }
    ```
    """
    result = WorldParser.parse_json(text)
    assert result is not None
    assert result["next_action"]["type"] == "click"

def test_parse_json_raw():
    text = '{"analysis": "test", "next_action": {"type": "type"}}'
    result = WorldParser.parse_json(text)
    assert result is not None
    assert result["next_action"]["type"] == "type"

def test_parse_invalid():
    text = "No json here."
    result = WorldParser.parse_json(text)
    assert result is None

def test_parse_json_with_trailing_comma():
    text = """
    ```json
    {
        "analysis": "Trailing comma test",
        "next_action": {"type": "click"},
    }
    ```
    """
    result = WorldParser.parse_json(text)
    assert result is not None
    assert result["analysis"] == "Trailing comma test"

def test_parse_json_unquoted_keys_failure():
    text = '{ analysis: "unquoted" }'
    result = WorldParser.parse_json(text)
    assert result is None