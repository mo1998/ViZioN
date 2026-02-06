import unittest
import json
from src.understanding.parser import WorldParser

class TestWorldParser(unittest.TestCase):
    def test_parse_json_markdown(self):
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
        self.assertIsNotNone(result)
        self.assertEqual(result["next_action"]["type"], "click")

    def test_parse_json_raw(self):
        text = '{"analysis": "test", "next_action": {"type": "type"}}'
        result = WorldParser.parse_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["next_action"]["type"], "type")

    def test_parse_invalid(self):
        text = "No json here."
        result = WorldParser.parse_json(text)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
