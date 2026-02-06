import unittest
import os
import shutil
import json
from src.reasoning.memory import MemoryManager

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data"
        self.memory_file = os.path.join(self.test_dir, "test_memory.json")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.manager = MemoryManager(memory_file=self.memory_file)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_short_term_memory(self):
        step = {
            "analysis": "Thinking...",
            "action": {"type": "click", "target_description": "Button", "coordinates": [100, 200]}
        }
        self.manager.add_short_term(step)
        context = self.manager.get_short_term_context()
        self.assertIn("Performed 'click' on 'Button'", context)
        
        coords = self.manager.get_last_action_coordinates()
        self.assertEqual(coords, [100, 200])

    def test_long_term_memory_save_and_load(self):
        goal = "test goal"
        step = {
            "analysis": "Thinking...",
            "action": {"type": "finish", "target_description": "Goal"}
        }
        self.manager.add_short_term(step)
        self.manager.save_successful_run(goal)
        
        # New manager to test loading
        new_manager = MemoryManager(memory_file=self.memory_file)
        experience = new_manager.retrieve_experience(goal)
        self.assertIsNotNone(experience)
        self.assertIn("finish on Goal", experience["steps_summary"])

if __name__ == '__main__':
    unittest.main()
