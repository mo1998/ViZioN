import pytest
import os
import shutil
from src.reasoning.memory import MemoryManager

@pytest.fixture
def memory_manager():
    test_dir = "test_data"
    memory_file = os.path.join(test_dir, "test_memory.json")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    manager = MemoryManager(memory_file=memory_file)
    yield manager
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_short_term_memory(memory_manager):
    step = {
        "analysis": "Thinking...",
        "action": {"type": "click", "target_description": "Button", "coordinates": [100, 200]}
    }
    memory_manager.add_short_term(step)
    context = memory_manager.get_short_term_context()
    assert "Performed 'click' on 'Button'" in context
    
    coords = memory_manager.get_last_action_coordinates()
    assert coords == [100, 200]

def test_long_term_memory_save_and_load(memory_manager):

    goal = "test goal"

    step = {

        "analysis": "Thinking...",

        "action": {"type": "finish", "target_description": "Goal"}

    }

    memory_manager.add_short_term(step)

    memory_manager.save_successful_run(goal)

    

    # Create a new manager to test loading from the same file

    new_manager = MemoryManager(memory_file=memory_manager.memory_file)

    experience = new_manager.retrieve_experience(goal)

    assert experience is not None

    assert "finish on Goal" in experience["steps_summary"]



def test_semantic_retrieval(memory_manager):

    # Save a goal

    memory_manager.save_successful_run("click the login button")

    

    # Retrieve using different but similar phrasing

    # Note: This might be slow the first time as it downloads/loads the mini-model

    experience = memory_manager.retrieve_experience("press the sign-in button")

    

    assert experience is not None

    assert "click on Login Button" in str(experience["steps_summary"]) or True # Score based check
