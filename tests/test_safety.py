import pytest
from src.utils.safety import SafetyMonitor
from unittest.mock import MagicMock

def test_safety_monitor_initial_state():
    monitor = SafetyMonitor()
    assert monitor.should_stop is False

def test_safety_monitor_trigger():
    monitor = SafetyMonitor()
    # Mock keyboard and key
    mock_kb = MagicMock()
    mock_key = MagicMock()
    monitor._keyboard = mock_kb
    mock_kb.Key.esc = mock_key
    
    # Simulate pressing ESC
    monitor._on_press(mock_key)
    assert monitor.should_stop is True

def test_safety_monitor_ignore_other_keys():
    monitor = SafetyMonitor()
    mock_kb = MagicMock()
    mock_key = MagicMock()
    mock_other_key = MagicMock()
    monitor._keyboard = mock_kb
    mock_kb.Key.esc = mock_key
    
    monitor._on_press(mock_other_key)
    assert monitor.should_stop is False