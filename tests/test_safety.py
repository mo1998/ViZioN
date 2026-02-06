import pytest
from src.utils.safety import SafetyMonitor
from pynput import keyboard

def test_safety_monitor_initial_state():
    monitor = SafetyMonitor()
    assert monitor.should_stop is False

def test_safety_monitor_trigger():
    monitor = SafetyMonitor()
    # Simulate pressing ESC
    monitor._on_press(keyboard.Key.esc)
    assert monitor.should_stop is True

def test_safety_monitor_ignore_other_keys():
    monitor = SafetyMonitor()
    monitor._on_press(keyboard.Key.space)
    assert monitor.should_stop is False
