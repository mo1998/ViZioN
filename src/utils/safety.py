import threading
import logging
from pynput import keyboard

logger = logging.getLogger(__name__)

class SafetyMonitor:
    """
    Monitors global hotkeys to trigger an emergency stop (Kill Switch).
    Default Hotkey: ESC
    """
    def __init__(self):
        self._stop_requested = False
        self.listener = None

    def start(self):
        """Starts the keyboard listener in a non-blocking thread."""
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        logger.info("Safety Monitor Active. Press 'ESC' to trigger Emergency Stop.")

    def stop(self):
        """Stops the listener."""
        if self.listener:
            self.listener.stop()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                logger.warning("🚨 EMERGENCY STOP TRIGGERED BY USER (ESC PRESSED) 🚨")
                self._stop_requested = True
                return False # Stop listener
        except AttributeError:
            pass

    @property
    def should_stop(self):
        return self._stop_requested
