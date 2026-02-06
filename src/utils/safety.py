import threading
import logging

logger = logging.getLogger(__name__)

class SafetyMonitor:
    """
    Monitors global hotkeys to trigger an emergency stop (Kill Switch).
    Default Hotkey: ESC
    """
    def __init__(self):
        self._stop_requested = False
        self.listener = None
        self._keyboard = None

    def _get_keyboard(self):
        if self._keyboard is None:
            try:
                from pynput import keyboard
                self._keyboard = keyboard
            except Exception as e:
                logger.warning(f"Safety Monitor: pynput keyboard backend not available ({e}). Kill Switch disabled.")
                return None
        return self._keyboard

    def start(self):
        """Starts the keyboard listener in a non-blocking thread."""
        kb = self._get_keyboard()
        if not kb:
            return

        try:
            self.listener = kb.Listener(on_press=self._on_press)
            self.listener.start()
            logger.info("Safety Monitor Active. Press 'ESC' to trigger Emergency Stop.")
        except Exception as e:
            logger.warning(f"Failed to start Safety Monitor listener: {e}")

    def stop(self):
        """Stops the listener."""
        if self.listener:
            self.listener.stop()

    def _on_press(self, key):
        kb = self._get_keyboard()
        if not kb:
            return
            
        try:
            if key == kb.Key.esc:
                logger.warning("🚨 EMERGENCY STOP TRIGGERED BY USER (ESC PRESSED) 🚨")
                self._stop_requested = True
                return False # Stop listener
        except AttributeError:
            pass

    @property
    def should_stop(self):
        return self._stop_requested
