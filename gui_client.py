import flet as ft
import requests
import io
import time
import threading
import pyautogui
import base64
from PIL import Image
import logging

# Configure logging for the client
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ViZioN-Client")

class VisionClientApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "ViZioN Remote Client"
        self.page.window_width = 600
        self.page.window_height = 800
        self.page.theme_mode = ft.ThemeMode.DARK
        
        self.running = False
        self.server_url = "http://localhost:8000" # Default
        
        self.setup_ui()

    def setup_ui(self):
        # Header
        self.page.add(ft.Text("ViZioN Remote Control", size=30, weight="bold"))
        
        # Server Config
        self.url_input = ft.TextField(label="Server URL", value=self.server_url, expand=True)
        self.page.add(ft.Row([self.url_input]))
        
        # Goal Input
        self.goal_input = ft.TextField(
            label="What is your goal?", 
            placeholder="e.g. Open Notepad and type Hello World",
            multiline=True,
            min_lines=3
        )
        self.page.add(self.goal_input)
        
        # Controls
        self.start_btn = ft.ElevatedButton("Start Automation", on_click=self.start_automation, icon=ft.icons.PLAY_ARROW)
        self.stop_btn = ft.ElevatedButton("Stop", on_click=self.stop_automation, icon=ft.icons.STOP, color="red", disabled=True)
        self.page.add(ft.Row([self.start_btn, self.stop_btn]))
        
        # Status & Logs
        self.status_text = ft.Text("Status: Idle", color="grey")
        self.page.add(self.status_text)
        
        self.log_area = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=True)
        self.page.add(ft.Container(content=self.log_area, border=ft.border.all(1, "grey"), height=300))

    def log(self, message, color="white"):
        self.log_area.controls.append(ft.Text(f"[{time.strftime('%H:%M:%S')}] {message}", color=color))
        self.page.update()

    def start_automation(self, e):
        if not self.goal_input.value:
            self.log("Error: Please enter a goal.", "red")
            return
        
        self.running = True
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.status_text.value = "Status: Running..."
        self.status_text.color = "green"
        self.server_url = self.url_input.value
        self.page.update()
        
        self.log(f"Starting automation: {self.goal_input.value}")
        # Start the loop in a background thread
        threading.Thread(target=self.automation_loop, daemon=True).start()

    def stop_automation(self, e):
        self.running = False
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.status_text.value = "Status: Stopped"
        self.status_text.color = "orange"
        self.log("Stopping automation...", "orange")
        self.page.update()

    def execute_local_action(self, action):
        act_type = action.get("type")
        coords = action.get("coordinates")
        
        if act_type == "click" and coords:
            self.log(f"Executing: Click at {coords}", "cyan")
            pyautogui.click(x=coords[0], y=coords[1])
        elif act_type == "type":
            text = action.get("text_content", "")
            self.log(f"Executing: Type '{text}'", "cyan")
            pyautogui.write(text)
        elif act_type == "finish":
            self.log("Server reports: Task Finished!", "green")
            self.stop_automation(None)
        else:
            self.log(f"Unknown or no action: {act_type}", "yellow")

    def automation_loop(self):
        try:
            while self.running:
                # 1. Capture Screen
                screenshot = pyautogui.screenshot()
                img_byte_arr = io.BytesIO()
                screenshot.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # 2. Send to Server
                self.log("Sending screenshot to server...")
                try:
                    files = {'file': ('screenshot.png', img_byte_arr, 'image/png')}
                    data = {'goal': self.goal_input.value}
                    
                    response = requests.post(f"{self.server_url}/process_step", files=files, data=data, timeout=60)
                    response.raise_for_status()
                    
                    plan = response.json()
                    
                    # 3. Log reasoning
                    reasoning = plan.get("reasoning", "No reasoning provided.")
                    self.log(f"Brain: {reasoning}")
                    
                    # 4. Execute Action
                    action = plan.get("next_action", {})
                    self.execute_local_action(action)
                    
                    if action.get("type") == "finish":
                        break
                        
                except Exception as e:
                    self.log(f"Server Error: {e}", "red")
                    self.stop_automation(None)
                    break
                
                time.sleep(2) # Interval
                
        except Exception as e:
            self.log(f"Client Loop Error: {e}", "red")
            self.stop_automation(None)

def main():
    ft.app(target=VisionClientApp)

if __name__ == "__main__":
    main()
