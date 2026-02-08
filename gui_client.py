import sys
import os
import builtins
import flet as ft
import requests
import io
import time
import threading
import pyautogui
import logging
import base64

# --- CRITICAL FIX FOR PYINSTALLER ---
builtins.exit = sys.exit
# -------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ViZioN-Client")
pyautogui.FAILSAFE = True 

class VisionClientApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "ViZioN Remote Client"
        
        # Modern Flet Window handling
        self.page.window.width = 450  
        self.page.window.height = 850
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.window.always_on_top = True 
        
        self.running = False
        self.server_url = "http://localhost:8052" 
        
        self.setup_ui()

    def setup_ui(self):
        self.page.add(ft.Text("ViZioN Remote Control", size=25, weight="bold"))
        
        self.url_input = ft.TextField(label="Server URL", value=self.server_url)
        self.page.add(self.url_input)
        
        self.goal_input = ft.TextField(
            label="What is your goal?", 
            hint_text="e.g. Open Notepad and type Hello World",
            multiline=True,
            min_lines=3,
            value="upgrade google plan" 
        )
        self.page.add(self.goal_input)
        
        # Icons as strings to avoid attribute errors
        self.start_btn = ft.ElevatedButton(
            "Start Automation", 
            on_click=self.start_automation, 
            icon="PLAY_ARROW" 
        )
        self.stop_btn = ft.ElevatedButton(
            "Stop", 
            on_click=self.stop_automation, 
            icon="STOP", 
            color="red", 
            disabled=True
        )
        self.page.add(ft.Row([self.start_btn, self.stop_btn]))
        
        self.status_text = ft.Text("Status: Idle", color="grey")
        self.page.add(self.status_text)

        self.page.add(ft.Text("Agent's View:", weight="bold"))
        
        # FIX: Latest Flet uses 'src' for both URLs and Base64 strings.
        # If it's base64, it just needs the correct header.
        self.screenshot_preview = ft.Image(
            src="",
            width=400,
            height=250,
            fit="contain",
            border_radius=10,
            visible=False
        )
        self.page.add(self.screenshot_preview)
        
        self.log_area = ft.ListView(expand=True, spacing=5, padding=10, auto_scroll=True)
        self.page.add(
            ft.Container(
                content=self.log_area, 
                border=ft.border.all(1, "grey"), 
                height=200,
                border_radius=5
            )
        )

    def log(self, message, color="white"):
        self.log_area.controls.append(ft.Text(f"[{time.strftime('%H:%M:%S')}] {message}", color=color, size=12))
        self.page.update()

    def start_automation(self, e):
        if not self.goal_input.value:
            self.log("Error: Please enter a goal.", "red")
            return
        
        self.running = True
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        
        # Transparency is better than minimizing for visibility
        self.page.window.opacity = 0.8
        self.status_text.value = "Status: Running"
        self.status_text.color = "green"
        self.server_url = self.url_input.value
        self.page.update()
        
        self.log(f"Starting automation: {self.goal_input.value}")
        threading.Thread(target=self.automation_loop, daemon=True).start()

    def stop_automation(self, e):
        self.running = False
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.status_text.value = "Status: Stopped"
        self.status_text.color = "orange"
        self.page.window.opacity = 1.0 
        self.log("Stopping automation...", "orange")
        self.page.update()

    def execute_local_action(self, action):
        if not action: return
            
        act_type = action.get("type")
        coords = action.get("coordinates")
        
        if act_type == "click" and coords:
            pyautogui.click(x=coords[0], y=coords[1])
        elif act_type == "type":
            text = action.get("text_content", "")
            pyautogui.write(text, interval=0.05)
        elif act_type == "switch":
            pyautogui.hotkey('alt', 'tab')
        elif act_type == "finish":
            self.log("Task Finished!", "green")
            self.stop_automation(None)

    def automation_loop(self):
        try:
            while self.running:
                # 1. Capture Screenshot
                screenshot = pyautogui.screenshot()
                
                # Convert to Base64 with the 'data:image/png;base64,' prefix
                # This works for 'src' in all modern Flet versions
                buffered = io.BytesIO()
                screenshot.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Update Preview using standard 'src' property
                self.screenshot_preview.src = f"data:image/png;base64,{img_str}"
                self.screenshot_preview.visible = True
                self.page.update()

                # 2. Communicate with Backend
                try:
                    img_byte_arr = buffered.getvalue()
                    files = {'file': ('screenshot.png', img_byte_arr, 'image/png')}
                    data = {'goal': self.goal_input.value}
                    
                    response = requests.post(f"{self.server_url}/process_step", files=files, data=data, timeout=30)
                    response.raise_for_status()
                    
                    plan = response.json()
                    self.log(f"Brain: {plan.get('reasoning', 'Thinking...')}", "yellow")
                    
                    # 3. Execute Action
                    self.execute_local_action(plan.get("next_action", {}))
                        
                except Exception as e:
                    self.log(f"Server Error: {e}", "red")
                    self.stop_automation(None)
                    break
                
                time.sleep(1) 
                
        except pyautogui.FailSafeException:
            self.log("Fail-safe triggered!", "red")
            self.stop_automation(None)
        except Exception as e:
            self.log(f"Critical Error: {e}", "red")
            self.stop_automation(None)

def main():
    ft.app(target=VisionClientApp)

if __name__ == "__main__":
    main()