# Building ViZioN Windows Client

This guide explains how to package the `gui_client.py` into a standalone Windows executable (`.exe`).

## Prerequisites

1.  A Windows machine with **Python 3.11** installed.
2.  Install the required packaging tools:
    ```bash
    pip install pyinstaller flet requests pyautogui Pillow
    ```

## Packaging Instructions

1.  Open a terminal (PowerShell or CMD) in the project root.
2.  Run PyInstaller:
    ```bash
    pyinstaller --onefile --windowed --name ViZioN-Client gui_client.py
    ```

### Argument breakdown:
*   `--onefile`: Packages everything into a single `.exe`.
*   `--windowed`: Prevents a console window from popping up when running the app.
*   `--name ViZioN-Client`: The name of the resulting executable.

## Distribution

After the process completes, you will find the `ViZioN-Client.exe` inside the `dist/` folder. You can move this file to any Windows machine and run it without needing Python installed (though it still needs network access to the ViZioN Linux server).

## Configuration

When you run the client:
1.  Enter the **Server URL** (e.g., `http://192.168.1.50:8000`).
2.  Enter your **Goal**.
3.  Click **Start Automation**.
