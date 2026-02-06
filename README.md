# ViZioN

<p align="center">
  <img src="Logo.png" width="250">
</p>

**ViZioN** is a production-ready Visual AI Agent designed to perceive, reason, and act within user interfaces. It leverages the cutting-edge **Qwen3-VL** Vision-Language Model to achieve human-level visual cognition.

## 🧠 Mental Model: The 5 Brains

ViZioN is architected around five core cognitive components, now enhanced with persistence:

1.  👁️ **Eyes (Perception):** The Vision-Language Model (`Qwen3-VL`) and optional structural detectors.
2.  🧩 **Parser (Understanding):** Extracts structured elements and repairs malformed JSON signals.
3.  🧠 **Reasoner (Memory & Logic):** Maintains state and learns from past interactions.
4.  🤔 **Planner (Strategy):** Decides actions based on the current screen and historical context.
5.  🖱️ **Hands (Execution):** Performs the actual clicks, typing, and API calls.

## 🛠️ Architecture

ViZioN operates on a high-fidelity "See-Think-Act" loop. For a deep dive into the system design, see [ARCHITECTURE.md](ARCHITECTURE.md).

*   **Semantic Understanding:** Uses `Qwen3-VL-8B-Instruct` for deep visual grounding.
*   **Structural Grounding:** Optional integration with `PaddleOCR` for verifiable text maps.
*   **Actionable UI Scene Graph:** Converts implicit visual knowledge into a first-class, verifiable representation.
*   **Action Layer:** Supports **Mock** (logging) and **Desktop** (PyAutoGUI) execution.

## 🚀 Getting Started

### Prerequisites

*   Linux / macOS / Windows
*   **Python 3.11**
*   NVIDIA GPU (Recommended: 24GB+ VRAM)
*   Conda

### Installation

1.  **Clone and Enter:**
    ```bash
    git clone https://github.com/your-username/ViZioN.git
    cd ViZioN
    ```

2.  **Create the Environment:**
    ```bash
    conda create -n vision_env python=3.11 -y
    conda activate vision_env
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Copy the template and add your Hugging Face token:
    ```bash
    cp .env.example .env
    # Edit .env to add your HF_TOKEN
    ```

### Usage

ViZioN now supports multi-step execution loops and real-time desktop interaction.

**Basic Run (Mock Mode):**
In mock mode, you must provide a static image. The agent will simulate actions based on this image.
```bash
conda run -n vision_env python main.py \
  --image "image.png" \
  --goal "run auto battle" \
  --mode mock \
  --max_steps 5
```

**Desktop Mode (Live Execution):**
In desktop mode, the agent captures your live screen and performs real-world actions using PyAutoGUI.
```bash
conda run -n vision_env python main.py \
  --goal "Login to the game and start farming" \
  --mode desktop \
  --max_steps 10
```

**Enable Dedicated OCR:**
For tasks requiring high precision in text detection (like reading small labels or logs):
```bash
conda run -n vision_env python main.py \
  --goal "Extract the total amount" \
  --use_ocr
```

## 💾 Memory & Learning

ViZioN is not stateless. It employs a three-tier memory system to improve reliability:

*   **Contextual History (Short-Term):** Tracks the last 5 actions in the current session. These are injected into the VLM prompt to prevent repetitive loops and provide temporal awareness.
*   **Visual Memory (Spatial):** Automatically draws a **Red Cross** on screenshots at the location of the previous interaction. This visually grounds the agent, allowing it to see its own past actions.
*   **Experience Store (Semantic Long-Term):** Successful task completions are persisted to `data/long_term_memory.json`. The agent uses **Semantic Search** (Vector Embeddings) to retrieve relevant "Strategy Hints" even if the user goal is phrased differently.

## 🛡️ Safety & Reliability

ViZioN is built for safe real-world deployment with active feedback loops:

*   **Verification Loop (Self-Correction):** For every action, the agent predicts an **Expected Visual Outcome**. Before taking the next step, it re-analyzes the screen to verify if the previous action succeeded.
*   **Safety Monitor (Kill Switch):** In `desktop` mode, a background listener monitors for the **ESC** key. Pressing it instantly terminates the agent to prevent runaway actions.
*   **Headless Safe:** The action layer uses lazy loading, allowing the agent to run in headless/mock environments without crashing.

## ⚡ Performance Optimizations

ViZioN includes several built-in optimizations to ensure low latency and high reliability:

*   **Smart Polling (State Detection):** Uses **Structural Similarity (SSIM)** to detect if the screen has changed. If the screen is static (e.g., loading or idle), the agent skips expensive VLM inference to save GPU resources and time.
*   **Hybrid Perception Infrastructure:** Includes a **Fast Eyes** module (OpenCV-based template matching) to track UI elements across frames without re-invoking the full VLM.
*   **Robust JSON Repair:** The parser automatically corrects common LLM output errors (like trailing commas or unquoted keys) to ensure the "See-Think-Act" loop never breaks due to formatting issues.
*   **Quantization Support:** Optimized for `bfloat16` and compatible with 4-bit/8-bit quantization for deployment on consumer hardware.

## 🏭 Production Deployment

### 1. Hardware
*   **GPU:** NVIDIA RTX 3090/4090 or A100/H100 (24GB VRAM minimum for 8B models).

### 2. System Dependencies (Linux)
```bash
sudo apt-get update
sudo apt-get install -y scrot python3-tk python3-dev libx11-dev
```

### 3. Model Access
Authentication is handled automatically via the `HF_TOKEN` in your `.env` file. ViZioN uses `AutoModelForVision2Seq` with `trust_remote_code=True` to support the latest Qwen3 architectures.

### 5. Running as a Service
For production API access, you can wrap the agent in a FastAPI server (see `src/api.py` - coming soon) and run with Gunicorn:
```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker src.api:app
```

## 📂 Project Structure

*   `src/perception`: VLM, OCR, Layout detection, and Scene Graph schema.
*   `src/reasoning`: Planner, Memory Manager, and decision-making logic.
*   `src/understanding`: Output parsing and JSON repair.
*   `src/action`: Execution engines (Mock, Desktop).
*   `src/utils`: Vision utilities, Safety monitors (Kill Switch).
*   `tests/`: Comprehensive Pytest suite covering all core modules.

## 🧪 Testing

The project uses `pytest` for automated testing. To run the suite:
```bash
python -m pytest
```
*(Or simply `pytest` if your environment respects `pytest.ini`)*

## 📜 License

[MIT License](LICENSE)
