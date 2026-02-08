import logging
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import json
from src.agent import VisualAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ViZioN Server API")

# Global agent instance (lazy initialization or per request)
# In a real production system, we might use sessions to maintain memory.
agent = None

def get_agent():
    global agent
    if agent is None:
        # Use a special 'remote' mode that doesn't try to use local GUI
        agent = VisualAgent(mode="remote", use_ocr=False)
    return agent

@app.get("/")
async def root():
    return {"status": "ViZioN Server is running"}

@app.post("/process_step")
async def process_step(
    goal: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Receives an image and a goal from the Windows client.
    Returns the next planned action.
    """
    logger.info(f"Received step request for goal: {goal}")
    
    # 1. Read Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # 2. Get Agent
    current_agent = get_agent()
    
    # 3. Run a single step (Planning)
    # Note: run_step in VisualAgent currently returns the plan dictionary
    plan = current_agent.run_step(image, goal)
    
    return plan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
