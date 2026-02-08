import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # VLLM Server Settings
    VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8051/v1/chat/completions")
    VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

    # Model Settings
    # Using the specific model requested by the user
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct" 
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Authentication
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Generation Settings
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.1 # Low temperature for precise actions/analysis
    TOP_P = 0.9

    # System Prompts
    SYSTEM_PROMPT = """You are ViZioN, an advanced visual AI agent. 
Your goal is to perceive the visual world, understand the structure of user interfaces and documents, and plan actions to fulfill user requests.
You output structured reasoning and plans.
"""
