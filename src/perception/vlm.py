import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from src.config import Config
import logging
import os

logger = logging.getLogger(__name__)

class VLMDetector:
    """
    Handles Semantic Understanding and VLM-based Visual Grounding.
    Uses the latest Qwen3 architecture via Auto classes.
    """
    def __init__(self, model_id=Config.MODEL_ID, device=Config.DEVICE):
        self.device = device
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading VLM: {self.model_id} on {self.device}...")
        
        # Use token from config if available
        hf_token = Config.HF_TOKEN
        
        try:
            # Using AutoModelForImageTextToText for broader compatibility with Qwen2/2.5/3 in transformers v5
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=Config.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                token=hf_token,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                token=hf_token,
                trust_remote_code=True
            )
            
            if self.device != "cuda" and self.model.device.type != "cuda":
                 self.model.to(self.device)
                 
            logger.info("VLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise

    def analyze(self, image: Image.Image, prompt_text: str) -> str:
        """
        Standard VLM inference.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text