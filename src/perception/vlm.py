import base64
import requests
from PIL import Image
from src.config import Config
import logging
import io

logger = logging.getLogger(__name__)

class VLMDetector:
    """
    Handles Semantic Understanding and VLM-based Visual Grounding using a vLLM server.
    """
    def __init__(self):
        logger.info(f"VLMDetector initialized to use vLLM server at {Config.VLLM_URL}")

    def analyze(self, image: Image.Image, prompt_text: str) -> str:
        """
        Sends an image and a prompt to the vLLM server for analysis.
        """
        try:
            # Encode image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            payload = {
                "model": Config.VLLM_MODEL_ID,
                "messages": messages,
                "max_tokens": Config.MAX_NEW_TOKENS,
                "temperature": Config.TEMPERATURE,
                "top_p": Config.TOP_P
            }

            headers = {"Content-Type": "application/json"}

            response = requests.post(Config.VLLM_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()
            output_text = response_data['choices'][0]['message']['content']
            return output_text

        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with vLLM server: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format from vLLM server: {e}")
            logger.error(f"Response: {response_data}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during VLM analysis: {e}")
            raise