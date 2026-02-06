import argparse
import sys
import logging
from src.agent import VisualAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="ViZioN: Visual AI Agent CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image (screenshot).")
    parser.add_argument("--goal", type=str, required=True, help="The user's high-level goal.")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "desktop"], help="Execution mode.")
    parser.add_argument("--use_ocr", action="store_true", help="Enable dedicated OCR (PaddleOCR).")
    
    args = parser.parse_args()

    print(f"Starting ViZioN with goal: '{args.goal}' on image: '{args.image}'")
    
    try:
        agent = VisualAgent(mode=args.mode, use_ocr=args.use_ocr)
        agent.run_step(args.image, args.goal)
    except Exception as e:
        logging.error(f"Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()