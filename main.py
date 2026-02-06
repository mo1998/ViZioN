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
    parser.add_argument("--image", type=str, help="Path to the input image (screenshot). Required for mock mode.")
    parser.add_argument("--goal", type=str, required=True, help="The user's high-level goal.")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "desktop"], help="Execution mode.")
    parser.add_argument("--use_ocr", action="store_true", help="Enable dedicated OCR (PaddleOCR).")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to run.")
    
    args = parser.parse_args()

    if args.mode == "mock" and not args.image:
        parser.error("--image is required when mode is 'mock'.")

    print(f"Starting ViZioN with goal: '{args.goal}'")
    if args.image:
        print(f"Initial image: '{args.image}'")
    
    try:
        agent = VisualAgent(mode=args.mode, use_ocr=args.use_ocr)
        # Pass image path for mock mode or as initial state
        agent.run(args.goal, initial_image_path=args.image, max_steps=args.max_steps)
    except Exception as e:
        logging.error(f"Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()