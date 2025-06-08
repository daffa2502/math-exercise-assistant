import logging
import sys
from pathlib import Path

# Add the parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(parent_dir / "math_assistant.log"),
    ],
)

# Import the UI module (this will be used when running this file directly)
from ui import main

if __name__ == "__main__":
    # Run the Streamlit application
    main()
