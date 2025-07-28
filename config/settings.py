import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Local model paths
LOCAL_QWEN_MODEL_PATH = MODELS_DIR / "qwen3-1.7b"

# Model settings - Use local Qwen model if available, fallback to HuggingFace
if LOCAL_QWEN_MODEL_PATH.exists():
    PROBLEM_GENERATOR_MODEL = str(LOCAL_QWEN_MODEL_PATH)
    EXPLANATION_MODEL = str(LOCAL_QWEN_MODEL_PATH)
else:
    # Fallback to online models
    PROBLEM_GENERATOR_MODEL = os.getenv("PROBLEM_GENERATOR_MODEL", "Qwen/Qwen3-1.7B")
    EXPLANATION_MODEL = os.getenv("EXPLANATION_MODEL", "Qwen/Qwen3-1.7B")

# Mock mode setting - set to True to run without loading models
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Application settings
DEFAULT_NUM_PROBLEMS = 5
DEFAULT_DIFFICULTY = "medium"  # easy, medium, hard
DEFAULT_TOPIC = "algebra"
DEFAULT_TIME_MINUTES = 10
AVAILABLE_TOPICS = [
    "algebra",
    "geometry",
    "trigonometry",
    "calculus",
    "statistics",
    "probability",
]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Cache settings
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
