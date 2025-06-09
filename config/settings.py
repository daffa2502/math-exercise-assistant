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

# Model settings
PROBLEM_GENERATOR_MODEL = os.getenv("PROBLEM_GENERATOR_MODEL", "google/flan-t5-base")
EXPLANATION_MODEL = os.getenv("EXPLANATION_MODEL", "google/flan-t5-large")

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
