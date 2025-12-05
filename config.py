"""
Configuration settings for the sentiment analysis application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv



# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# Processing Configuration
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

# Rate Limiting (GPT-4o-mini tier 1 defaults)
RATE_LIMIT_RPM = 500  # Requests per minute
RATE_LIMIT_TPM = 200000  # Tokens per minute

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
RETRY_MULTIPLIER = 2  # exponential backoff multiplier

# Cost Configuration (per 1M tokens)
COST_INPUT_PER_1M = 0.150  # $0.15 per 1M input tokens
COST_OUTPUT_PER_1M = 0.600  # $0.60 per 1M output tokens

# Sentiment Categories
SENTIMENT_CATEGORIES = ["Positive", "Neutral", "Negative"]

# File Configuration
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
