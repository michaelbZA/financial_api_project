"""
Configuration settings for the Financial API Integration project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Default settings
DEFAULT_STOCKS = os.getenv("DEFAULT_STOCKS", "AAPL,MSFT,GOOGL,AMZN,BRK-B").split(",")
DEFAULT_INDICES = os.getenv("DEFAULT_INDICES", "^GSPC,^DJI,^IXIC").split(",")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1y")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# API rate limits
ALPHA_VANTAGE_CALLS_PER_MINUTE = 5  # Free tier limit
YAHOO_FINANCE_CALLS_PER_MINUTE = 100  # Conservative estimate

# Technical analysis settings
MOVING_AVERAGE_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20