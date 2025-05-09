"""
Logging utility for the Financial API Integration project.
"""

import logging
import sys
from pathlib import Path
from config.settings import LOG_LEVEL, BASE_DIR

# Create logs directory if it doesn't exist
logs_dir = BASE_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Name of the logger (typically __name__ from the calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure handlers if they haven't been set up yet
    if not logger.handlers:
        # Set log level from configuration
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # File handler
        file_handler = logging.FileHandler(logs_dir / "financial_api.log")
        file_handler.setLevel(level)
        
        # Format for both handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger