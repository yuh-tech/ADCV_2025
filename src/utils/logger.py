"""
Logging utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = 'land_cover_segmentation',
    log_file: Optional[Path] = None,
    log_level: str = 'INFO',
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_string: Custom format string (optional)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'land_cover_segmentation') -> logging.Logger:
    """
    Get existing logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

