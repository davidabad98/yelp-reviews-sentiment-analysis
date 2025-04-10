"""
Logging utilities for the Yelp sentiment analysis project.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (None for console only)
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
