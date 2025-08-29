import logging
import sys


def get_logger(name: str = "flekspy", level: str = "INFO") -> logging.Logger:
    """Helper function to set up a logger."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    # Prevent duplicate messages
    logger.propagate = False
    return logger
