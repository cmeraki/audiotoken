import os
import sys
import logging
from typing import Optional
from logging.handlers import RotatingFileHandler

def get_logger(name:str, log_file:Optional[str] = "app.log", level:str = "ERROR"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove all handlers associated with the logger object
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s | %(processName)s | %(thread)d | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S.%f'

    if log_file:
        os.makedirs("logs", exist_ok=True)
        log_file = os.path.join("logs", log_file)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

# Usage example
if __name__ == "__main__":
    logger = get_logger(__name__, log_file=None, level="DEBUG")

    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
