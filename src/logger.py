import os
import sys
import json
from loguru import logger


def serialize_log(record):
    subset = {
        "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "level": record["level"].name,
        "message": record["message"],
        "thread_id": record["thread"].id,
        "process_id": record["process"].id
    }
    # You can add more fields here if needed
    return subset

def get_logger(log_file="app.log"):

    os.makedirs("logs", exist_ok=True)
    logger.remove()

    log_file = os.path.join("logs", log_file)

    format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message} | {thread.id} | {process.name}"

    logger.add(
        log_file,
        format=format,
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        enqueue=True,
    )

    logger.add(
        sys.stderr,
        format=format,
        level="ERROR",
    )


get_logger(log_file="app.log")

# Usage example
if __name__ == "__main__":
    get_logger()
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
