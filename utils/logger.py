import logging
from logging.handlers import RotatingFileHandler

from utils.directories import PIPELINE_LOG_DIR

def get_logger(name):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        PIPELINE_LOG_DIR,
        maxBytes=10 * 1024 * 1024,
        backupCount=10
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger