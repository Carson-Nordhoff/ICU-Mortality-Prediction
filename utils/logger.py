import logging
from logging.handlers import RotatingFileHandler

from pathlib import Path

from utils.directories import ARTIFACT_DIR, LOG_DIR, ML_PIPELINE_LOG_DIR, SQL_PIPELINE_LOG_DIR

def get_logger(name, output_file):

    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

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
        output_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8" #avoid special char crashes
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_ml_system_logger(name):
    return get_logger(name, ML_PIPELINE_LOG_DIR)

def get_sql_system_logger(name):
    return get_logger(name, SQL_PIPELINE_LOG_DIR)