import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

def setup_logger(name, log_file="trading.log"):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_filename = f"logs/{log_file}"
    file_handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1, backupCount=30)  # 保留30天
    file_handler.setLevel(logging.DEBUG)
    file_handler.suffix = "%Y-%m-%d"
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger