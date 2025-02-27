import logging
from logging.handlers import TimedRotatingFileHandler
import os

def setup_logger(name, log_file="trading.log"):
    """配置并返回日志器"""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    _logger = logging.getLogger(name)  # 重命名避免shadowing
    _logger.setLevel(logging.DEBUG)
    log_filename = f"logs/{log_file}"
    file_handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1, backupCount=30)
    file_handler.setLevel(logging.DEBUG)
    file_handler.suffix = "%Y-%m-%d"
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger

if __name__ == "__main__":
    logger = setup_logger("test")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.error("This is an error message")