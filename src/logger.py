import logging
import os
from datetime import datetime 

LOG_FILE_NAME = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)  

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    
    # Add file handler if not already added
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test_logger")
    logger.info("Logger initialized successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print(f"Log file created at: {LOG_FILE_PATH}")