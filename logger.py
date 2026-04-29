from src.logger import get_logger

if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test_logger")
    logger.info("Logger initialized successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print("Logger test completed - check the logs directory for the log file")