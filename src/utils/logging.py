import logging
import sys
import os

def get_logger(name="CAMUS", log_file=None):
    """
    Retrieves a logger instance configured with stream and optional file handlers.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Optional path to a log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check rank
    rank = int(os.environ.get("RANK", 0))
    
    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    if not logger.handlers:
        # Stream Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    # File Handler (add if log_file is provided and we are rank 0)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger