import logging

def setup_logger(name='qjn'):
    """Configure and return a logger instance."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger