import logging
import sys
import psutil

def setup_detailed_logging():
    """Configure detailed logging with custom formatting"""
    logger = logging.getLogger(__name__)
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pdf_processor.log"),
        ],
    )
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s"
    )

    # Update existing handlers with new formatter
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # Add memory usage to log messages
    def memory_usage_mb():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    # Add custom filter to include memory usage
    class MemoryFilter(logging.Filter):
        def filter(self, record):
            record.memory_mb = f"{memory_usage_mb():.2f}MB"
            return True

    logger.addFilter(MemoryFilter())
    
    # Set specific loggers
    logging.getLogger("unstructured").setLevel(logging.DEBUG)
    logging.getLogger("processor").setLevel(logging.DEBUG)
    
    return logger
