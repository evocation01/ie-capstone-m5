import logging
import sys


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Configures a simple logger with stdout handler.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if function is called multiple times
    if not logger.handlers:
        logger.setLevel(log_level)

        # Create handler that writes to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger
    return logger
