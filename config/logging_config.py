"""Logging setup shared across modules.

    from config.logging_config import setup_logger
    logger = setup_logger("smartdesk.synthesizer")

Level comes from LOG_LEVEL in .env (default INFO). Handlers are added once per logger name.
"""
import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    return logger
