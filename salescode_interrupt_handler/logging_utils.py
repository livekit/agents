import logging
import os

LOG_LEVEL = os.getenv("INTERRUPT_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("salescode.interrupts")


def get_logger(name: str = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"salescode.interrupts.{name}")
    return logger
