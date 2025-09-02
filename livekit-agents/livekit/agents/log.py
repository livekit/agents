"""
Logging configuration for LiveKit Agents.

This module sets up logging for the LiveKit Agents framework.

Environment Variables:
    LIVEKIT_LOGGER_NAME: Custom logger name to use instead of the default one.
                         This allows applications to customize the logger name for
                         better integration with their logging infrastructure.
"""

import logging
import os

DEV_LEVEL = 23
logging.addLevelName(DEV_LEVEL, "DEV")

DEFAULT_LOGGER_NAME = "livekit.agents"
logger = logging.getLogger(os.environ.get("LIVEKIT_LOGGER_NAME", DEFAULT_LOGGER_NAME))
