import logging

DEV_LEVEL = 23
logging.addLevelName(DEV_LEVEL, "DEV")

logger = logging.getLogger("livekit.agents")
