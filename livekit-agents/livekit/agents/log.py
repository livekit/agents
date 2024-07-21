import logging

DEV_LEVEL = 25
logging.addLevelName(DEV_LEVEL, "DEV")

logger = logging.getLogger("livekit.agents")
