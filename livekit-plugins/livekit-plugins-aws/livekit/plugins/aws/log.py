import logging

logger = logging.getLogger("livekit.plugins.aws")
for logger_name in ["botocore", "aiobotocore"]:
    logging.getLogger(logger_name).setLevel(logging.INFO)
