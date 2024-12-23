import logging

logger = logging.getLogger("livekit.plugins.playai")
# suppress verbose websocket logs
logging.getLogger("websockets.client").setLevel(logging.INFO)
