from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


@dataclass
class Message:
    """
    Dataclass abstraction for message
    """
    event: str
    payload: Dict[str, Any]
    ref: Any
    topic: str

    def __hash__(self):
        return hash((self.event, tuple(list(self.payload.values())), self.ref, self.topic))


class ChannelEvents(str, Enum):
    """
    ChannelEvents are a bunch of constant strings that are defined according to
    what the Phoenix realtime server expects.
    """

    close = "phx_close"
    error = "phx_error"
    join = "phx_join"
    reply = "phx_reply"
    leave = "phx_leave"
    heartbeat = "heartbeat"


PHOENIX_CHANNEL = "phoenix"
HEARTBEAT_PAYLOAD = {"msg": "ping"}
