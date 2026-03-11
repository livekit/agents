"""PersonaPlex experimental support for NVIDIA LiveKit plugin.

Support for NVIDIA PersonaPlex full-duplex conversational AI model.
"""

from . import realtime
from .models import PersonaplexVoice
from .realtime.realtime_model import RealtimeModel, RealtimeSession

__all__ = [
    "PersonaplexVoice",
    "realtime",
    "RealtimeModel",
    "RealtimeSession",
]
