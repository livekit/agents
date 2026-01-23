from livekit import rtc

from . import aio, audio, codecs, encryption, http_context, http_server, hw, images
from .audio import AudioBuffer, combine_frames, merge_frames
from .bounded_dict import BoundedDict
from .connection_pool import ConnectionPool
from .exp_filter import ExpFilter
from .log import log_exceptions
from .misc import is_given, nodename, shortuuid, time_ms
from .moving_average import MovingAverage
from .participant import wait_for_participant, wait_for_track_publication

# Optional session_store module (requires apsw)
try:
    from . import session_store, session_store_integration
    from .session_store import SessionStore, StoredState
    from .session_store_integration import (
        create_store_for_session,
        rehydrate_agent_session_from_store,
        serialize_agent_session_to_store,
    )

    _HAS_SESSION_STORE = True
except ImportError:
    _HAS_SESSION_STORE = False
    session_store = None  # type: ignore
    session_store_integration = None  # type: ignore

EventEmitter = rtc.EventEmitter

__all__ = [
    "AudioBuffer",
    "merge_frames",
    "combine_frames",
    "time_ms",
    "nodename",
    "shortuuid",
    "http_context",
    "http_server",
    "ExpFilter",
    "MovingAverage",
    "BoundedDict",
    "EventEmitter",
    "log_exceptions",
    "codecs",
    "images",
    "audio",
    "aio",
    "hw",
    "is_given",
    "ConnectionPool",
    "wait_for_participant",
    "wait_for_track_publication",
    "encryption",
]

# Add session_store exports if available
if _HAS_SESSION_STORE:
    __all__.extend(
        [
            "session_store",
            "session_store_integration",
            "SessionStore",
            "StoredState",
            "serialize_agent_session_to_store",
            "rehydrate_agent_session_from_store",
            "create_store_for_session",
        ]
    )

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
