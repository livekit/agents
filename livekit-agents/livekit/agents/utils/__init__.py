from livekit import rtc

from . import aio, audio, codecs, http_context, http_server, hw, images, large_payload
from .audio import AudioArrayBuffer, AudioBuffer, combine_frames, merge_frames
from .bounded_dict import BoundedDict
from .connection_pool import ConnectionPool
from .env import resolve_env_var
from .exp_filter import ExpFilter
from .large_payload import (
    DEFAULT_INLINE_PAYLOAD_LIMIT,
    DEFAULT_STREAM_PAYLOAD_LIMIT,
    LargePayloadChecksumError,
    LargePayloadDescriptor,
    LargePayloadDescriptorTooLargeError,
    LargePayloadError,
    LargePayloadInfo,
    LargePayloadTooLargeError,
    parse_large_payload_descriptor,
    publish_large_payload,
    read_large_payload_stream,
)
from .log import log_exceptions
from .misc import is_dev_mode, is_given, is_hosted, nodename, shortuuid, time_ms
from .moving_average import MovingAverage
from .participant import wait_for_agent, wait_for_participant, wait_for_track_publication

EventEmitter = rtc.EventEmitter

__all__ = [
    "AudioBuffer",
    "AudioArrayBuffer",
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
    "large_payload",
    "is_dev_mode",
    "is_given",
    "is_hosted",
    "ConnectionPool",
    "wait_for_agent",
    "wait_for_participant",
    "wait_for_track_publication",
    "resolve_env_var",
    "DEFAULT_INLINE_PAYLOAD_LIMIT",
    "DEFAULT_STREAM_PAYLOAD_LIMIT",
    "LargePayloadChecksumError",
    "LargePayloadDescriptor",
    "LargePayloadDescriptorTooLargeError",
    "LargePayloadError",
    "LargePayloadInfo",
    "LargePayloadTooLargeError",
    "parse_large_payload_descriptor",
    "publish_large_payload",
    "read_large_payload_stream",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
