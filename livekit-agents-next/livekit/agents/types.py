from typing import TypeAlias, TypeVar

from ._generated.connect import (
    DEFAULT_API_CONNECT_OPTIONS as DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions as APIConnectOptions,
)
from ._generated.transcript import TranscriptChunk as TranscriptChunk
from ._generated.types import (
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    NOT_GIVEN as NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    FlushSentinel as FlushSentinel,
    NotGiven as NotGiven,
)

__all__ = [
    "ATTRIBUTE_PUBLISH_ON_BEHALF",
    "USERDATA_TIMED_TRANSCRIPT",
    "USERDATA_TTS_STARTED_TIME",
    "APIConnectOptions",
    "DEFAULT_API_CONNECT_OPTIONS",
    "FlushSentinel",
    "NOT_GIVEN",
    "NotGiven",
    "NotGivenOr",
    "TranscriptChunk",
]


_T = TypeVar("_T")


NotGivenOr: TypeAlias = _T | NotGiven
