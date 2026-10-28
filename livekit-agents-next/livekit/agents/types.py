from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from ._generated.types import (
    ATTRIBUTE_AGENT_NAME,
    ATTRIBUTE_AGENT_STATE,
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    ATTRIBUTE_REDACTION_ENABLED,
    ATTRIBUTE_SIMULATION_ENABLED,
    ATTRIBUTE_SIMULATION_JOB_ID,
    ATTRIBUTE_SIMULATION_RUN_ID,
    ATTRIBUTE_SIMULATOR,
    ATTRIBUTE_SIMULATOR_DISPATCH,
    ATTRIBUTE_TRANSCRIPTION_EXPRESSION,
    ATTRIBUTE_TRANSCRIPTION_FINAL,
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
    CLIENT_PROTOCOL_TRANSCRIPTION_STREAMS,
    TOPIC_CHAT,
    TOPIC_TRANSCRIPTION,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
)

__all__ = [
    "ATTRIBUTE_AGENT_NAME",
    "ATTRIBUTE_AGENT_STATE",
    "ATTRIBUTE_PUBLISH_ON_BEHALF",
    "ATTRIBUTE_REDACTION_ENABLED",
    "ATTRIBUTE_SIMULATION_ENABLED",
    "ATTRIBUTE_SIMULATION_JOB_ID",
    "ATTRIBUTE_SIMULATION_RUN_ID",
    "ATTRIBUTE_SIMULATOR",
    "ATTRIBUTE_SIMULATOR_DISPATCH",
    "ATTRIBUTE_TRANSCRIPTION_EXPRESSION",
    "ATTRIBUTE_TRANSCRIPTION_FINAL",
    "ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID",
    "ATTRIBUTE_TRANSCRIPTION_TRACK_ID",
    "CLIENT_PROTOCOL_TRANSCRIPTION_STREAMS",
    "TOPIC_CHAT",
    "TOPIC_TRANSCRIPTION",
    "USERDATA_TIMED_TRANSCRIPT",
    "USERDATA_TTS_STARTED_TIME",
    "APIConnectOptions",
    "DEFAULT_API_CONNECT_OPTIONS",
    "FlushSentinel",
    "NOT_GIVEN",
    "NotGiven",
    "NotGivenOr",
    "TimedString",
]


_T = TypeVar("_T")


class FlushSentinel:
    pass


class NotGiven:
    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.is_instance_schema(cls)


NotGivenOr: TypeAlias = _T | NotGiven
NOT_GIVEN = NotGiven()


@dataclass(frozen=True)
class APIConnectOptions:
    max_retry: int = 3
    """
    Maximum number of retries to connect to the API.
    """

    retry_interval: float = 2.0
    """
    Interval between retries to connect to the API in seconds.
    """

    timeout: float = 10.0
    """
    Timeout for connecting to the API in seconds.
    """

    def __post_init__(self) -> None:
        if self.max_retry < 0:
            raise ValueError("max_retry must be greater than or equal to 0")

        if self.retry_interval < 0:
            raise ValueError("retry_interval must be greater than or equal to 0")

        if self.timeout < 0:
            raise ValueError("timeout must be greater than or equal to 0")

    def _interval_for_retry(self, num_retries: int) -> float:
        """
        Return the interval for the given number of retries.

        The first retry is immediate, and then uses specified retry_interval
        """
        from . import _core

        return _core.api_retry_interval(self.retry_interval, num_retries)  # type: ignore[no-any-return]


DEFAULT_API_CONNECT_OPTIONS = APIConnectOptions()


class TimedString(str):
    """A string with optional start and end timestamps for word-level alignment.

    Attributes:
        start_time: Word start time in seconds (NOT_GIVEN when unavailable).
        end_time: Word end time in seconds (NOT_GIVEN when unavailable).
        confidence: Per-word confidence score (NOT_GIVEN when unavailable).
        start_time_offset: Offset in seconds relative to the start of the audio
            input stream or session. Used by STT plugins to align words against
            the session timeline (NOT_GIVEN when unavailable).
        speaker_id: Speaker identifier when the provider supports diarization.
            Uses ``str | None`` rather than ``NotGivenOr[str]`` because the
            absence of a speaker is a routine, expected case across all
            providers — not a "not given" boundary condition — and downstream
            consumers gate on ``speaker_id is None`` rather than ``is_given``.
    """

    start_time: NotGivenOr[float]
    end_time: NotGivenOr[float]
    confidence: NotGivenOr[float]
    start_time_offset: NotGivenOr[float]
    speaker_id: str | None

    def __new__(
        cls,
        text: str,
        start_time: NotGivenOr[float] = NOT_GIVEN,
        end_time: NotGivenOr[float] = NOT_GIVEN,
        confidence: NotGivenOr[float] = NOT_GIVEN,
        start_time_offset: NotGivenOr[float] = NOT_GIVEN,
        speaker_id: str | None = None,
    ) -> "TimedString":
        obj = super().__new__(cls, text)
        obj.start_time = start_time
        obj.end_time = end_time
        obj.confidence = confidence
        obj.start_time_offset = start_time_offset
        obj.speaker_id = speaker_id
        return obj
