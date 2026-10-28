import asyncio
import contextlib
from collections.abc import AsyncIterator
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
    """Per-request connection policy: retries and the limit of each wait.

    Each ``*_timeout`` is a finite number of seconds >= 0, or ``None`` to disable that
    limit. Invalid values raise ``ValueError`` naming the field before any work starts.
    """

    max_retry: int = 3
    """Attempts after the first."""

    retry_interval: float = 2.0
    """Seconds between retries (the first retry waits 0.1 s)."""

    pool_timeout: float | None = 10.0
    """Wait for transport capacity or a pooled connection."""

    connect_timeout: float | None = 10.0
    """Establish a usable connection, including DNS and TLS."""

    write_timeout: float | None = 10.0
    """Wait without progress while writing the request."""

    read_timeout: float | None = 10.0
    """Wait without progress while a response is expected."""

    total_timeout: float | None = None
    """The whole operation, including retries and fallback."""

    def __post_init__(self) -> None:
        from . import _core

        if isinstance(self.max_retry, bool) or not isinstance(self.max_retry, int):
            raise ValueError("conn_options.max_retry must be an integer >= 0")
        _core.api_connect_options_check(
            self.max_retry,
            self.retry_interval,
            self.pool_timeout,
            self.connect_timeout,
            self.write_timeout,
            self.read_timeout,
            self.total_timeout,
        )

    @contextlib.asynccontextmanager
    async def total_deadline(self) -> AsyncIterator[None]:
        """Bound an operation, retries included, by ``total_timeout``.

        Raises:
            APITimeoutError: with ``phase="total"`` once the limit expires.
        """
        from ._exceptions import APITimeoutError

        try:
            async with asyncio.timeout(self.total_timeout) as deadline:
                yield
        except TimeoutError as e:
            if not deadline.expired():
                raise
            raise APITimeoutError(
                "Operation exceeded total_timeout", retryable=False, phase="total"
            ) from e

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
