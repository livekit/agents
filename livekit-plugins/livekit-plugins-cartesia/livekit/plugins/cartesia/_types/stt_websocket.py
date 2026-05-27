from typing import Literal

from typing_extensions import NotRequired, TypedDict


class STTWord(TypedDict):
    """Word-level timestamp from Cartesia STT.

    Attributes:
        word: The transcribed word.
        start: Start time in seconds.
        end: End time in seconds.
    """

    word: str
    start: float
    end: float


class STTTranscriptEvent(TypedDict):
    """Transcript chunk for the current connection.

    Each event is a delta from the last chunk with ``is_final=True``, not the
    cumulative transcript.

    Attributes:
        type: Event discriminator.
        is_final: Whether ``text`` is finalized.
        request_id: Unique identifier for this WebSocket connection.
        text: Transcribed text delta.
        duration: Duration of the audio in seconds.
        words: Optional word-level timestamps.
    """

    type: Literal["transcript"]
    is_final: bool
    request_id: str
    text: str
    duration: NotRequired[float]
    words: NotRequired[list[STTWord]]


class STTFlushDoneEvent(TypedDict):
    """Acknowledgment for the ``finalize`` command.

    Attributes:
        type: Event discriminator.
        request_id: Unique identifier for this WebSocket connection.
    """

    type: Literal["flush_done"]
    request_id: str


class STTDoneEvent(TypedDict):
    """Acknowledgment for the ``close`` command; session is closing.

    Attributes:
        type: Event discriminator.
        request_id: Unique identifier for this WebSocket connection.
    """

    type: Literal["done"]
    request_id: str


class STTErrorEvent(TypedDict):
    """Error event sent by the server.

    Attributes:
        type: Event discriminator.
        code: HTTP-style status code; values >= 500 are treated as retryable.
        message: Human-readable error message.
        request_id: Unique identifier for this WebSocket connection.
    """

    type: Literal["error"]
    code: NotRequired[int]
    message: NotRequired[str]
    request_id: NotRequired[str]


STTEventMessage = STTTranscriptEvent | STTFlushDoneEvent | STTDoneEvent | STTErrorEvent
"""Server-sent message on the ``/stt/websocket`` endpoint."""
