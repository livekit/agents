"""The transcript response shared by Baseten Whisper and Qwen3 deployments."""

from typing import TypedDict


class WordTimestamp(TypedDict, total=False):
    word: str
    start_time: float
    end_time: float
    prob: float


class Segment(TypedDict, total=False):
    text: str
    start_time: float
    end_time: float
    word_timestamps: list[WordTimestamp] | None


class _EndAudioBody(TypedDict, total=False):
    status: str


class TranscriptResponse(TypedDict, total=False):
    """A message, untagged: ``type`` defaults to ``"transcription"``."""

    type: str
    body: _EndAudioBody
    message: str
    segments: list[Segment] | None
    transcript: str | None
    confidence: float
    language_code: str | None
    is_final: bool | None
    is_end_of_audio_flush: bool
