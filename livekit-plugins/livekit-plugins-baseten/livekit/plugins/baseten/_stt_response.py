"""The transcript response shared by Baseten Whisper and Qwen3 deployments."""

from pydantic import BaseModel, ConfigDict, ValidationError

from livekit.agents import APIConnectionError


class WordTimestamp(BaseModel):
    model_config = ConfigDict(strict=True)

    word: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    prob: float = 0.0


class Segment(BaseModel):
    model_config = ConfigDict(strict=True)

    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    word_timestamps: list[WordTimestamp] | None = None


class TranscriptResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    type: str = "transcription"
    message: str = "unknown STT error"
    segments: list[Segment] | None = None
    transcript: str | None = None
    confidence: float = 0.0
    language_code: str | None = None
    is_final: bool | None = None
    is_end_of_audio_flush: bool = False


def decode_transcript(payload: str) -> TranscriptResponse:
    try:
        return TranscriptResponse.model_validate_json(payload)
    except ValidationError as error:
        raise APIConnectionError(
            "Baseten returned an invalid transcript", retryable=False
        ) from error
