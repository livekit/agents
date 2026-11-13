"""The transcript response shared by Baseten Whisper and Qwen3 deployments."""

from pydantic import BaseModel, ValidationError

from livekit.agents import APIError


class WordTimestamp(BaseModel):
    word: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    prob: float = 0.0


class Segment(BaseModel):
    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    word_timestamps: list[WordTimestamp] | None = None


class TranscriptResponse(BaseModel):
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
        raise APIError(
            "Baseten returned an invalid transcript", kind="protocol", detail=payload
        ) from error
