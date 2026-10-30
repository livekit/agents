"""SDK values converted to the generated realtime option record."""

from __future__ import annotations

from pydantic import BaseModel

from .._generated.realtime_options import RealtimeOptions
from ..types import NOT_GIVEN, NotGiven, NotGivenOr
from .tool_context import ToolChoice


def _json(value: object) -> object:
    """A provider's pydantic option as the JSON it sends; ``NOT_GIVEN`` and ``None`` as they are."""
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, exclude_unset=True)
    return value


def realtime_options(
    *,
    input_audio_transcription: object = NOT_GIVEN,
    input_audio_noise_reduction: object = NOT_GIVEN,
    turn_detection: object = NOT_GIVEN,
    tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
    speed: NotGivenOr[float] = NOT_GIVEN,
    tracing: object = NOT_GIVEN,
    truncation: object = NOT_GIVEN,
    reasoning: object = NOT_GIVEN,
    max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
    voice: NotGivenOr[str] = NOT_GIVEN,
    max_response_output_tokens: object = NOT_GIVEN,
) -> RealtimeOptions:
    return RealtimeOptions(
        input_audio_transcription=_json(input_audio_transcription),
        input_audio_noise_reduction=_json(input_audio_noise_reduction),
        turn_detection=_json(turn_detection),
        tool_choice=tool_choice if isinstance(tool_choice, NotGiven) else tool_choice or None,
        speed=speed,
        tracing=_json(tracing),
        truncation=_json(truncation),
        reasoning=_json(reasoning),
        max_session_duration=max_session_duration,
        voice=voice,
        max_response_output_tokens=_json(max_response_output_tokens),
    )
