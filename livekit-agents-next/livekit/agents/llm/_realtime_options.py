"""SDK values converted to the generated realtime option records."""

from __future__ import annotations

from pydantic import BaseModel

from .._generated.realtime_options import (
    RealtimeDurationOption,
    RealtimeJsonOption,
    RealtimeOptions,
    RealtimeToolChoice,
)
from ..types import NOT_GIVEN, NotGiven, NotGivenOr
from .tool_context import ToolChoice


def _json_option(value: object) -> RealtimeJsonOption | None:
    if isinstance(value, NotGiven):
        return None
    if isinstance(value, BaseModel):
        value = value.model_dump(by_alias=True, exclude_unset=True)
    return RealtimeJsonOption(value=value)


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
        input_audio_transcription=_json_option(input_audio_transcription),
        input_audio_noise_reduction=_json_option(input_audio_noise_reduction),
        turn_detection=_json_option(turn_detection),
        tool_choice=None
        if isinstance(tool_choice, NotGiven)
        else RealtimeToolChoice(value=tool_choice or None),
        speed=None if isinstance(speed, NotGiven) else speed,
        tracing=_json_option(tracing),
        truncation=_json_option(truncation),
        reasoning=_json_option(reasoning),
        max_session_duration=None
        if isinstance(max_session_duration, NotGiven)
        else RealtimeDurationOption(value=max_session_duration),
        voice=None if isinstance(voice, NotGiven) else voice,
        max_response_output_tokens=_json_option(max_response_output_tokens),
    )
