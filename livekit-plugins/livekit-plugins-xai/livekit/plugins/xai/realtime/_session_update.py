from __future__ import annotations

from openai.types.realtime import (
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeAudioConfigOutput,
    RealtimeSessionCreateRequest,
)


def lift_xai_session_fields(session: RealtimeSessionCreateRequest) -> None:
    """Move voice/turn_detection to top-level session fields for xAI."""
    audio = session.audio
    if not isinstance(audio, RealtimeAudioConfig):
        return
    output = audio.output
    if isinstance(output, RealtimeAudioConfigOutput) and "voice" in output.model_fields_set:
        session.voice = output.voice  # type: ignore[attr-defined]
        output.model_fields_set.discard("voice")
    audio_input = audio.input
    if (
        isinstance(audio_input, RealtimeAudioConfigInput)
        and "turn_detection" in audio_input.model_fields_set
    ):
        session.turn_detection = audio_input.turn_detection  # type: ignore[attr-defined]
        audio_input.model_fields_set.discard("turn_detection")
    out_set = isinstance(output, RealtimeAudioConfigOutput) and bool(output.model_fields_set)
    in_set = isinstance(audio_input, RealtimeAudioConfigInput) and bool(
        audio_input.model_fields_set
    )
    if not out_set and not in_set:
        session.model_fields_set.discard("audio")
