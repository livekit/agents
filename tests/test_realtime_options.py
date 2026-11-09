from __future__ import annotations

from dataclasses import replace

import pytest

from livekit.agents import NOT_GIVEN, _core
from livekit.agents.llm._realtime import AudioTranscription, RealtimeModel, ServerVad
from livekit.agents.llm.realtime import RealtimeModel as HostRealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


def test_omitted_and_null_options_preserve_or_disable_transcription() -> None:
    model = RealtimeModel(api_key="test")
    assert model.capabilities.user_transcription

    model.update_options(input_audio_transcription=None)
    assert not model.capabilities.user_transcription

    model.update_options(voice="marin")
    assert not model.capabilities.user_transcription

    model.update_options(
        input_audio_transcription=AudioTranscription(model="gpt-4o-mini-transcribe")
    )
    assert model.capabilities.user_transcription

    # An explicit NOT_GIVEN is omitted, not None.
    model.update_options(input_audio_transcription=NOT_GIVEN)
    assert model.capabilities.user_transcription

    disabled = RealtimeModel(api_key="test", input_audio_transcription=None)
    assert not disabled.capabilities.user_transcription


def test_turn_detection_records_reach_the_core() -> None:
    model = RealtimeModel(
        api_key="test", turn_detection=ServerVad(create_response=False, interrupt_response=False)
    )
    assert not model.capabilities.turn_detection
    model.update_options(turn_detection=ServerVad())
    assert model.capabilities.turn_detection


def test_option_dicts_are_refused() -> None:
    with pytest.raises(TypeError):
        RealtimeModel(api_key="test", turn_detection={"type": "server_vad"})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fallback_accepts_native_and_host_models_and_rejects_capability_mismatch() -> None:
    class HostModel(HostRealtimeModel):
        def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
            raise AssertionError("constructing a fallback must not open provider sessions")

    native = RealtimeModel(api_key="test")
    host = HostModel(capabilities=native.capabilities)
    fallback = _core.RealtimeFallbackModel([native._core_leg(), host])
    assert fallback.capabilities.audio_output
    await fallback.restart(switch_model=True)

    incompatible = HostModel(capabilities=replace(native.capabilities, audio_output=False))
    with pytest.raises(ValueError, match="audio_output"):
        _core.RealtimeFallbackModel([native._core_leg(), incompatible])


@pytest.mark.parametrize("cooldown", [-1.0, float("inf"), float("nan"), 1e300])
def test_fallback_rejects_invalid_cooldown_before_opening_a_session(cooldown: float) -> None:
    native = RealtimeModel(api_key="test")
    with pytest.raises(ValueError, match="cooldown"):
        _core.RealtimeFallbackModel([native._core_leg()], cooldown=cooldown)


# Catches: a message cut before any word was heard (transcript "") truncated as "not given",
# which keeps the whole text on the plugin's side.
def test_an_empty_truncate_transcript_reaches_the_plugin_as_empty() -> None:
    from livekit.agents._generated import realtime_wire

    from .fake_realtime import FakeRealtimeModel

    session = FakeRealtimeModel().session()
    for transcript in ("", None):
        session._core_send(
            realtime_wire.Truncate(
                message_id="m1", has_audio=False, audio_end_ms=0, audio_transcript=transcript
            )
        )
    assert session.truncated == ["", NOT_GIVEN]
