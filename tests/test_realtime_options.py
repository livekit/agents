from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from livekit.agents import _core
from livekit.agents.llm._realtime import RealtimeModel
from livekit.agents.llm.realtime import RealtimeModel as HostRealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


def test_omitted_and_null_options_preserve_or_disable_transcription() -> None:
    model = RealtimeModel(api_key="test")
    assert model.capabilities.user_transcription

    model.update_options(input_audio_transcription=None)
    assert not model.capabilities.user_transcription

    model.update_options(voice="marin")
    assert not model.capabilities.user_transcription

    model.update_options(input_audio_transcription={"model": "gpt-4o-mini-transcribe"})
    assert model.capabilities.user_transcription

    disabled = RealtimeModel(api_key="test", input_audio_transcription=None)
    assert not disabled.capabilities.user_transcription


def test_pydantic_turn_detection_uses_aliases_and_explicit_values() -> None:
    class TurnDetection(BaseModel):
        type: Literal["server_vad"] = "server_vad"
        create: bool = Field(alias="create_response")
        interrupt_response: bool = False

    model = RealtimeModel(api_key="test", turn_detection=TurnDetection(create_response=False))
    assert not model.capabilities.turn_detection
    model.update_options(turn_detection=TurnDetection(create_response=True))
    assert model.capabilities.turn_detection


@pytest.mark.asyncio
async def test_fallback_accepts_native_and_host_models_and_rejects_capability_mismatch() -> None:
    class HostModel(HostRealtimeModel):
        def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
            raise AssertionError("constructing a fallback must not open provider sessions")

    native = RealtimeModel(api_key="test")
    host = HostModel(capabilities=native.capabilities)
    fallback = _core.RealtimeFallbackModel([native._core_native(), host])
    assert fallback.capabilities()["audio_output"]
    await fallback.restart(switch_model=True)

    incompatible = HostModel(capabilities=replace(native.capabilities, audio_output=False))
    with pytest.raises(ValueError, match="audio_output"):
        _core.RealtimeFallbackModel([native._core_native(), incompatible])


@pytest.mark.parametrize("cooldown", [-1.0, float("inf"), float("nan"), 1e300])
def test_fallback_rejects_invalid_cooldown_before_opening_a_session(cooldown: float) -> None:
    native = RealtimeModel(api_key="test")
    with pytest.raises(ValueError, match="cooldown"):
        _core.RealtimeFallbackModel([native._core_native()], cooldown=cooldown)
