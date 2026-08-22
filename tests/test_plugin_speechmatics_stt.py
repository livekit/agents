"""Unit tests for Speechmatics STT plugin configuration."""

from __future__ import annotations

import pytest
from speechmatics.voice import (
    EndOfTurnConfig,
    VoiceActivityConfig,
    VoiceAgentConfigPreset,
)

from livekit.plugins.speechmatics import stt as speechmatics_stt
from livekit.plugins.speechmatics.stt import TurnDetectionMode

pytestmark = pytest.mark.plugin("speechmatics")


def _stt(**kwargs: object) -> speechmatics_stt.STT:
    return speechmatics_stt.STT(api_key="test-key", **kwargs)


def _adaptive_config(**kwargs: object) -> object:
    return _stt(turn_detection_mode=TurnDetectionMode.ADAPTIVE, **kwargs)._prepare_config()


def test_end_of_turn_config_defaults_to_preset() -> None:
    config = _adaptive_config()
    preset = VoiceAgentConfigPreset.load(TurnDetectionMode.ADAPTIVE.value)

    assert config.end_of_turn_config == preset.end_of_turn_config  # type: ignore[attr-defined]


def test_vad_config_defaults_to_preset() -> None:
    config = _adaptive_config()

    assert config.vad_config.enabled is True  # type: ignore[attr-defined]
    assert config.vad_config.silence_duration == pytest.approx(0.18)  # type: ignore[attr-defined]


def test_end_of_turn_config_overrides_preset() -> None:
    eou = EndOfTurnConfig(min_end_of_turn_delay=0.15, use_forced_eou=True)
    config = _adaptive_config(end_of_turn_config=eou)

    assert config.end_of_turn_config is eou  # type: ignore[attr-defined]


def test_vad_config_overrides_preset() -> None:
    vad_cfg = VoiceActivityConfig(enabled=True, silence_duration=0.3)
    config = _adaptive_config(vad_config=vad_cfg)

    assert config.vad_config is vad_cfg  # type: ignore[attr-defined]


def test_options_store_configs() -> None:
    eou = EndOfTurnConfig(min_end_of_turn_delay=0.15)
    vad_cfg = VoiceActivityConfig(silence_duration=0.3)
    instance = _stt(end_of_turn_config=eou, vad_config=vad_cfg)

    assert instance._stt_options.end_of_turn_config is eou
    assert instance._stt_options.vad_config is vad_cfg


def test_enabled_vad_config_rejected_with_external_mode() -> None:
    # The external VAD (or manual finalize()) already controls turn boundaries;
    # an enabled client-side VAD would endpoint the same audio a second time.
    with pytest.raises(ValueError, match="turn_detection_mode=EXTERNAL"):
        speechmatics_stt.STT(
            api_key="test-key",
            turn_detection_mode=TurnDetectionMode.EXTERNAL,
            vad=None,
            vad_config=VoiceActivityConfig(enabled=True, silence_duration=0.3),
        )


def test_disabled_vad_config_allowed_with_external_mode() -> None:
    instance = speechmatics_stt.STT(
        api_key="test-key",
        turn_detection_mode=TurnDetectionMode.EXTERNAL,
        vad=None,
        vad_config=VoiceActivityConfig(enabled=False, threshold=0.5),
    )
    assert instance._stt_options.vad_config is not None


def test_omitted_configs_change_nothing_for_existing_users() -> None:
    # A default-constructed STT must produce exactly the preset values.
    config = _stt()._prepare_config()
    preset = VoiceAgentConfigPreset.load(TurnDetectionMode.EXTERNAL.value)

    assert config.end_of_turn_config == preset.end_of_turn_config
    assert config.vad_config == preset.vad_config
