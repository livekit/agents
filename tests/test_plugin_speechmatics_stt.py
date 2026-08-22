from __future__ import annotations

import pytest

pytestmark = pytest.mark.plugin("speechmatics")


def _make_stt(**kwargs):
    from livekit.plugins import speechmatics
    from livekit.plugins.speechmatics.stt import TurnDetectionMode

    # ADAPTIVE avoids the EXTERNAL-mode Silero auto-load; no connection is opened.
    kwargs.setdefault("turn_detection_mode", TurnDetectionMode.ADAPTIVE)
    return speechmatics.STT(api_key="fake-api-key", **kwargs)


def test_end_of_turn_config_passthrough():
    from speechmatics.voice import EndOfTurnConfig

    eot = EndOfTurnConfig(
        base_multiplier=1.5,
        min_end_of_turn_delay=0.2,
        use_forced_eou=True,
    )
    config = _make_stt(end_of_turn_config=eot)._prepare_config()

    assert config.end_of_turn_config.base_multiplier == 1.5
    assert config.end_of_turn_config.min_end_of_turn_delay == 0.2
    assert config.end_of_turn_config.use_forced_eou is True


def test_vad_config_passthrough():
    from speechmatics.voice import VoiceActivityConfig

    vad_config = VoiceActivityConfig(enabled=True, silence_duration=0.35, threshold=0.5)
    config = _make_stt(vad_config=vad_config)._prepare_config()

    assert config.vad_config.enabled is True
    assert config.vad_config.silence_duration == 0.35
    assert config.vad_config.threshold == 0.5


def test_preset_defaults_kept_when_not_given():
    config = _make_stt()._prepare_config()

    # ADAPTIVE preset values must be untouched when the new kwargs are omitted.
    assert config.vad_config is not None and config.vad_config.enabled is True
    assert config.vad_config.silence_duration == pytest.approx(0.18)
    assert config.end_of_turn_config.use_forced_eou is True
    assert config.end_of_turn_config.min_end_of_turn_delay == pytest.approx(0.01)
