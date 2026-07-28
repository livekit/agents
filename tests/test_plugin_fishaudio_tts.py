"""Tests for Fish Audio TTS plugin configuration and request building."""

from __future__ import annotations

import pytest

from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.plugin("fishaudio")


def test_default_model_is_s2_1_pro():
    """A bare TTS() defaults to the s2.1-pro model."""
    from livekit.plugins.fishaudio import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "s2.1-pro"


def test_model_variants_accepted():
    """s1, s2-pro, and s2.1-pro are all accepted."""
    from livekit.plugins.fishaudio import TTS

    for model in ("s1", "s2-pro", "s2.1-pro"):
        assert TTS(api_key="test-key", model=model).model == model


def test_prosody_omitted_by_default():
    """Without speed/volume the request sends prosody=None (unchanged default)."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS_opts()
    assert tts.speed is NOT_GIVEN
    assert tts.volume is NOT_GIVEN
    assert _build_tts_request(tts, text="hi")["prosody"] is None


def test_speed_sets_prosody():
    """speed populates only prosody.speed."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    assert _build_tts_request(TTS_opts(speed=0.8), text="hi")["prosody"] == {"speed": 0.8}


def test_volume_sets_prosody():
    """volume populates only prosody.volume."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    assert _build_tts_request(TTS_opts(volume=-3.0), text="hi")["prosody"] == {"volume": -3.0}


def test_speed_and_volume_set_prosody():
    """speed and volume together populate both prosody fields."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    prosody = _build_tts_request(TTS_opts(speed=1.2, volume=-3.0), text="hi")["prosody"]
    assert prosody == {"speed": 1.2, "volume": -3.0}


def test_update_options_sets_speed_and_volume():
    """update_options forwards speed/volume onto the request."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(api_key="test-key")
    assert _build_tts_request(tts._opts, text="hi")["prosody"] is None

    tts.update_options(speed=0.9, volume=2.0)
    assert _build_tts_request(tts._opts, text="hi")["prosody"] == {"speed": 0.9, "volume": 2.0}


def test_sampling_and_encoding_defaults_unchanged():
    """Without overrides the request keeps the plugin's historical defaults."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    request = _build_tts_request(TTS_opts(), text="hi")
    assert request["temperature"] == 0.7
    assert request["top_p"] == 0.7
    assert request["mp3_bitrate"] == 64
    assert request["opus_bitrate"] == 64000
    assert request["normalize"] is True


def test_constructor_sets_sampling_and_encoding_params():
    """temperature/top_p/mp3_bitrate/opus_bitrate/normalize flow onto the request."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(
        api_key="test-key",
        temperature=0.3,
        top_p=0.9,
        mp3_bitrate=192,
        opus_bitrate=32000,
        normalize=False,
    )
    request = _build_tts_request(tts._opts, text="hi")
    assert request["temperature"] == 0.3
    assert request["top_p"] == 0.9
    assert request["mp3_bitrate"] == 192
    assert request["opus_bitrate"] == 32000
    assert request["normalize"] is False


def test_update_options_sets_sampling_and_encoding_params():
    """update_options forwards the new params onto the request."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(api_key="test-key")
    tts.update_options(
        temperature=0.5, top_p=0.8, mp3_bitrate=128, opus_bitrate=-1000, normalize=False
    )
    request = _build_tts_request(tts._opts, text="hi")
    assert request["temperature"] == 0.5
    assert request["top_p"] == 0.8
    assert request["mp3_bitrate"] == 128
    assert request["opus_bitrate"] == -1000
    assert request["normalize"] is False


def test_temperature_and_top_p_validated():
    """Out-of-range temperature/top_p raise, in the constructor and update_options."""
    from livekit.plugins.fishaudio import TTS

    with pytest.raises(ValueError):
        TTS(api_key="test-key", temperature=1.5)
    with pytest.raises(ValueError):
        TTS(api_key="test-key", top_p=-0.1)

    tts = TTS(api_key="test-key")
    with pytest.raises(ValueError):
        tts.update_options(temperature=1.5)
    with pytest.raises(ValueError):
        tts.update_options(top_p=-0.1)


def test_normalize_loudness_sets_prosody():
    """normalize_loudness rides inside prosody, alongside speed/volume."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    prosody = _build_tts_request(TTS_opts(normalize_loudness=False), text="hi")["prosody"]
    assert prosody == {"normalize_loudness": False}


def test_normalize_loudness_with_speed_and_volume():
    """All three prosody fields are populated together."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    prosody = _build_tts_request(
        TTS_opts(speed=1.2, volume=-3.0, normalize_loudness=True), text="hi"
    )["prosody"]
    assert prosody == {"speed": 1.2, "volume": -3.0, "normalize_loudness": True}


def test_generation_tuning_omitted_by_default():
    """Unset tuning params are absent from the request so Fish's defaults apply."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    request = _build_tts_request(TTS_opts(), text="hi")
    assert "max_new_tokens" not in request
    assert "min_chunk_length" not in request
    assert "condition_on_previous_chunks" not in request
    assert "early_stop_threshold" not in request


def test_constructor_sets_generation_tuning_params():
    """max_new_tokens/min_chunk_length/condition_on_previous_chunks/early_stop_threshold
    flow onto the request."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(
        api_key="test-key",
        max_new_tokens=512,
        min_chunk_length=25,
        condition_on_previous_chunks=False,
        early_stop_threshold=0.8,
    )
    request = _build_tts_request(tts._opts, text="hi")
    assert request["max_new_tokens"] == 512
    assert request["min_chunk_length"] == 25
    assert request["condition_on_previous_chunks"] is False
    assert request["early_stop_threshold"] == 0.8


def test_update_options_sets_generation_tuning_params():
    """update_options forwards the tuning params onto the request."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(api_key="test-key")
    tts.update_options(
        max_new_tokens=2048,
        min_chunk_length=30,
        condition_on_previous_chunks=False,
        early_stop_threshold=0.5,
        normalize_loudness=False,
    )
    request = _build_tts_request(tts._opts, text="hi")
    assert request["max_new_tokens"] == 2048
    assert request["min_chunk_length"] == 30
    assert request["condition_on_previous_chunks"] is False
    assert request["early_stop_threshold"] == 0.5
    assert request["prosody"] == {"normalize_loudness": False}


def test_generation_tuning_validated():
    """Out-of-range tuning params raise, in the constructor and update_options."""
    from livekit.plugins.fishaudio import TTS

    with pytest.raises(ValueError):
        TTS(api_key="test-key", max_new_tokens=-1)
    with pytest.raises(ValueError):
        TTS(api_key="test-key", min_chunk_length=150)
    with pytest.raises(ValueError):
        TTS(api_key="test-key", early_stop_threshold=1.5)

    tts = TTS(api_key="test-key")
    with pytest.raises(ValueError):
        tts.update_options(max_new_tokens=-1)
    with pytest.raises(ValueError):
        tts.update_options(min_chunk_length=-5)
    with pytest.raises(ValueError):
        tts.update_options(early_stop_threshold=-0.1)


def test_normalize_loudness_dropped_on_s1_constructor():
    """On the s1 model normalize_loudness is dropped (warn log), not sent."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(api_key="test-key", model="s1", normalize_loudness=True)
    assert _build_tts_request(tts._opts, text="hi")["prosody"] is None


def test_normalize_loudness_dropped_on_s1_update_options():
    """update_options refuses normalize_loudness while the model is s1."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(api_key="test-key", model="s1")
    tts.update_options(normalize_loudness=True)
    assert _build_tts_request(tts._opts, text="hi")["prosody"] is None


def test_normalize_loudness_dropped_on_model_switch_to_s1():
    """Switching to s1 drops a previously set normalize_loudness."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(api_key="test-key", model="s2-pro", normalize_loudness=True)
    assert _build_tts_request(tts._opts, text="hi")["prosody"] == {"normalize_loudness": True}

    tts.update_options(model="s1")
    assert _build_tts_request(tts._opts, text="hi")["prosody"] is None


def test_normalize_loudness_kept_on_s2_pro_family():
    """s2-pro and s2.1-pro keep normalize_loudness."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    for model in ("s2-pro", "s2.1-pro"):
        tts = TTS(api_key="test-key", model=model, normalize_loudness=False)
        prosody = _build_tts_request(tts._opts, text="hi")["prosody"]
        assert prosody == {"normalize_loudness": False}


def TTS_opts(**overrides):
    """Build a _TTSOptions with sensible test defaults, overriding as needed."""
    from livekit.plugins.fishaudio.tts import DEFAULT_BASE_URL, DEFAULT_MODEL, _TTSOptions

    opts = {
        "model": DEFAULT_MODEL,
        "output_format": "wav",
        "sample_rate": 24000,
        "voice_id": NOT_GIVEN,
        "base_url": DEFAULT_BASE_URL,
        "api_key": "test-key",
        "latency_mode": "balanced",
        "chunk_length": 100,
        "speed": NOT_GIVEN,
        "volume": NOT_GIVEN,
        "temperature": 0.7,
        "top_p": 0.7,
        "mp3_bitrate": 64,
        "opus_bitrate": 64000,
        "normalize": True,
        "normalize_loudness": NOT_GIVEN,
        "max_new_tokens": NOT_GIVEN,
        "min_chunk_length": NOT_GIVEN,
        "condition_on_previous_chunks": NOT_GIVEN,
        "early_stop_threshold": NOT_GIVEN,
    }
    opts.update(overrides)
    return _TTSOptions(**opts)
