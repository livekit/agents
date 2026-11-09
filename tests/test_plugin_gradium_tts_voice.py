"""Gradium TTS voice selection: `voice` and `voice_id` are alternatives, never both."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_voice_id_when_neither_is_given():
    from livekit.plugins.gradium import TTS
    from livekit.plugins.gradium.tts import DEFAULT_VOICE_ID

    tts = TTS(api_key="test-key")
    assert tts._opts.voice_id == DEFAULT_VOICE_ID
    # pinned on purpose: the default speaker is user-visible behaviour
    assert DEFAULT_VOICE_ID == "vjFyp2BcCyTlUmYb"
    assert tts._opts.voice is None


def test_voice_name_is_not_overridden_by_a_default_voice_id():
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key", voice="narrator")
    assert tts._opts.voice == "narrator"
    assert tts._opts.voice_id is None


def test_voice_id_alone_is_kept():
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key", voice_id="abc")
    assert tts._opts.voice_id == "abc"
    assert tts._opts.voice is None


def test_voice_and_voice_id_are_exclusive():
    from livekit.plugins.gradium import TTS

    with pytest.raises(ValueError, match="mutually exclusive"):
        TTS(api_key="test-key", voice="narrator", voice_id="abc")


def test_update_options_voice_clears_voice_id():
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(voice="narrator")
    assert tts._opts.voice == "narrator"
    assert tts._opts.voice_id is None


def test_update_options_voice_id_clears_voice():
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key", voice="narrator")
    tts.update_options(voice_id="abc")
    assert tts._opts.voice_id == "abc"
    assert tts._opts.voice is None


def test_update_options_rejects_both():
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key")
    with pytest.raises(ValueError, match="mutually exclusive"):
        tts.update_options(voice="narrator", voice_id="abc")
