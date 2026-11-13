"""Capabilities are fixed at construction: a session wires a provider by them once (wraps a
non-streaming STT, expects aligned words, expects a realtime model's audio), so an option that
would change one is refused by ``update_options`` and taken by the constructor only."""

from __future__ import annotations

import pytest

from livekit.plugins import smallestai, ultravox


# Catches: an option that changes a capability applied after the session wired the provider by
# it (wrapped a non-streaming STT, expected aligned words, expected audio from a realtime model).
def test_an_option_that_would_change_a_capability_is_refused() -> None:
    speaker = smallestai.TTS(api_key="key", word_timestamps=False)
    with pytest.raises(ValueError, match="word_timestamps is fixed at construction"):
        speaker.update_options(word_timestamps=True)
    assert speaker.capabilities.aligned_transcript is False
    speaker.update_options(word_timestamps=False)  # unchanged: accepted

    listener = smallestai.STT(api_key="key", model="pulse")
    with pytest.raises(ValueError, match="create a new STT"):
        listener.update_options(model="lightning")
    assert listener.capabilities.streaming is True

    realtime = ultravox.realtime.RealtimeModel(api_key="key", output_medium="voice")
    with pytest.raises(ValueError, match="output_medium is fixed at construction"):
        realtime.update_options(output_medium="text")
    assert realtime.capabilities.audio_output is True
