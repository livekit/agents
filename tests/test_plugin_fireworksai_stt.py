from __future__ import annotations

import pytest

pytestmark = pytest.mark.plugin("fireworksai")


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.fireworksai.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_stream_language_overrides_stt_default():
    from livekit.plugins.fireworksai.stt import STT

    provider = STT(api_key="test-key", language="en", http_session=object())  # type: ignore[arg-type]
    stream = provider.stream(language="fr")
    try:
        assert stream._opts.language == "fr"
        assert provider._opts.language == "en"
        assert provider.stream()._opts.language == "en"
    finally:
        await stream.aclose()
