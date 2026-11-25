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


class _CapturingSession:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def ws_connect(self, url: str, **kwargs):
        self.urls.append(url)
        return object()


async def test_stream_language_overrides_default():
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.fireworksai.stt import STT

    session = _CapturingSession()
    provider = STT(api_key="test-key", language="en", http_session=session)  # type: ignore[arg-type]
    stream = provider.stream(language="fr")
    try:
        await stream._connect_ws()
        query = parse_qs(urlparse(session.urls[-1]).query)
        assert query["language"] == ["fr"]
        # the per-stream language does not leak into the provider's options
        assert provider._opts.language == "en"
    finally:
        await stream.aclose()
        await provider.aclose()
