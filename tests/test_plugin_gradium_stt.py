"""Tests for Gradium STT plugin configuration and behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.plugin("gradium")


@pytest.mark.parametrize("stop_s", [0.4, 0.0, None])
async def test_endpoint_uses_end_text_timing(monkeypatch, stop_s):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import aiohttp

    from livekit.agents import stt
    from livekit.plugins.gradium import stt as gradium_stt

    messages = asyncio.Queue()

    def send(data):
        messages.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(data), ""))

    send({"type": "text", "text": "hello", "start_s": 0.1})
    if stop_s is not None:
        send({"type": "end_text", "stop_s": stop_s})
    for _ in range(7):
        send({"type": "step", "vad": [{}, {}, {"inactivity_prob": 1.0}]})
    send({"type": "text", "text": "again", "start_s": 1.0})
    for _ in range(7):
        send({"type": "step", "vad": [{}, {}, {"inactivity_prob": 1.0}]})
    ws = SimpleNamespace(receive=messages.get, send_str=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(gradium_stt.SpeechStream, "_connect_ws", AsyncMock(return_value=ws))
    provider = gradium_stt.STT(api_key="test-key", http_session=SimpleNamespace(), vad_flush=False)
    stream = provider.stream()
    stream.start_time_offset = 7.0
    endpoints = 0
    try:
        async with asyncio.timeout(2):
            async for event in stream:
                if event.type == stt.SpeechEventType.END_OF_SPEECH:
                    endpoints += 1
                    if endpoints == 1 and stop_s:
                        assert event.alternatives[0].end_time == pytest.approx(7.4)
                    else:
                        assert event.alternatives == []
                    if endpoints == 2:
                        break
            else:
                pytest.fail("missing endpoint")
    finally:
        await stream.aclose()
        await provider.aclose()


def test_stt_default_language():
    """STT defaults to English."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key")

    assert stt._opts.language == "en"


def test_stt_custom_language():
    """STT accepts custom language."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr")

    assert stt._opts.language == "fr"


def test_stream_language_overrides_stream_options():
    """stream language overrides the copied stream config."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr", http_session=MagicMock())

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream(language="de")

    assert stream._opts.language == "de"
    assert stt._opts.language == "fr"


@pytest.mark.asyncio
async def test_speech_stream_sends_language_in_json_config():
    """SpeechStream sends language in setup json_config."""
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, LanguageCode
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    sent_messages: list[str] = []

    class FakeWebSocket:
        async def send_str(self, message: str) -> None:
            sent_messages.append(message)

    class FakeSession:
        async def ws_connect(self, url, headers):
            return FakeWebSocket()

    stt = STT(api_key="test-key")
    opts = STTOptions(language=LanguageCode("pt"), temperature=0.3)

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            model_endpoint="wss://api.gradium.ai/api/speech/asr",
            model_name="default",
            http_session=FakeSession(),
        )

    await stream._connect_ws()

    setup_msg = json.loads(sent_messages[0])
    assert setup_msg["json_config"] == {"language": "pt", "temp": 0.3}


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.gradium.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(buffer_size_seconds=0.1))
    finally:
        await stream.aclose()
        await provider.aclose()
