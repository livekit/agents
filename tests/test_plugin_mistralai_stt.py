from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import stt, vad
from livekit.plugins.mistralai import stt as mistral_stt

pytestmark = pytest.mark.plugin("mistralai")


async def test_endpoint_preserves_vad_time_before_flush(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(mistral_stt.time, "time", lambda: now)
    stream = object.__new__(mistral_stt.SpeechStream)
    stream._speaking = True
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)

    async def vad_events():
        yield vad.VADEvent(
            type=vad.VADEventType.END_OF_SPEECH,
            samples_index=0,
            timestamp=1.0,
            speech_duration=0.5,
            silence_duration=0.3,
            inference_duration=0.05,
        )

    async def flush():
        nonlocal now
        now += 2.0

    await stream._vad_task(vad_events(), SimpleNamespace(flush_audio=flush))
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert events[-1].speech_end_time == pytest.approx(999.65)


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from contextlib import asynccontextmanager

    from livekit.agents import APIConnectOptions

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)

    @asynccontextmanager
    async def connection(**kwargs):
        async with clock.ws_connect() as ws:
            yield SimpleNamespace(send_audio=ws.send_bytes, events=ws.__aiter__)

    provider = mistral_stt.STT(api_key="test-key")
    stream = mistral_stt.SpeechStream(
        stt=provider,
        conn_options=APIConnectOptions(max_retry=0),
        pool=SimpleNamespace(connection=connection),
    )
    try:
        await clock.check(stream, stream._reconnect_event.set)
    finally:
        await stream.aclose()
        await provider.aclose()
