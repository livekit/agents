from __future__ import annotations

import json
import types
from types import SimpleNamespace
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import stt, vad
from livekit.plugins.mistralai import STT, stt as mistral_stt

pytestmark = pytest.mark.plugin("mistralai")


async def test_endpoint_preserves_vad_time_before_flush(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(mistral_stt.time, "time", lambda: now)
    stream = object.__new__(mistral_stt.SpeechStream)
    stream._speaking = True
    stream._audio_pending = True
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
            yield SimpleNamespace(send_audio=ws.send_bytes, request_id="test", _websocket=ws)

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


def _frame(ms: int = 100, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _FakeTranscriptions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def complete_async(self, **kwargs: Any) -> object:
        self.requests.append(kwargs)
        return types.SimpleNamespace(text="hello", language=None, segments=[])


def _stt(transcriptions: _FakeTranscriptions, **kwargs: Any) -> STT:
    client = types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=transcriptions))
    kwargs.setdefault("model", "voxtral-mini-2602")  # the default model is a realtime one
    return STT(client=client, **kwargs)  # type: ignore[arg-type]


async def test_recognize_language_override_applies_to_one_call() -> None:
    transcriptions = _FakeTranscriptions()
    batch_stt = _stt(transcriptions, language="en")

    overridden = await batch_stt.recognize([_frame()], language="fr")
    default = await batch_stt.recognize([_frame()])

    assert [r["language"] for r in transcriptions.requests] == ["fr", "en"]
    assert overridden.alternatives[0].language == "fr"
    assert default.alternatives[0].language == "en"
    assert batch_stt._opts.language == "en"


async def test_recognize_language_override_keeps_auto_detection() -> None:
    # without a configured language the model detects it;
    # a one-off language must not stick for the calls after it
    transcriptions = _FakeTranscriptions()
    batch_stt = _stt(transcriptions)

    await batch_stt.recognize([_frame()], language="fr")
    await batch_stt.recognize([_frame()])

    first, second = transcriptions.requests
    assert first["language"] == "fr"
    assert first["timestamp_granularities"] == ["segment"]
    assert second["language"] is None
    assert second["timestamp_granularities"] == ["segment"]
    assert batch_stt._opts.language is None


@pytest.mark.parametrize("processing", [False, True])
async def test_provider_progress_resets_retries_without_transcript(processing):
    from livekit.plugins.mistralai import stt as mistral_stt

    message = (
        {"type": "transcription.language", "audio_language": "en"}
        if processing
        else {"type": "session.created", "session": {}}
    )

    class Websocket:
        async def __aiter__(self):
            yield json.dumps(message)

    class Connection:
        request_id = ""
        _websocket = Websocket()

    stream = mistral_stt.SpeechStream.__new__(mistral_stt.SpeechStream)
    stream._on_provider_progress = None
    stream._num_retries = 2
    await stream._recv_task(Connection())
    assert stream._num_retries == (0 if processing else 2)
