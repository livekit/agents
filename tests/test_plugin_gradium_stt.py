"""Tests for Gradium STT plugin configuration and behavior."""

from __future__ import annotations

import asyncio
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


@pytest.mark.no_concurrent
@pytest.mark.parametrize("vad_flush", [True, False])
async def test_synthetic_vad_silence_does_not_shift_later_turns(monkeypatch, vad_flush):
    import base64

    from livekit.agents import APIConnectOptions, stt
    from livekit.plugins.gradium import STT

    from .stt_reconnect import ReconnectClock, wait_until

    clock = ReconnectClock(monkeypatch)
    options = {} if vad_flush else {"vad_flush": False}
    provider = STT(api_key="test-key", http_session=clock, **options)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.start_time_offset = 7.0

    def sent_duration(ws):
        return sum(
            len(base64.b64decode(data["audio"])) / (24000 * 2)
            for message in ws.sent
            if isinstance(message, str) and (data := json.loads(message))["type"] == "audio"
        )

    async def check_turn(index, *, captured_at, provider_start, expected_end):
        await clock.push(stream, captured_at=captured_at)
        stream.flush()
        ws = clock.sockets[-1]
        await wait_until(lambda: sent_duration(ws) >= provider_start + 1.0 - 1e-6)
        ws.respond({"type": "text", "text": "hello", "start_s": provider_start + 0.1})
        step = {"type": "step", "vad": [{}, {}, {"inactivity_prob": 1.0}]}
        ws.respond(step)
        if vad_flush:
            await wait_until(lambda: sent_duration(ws) >= provider_start + 1.48 - 1e-6)
        # The boundary precedes the padding, even when its reply arrives after it.
        ws.respond({"type": "end_text", "stop_s": provider_start + 0.4})
        for _ in range(6):
            ws.respond(step)
        async for event in stream:
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                assert event.alternatives[0].start_time == pytest.approx(expected_end - 0.3)
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                assert event.alternatives[0].end_time == pytest.approx(expected_end)
                return
        pytest.fail(f"missing endpoint for turn {index}")

    try:
        import asyncio

        async with asyncio.timeout(5):
            for index in range(3):
                await check_turn(
                    index,
                    captured_at=1000.0 + index,
                    provider_start=index * (1.48 if vad_flush else 1.0),
                    expected_end=7.4 + index,
                )
            stream.update_options(buffer_size_seconds=0.1)
            await wait_until(lambda: len(clock.sockets) == 2)
            await check_turn(3, captured_at=1010.0, provider_start=0.0, expected_end=17.4)
    finally:
        await stream.aclose()
        await provider.aclose()


def test_padded_timeline_settles_padding_behind_each_timestamp():
    from livekit.plugins.gradium.stt import _PaddedTimeline

    sample_rate = 24000
    timeline = _PaddedTimeline(sample_rate)
    turns = 1000
    for turn in range(turns):
        provider_start = turn * 1.48
        timeline.add_audio(sample_rate)
        assert timeline.remove_padding(provider_start + 0.1) == pytest.approx(turn + 0.1)
        timeline.add_padding(int(0.48 * sample_rate))
        # the boundary precedes padding that was sent before its reply arrived
        assert timeline.remove_padding(provider_start + 0.4) == pytest.approx(turn + 0.4)
        # a timestamp inside the padding stays at the padding's start
        assert timeline.remove_padding(provider_start + 1.2) == pytest.approx(turn + 1.0)
        assert len(timeline._pending_padding) <= 1

    assert timeline.remove_padding(turns * 1.48 + 0.1) == pytest.approx(turns + 0.1)
    assert not timeline._pending_padding

    timeline.reset()
    timeline.add_audio(sample_rate)
    assert timeline.remove_padding(0.5) == pytest.approx(0.5)


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
        async def ws_connect(self, url, headers, **kwargs):
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


class _FakeMessage:
    def __init__(self, payload: dict) -> None:
        import aiohttp

        self.type = aiohttp.WSMsgType.TEXT
        self.data = json.dumps(payload)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[dict] = asyncio.Queue()
        self.close_code = None

    async def send_str(self, message: str) -> None:
        pass

    async def ping(self) -> None:
        pass

    async def receive(self) -> _FakeMessage:
        return _FakeMessage(await self.incoming.get())

    async def close(self) -> None:
        pass


class _FakeSession:
    def __init__(self) -> None:
        self.ws = _FakeWebSocket()

    async def ws_connect(self, url, headers, **kwargs) -> _FakeWebSocket:
        return self.ws


def _step(inactivity_prob: float, total_duration_s: float = 1.0) -> dict:
    return {
        "type": "step",
        "total_duration_s": total_duration_s,
        "vad": [
            {"horizon_s": horizon_s, "inactivity_prob": inactivity_prob}
            for horizon_s in (0.5, 1.0, 2.0)
        ],
    }


class _EventCollector:
    """Collects speech events from a stream in the background."""

    def __init__(self, stream) -> None:
        self.events: list = []
        self._task = asyncio.create_task(self._collect(stream))

    async def _collect(self, stream) -> None:
        from livekit.agents.stt import SpeechEventType

        async for ev in stream:
            if ev.type != SpeechEventType.RECOGNITION_USAGE:
                self.events.append(ev)

    async def wait_for(self, count: int) -> list:
        for _ in range(500):
            if len(self.events) >= count:
                return self.events
            await asyncio.sleep(0.01)
        raise AssertionError(f"expected {count} events, got {self.events}")

    async def drain(self, ws: _FakeWebSocket) -> None:
        while not ws.incoming.empty():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_vad_start_of_speech_opens_a_turn_before_any_text():
    """With vad_start_threshold, VAD activity starts speech, ignoring warm-up steps."""
    from livekit.agents.stt import SpeechEventType
    from livekit.plugins.gradium import STT

    session = _FakeSession()
    stt = STT(api_key="test-key", http_session=session, vad_start_threshold=0.4)
    stream = stt.stream()
    collector = _EventCollector(stream)
    try:
        # warm-up steps sent before any audio was processed read as activity
        for prob in (0.1, 0.1):
            session.ws.incoming.put_nowait(_step(prob, total_duration_s=0.0))
        await collector.drain(session.ws)
        assert collector.events == []

        # speech right from the first processed audio opens the turn
        session.ws.incoming.put_nowait(_step(0.2, total_duration_s=0.08))
        events = await collector.wait_for(1)
        assert [ev.type for ev in events] == [SpeechEventType.START_OF_SPEECH]

        # text in the open turn does not start it again
        session.ws.incoming.put_nowait({"type": "text", "text": "hello", "start_s": 0.5})
        events = await collector.wait_for(2)
        assert [ev.type for ev in events] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.INTERIM_TRANSCRIPT,
        ]
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_vad_start_of_speech_without_text_still_ends_the_turn():
    """A VAD-opened turn without text ends on inactivity and the next one can open."""
    from livekit.agents.stt import SpeechEventType
    from livekit.plugins.gradium import STT

    session = _FakeSession()
    stt = STT(api_key="test-key", http_session=session, vad_start_threshold=0.4)
    stream = stt.stream()
    collector = _EventCollector(stream)
    try:
        session.ws.incoming.put_nowait(_step(0.2))
        # the first positive step starts the countdown, delay_in_tokens (6) more end the turn
        for _ in range(7):
            session.ws.incoming.put_nowait(_step(0.95))
        session.ws.incoming.put_nowait(_step(0.2))

        events = await collector.wait_for(4)
        assert [ev.type for ev in events] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
            SpeechEventType.START_OF_SPEECH,
        ]
        assert events[1].alternatives[0].text == ""
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_vad_start_of_speech_is_off_by_default():
    """Without vad_start_threshold, speech only starts on the first transcribed text."""
    from livekit.agents.stt import SpeechEventType
    from livekit.plugins.gradium import STT

    session = _FakeSession()
    stt = STT(api_key="test-key", http_session=session)
    stream = stt.stream()
    collector = _EventCollector(stream)
    try:
        for prob in (0.9, 0.2, 0.2):
            session.ws.incoming.put_nowait(_step(prob))
        await collector.drain(session.ws)
        assert collector.events == []

        session.ws.incoming.put_nowait({"type": "text", "text": "hello", "start_s": 0.5})
        events = await collector.wait_for(2)
        assert [ev.type for ev in events] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.INTERIM_TRANSCRIPT,
        ]
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_vad_start_of_speech_with_vad_bucket_zero():
    """vad_bucket=0 selects the first horizon for both the start and the end of a turn."""
    from livekit.agents.stt import SpeechEventType
    from livekit.plugins.gradium import STT

    session = _FakeSession()
    stt = STT(api_key="test-key", http_session=session, vad_bucket=0, vad_start_threshold=0.4)
    stream = stt.stream()
    collector = _EventCollector(stream)
    try:

        def bucket_zero_step(inactivity_prob: float) -> dict:
            # the other buckets disagree, so only bucket 0 can drive the turn
            step = _step(1.0 - inactivity_prob)
            step["vad"][0]["inactivity_prob"] = inactivity_prob
            return step

        session.ws.incoming.put_nowait(bucket_zero_step(0.2))
        # the first positive step starts the countdown, delay_in_tokens (6) more end the turn
        for _ in range(7):
            session.ws.incoming.put_nowait(bucket_zero_step(0.95))

        events = await collector.wait_for(3)
        assert [ev.type for ev in events] == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
        ]
    finally:
        await stream.aclose()


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
