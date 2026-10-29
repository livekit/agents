"""Tests for Gradium STT plugin configuration and behavior."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.plugin("gradium")


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
    """vad_bucket=0 selects the first horizon, it does not disable the VAD start of speech."""
    from livekit.agents.stt import SpeechEventType
    from livekit.plugins.gradium import STT

    session = _FakeSession()
    stt = STT(api_key="test-key", http_session=session, vad_bucket=0, vad_start_threshold=0.4)
    stream = stt.stream()
    collector = _EventCollector(stream)
    try:
        step = _step(0.9)
        step["vad"][0]["inactivity_prob"] = 0.2
        session.ws.incoming.put_nowait(step)

        events = await collector.wait_for(1)
        assert [ev.type for ev in events] == [SpeechEventType.START_OF_SPEECH]
    finally:
        await stream.aclose()
