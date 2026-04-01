from __future__ import annotations

import asyncio
from collections import deque

import pytest

from livekit import rtc
from livekit.agents import LanguageCode
from livekit.agents.stt import SpeechEventType
from livekit.agents.utils.aio.channel import ChanEmpty
from mistralai.client.models.transcriptionstreamdone import TranscriptionStreamDone
from mistralai.client.models.usageinfo import UsageInfo

from .conftest import TEST_CONNECT_OPTIONS


class _QueueWebSocket:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.close_code = 1000

    def __aiter__(self) -> _QueueWebSocket:
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self._queue.put_nowait(None)


class _DoneAfterFlushConnection:
    def __init__(self, *, text: str, audio_seconds: int) -> None:
        self.request_id = "req-test"
        self.is_closed = False
        self._events: asyncio.Queue[object | None] = asyncio.Queue()
        self._initial_events: deque[object] = deque()
        self._text = text
        self._audio_seconds = audio_seconds
        self._flush_tasks: list[asyncio.Task[None]] = []

    def __aiter__(self) -> _DoneAfterFlushConnection:
        return self

    async def __anext__(self):
        item = await self._events.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send_audio(self, audio_bytes: bytes) -> None:
        del audio_bytes

    async def flush_audio(self) -> None:
        async def _emit_done() -> None:
            await asyncio.sleep(0.01)
            event = TranscriptionStreamDone(
                model="voxtral-mini-transcribe-realtime-2602",
                text=self._text,
                usage=UsageInfo(prompt_audio_seconds=self._audio_seconds),
                language="en",
            )
            await self._events.put(event)
            await self._events.put(None)

        self._flush_tasks.append(asyncio.create_task(_emit_done()))

    async def end_audio(self) -> None:
        return None

    async def close(self) -> None:
        self.is_closed = True
        for task in self._flush_tasks:
            await asyncio.gather(task, return_exceptions=True)
        await self._events.put(None)


class _IdleWebSocket:
    def __init__(self) -> None:
        self._closed = asyncio.Event()
        self.close_code = 1000

    def __aiter__(self) -> _IdleWebSocket:
        return self

    async def __anext__(self) -> str:
        await self._closed.wait()
        raise StopAsyncIteration

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self._closed.set()


class _IdleConnection:
    def __init__(self) -> None:
        self.request_id = "req-idle"
        self.is_closed = False
        self._websocket = _IdleWebSocket()
        self._initial_events: deque[object] = deque()

    async def send_audio(self, audio_bytes: bytes) -> None:
        del audio_bytes

    async def flush_audio(self) -> None:
        return None

    async def end_audio(self) -> None:
        return None

    async def close(self) -> None:
        self.is_closed = True
        await self._websocket.close()


class _IteratorOnlyConnection:
    def __init__(self) -> None:
        self.request_id = "req-iter"
        self.is_closed = False
        self._initial_events: deque[object] = deque()

    def __aiter__(self) -> _IteratorOnlyConnection:
        return self

    async def __anext__(self):
        if self.is_closed:
            raise StopAsyncIteration

        self.is_closed = True
        return TranscriptionStreamDone(
            model="voxtral-mini-transcribe-realtime-2602",
            text="iterator transcript",
            usage=UsageInfo(prompt_audio_seconds=1),
            language="en",
        )

    async def send_audio(self, audio_bytes: bytes) -> None:
        del audio_bytes

    async def flush_audio(self) -> None:
        return None

    async def end_audio(self) -> None:
        return None

    async def close(self) -> None:
        self.is_closed = True


class _FakePool:
    def __init__(self, conn: object) -> None:
        self.conn = conn
        self.invalidate_calls = 0
        self.prewarm_calls = 0
        self.put_calls = 0
        self.remove_calls = 0

    async def get(self, *, timeout: float) -> object:
        del timeout
        return self.conn

    def put(self, conn: object) -> None:
        assert conn is self.conn
        self.put_calls += 1

    def remove(self, conn: object) -> None:
        assert conn is self.conn
        self.remove_calls += 1

    def invalidate(self) -> None:
        self.invalidate_calls += 1

    def prewarm(self) -> None:
        self.prewarm_calls += 1

    async def aclose(self) -> None:
        close = getattr(self.conn, "close", None)
        if close is not None:
            await close()


def _audio_frame(*, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples_per_channel = sample_rate // 100
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples_per_channel,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples_per_channel,
    )


def _drain_events(stream) -> list:
    events = []
    while True:
        try:
            events.append(stream._event_ch.recv_nowait())
        except ChanEmpty:
            return events


@pytest.mark.asyncio
async def test_stream_waits_for_done_after_end_input():
    from livekit.plugins.mistralai import STT

    stt = STT(client=object(), model="voxtral-mini-transcribe-realtime-2602")
    stt._pool = _FakePool(_DoneAfterFlushConnection(text="hello world", audio_seconds=1))

    async with stt.stream(conn_options=TEST_CONNECT_OPTIONS) as stream:
        stream.push_frame(_audio_frame())
        stream.end_input()
        events = [event async for event in stream]
    await asyncio.sleep(0.02)

    final_texts = [
        ev.alternatives[0].text for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT
    ]
    usage_durations = [
        ev.recognition_usage.audio_duration
        for ev in events
        if ev.type == SpeechEventType.RECOGNITION_USAGE and ev.recognition_usage is not None
    ]

    assert final_texts == ["hello world"]
    assert usage_durations == [1.0]


@pytest.mark.asyncio
async def test_recv_task_uses_public_connection_iterator():
    from livekit.plugins.mistralai import STT

    stt = STT(client=object(), model="voxtral-mini-transcribe-realtime-2602")
    stt._pool = _FakePool(_IteratorOnlyConnection())

    async with stt.stream(conn_options=TEST_CONNECT_OPTIONS) as stream:
        events = [event async for event in stream]

    final_texts = [
        ev.alternatives[0].text for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT
    ]
    usage_durations = [
        ev.recognition_usage.audio_duration
        for ev in events
        if ev.type == SpeechEventType.RECOGNITION_USAGE and ev.recognition_usage is not None
    ]

    assert final_texts == ["iterator transcript"]
    assert usage_durations == [1.0]


@pytest.mark.asyncio
async def test_update_options_reconfigures_active_streams():
    from livekit.plugins.mistralai import STT

    stt = STT(
        client=object(),
        model="voxtral-mini-transcribe-realtime-2602",
        sample_rate=16000,
        interim_results=True,
        language="en",
    )
    stt._pool = _FakePool(_IdleConnection())

    async with stt.stream(conn_options=TEST_CONNECT_OPTIONS) as stream:
        stt.update_options(sample_rate=8000, interim_results=False, language="fr")

        assert stt._pool.invalidate_calls == 1
        assert stream._reconnect_event.is_set()
        assert stream._opts.sample_rate == 8000
        assert stream._needed_sr == 8000
        assert stream._opts.interim_results is False
        assert stream._opts.language == LanguageCode("fr")


@pytest.mark.asyncio
async def test_usage_metrics_are_emitted_for_each_completed_turn():
    from livekit.plugins.mistralai import STT

    stt = STT(client=object(), model="voxtral-mini-transcribe-realtime-2602")
    stt._pool = _FakePool(_IdleConnection())

    async with stt.stream(conn_options=TEST_CONNECT_OPTIONS) as stream:
        stream._handle_interim_text(" hello")
        stream._process_event(
            TranscriptionStreamDone(
                model="voxtral-mini-transcribe-realtime-2602",
                text="hello",
                usage=UsageInfo(prompt_audio_seconds=1),
                language="en",
            )
        )

        stream._handle_interim_text(" world")
        stream._process_event(
            TranscriptionStreamDone(
                model="voxtral-mini-transcribe-realtime-2602",
                text="world",
                usage=UsageInfo(prompt_audio_seconds=2),
                language="en",
            )
        )

        usage_durations = [
            ev.recognition_usage.audio_duration
            for ev in _drain_events(stream)
            if ev.type == SpeechEventType.RECOGNITION_USAGE and ev.recognition_usage is not None
        ]

    assert usage_durations == [1.0, 2.0]


@pytest.mark.asyncio
async def test_prewarm_only_warms_realtime_models():
    from livekit.plugins.mistralai import STT

    batch_stt = STT(client=object(), model="voxtral-mini-latest")
    batch_stt._pool = _FakePool(_IdleConnection())
    batch_stt.prewarm()
    assert batch_stt._pool.prewarm_calls == 0

    realtime_stt = STT(client=object(), model="voxtral-mini-transcribe-realtime-2602")
    realtime_stt._pool = _FakePool(_IdleConnection())
    realtime_stt.prewarm()
    assert realtime_stt._pool.prewarm_calls == 1
