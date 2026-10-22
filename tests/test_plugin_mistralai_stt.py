"""Tests for Mistral realtime STT server-side turn detection.

Runs a real `SpeechStream` against a fake `RealtimeConnection`, so the session
update and the mapping of `turn.*` events are assertable without a network.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mistralai.client.models import (
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
    UsageInfo,
)
from mistralai.extra.realtime import UnknownRealtimeEvent

from livekit.agents import APIConnectOptions, stt

pytestmark = pytest.mark.plugin("mistralai")

_MODEL = "voxtral-mini-transcribe-realtime-3"
_CONN_OPTIONS = APIConnectOptions(max_retry=0, timeout=1.0)


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream"
        await asyncio.sleep(0.01)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.keepalive_task = None

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))


class _FakeConnection:
    def __init__(self) -> None:
        self._websocket = _FakeWebSocket()
        self.flushes = 0
        self._events: asyncio.Queue[Any] = asyncio.Queue()

    def push(self, event: Any) -> None:
        self._events.put_nowait(event)

    async def send_audio(self, audio: bytes) -> None:
        pass

    async def flush_audio(self) -> None:
        self.flushes += 1

    async def end_audio(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def events(self) -> AsyncIterator[Any]:
        while True:
            yield await self._events.get()


class _FakeTranscription:
    connections: list[_FakeConnection] = []

    def __init__(self, sdk_config: Any) -> None:
        pass

    async def connect(self, **kwargs: Any) -> _FakeConnection:
        connection = _FakeConnection()
        self.connections.append(connection)
        return connection


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> list[_FakeConnection]:
    from livekit.plugins.mistralai import stt as mistral_stt

    _FakeTranscription.connections = []
    monkeypatch.setattr(mistral_stt, "RealtimeTranscription", _FakeTranscription)
    return _FakeTranscription.connections


def _stt(**kwargs: Any):
    from livekit.plugins.mistralai import STT

    return STT(api_key="test-key", **kwargs)


def _turn(event_type: str) -> UnknownRealtimeEvent:
    return UnknownRealtimeEvent(type=event_type, content={"type": event_type})


def _done(text: str) -> TranscriptionStreamDone:
    return TranscriptionStreamDone(
        model=_MODEL,
        text=text,
        language=None,
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, prompt_audio_seconds=1),
    )


async def test_turn_detection_is_sent_in_a_session_update(connections):
    instance = _stt(model=_MODEL, turn_detection={"min_silence_ms": 300})
    assert instance._vad is None

    stream = instance.stream(conn_options=_CONN_OPTIONS)
    try:
        await _wait_until(lambda: len(connections) == 1)
        assert connections[0]._websocket.sent == [
            {"type": "session.update", "session": {"turn_detection": {"min_silence_ms": 300}}}
        ]
    finally:
        await stream.aclose()


async def test_turn_events_drive_speech_events(connections):
    stream = _stt(model=_MODEL, turn_detection={}).stream(conn_options=_CONN_OPTIONS)
    events: list[stt.SpeechEvent] = []

    async def collect() -> None:
        async for event in stream:
            events.append(event)

    collector = asyncio.create_task(collect())
    try:
        await _wait_until(lambda: len(connections) == 1)
        connection = connections[0]
        for event in (
            _turn("turn.start"),
            TranscriptionStreamTextDelta(text="Hello"),
            _turn("turn.eager_end"),
            _turn("turn.resume"),
            TranscriptionStreamTextDelta(text=" world"),
            _turn("turn.end"),
        ):
            connection.push(event)
        await _wait_until(lambda: connection.flushes == 1)

        connection.push(_done("Hello world"))
        await _wait_until(
            lambda: bool(events) and events[-1].type == stt.SpeechEventType.END_OF_SPEECH
        )
    finally:
        await stream.aclose()
        collector.cancel()

    assert [(e.type, e.alternatives[0].text if e.alternatives else None) for e in events] == [
        (stt.SpeechEventType.START_OF_SPEECH, None),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "Hello"),
        (stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, "Hello"),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "Hello"),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "Hello world"),
        (stt.SpeechEventType.FINAL_TRANSCRIPT, "Hello world"),
        (stt.SpeechEventType.RECOGNITION_USAGE, None),
        (stt.SpeechEventType.END_OF_SPEECH, None),
    ]


async def test_a_turn_starting_before_the_previous_one_is_transcribed(connections):
    stream = _stt(model=_MODEL, turn_detection={}).stream(conn_options=_CONN_OPTIONS)
    events: list[stt.SpeechEvent] = []

    async def collect() -> None:
        async for event in stream:
            events.append(event)

    collector = asyncio.create_task(collect())
    try:
        await _wait_until(lambda: len(connections) == 1)
        connection = connections[0]
        for event in (
            _turn("turn.start"),
            TranscriptionStreamTextDelta(text="Hello"),
            _turn("turn.end"),
            _turn("turn.start"),
            _done("Hello"),
        ):
            connection.push(event)
        await _wait_until(lambda: len(events) == 6)
    finally:
        await stream.aclose()
        collector.cancel()

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.RECOGNITION_USAGE,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.START_OF_SPEECH,
    ]


@pytest.mark.parametrize(
    "turn_detection",
    [
        {"eager_end_threshold": 0.6},
        {"end_threshold": 0.5},
        {"min_silence_ms": 4000},
    ],
)
def test_contradictory_turn_detection_raises(turn_detection):
    with pytest.raises(ValueError):
        _stt(model=_MODEL, turn_detection=turn_detection)


@pytest.mark.parametrize("model", ["voxtral-mini-latest", "voxtral-mini-transcribe-realtime-2602"])
def test_turn_detection_is_ignored_on_unsupported_models(model):
    instance = _stt(model=model, turn_detection={}, vad=object())  # type: ignore[arg-type]
    assert instance._opts.turn_detection is None


def test_an_explicit_vad_wins_over_turn_detection():
    vad = object()
    instance = _stt(model=_MODEL, turn_detection={}, vad=vad)  # type: ignore[arg-type]
    assert instance._opts.turn_detection is None
    assert instance._vad is vad
