"""Bounded protocol lifecycle cases with generated PCM and in-memory sockets."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.agents import APIError, stt
from livekit.plugins.oruk import RealtimeSTT
from livekit.plugins.oruk._realtime_protocol import RealtimeOptions, TurnResult

from .test_plugin_oruk_realtime import EMOTION, OPTIONS, TRANSCRIPT, Session, Socket, audio

pytestmark = pytest.mark.unit


class ControlledSocket(Socket):
    def __init__(self, request_id, mode):
        super().__init__(request_id, mode)
        self.committed = asyncio.Event()

    async def send_json(self, event):
        if event["type"] != "input_audio_buffer.commit":
            return await super().send_json(event)
        self.commands.append(event)
        self.committed.set()
        final = {"type": TRANSCRIPT + "completed", "transcript": "hello"}
        phrase = {
            "type": EMOTION + "completed",
            "phrase_id": "phrase_1",
            "start": 0.0,
            "end": 0.032,
            "text": "hello",
            "emotions": [{"label": "calm", "score": 0.25}],
        }
        usage = {"type": "session.usage", "usage": {"audio_seconds": 0.032}}
        if self.mode == "usage_before_final":
            self.emit(usage)
        self.emit(final)
        if self.mode == "wait_for_late":
            return
        self.emit(phrase)
        if self.mode == "identical_duplicates":
            self.emit(final.copy())
            self.emit(phrase.copy())
        elif self.mode == "conflicting_final":
            self.emit({**final, "transcript": "different"})
        elif self.mode == "conflicting_phrase":
            self.emit({**phrase, "emotions": [{"label": "calm", "score": 0.75}]})
        elif self.mode == "phrase_overflow":
            for index in range(2, 258):
                self.emit({**phrase, "phrase_id": str(index)})
        self.emit(usage)
        if self.mode == "identical_duplicates":
            self.emit(usage.copy())
        elif self.mode == "conflicting_usage":
            self.emit({"type": "session.usage", "usage": {"audio_seconds": 0.064}})
        if self.mode != "never_close":
            self.incoming.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))


class ControlledSession(Session):
    async def ws_connect(self, endpoint, *, headers, **kwargs):
        assert headers["Authorization"] == "Bearer test"
        socket = ControlledSocket(headers["X-Request-ID"], self.mode)
        self.sockets.append(socket)
        return socket


@pytest.fixture
def sockets(monkeypatch):
    session = ControlledSession()
    monkeypatch.setattr("livekit.plugins.oruk.realtime.new_http_session", lambda: session)
    return session


async def test_identical_duplicate_events_are_idempotent(sockets):
    sockets.mode = "identical_duplicates"
    owner = RealtimeSTT(api_key="test")
    try:
        async with owner.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(audio())
            stream.end_input()
            final = [
                item async for item in stream if item.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            ]
        assert len(final) == 1
        receipt = owner.take_turn(final[0].request_id)
        assert len(receipt.phrases) == 1
    finally:
        await owner.aclose()


@pytest.mark.parametrize(
    "mode,code",
    [
        ("conflicting_final", "conflicting_final_transcript"),
        ("conflicting_phrase", "conflicting_phrase_event"),
        ("conflicting_usage", "conflicting_usage"),
        ("usage_before_final", "invalid_usage"),
        ("phrase_overflow", "phrase_metadata_limit"),
    ],
)
async def test_conflicting_or_overflowing_metadata_fails_without_partial_final(sockets, mode, code):
    sockets.mode = mode
    owner = RealtimeSTT(api_key="test")
    seen = []
    try:
        async with owner.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(audio())
            stream.end_input()
            with pytest.raises(APIError, match=code):
                async for item in stream:
                    seen.append(item)
        assert not any(item.type == stt.SpeechEventType.FINAL_TRANSCRIPT for item in seen)
        assert not owner._receipts
        assert len(sockets.sockets) == 1 and sockets.sockets[0].closed
        assert sockets.sockets[0].audio == [bytes(audio().data)]
    finally:
        await owner.aclose()


async def test_completion_timeout_closes_and_joins_without_replay(sockets, monkeypatch):
    sockets.mode = "never_close"
    monkeypatch.setattr(
        "livekit.plugins.oruk.realtime.RealtimeOptions",
        lambda: RealtimeOptions(finish_timeout=0.02, max_turn_seconds=1),
    )
    owner = RealtimeSTT(api_key="test")
    try:
        async with owner.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(audio())
            stream.end_input()
            with pytest.raises(APIError, match="realtime_connection_failed"):
                async for _ in stream:
                    pass
        assert stream._task.done() and stream._metrics_task.done()
        assert len(sockets.sockets) == 1 and sockets.sockets[0].closed
        assert not owner._receipts
    finally:
        await owner.aclose()


async def test_cancel_after_final_text_before_usage_has_no_receipt(sockets):
    sockets.mode = "wait_for_late"
    owner = RealtimeSTT(api_key="test")
    stream = owner.stream(conn_options=OPTIONS)
    try:
        stream.push_frame(audio())
        stream.end_input()
        await asyncio.wait_for(anext(stream), 1)
        await asyncio.wait_for(anext(stream), 1)
        await asyncio.wait_for(sockets.sockets[0].committed.wait(), 1)
        await stream.aclose()
        assert stream._task.done() and stream._metrics_task.done()
        assert sockets.sockets[0].closed and not owner._receipts
        assert len(sockets.sockets) == 1
        assert (
            sum(
                command["type"] == "input_audio_buffer.commit"
                for command in sockets.sockets[0].commands
            )
            == 1
        )
    finally:
        await stream.aclose()
        await owner.aclose()


async def test_receipt_eviction_and_stream_count_bounds():
    owner = RealtimeSTT(api_key="test")
    streams = [owner.stream(conn_options=OPTIONS) for _ in range(4)]
    try:
        with pytest.raises(RuntimeError, match="more than four"):
            owner.stream(conn_options=OPTIONS)
        for index in range(5):
            owner._record_turn(TurnResult(str(index), transcript="synthetic"))
        assert owner.take_turn("0") is None
        assert len(owner._receipts) == 4
        assert owner.take_turns(["1", "2", "3", "4"]) is not None
        assert not owner._receipts
    finally:
        await owner.aclose()
    assert all(stream._task.done() and stream._metrics_task.done() for stream in streams)


async def test_input_backlog_and_boundary_overflow_are_explicit():
    owner = RealtimeSTT(api_key="test")
    stream = owner.stream(conn_options=OPTIONS)
    try:
        for _ in range(31):
            stream.push_frame(audio(samples=5120))
        with pytest.raises(RuntimeError, match="ten audio seconds"):
            stream.push_frame(audio(samples=5120))
        for _ in range(8):
            stream.flush()
        with pytest.raises(RuntimeError, match="too many pending turn boundaries"):
            stream.flush()
        assert stream._queued_bytes == 31 * 10240
        assert len(stream._boundary_times) == 8
    finally:
        await stream.aclose()
        await owner.aclose()
