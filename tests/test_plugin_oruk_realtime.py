"""Synthetic protocol and segmentation tests; no provider, credentials or VAD model."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, stt, vad
from livekit.plugins.oruk import RealtimeSTT, vad_stream_node

pytestmark = pytest.mark.unit
OPTIONS = APIConnectOptions(max_retry=3, timeout=1, retry_interval=0)
TRANSCRIPT = "conversation.item.input_audio_transcription."
EMOTION = "conversation.item.input_audio_emotion."


def audio(value=1, samples=512):
    return rtc.AudioFrame(bytes([value, 0]) * samples, 16000, 1, samples)


class Socket:
    def __init__(self, request_id, mode):
        self.id = request_id
        self.mode = mode
        self.audio = []
        self.commands = []
        self.incoming = asyncio.Queue()
        self.close_code = None
        self.closed = False
        self.emit({"type": "session.created"})

    def emit(self, event):
        event.setdefault("request_id", self.id)
        self.incoming.put_nowait(
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))
        )

    async def receive(self):
        message = await self.incoming.get()
        if message.type == aiohttp.WSMsgType.CLOSE:
            self.close_code = 1000
        return message

    async def send_json(self, event):
        self.commands.append(event)
        if event["type"] == "session.update":
            if self.mode in ("dict_error_code", "list_error_code"):
                self.emit(
                    {
                        "type": "error",
                        "error": {
                            "code": {"unexpected": "object"}
                            if self.mode == "dict_error_code"
                            else []
                        },
                    }
                )
                return
            self.emit(
                {
                    "type": "session.updated",
                    "request_id": "wrong" if self.mode == "wrong_id" else self.id,
                }
            )
        elif event["type"] == "input_audio_buffer.commit":
            if self.mode != "missing_final":
                self.emit({"type": TRANSCRIPT + "completed", "transcript": "hello"})
            self.emit(
                {
                    "type": EMOTION + "completed",
                    "phrase_id": "phrase_1",
                    "start": 0.0,
                    "end": 0.032,
                    "text": "hello",
                    "emotions": [
                        {
                            "label": "calm",
                            "score": float("nan") if self.mode == "invalid_score" else 0.25,
                        }
                    ],
                }
            )
            if self.mode != "missing_usage":
                self.emit({"type": "session.usage", "usage": {"audio_seconds": 0.032}})
            if self.mode == "error_after_usage":
                self.emit({"type": "error", "error": {"code": "cleanup_failed"}})
            self.incoming.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))

    async def send_bytes(self, pcm):
        self.audio.append(pcm)
        if self.mode == "ambiguous_write":
            raise aiohttp.ClientConnectionError("synthetic ambiguous delivery")
        self.emit({"type": TRANSCRIPT + "delta", "delta": "hello"})

    async def close(self):
        self.closed = True


class Session:
    def __init__(self, mode="normal"):
        self.mode = mode
        self.sockets = []
        self.closed = False

    async def ws_connect(self, endpoint, *, headers, **kwargs):
        assert headers["Authorization"] == "Bearer test"
        socket = Socket(headers["X-Request-ID"], self.mode)
        self.sockets.append(socket)
        return socket

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_session(monkeypatch):
    session = Session()
    monkeypatch.setattr("livekit.plugins.oruk.realtime.new_http_session", lambda: session)
    return session


@pytest.mark.asyncio
async def test_native_interim_precedes_commit_and_final_includes_late_phrase(fake_session):
    owner = RealtimeSTT(api_key="test")
    async with owner.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio())
        assert (
            await asyncio.wait_for(anext(stream), 1)
        ).type == stt.SpeechEventType.START_OF_SPEECH
        interim = await asyncio.wait_for(anext(stream), 1)
        assert interim.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
        assert not any(
            c["type"] == "input_audio_buffer.commit" for c in fake_session.sockets[0].commands
        )
        stream.end_input()
        events = [event async for event in stream]
    final = next(e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    metadata = final.alternatives[0].metadata["oruk"]
    assert metadata["phrases"][0]["emotions"] == [{"label": "calm", "score": 0.25}]
    assert metadata["detected_language"] is None
    assert metadata["asr_confidence"] is None
    assert final.alternatives[0].language == ""
    assert owner.take_turn(final.request_id).transcript == "hello"
    assert owner.take_turn(final.request_id) is None
    await owner.aclose()
    assert fake_session.closed and all(socket.closed for socket in fake_session.sockets)


@pytest.mark.asyncio
async def test_flush_rotates_socket_without_replaying_prior_turn(fake_session):
    owner = RealtimeSTT(api_key="test")
    async with owner.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio(1))
        stream.flush()
        stream.push_frame(audio(2))
        stream.end_input()
        events = [event async for event in stream]
    assert len(fake_session.sockets) == 2
    assert fake_session.sockets[0].audio == [bytes(audio(1).data)]
    assert fake_session.sockets[1].audio == [bytes(audio(2).data)]
    finals = [event for event in events if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 2 and finals[0].request_id != finals[1].request_id
    await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "ambiguous_write",
        "error_after_usage",
        "missing_final",
        "missing_usage",
        "invalid_score",
        "wrong_id",
        "dict_error_code",
        "list_error_code",
    ],
)
async def test_failed_turn_never_replays_or_releases_final(fake_session, mode):
    fake_session.mode = mode
    owner = RealtimeSTT(api_key="test")
    seen = []
    async with owner.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio())
        stream.end_input()
        with pytest.raises(APIError) as caught:
            async for event in stream:
                seen.append(event)
    assert not caught.value.retryable
    if mode in ("dict_error_code", "list_error_code"):
        assert "realtime_error" in str(caught.value)
    assert len(fake_session.sockets) == 1
    assert not any(event.type == stt.SpeechEventType.FINAL_TRANSCRIPT for event in seen)
    assert owner.take_turn(fake_session.sockets[0].id) is None
    await owner.aclose()


@pytest.mark.asyncio
async def test_recognizer_close_cleans_session_and_receipts_after_child_failure(fake_session):
    from livekit.plugins.oruk._realtime_protocol import TurnResult

    owner = RealtimeSTT(api_key="test")
    owner._session = fake_session
    owner._record_turn(TurnResult("cached", transcript="synthetic"))
    closed = []

    class Child:
        def __init__(self, fails):
            self.fails = fails

        async def aclose(self):
            closed.append(self.fails)
            if self.fails:
                raise RuntimeError("synthetic child close failure")

    children = [Child(True), Child(False)]
    owner._streams.update(children)
    with pytest.raises(RuntimeError, match="synthetic child close failure"):
        await owner.aclose()
    assert sorted(closed) == [False, True]
    assert owner._closed and fake_session.closed
    assert owner._receipts == {}


@pytest.mark.asyncio
async def test_cancel_does_not_commit_or_replay_consumed_audio(fake_session):
    owner = RealtimeSTT(api_key="test")
    stream = owner.stream(conn_options=OPTIONS)
    stream.push_frame(audio())
    await asyncio.wait_for(anext(stream), 1)
    await asyncio.wait_for(anext(stream), 1)
    await stream.aclose()
    assert len(fake_session.sockets) == 1
    socket = fake_session.sockets[0]
    assert socket.audio == [bytes(audio().data)]
    assert not any(command["type"] == "input_audio_buffer.commit" for command in socket.commands)
    assert socket.closed and owner.take_turn(socket.id) is None
    await owner.aclose()


@pytest.mark.asyncio
async def test_paused_consumer_cannot_grow_metrics_tee_without_bound(fake_session):
    owner = RealtimeSTT(api_key="test")
    stream = owner.stream(conn_options=OPTIONS)

    async def settled_or_failed(count):
        while not stream._task.done():
            if len(fake_session.sockets) >= count and fake_session.sockets[count - 1].closed:
                return
            await asyncio.sleep(0)

    try:
        # Do not read stream events: the framework metrics peer still drains.
        for count in range(1, 21):
            if stream._task.done():
                break
            stream.push_frame(audio())
            stream.flush()
            await asyncio.wait_for(settled_or_failed(count), 1)
        assert stream._task.done()
        events = []
        with pytest.raises(APIError, match="output backlog"):
            async for event in stream:
                events.append(event)
        assert 0 < len(events) <= 64
        assert len(fake_session.sockets) < 20
        assert all(socket.closed for socket in fake_session.sockets)
    finally:
        await stream.aclose()
        await owner.aclose()


@pytest.mark.asyncio
async def test_boundary_time_is_captured_before_queued_turn_is_transmitted(
    fake_session, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr(
        "livekit.plugins.oruk.realtime.time",
        SimpleNamespace(time=lambda: now[0], monotonic=time.monotonic),
    )
    owner = RealtimeSTT(api_key="test")
    async with owner.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio(1))
        stream.flush()
        now[0] = 200.0
        stream.push_frame(audio(2))
        stream.end_input()
        now[0] = 1000.0
        finals = [
            event async for event in stream if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
    assert [e.alternatives[0].metadata["oruk"]["input_boundary_time"] for e in finals] == [
        100.0,
        200.0,
    ]
    assert all(e.speech_end_time is None for e in finals)
    await owner.aclose()


class ScriptedVAD(vad.VAD):
    """Three windows: START, END, START with an overlapping prefix; then EOF tail."""

    def __init__(self):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))

    def stream(self):
        return ScriptedVADStream(self)


class ScriptedVADStream(vad.VADStream):
    async def _main_task(self):
        frames = []
        async for frame in self._input_ch:
            if isinstance(frame, self._FlushSentinel):
                continue
            frames.append(frame)
            if len(frames) == 4:
                continue  # mimic an incomplete inference window at EOF
            fields = {
                "samples_index": len(frames) * 512,
                "timestamp": len(frames) * 0.032,
                "speech_duration": 0.032,
                "silence_duration": 0.0,
            }
            self._event_ch.send_nowait(
                vad.VADEvent(vad.VADEventType.INFERENCE_DONE, frames=[frame], **fields)
            )
            if len(frames) in (1, 3):
                prefix = frames[-2:] if len(frames) == 3 else frames[:]
                self._event_ch.send_nowait(
                    vad.VADEvent(vad.VADEventType.START_OF_SPEECH, frames=prefix, **fields)
                )
            else:
                self._event_ch.send_nowait(
                    vad.VADEvent(vad.VADEventType.END_OF_SPEECH, frames=frames[:], **fields)
                )


@pytest.mark.asyncio
async def test_vad_bridge_no_whole_utterance_replay_prefix_overlap_or_lost_eof_tail(fake_session):
    owner = RealtimeSTT(api_key="test")

    async def source():
        for value in (1, 2, 3, 4):
            yield audio(value)
            await asyncio.sleep(0)

    events = [event async for event in vad_stream_node(owner, source(), detector=ScriptedVAD())]
    assert len(fake_session.sockets) == 2
    assert b"".join(fake_session.sockets[0].audio) == bytes(audio(1).data) + bytes(audio(2).data)
    assert b"".join(fake_session.sockets[1].audio) == bytes(audio(3).data) + bytes(audio(4).data)
    assert len([e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]) == 2
    await owner.aclose()
