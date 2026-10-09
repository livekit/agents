"""Combined native/core candidate tests: synthetic sockets, real AgentSession hooks.

Run with the separately reviewed STT turn-identity core source on PYTHONPATH.
Stock core is intentionally not sufficient: these tests must fail without the
identity propagation patch, rather than silently skip the missing feature.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from examples.other.oruk_bound_turn_hook import BoundTurnAgent
from livekit.agents import Agent, AgentSession, llm, stt, vad
from livekit.plugins.oruk import RealtimeSTT
from livekit.plugins.oruk._realtime_protocol import TurnResult

from .fake_io import FakeAudioInput
from .fake_stt import FakeSTT
from .test_plugin_oruk_realtime import OPTIONS, Session, audio

pytestmark = pytest.mark.unit


@pytest.fixture
def sockets(monkeypatch):
    session = Session()
    monkeypatch.setattr("livekit.plugins.oruk.realtime.new_http_session", lambda: session)
    return session


async def wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("session condition did not complete")


class PairedVAD(vad.VAD):
    """Each pair of synthetic processed windows forms a START/END segment."""

    def __init__(self):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))

    def stream(self):
        return PairedVADStream(self)


class PairedVADStream(vad.VADStream):
    async def _main_task(self):
        count = 0
        segment = []
        async for frame in self._input_ch:
            if isinstance(frame, self._FlushSentinel):
                continue
            count += 1
            segment.append(frame)
            fields = {
                "samples_index": count * 512,
                "timestamp": count * 0.032,
                "speech_duration": 0.032,
                "silence_duration": 0.0,
            }
            self._event_ch.send_nowait(
                vad.VADEvent(vad.VADEventType.INFERENCE_DONE, frames=[frame], **fields)
            )
            if count % 2:
                self._event_ch.send_nowait(
                    vad.VADEvent(vad.VADEventType.START_OF_SPEECH, frames=[frame], **fields)
                )
            else:
                self._event_ch.send_nowait(
                    vad.VADEvent(vad.VADEventType.END_OF_SPEECH, frames=segment[:], **fields)
                )
                segment.clear()


class CapturingBoundAgent(BoundTurnAgent):
    def __init__(self, recognizer):
        super().__init__(recognizer, PairedVAD(), instructions="offline test")
        self.completed = asyncio.Queue()

    async def on_user_turn_completed(self, turn_ctx, new_message):
        await super().on_user_turn_completed(turn_ctx, new_message)
        self.completed.put_nowait((new_message.model_copy(deep=True), turn_ctx.copy()))


async def test_vad_pcm_socket_final_late_affect_reaches_real_hook_twice(sockets, caplog):
    recognizer = RealtimeSTT(api_key="test")
    agent = CapturingBoundAgent(recognizer)
    session = AgentSession(
        aec_warmup_duration=None,
        turn_handling={"endpointing": {"min_delay": 0.0, "max_delay": 0.0}},
    )
    source = FakeAudioInput()
    session.input.audio = source
    await session.start(agent)
    try:
        seen = []
        for value in (1, 3):
            source.push(audio(value))
            source.push(audio(value + 1))
            message, context = await asyncio.wait_for(agent.completed.get(), 2.0)
            assert message.raw_text_content == "hello"
            request_id = sockets.sockets[-1].id
            assert message.extra["stt_request_ids"] == [request_id]
            assert message.extra["stt_request_ids_complete"] is True
            signal = message.extra["oruk"]["turns"][0]
            assert signal["request_id"] == request_id
            assert signal["observed_phrases"][0]["emotions"] == [{"label": "calm", "score": 0.25}]
            assert context.items[-1].role == "system"
            assert request_id in context.items[-1].text_content
            assert recognizer.take_turn(request_id) is None
            seen.append(request_id)
        assert seen[0] != seen[1]
        assert [b"".join(socket.audio) for socket in sockets.sockets] == [
            bytes(audio(1).data) + bytes(audio(2).data),
            bytes(audio(3).data) + bytes(audio(4).data),
        ]
    finally:
        await session.aclose()
        await recognizer.aclose()
    assert not [record for record in caplog.records if record.levelno >= 40]


class ReceiptAgent(Agent):
    """Real manual session hook running the exact recipe against provider results."""

    def __init__(self, recognizer):
        super().__init__(instructions="offline test")
        self._oruk = recognizer
        self.completed = asyncio.Queue()

    async def on_user_turn_completed(self, turn_ctx, new_message):
        await BoundTurnAgent.on_user_turn_completed(self, turn_ctx, new_message)
        self.completed.put_nowait(new_message.model_copy(deep=True))


async def native_final(recognizer, value):
    async with recognizer.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio(value))
        stream.end_input()
        events = [item async for item in stream]
    return next(item for item in events if item.type == stt.SpeechEventType.FINAL_TRANSCRIPT)


@pytest.mark.parametrize("discard_first", [False, True])
async def test_merged_or_cleared_native_receipts_use_exact_real_hook_ids(sockets, discard_first):
    recognizer = RealtimeSTT(api_key="test")
    provider = FakeSTT()
    session = AgentSession(
        stt=provider,
        aec_warmup_duration=None,
        turn_handling={
            "turn_detection": "manual",
            "endpointing": {"min_delay": 0.0, "max_delay": 0.0},
        },
    )
    source = FakeAudioInput()
    session.input.audio = source
    agent = ReceiptAgent(recognizer)
    await session.start(agent)
    source.push(audio())
    try:
        stream = await asyncio.wait_for(provider.stream_ch.recv(), 1.0)
        recognition = session._activity._audio_recognition
        first = await native_final(recognizer, 1)
        stream._event_ch.send_nowait(first)
        await wait_until(lambda: recognition._audio_transcript == "hello")
        if discard_first:
            session.clear_user_turn()
            stream = await asyncio.wait_for(provider.stream_ch.recv(), 1.0)
        second = await native_final(recognizer, 2)
        stream._event_ch.send_nowait(second)
        text = "hello" if discard_first else "hello hello"
        await wait_until(lambda: recognition._audio_transcript == text)
        await session.commit_user_turn(transcript_timeout=0, stt_flush_duration=0)
        message = await asyncio.wait_for(agent.completed.get(), 1.0)
        expected = [second.request_id] if discard_first else [first.request_id, second.request_id]
        assert message.extra["stt_request_ids"] == expected
        assert [item["request_id"] for item in message.extra["oruk"]["turns"]] == expected
        if discard_first:
            assert recognizer.take_turn(first.request_id) is not None
    finally:
        await session.aclose()
        await recognizer.aclose()


async def test_receipt_group_missing_member_is_atomic_and_expiry_is_fail_closed(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "livekit.plugins.oruk.realtime.time",
        SimpleNamespace(time=time.time, monotonic=lambda: now[0]),
    )
    recognizer = RealtimeSTT(api_key="test")
    try:
        recognizer._record_turn(TurnResult("one", transcript="hello"))
        assert recognizer.take_turns(["one", "missing"]) is None
        assert recognizer.take_turns(["one", "one"]) is None
        assert recognizer.take_turn("one") is not None
        recognizer._record_turn(TurnResult("expired", transcript="hello"))
        now[0] += 31
        assert recognizer.take_turns(["expired"]) is None
    finally:
        await recognizer.aclose()


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"stt_request_ids": ["one"], "stt_request_ids_complete": False},
        {"stt_request_ids": ["wrong"], "stt_request_ids_complete": True},
    ],
)
async def test_recipe_never_uses_latest_or_same_text_when_binding_missing(extra):
    recognizer = RealtimeSTT(api_key="test")
    try:
        recognizer._record_turn(TurnResult("one", transcript="hello"))
        message = llm.ChatMessage(role="user", content=["hello"], extra=extra)
        context = llm.ChatContext.empty()
        agent = ReceiptAgent(recognizer)
        await agent.on_user_turn_completed(context, message)
        assert "oruk" not in message.extra
        assert context.items == []
        assert recognizer.take_turn("one") is not None
    finally:
        await recognizer.aclose()
