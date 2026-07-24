"""Hermetic tests for the OpenAI Realtime Translation plugin.

These never touch the network: ``_create_ws_conn`` is monkeypatched to return a
fake websocket whose inbound queue we drive directly, asserting that the
continuous translation stream is correctly mapped onto LiveKit realtime events.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, llm
from livekit.plugins.openai.realtime.translation_model import (
    RealtimeTranslationModel,
    RealtimeTranslationSession,
    process_translation_url,
)

pytestmark = pytest.mark.plugin("openai")


class _FakeWS:
    def __init__(self) -> None:
        self._in: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False

    def feed(self, event: dict) -> None:
        self._in.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), ""))

    async def receive(self) -> aiohttp.WSMessage:
        return await self._in.get()

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True


async def _wait_for(cond: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.005)


def _make_session(
    monkeypatch: pytest.MonkeyPatch, **model_kwargs: object
) -> tuple[RealtimeTranslationModel, RealtimeTranslationSession, _FakeWS]:
    fake = _FakeWS()

    async def _fake_conn(self: RealtimeTranslationSession) -> _FakeWS:
        return fake

    monkeypatch.setattr(RealtimeTranslationSession, "_create_ws_conn", _fake_conn)

    model = RealtimeTranslationModel(
        target_language=str(model_kwargs.pop("target_language", "es")),
        api_key="sk-test",
        output_segment_idle=float(model_kwargs.pop("output_segment_idle", 0.05)),  # type: ignore[arg-type]
        input_segment_idle=float(model_kwargs.pop("input_segment_idle", 0.05)),  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=0, timeout=0.5),
        **model_kwargs,  # type: ignore[arg-type]
    )
    session = model.session()
    return model, session, fake


async def _collect(
    ev: llm.GenerationCreatedEvent,
) -> tuple[list[str], list]:
    texts: list[str] = []
    frames: list = []

    async def _drain_text(stream) -> None:
        async for delta in stream:
            texts.append(delta)

    async def _drain_audio(stream) -> None:
        async for frame in stream:
            frames.append(frame)

    async for msg in ev.message_stream:
        await asyncio.gather(_drain_text(msg.text_stream), _drain_audio(msg.audio_stream))

    return texts, frames


def test_translation_url() -> None:
    assert (
        process_translation_url("https://api.openai.com/v1", "gpt-realtime-translate")
        == "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
    )


async def test_session_update_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    model, session, fake = _make_session(
        monkeypatch,
        target_language="es",
        transcription_model="gpt-realtime-whisper",
        input_audio_noise_reduction="near_field",
    )
    try:
        await _wait_for(lambda: len(fake.sent) >= 1)
        ev = fake.sent[0]
        assert ev["type"] == "session.update"
        audio = ev["session"]["audio"]
        assert audio["output"]["language"] == "es"
        assert audio["input"]["transcription"] == {"model": "gpt-realtime-whisper"}
        assert audio["input"]["noise_reduction"] == {"type": "near_field"}
    finally:
        await session.aclose()


async def test_output_segment_streams_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    model, session, fake = _make_session(monkeypatch)
    gens: list[llm.GenerationCreatedEvent] = []
    metrics: list = []
    session.on("generation_created", gens.append)
    session.on("metrics_collected", metrics.append)
    try:
        await _wait_for(lambda: len(fake.sent) >= 1)
        fake.feed({"type": "session.created", "session": {"id": "sess_1"}})
        fake.feed({"type": "session.output_transcript.delta", "delta": "Hello "})
        await _wait_for(lambda: len(gens) == 1)

        collected = asyncio.create_task(_collect(gens[0]))
        audio_b64 = base64.b64encode(b"\x00" * 960).decode()  # 480 mono pcm16 samples
        fake.feed(
            {
                "type": "session.output_audio.delta",
                "delta": audio_b64,
                "sample_rate": 24000,
                "channels": 1,
            }
        )
        fake.feed({"type": "session.output_transcript.delta", "delta": "world"})

        texts, frames = await collected  # completes when the idle timer closes the segment
        assert texts == ["Hello ", "world"]
        assert len(frames) == 1
        assert frames[0].sample_rate == 24000
        assert frames[0].samples_per_channel == 480

        await _wait_for(lambda: len(metrics) == 1)
        # response_id and metrics request_id must match (keeps trace spans balanced)
        assert metrics[0].request_id == gens[0].response_id
    finally:
        await session.aclose()


async def test_back_to_back_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    model, session, fake = _make_session(monkeypatch)
    gens: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", gens.append)
    try:
        await _wait_for(lambda: len(fake.sent) >= 1)
        fake.feed({"type": "session.output_transcript.delta", "delta": "one"})
        await _wait_for(lambda: len(gens) == 1)
        await asyncio.sleep(0.08)  # let the first segment go idle and close
        fake.feed({"type": "session.output_transcript.delta", "delta": "two"})
        await _wait_for(lambda: len(gens) == 2)
        assert gens[0].response_id != gens[1].response_id
    finally:
        await session.aclose()


async def test_input_transcription_interim_then_final(monkeypatch: pytest.MonkeyPatch) -> None:
    model, session, fake = _make_session(monkeypatch)
    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)
    try:
        await _wait_for(lambda: len(fake.sent) >= 1)
        fake.feed({"type": "session.input_transcript.delta", "delta": "Hola "})
        fake.feed({"type": "session.input_transcript.delta", "delta": "mundo"})
        await _wait_for(lambda: len(events) >= 2)
        assert events[0].transcript == "Hola " and events[0].is_final is False
        assert events[1].transcript == "Hola mundo" and events[1].is_final is False

        await _wait_for(lambda: any(e.is_final for e in events))
        final = next(e for e in reversed(events) if e.is_final)
        assert final.transcript == "Hola mundo"
        assert final.item_id == events[0].item_id  # same item id across the utterance
    finally:
        await session.aclose()


async def test_late_delta_reopens_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    # long idle so the timer can't interfere; we simulate the close/late-delta race
    model, session, fake = _make_session(monkeypatch, output_segment_idle=5.0)
    gens: list[llm.GenerationCreatedEvent] = []
    metrics: list = []
    session.on("generation_created", gens.append)
    session.on("metrics_collected", metrics.append)
    try:
        await _wait_for(lambda: len(fake.sent) >= 1)
        fake.feed({"type": "session.output_transcript.delta", "delta": "a"})
        await _wait_for(lambda: len(gens) == 1)

        # force the "channels closed but reference not yet cleared" state and
        # confirm the next delta does not raise and opens a fresh segment
        seg = session._current_segment
        assert seg is not None
        seg.text_ch.close()
        seg.audio_ch.close()
        fake.feed({"type": "session.output_transcript.delta", "delta": "b"})
        await _wait_for(lambda: len(gens) == 2)

        # the reopened segment must still record a first-token timestamp, so its
        # ttft metric is real (not -1) once it closes
        reopened = session._current_segment
        assert reopened is not None
        assert reopened.first_token_timestamp is not None
        session._close_segment()
        await _wait_for(lambda: any(m.request_id == gens[1].response_id for m in metrics))
        reopened_metric = next(m for m in metrics if m.request_id == gens[1].response_id)
        assert reopened_metric.ttft >= 0
    finally:
        await session.aclose()


async def test_generate_reply_returns_failed_future(monkeypatch: pytest.MonkeyPatch) -> None:
    model, session, fake = _make_session(monkeypatch)
    try:
        fut = session.generate_reply()
        assert fut.done()
        with pytest.raises(llm.RealtimeError):
            fut.result()
    finally:
        await session.aclose()


def test_capabilities() -> None:
    model = RealtimeTranslationModel(target_language="es", api_key="sk-test")
    caps = model.capabilities
    assert caps.turn_detection is True
    assert caps.audio_output is True
    assert caps.user_transcription is True
    assert caps.message_truncation is False
    assert caps.mutable_chat_context is False
