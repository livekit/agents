"""Short rejected utterances must not accumulate across distinct speech onsets."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit import rtc
from livekit.agents import Agent, AgentSession, StopResponse, llm, utils, vad
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioInput, FakeAudioOutput
from .fake_llm import FakeLLM
from .fake_stt import FakeRecognizeStream, FakeSTT

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    original = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original)
        provider.shutdown()


class _ManualVAD(vad.VAD):
    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.02))
        self.events = utils.aio.Chan[vad.VADEvent]()

    def stream(self) -> vad.VADStream:
        return _ManualVADStream(self)

    def send(self, event: vad.VADEventType) -> None:
        self.events.send_nowait(
            vad.VADEvent(
                type=event,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=0.0 if event == vad.VADEventType.START_OF_SPEECH else 0.02,
                silence_duration=0.0,
            )
        )


class _ManualVADStream(vad.VADStream):
    async def _main_task(self) -> None:
        assert isinstance(self._vad, _ManualVAD)
        async for event in self._vad.events:
            self._event_ch.send_nowait(event)


class _CapturingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")
        self.turns: list[str] = []

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.turns.append(new_message.text_content or "")
        raise StopResponse()


class _CountingOutput(FakeAudioOutput):
    def __init__(self) -> None:
        super().__init__()
        self.frames = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.frames += 1


async def _wait_for(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate(), "session did not reach the expected state"


@dataclass
class _Harness:
    session: AgentSession
    agent: _CapturingAgent
    vad: _ManualVAD
    stt: FakeRecognizeStream
    output: _CountingOutput
    finals: list[str]
    stop_audio: asyncio.Event
    recognizer: FakeSTT
    mode: Literal["vad", "stt"]

    async def start_speech(self, *, request_id: str = "") -> None:
        if self.mode == "stt":
            self.stt.send_fake_event(
                SpeechEvent(type=SpeechEventType.START_OF_SPEECH, request_id=request_id)
            )
        else:
            self.vad.send(vad.VADEventType.START_OF_SPEECH)
        await _wait_for(lambda: self.session.user_state == "speaking")

    async def final(self, text: str) -> None:
        count = len(self.finals)
        self.stt.send_fake_transcript(text)
        await _wait_for(lambda: len(self.finals) == count + 1)

    async def end_speech(self) -> None:
        if self.mode == "stt":
            self.stt.send_fake_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
        else:
            self.vad.send(vad.VADEventType.END_OF_SPEECH)
        await _wait_for(lambda: self.session.user_state == "listening")

    async def utterance(self, text: str) -> None:
        await self.start_speech()
        await self.final(text)
        await self.end_speech()
        # Complete the configured endpointing delay, using the suite's virtual clock.
        await asyncio.sleep(0.2)

    async def say(
        self,
        *,
        allow_interruptions: bool = True,
        playout_gate: asyncio.Event | None = None,
    ) -> SpeechHandle:
        async def audio() -> AsyncIterator[rtc.AudioFrame]:
            if playout_gate is not None:
                await playout_gate.wait()
            # Finite audio avoids hanging teardown or relying on a naturally completed reply.
            for _ in range(1500):
                if self.stop_audio.is_set():
                    return
                yield rtc.AudioFrame(
                    data=b"\x00\x20" * 320,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=320,
                )
                await asyncio.sleep(0.02)

        speech = self.session.say(
            "original reply", audio=audio(), allow_interruptions=allow_interruptions
        )
        if playout_gate is not None:
            await asyncio.sleep(0.1)
            assert self.output.frames == 0
        else:
            await _wait_for(lambda: self.output.frames > 0)
            assert self.session.agent_state == "speaking"
        return speech


@asynccontextmanager
async def _session(
    mode: Literal["vad", "stt"] = "vad", *, min_words: int = 2
) -> AsyncIterator[_Harness]:
    recognizer = FakeSTT()
    detector = _ManualVAD()
    session = AgentSession(
        stt=recognizer,
        vad=detector if mode == "vad" else None,
        llm=FakeLLM(),
        aec_warmup_duration=None,
        session_close_transcript_timeout=0,
        turn_handling={
            "turn_detection": mode,
            "endpointing": {"min_delay": 0.1, "max_delay": 0.1},
            "interruption": {"min_words": min_words, "min_duration": 0.5},
            "preemptive_generation": {"enabled": False},
        },
    )
    source = FakeAudioInput()
    output = _CountingOutput()
    session.input.audio = source
    session.output.audio = output
    finals: list[str] = []
    session.on(
        "user_input_transcribed",
        lambda event: finals.append(event.transcript) if event.is_final else None,
    )
    agent = _CapturingAgent()
    stop_audio = asyncio.Event()
    try:
        await session.start(agent)
        source.push(0.02)
        stream = await asyncio.wait_for(recognizer.stream_ch.recv(), timeout=2)
        yield _Harness(
            session, agent, detector, stream, output, finals, stop_audio, recognizer, mode
        )
    finally:
        stop_audio.set()
        await session.aclose()


@pytest.mark.parametrize(("ack", "interrupt"), [("okay", "please stop"), ("ألو", "توقف الآن")])
@pytest.mark.parametrize("mode", ["vad", "stt"])
async def test_distinct_short_overlaps_do_not_interrupt(
    ack: str, interrupt: str, mode: Literal["vad", "stt"]
) -> None:
    async with _session(mode) as h:
        original = await h.say()
        for _ in range(3):
            frames = h.output.frames
            await h.utterance(ack)
            assert not original.interrupted
            assert not original.done()
            assert h.output.frames > frames
            assert h.session.agent_state == "speaking"
            assert h.agent.turns == []
        assert h.finals == [ack] * 3
        with pytest.raises(ChanEmpty):
            h.recognizer.stream_ch.recv_nowait()  # the original stream is still the only one

        await h.utterance(interrupt)
        await _wait_for(lambda: h.agent.turns == [interrupt])
        assert original.interrupted


@pytest.mark.parametrize(("ack", "interrupt"), [("okay", "please stop"), ("ألو", "توقف الآن")])
async def test_rejected_overlap_does_not_prefix_a_genuine_interrupt(
    ack: str, interrupt: str
) -> None:
    async with _session() as h:
        original = await h.say()
        await h.utterance(ack)
        assert not original.interrupted
        await h.utterance(interrupt)
        assert h.agent.turns == [interrupt]
        assert original.interrupted


@pytest.mark.parametrize("second_final_after_eou", [False, True])
@pytest.mark.parametrize(("first", "second"), [("please", "stop"), ("توقف", "الآن")])
async def test_split_finals_in_one_utterance_still_interrupt(
    first: str, second: str, second_final_after_eou: bool
) -> None:
    async with _session() as h:
        original = await h.say()
        await h.start_speech()
        await h.final(first)
        if second_final_after_eou:
            await h.end_speech()
            await asyncio.sleep(0.2)
            assert not original.interrupted
        await h.final(second)
        if not second_final_after_eou:
            await h.end_speech()
        await _wait_for(lambda: h.agent.turns == [f"{first} {second}"])
        assert original.interrupted


async def test_renewed_speech_before_endpoint_keeps_the_pending_words() -> None:
    async with _session() as h:
        original = await h.say()
        await h.start_speech()
        await h.final("please")
        await h.end_speech()
        await asyncio.sleep(0.02)
        await h.start_speech()
        await h.final("stop")
        await h.end_speech()
        await _wait_for(lambda: h.agent.turns == ["please stop"])
        assert original.interrupted


@pytest.mark.parametrize(("first", "second"), [("please", "stop"), ("توقف", "الآن")])
async def test_stt_late_final_before_repeated_end_keeps_the_same_utterance(
    first: str, second: str
) -> None:
    async with _session("stt") as h:
        original = await h.say()
        await h.utterance(first)
        assert not original.interrupted
        # No new START: the late segment still belongs to the rejected utterance.
        await h.final(second)
        # STT mode owns endpointing through END events, including this later verdict.
        await h.end_speech()
        await _wait_for(lambda: h.agent.turns == [f"{first} {second}"])
        assert original.interrupted


@pytest.mark.parametrize("word", ["yes", "نعم"])
async def test_one_word_while_listening_is_accepted(word: str) -> None:
    async with _session() as h:
        await h.utterance(word)
        assert h.agent.turns == [word]


@pytest.mark.parametrize("queued", [False, True])
async def test_short_turn_during_generation_is_retained(queued: bool) -> None:
    async with _session() as h:
        playout_gate = asyncio.Event()
        original = await h.say(playout_gate=playout_gate)
        pending = await h.say(playout_gate=playout_gate) if queued else None
        await h.utterance("please")
        assert not original.interrupted
        if pending is not None:
            assert not pending.interrupted
        await h.utterance("stop")
        assert h.agent.turns == ["please stop"]
        assert original.interrupted
        assert h.output.frames == 0


async def test_protected_speech_is_not_interrupted() -> None:
    async with _session() as h:
        original = await h.say(allow_interruptions=False)
        for _ in range(3):
            await h.utterance("okay")
            assert not original.interrupted
            assert not original.done()
        assert h.agent.turns == []


async def test_rejected_overlap_does_not_prefix_a_later_listening_answer() -> None:
    async with _session() as h:
        original = await h.say()
        await h.utterance("okay")
        assert not original.interrupted
        h.stop_audio.set()
        await original
        await _wait_for(lambda: h.session.agent_state == "listening")
        await h.utterance("yes")
        assert h.agent.turns == ["yes"]


async def test_user_speech_beginning_before_playout_is_retained() -> None:
    async with _session() as h:
        await h.start_speech()
        await h.final("please")
        await h.end_speech()
        original = await h.say()
        await asyncio.sleep(0.2)
        assert not original.interrupted
        await h.utterance("stop")
        assert h.agent.turns == ["please stop"]
        assert original.interrupted


@pytest.mark.parametrize(
    ("min_words", "playout_after_first"),
    [(2, True), (3, True), (3, False)],
    ids=["two-word-hybrid", "three-word-hybrid", "three-word-generating"],
)
async def test_preplayout_words_survive_later_rejected_overlaps(
    min_words: int, playout_after_first: bool
) -> None:
    async with _session(min_words=min_words) as h:
        playout_gate = asyncio.Event()
        original = await h.say(playout_gate=playout_gate)
        await h.utterance("please")
        assert not original.interrupted
        assert h.agent.turns == []
        if playout_after_first:
            playout_gate.set()
            await _wait_for(lambda: h.output.frames > 0)

        words = ["please", "stop", "now"][:min_words]
        for index, word in enumerate(words[1:], start=2):
            await h.utterance(word)
            if index < min_words:
                assert not original.interrupted
                assert h.agent.turns == []

        await _wait_for(lambda: h.agent.turns == [" ".join(words)])
        assert original.interrupted
        assert h.finals == words
        if not playout_after_first:
            assert h.output.frames == 0


async def test_rejected_turns_are_dropped_not_traced_as_committed(
    spans: InMemorySpanExporter,
) -> None:
    async with _session() as h:
        original = await h.say()
        for _ in range(3):
            await h.utterance("okay")
            assert not original.interrupted
        await h.utterance("please stop")
        assert h.agent.turns == ["please stop"]

    waits = [span for span in spans.get_finished_spans() if span.name == "eou_wait"]
    outcomes = [(span.attributes or {})[trace_types.ATTR_EOU_OUTCOME] for span in waits]
    assert outcomes == ["dropped", "dropped", "dropped", "committed"]


async def test_stt_onset_request_id_belongs_to_the_new_turn(spans: InMemorySpanExporter) -> None:
    async with _session("stt") as h:
        original = await h.say()
        await h.utterance("okay")
        assert not original.interrupted
        await h.start_speech(request_id="new-utterance-request")
        await h.final("please stop")
        await h.end_speech()
        await _wait_for(lambda: h.agent.turns == ["please stop"])

    [turn] = [
        span
        for span in spans.get_finished_spans()
        if span.name == "user_turn"
        and (span.attributes or {}).get(trace_types.ATTR_USER_TRANSCRIPT) == "please stop"
    ]
    assert list((turn.attributes or {}).get(trace_types.ATTR_PROVIDER_REQUEST_IDS, ())) == [
        "new-utterance-request"
    ]


async def test_provider_onset_before_vad_keeps_the_new_turn_clean() -> None:
    async with _session() as h:
        original = await h.say()
        await h.utterance("okay")
        assert not original.interrupted
        # Provider START/FINAL can beat the VAD's asynchronous onset callback.
        h.stt.send_fake_event(SpeechEvent(type=SpeechEventType.START_OF_SPEECH))
        await h.final("please stop")
        await h.start_speech()
        await h.end_speech()
        await asyncio.sleep(0.2)
        assert h.agent.turns == ["please stop"]
        assert original.interrupted
