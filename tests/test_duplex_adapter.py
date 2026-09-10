"""
Tests for the duplex adapter: a full-duplex model emits audio continuously and reports only
fragments of its transcript, so the framework gates the audio against the model's own noise floor,
cuts it into generations, attaches the words to the sound that carries them and keeps the chat
context the framework sees. Output the model never transcribes still plays, it simply produces no
chat item.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import llm
from livekit.agents.llm.duplex_adapter import (
    AdaptiveNoiseGate,
    FixedGate,
    _DuplexRealtimeSession,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import aio

pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000
FRAME_MS = 100


def _frame(level: float, *, duration_ms: int = FRAME_MS) -> rtc.AudioFrame:
    """A frame whose RMS is exactly ``level`` (0..1), from an alternating square wave."""
    num_samples = SAMPLE_RATE * duration_ms // 1000
    amplitude = int(level * 32767)
    samples = np.empty(num_samples, dtype=np.int16)
    samples[0::2] = amplitude
    samples[1::2] = -amplitude
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=num_samples,
    )


class _FakeDuplexModel(llm.DuplexModel):
    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=True, auto_tool_reply_generation=True
            )
        )
        self.session_obj: _FakeDuplexSession | None = None
        self.askable = False

    @property
    def model(self) -> str:
        return "fake-duplex"

    @property
    def provider(self) -> str:
        return "fake"

    def session(self) -> _FakeDuplexSession:
        self.session_obj = _FakeDuplexSession(self)
        return self.session_obj

    async def aclose(self) -> None:
        pass


class _FakeDuplexSession(llm.DuplexSession):
    def __init__(self, model: _FakeDuplexModel) -> None:
        super().__init__(model)
        self.audio_ch = aio.Chan[llm.DuplexAudioFrame]()
        self._tools = llm.ToolContext([])
        self.config_batches: list[tuple[object, object, object]] = []
        self.appended: list[llm.ChatItem] = []
        self.replies_requested: list[object] = []
        self.fail_instructions = False

    @property
    def audio_stream(self) -> aio.Chan[llm.DuplexAudioFrame]:
        return self.audio_ch

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    async def _update_instructions(self, instructions: str) -> None:
        if self.fail_instructions:
            raise llm.RealtimeError("no")

    async def _append_items(self, items: list[llm.ChatItem]) -> None:
        self.appended.extend(items)

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        pass

    async def aclose(self) -> None:
        await super().aclose()
        if not self.audio_ch.closed:
            self.audio_ch.close()

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        if not self._duplex_model.askable:
            super()._generate_reply(instructions=instructions, tool_choice=tool_choice, tools=tools)
        self.replies_requested.append(instructions)

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        self.config_batches.append((instructions, chat_ctx, tools))
        await super()._update_session(instructions=instructions, chat_ctx=chat_ctx, tools=tools)

    # test helpers

    def push(self, level: float, *, count: int = 1, start_ms: int | None = None) -> None:
        for i in range(count):
            self.audio_ch.send_nowait(
                llm.DuplexAudioFrame(
                    _frame(level), start_ms=None if start_ms is None else start_ms + i * FRAME_MS
                )
            )

    def say(self, text: str, *, start_ms: int | None = None, end_ms: int | None = None) -> None:
        self.emit(
            "transcript_delta",
            llm.DuplexOutputTranscriptDelta(text=text, start_ms=start_ms, end_ms=end_ms),
        )

    def heard(self, item_id: str, text: str) -> None:
        """The plugin's account of one finished user turn."""
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=text, is_final=True),
        )
        self.emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
        )


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _read(ev: llm.GenerationCreatedEvent) -> tuple[int, str]:
    """Total forwarded frames and transcript of a finished generation."""
    frames, text = 0, ""
    async for msg in ev.message_stream:
        async for _ in msg.audio_stream:
            frames += 1
        async for chunk in msg.text_stream:
            text += chunk
    return frames, text


async def _chunks(ev: llm.GenerationCreatedEvent) -> list[str]:
    chunks: list[str] = []
    async for msg in ev.message_stream:
        async for _ in msg.audio_stream:
            pass
        async for chunk in msg.text_stream:
            chunks.append(chunk)
    return chunks


@pytest.fixture
async def duplex() -> tuple[_FakeDuplexSession, _DuplexRealtimeSession, list]:
    model = _FakeDuplexModel()
    session = llm.DuplexRealtimeAdapter(model).session()
    assert isinstance(session, _DuplexRealtimeSession)
    fake = model.session_obj
    assert fake is not None

    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)

    yield fake, session, generations

    await session.aclose()


# the gate


def test_gate_stays_closed_on_silence() -> None:
    gate = AdaptiveNoiseGate()
    assert not any(gate.update(_frame(0.0)) for _ in range(20))


def test_gate_stays_closed_on_steady_room_tone() -> None:
    # a constant floor is never "output", however loud it is in absolute terms
    gate = AdaptiveNoiseGate()
    assert not any(gate.update(_frame(0.02)) for _ in range(40))


def test_gate_opens_on_speech_over_room_tone() -> None:
    gate = AdaptiveNoiseGate()
    for _ in range(30):
        gate.update(_frame(0.02))
    assert gate.update(_frame(0.3))


def test_gate_holds_through_the_silence_window_then_closes() -> None:
    gate = AdaptiveNoiseGate(min_silence_duration=0.25)
    for _ in range(30):
        gate.update(_frame(0.001))
    assert gate.update(_frame(0.3))

    # two 100 ms quiet frames are inside the 250 ms window, the third crosses it
    assert gate.update(_frame(0.001))
    assert gate.update(_frame(0.001))
    assert not gate.update(_frame(0.001))


def test_speech_longer_than_the_window_does_not_raise_the_floor_to_its_own_level() -> None:
    """Sustained delivery must not wedge the gate: the floor is the quiet speech leaves behind."""
    gate = AdaptiveNoiseGate(window=1.0)
    for _ in range(10):
        gate.update(_frame(0.001))
    assert gate.update(_frame(0.3))

    # ten seconds of speech against a one second window, quiet only in the gaps between words
    for _ in range(10):
        for _ in range(9):
            assert gate.update(_frame(0.3))
        assert gate.update(_frame(0.002))
    assert gate.update(_frame(0.3))

    for _ in range(4):
        assert gate.update(_frame(0.001))
    assert not gate.update(_frame(0.001))
    assert gate.update(_frame(0.3))


def test_a_gapless_monologue_never_wedges_the_gate_shut() -> None:
    """Speech must not teach the gate its own level, however long the model runs on for."""
    gate = AdaptiveNoiseGate()
    for _ in range(30):
        gate.update(_frame(0.002))

    assert all(gate.update(_frame(0.2)) for _ in range(400))

    for _ in range(5):
        gate.update(_frame(0.002))
    assert not gate.update(_frame(0.002))


def test_one_dropped_frame_does_not_define_the_floor() -> None:
    """A lost packet is not evidence the model went quieter; only a whole stretch counts."""
    gate = AdaptiveNoiseGate()
    for _ in range(30):
        gate.update(_frame(0.002))

    gate.update(_frame(0.0))
    assert not any(gate.update(_frame(0.002)) for _ in range(300))


def test_the_floor_recovers_once_the_model_is_really_silent() -> None:
    gate = AdaptiveNoiseGate()
    # the session opens mid-speech, so the gate starts out with speech for a floor
    assert not any(gate.update(_frame(0.2)) for _ in range(30))

    for _ in range(20):
        gate.update(_frame(0.002))
    assert gate.update(_frame(0.2))


def test_a_declared_silence_needs_no_learning() -> None:
    gate = FixedGate(0.002)
    assert gate.update(_frame(0.2))
    assert all(gate.update(_frame(0.2)) for _ in range(600))

    for _ in range(5):
        gate.update(_frame(0.002))
    assert not gate.update(_frame(0.002))


def test_a_declared_silence_still_ignores_steady_room_tone() -> None:
    gate = FixedGate(0.02)
    assert not any(gate.update(_frame(0.02)) for _ in range(40))
    assert gate.update(_frame(0.3))


async def test_the_adapter_prefers_the_gate_the_model_declares() -> None:
    async def gate_of(adapter: llm.DuplexRealtimeAdapter) -> object:
        session = adapter.session()
        assert isinstance(session, _DuplexRealtimeSession)
        gate = session._gate
        await session.aclose()
        return gate

    model = _FakeDuplexModel()
    assert isinstance(await gate_of(llm.DuplexRealtimeAdapter(model)), AdaptiveNoiseGate)

    declared = FixedGate(0.002)
    model.audio_gate = lambda: declared  # type: ignore[method-assign]
    assert await gate_of(llm.DuplexRealtimeAdapter(model)) is declared

    # an explicit gate still wins
    explicit = llm.DuplexRealtimeAdapter(model, gate=AdaptiveNoiseGate)
    assert isinstance(await gate_of(explicit), AdaptiveNoiseGate)


async def test_a_stream_that_stops_mid_burst_still_ends_the_generation(duplex) -> None:
    """A provider may stop sending instead of streaming its own silence; the burst must not hang."""
    fake, session, generations = duplex
    session._audio_timeout = 0.05
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    await _settle()
    assert len(generations) == 1
    assert session._burst is not None

    await asyncio.sleep(0.25)  # no further frames arrive; the timer also waits out the last one
    assert session._burst is None
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1))[0] == 3

    # the gate went with it, so the next frames open a new burst rather than joining the old one
    fake.push(0.3, count=3)
    await _settle()
    assert len(generations) == 2


def test_gate_decides_on_audio_duration_not_on_frame_count() -> None:
    """The same sound streamed at two frame sizes is gated identically."""
    levels = [0.001] * 12 + [0.3] * 6 + [0.001] * 12
    decisions: list[list[bool]] = []
    for frame_ms in (20, FRAME_MS):
        gate = AdaptiveNoiseGate(window=1.0, min_silence_duration=0.45)
        states: list[bool] = []
        for level in levels:
            for _ in range(FRAME_MS // frame_ms):
                state = gate.update(_frame(level, duration_ms=frame_ms))
            states.append(state)
        decisions.append(states)

    assert decisions[0] == decisions[1]
    # the run exercises both edges rather than agreeing on a gate that never moved
    assert decisions[0][12] and not decisions[0][-1]


# the segmenter


async def test_silence_alone_produces_no_generation(duplex) -> None:
    fake, _session, generations = duplex
    fake.push(0.0, count=20)
    await _settle()
    assert generations == []


async def test_untranscribed_burst_plays_and_carries_no_transcript(duplex) -> None:
    """A backchannel: the model vocalises, nothing describes it, it still reaches the room."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    fake.push(0.001, count=5)
    await _settle()

    assert len(generations) == 1
    frames, text = await _read(generations[0])
    assert frames >= 3
    assert text == ""


async def test_two_stretches_of_speech_are_two_generations(duplex) -> None:
    """Only the sound stopping ends a burst: there is no turn event to do it."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    fake.push(0.001, count=8)
    await _settle()
    fake.push(0.3, count=3)
    fake.push(0.001, count=8)
    await _settle()

    assert len(generations) == 2


async def test_a_pause_inside_an_utterance_stays_in_one_generation(duplex) -> None:
    """A burst carries its own quiet stretches, so what it forwards plays back in real time."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)  # let the gate learn the model's floor
    fake.push(0.3, count=3)
    fake.push(0.001, count=3)  # shorter than the gate's silence window
    fake.push(0.3, count=3)
    fake.push(0.001, count=8)
    await _settle()

    assert len(generations) == 1
    frames, _ = await asyncio.wait_for(_read(generations[0]), timeout=1)
    assert frames >= 9  # both stretches and the pause between them


async def test_fragments_join_the_burst_that_is_open(duplex) -> None:
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=2)
    await _settle()
    fake.say("Sure,", start_ms=2000, end_ms=2200)
    fake.push(0.3, count=2)
    fake.say(" I can.", start_ms=2200, end_ms=2400)
    fake.push(0.001, count=8)
    await _settle()

    assert len(generations) == 1
    frames, text = await asyncio.wait_for(_read(generations[0]), timeout=1)
    assert frames >= 4
    assert text == "Sure, I can."


async def test_transcript_is_timed_against_the_burst_it_joins(duplex) -> None:
    """The first fragment anchors the model's span clock to the sound that opened the burst, so
    every fragment is placed where it is spoken in the forwarded audio."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=2)
    await _settle()
    fake.say("hello", start_ms=7000, end_ms=7200)
    fake.say(" there", start_ms=7200, end_ms=7400)
    fake.push(0.3, count=3)
    fake.audio_ch.close()
    await _settle()

    first, second = await _chunks(generations[0])
    assert isinstance(first, TimedString) and isinstance(second, TimedString)
    assert (first.start_time, first.end_time) == (pytest.approx(0.0), pytest.approx(0.2))
    assert (second.start_time, second.end_time) == (pytest.approx(0.2), pytest.approx(0.4))


async def test_stamped_audio_supplies_the_clock(duplex) -> None:
    """A provider that times its audio needs no inference: the stamps are the adapter's clock."""
    fake, session, generations = duplex
    fake.push(0.001, count=20, start_ms=5000)
    fake.push(0.3, count=2, start_ms=7000)
    await _settle()
    assert session._audio_ms == 7200
    fake.say("hello", start_ms=7000, end_ms=7200)
    fake.push(0.3, count=3, start_ms=7200)
    fake.audio_ch.close()
    await _settle()

    (first,) = await _chunks(generations[0])
    assert isinstance(first, TimedString)
    assert (first.start_time, first.end_time) == (pytest.approx(0.0), pytest.approx(0.2))


async def test_a_fragment_that_leads_its_audio_waits_for_the_burst_that_carries_it(
    duplex, caplog
) -> None:
    """A transcript can beat its own audio to the wire by half a second or more; it must neither
    open a generation of silence nor be lost."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    await _settle()
    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        fake.say(" Sure.", start_ms=2000, end_ms=2200)
        fake.push(0.001, count=5)
        await _settle()
        assert generations == []
        assert session._fragments  # held for the sound to come

        fake.push(0.3, count=3)
        fake.push(0.001, count=8)
        await _settle()

    assert [r.message for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert len(generations) == 1
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1)) == (7, " Sure.")


async def test_a_fragment_the_burst_ends_before_reaching_rolls_over_to_the_next(duplex) -> None:
    """Text for the next utterance arrives while the current one is still sounding; the current
    burst never reaches its position, so it anchors the burst that follows."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    await _settle()
    fake.say(" Checking the current conditions.", start_ms=2000, end_ms=2800)
    fake.say(" It's", start_ms=4200, end_ms=4400)  # 1.4 s after the onset
    fake.push(0.001, count=8)  # the burst closes at 1.1 s, short of it
    await _settle()
    assert len(generations) == 1
    assert len(session._fragments) == 1
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1))[1] == (
        " Checking the current conditions."
    )

    fake.push(0.3, count=3)
    await _settle()
    fake.say(" 62 degrees.", start_ms=4400, end_ms=5000)
    fake.push(0.3, count=3)
    fake.push(0.001, count=8)
    await _settle()
    assert len(generations) == 2
    assert (await asyncio.wait_for(_read(generations[1]), timeout=1))[1] == " It's 62 degrees."


async def test_a_fragment_that_lags_its_audio_anchors_at_the_bursts_onset(duplex) -> None:
    """Sound first, words later: the words still describe the onset, not the moment they arrived."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=5)
    await _settle()
    fake.say("Hello there.", start_ms=9000, end_ms=9400)
    fake.push(0.3, count=1)
    fake.audio_ch.close()
    await _settle()

    (first,) = await _chunks(generations[0])
    assert isinstance(first, TimedString)
    assert first.start_time == pytest.approx(0.0)


async def test_a_fragment_is_attached_when_the_audio_reaches_it(duplex) -> None:
    """The model's words run ahead of its voice; a span the sound has not reached yet waits, even
    across a pause in the transcript, and joins this burst if the gate rides the pause out."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)  # the audio clock reads 2000 when the burst opens
    fake.push(0.3, count=3)
    await _settle()
    fake.say(" Yep, I've got order A1042", start_ms=5000, end_ms=5600)
    fake.say(" on file.", start_ms=6500, end_ms=6900)  # a 900 ms pause in the model's own words
    fake.push(0.3, count=1)
    await _settle()
    assert len(session._fragments) == 1  # due 1.5 s after the onset, and the audio is at 0.4 s

    fake.push(0.3, count=2)  # the pause the gate rides out, then the sound resumes
    fake.push(0.001, count=4)
    fake.push(0.3, count=3)
    await _settle()
    assert not session._fragments
    fake.push(0.001, count=8)
    await _settle()

    assert len(generations) == 1
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1))[1] == (
        " Yep, I've got order A1042 on file."
    )


async def test_transcript_no_audio_ever_claims_is_emitted_rather_than_lost(duplex, caplog) -> None:
    """Losing transcript is worse than an odd chat item; the model's silence is the clock."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    await _settle()
    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        fake.say("Lost words.", start_ms=2000, end_ms=2400)
        fake.push(0.001, count=29)
        await _settle()
        assert generations == []  # 2.9 s of silence is not yet the timeout
        fake.push(0.001, count=1)
        await _settle()

    assert [r.message for r in caplog.records if r.levelno >= logging.ERROR] == [
        "duplex transcript outlived the audio it describes"
    ]
    assert len(generations) == 1
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1)) == (0, "Lost words.")
    assert not session._fragments


async def test_a_function_call_joins_the_speech_in_flight(duplex) -> None:
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    await _settle()
    call = llm.FunctionCall(call_id="c1", name="lookup", arguments="{}")
    fake.emit("function_call", call)
    await _settle()

    assert len(generations) == 1
    assert session._burst is not None  # the speech carries on, the call rides in it

    fake.push(0.001, count=8)
    await _settle()
    frames, _ = await asyncio.wait_for(_read(generations[0]), timeout=1)
    assert frames >= 3
    assert [c async for c in generations[0].function_stream] == [call]


async def test_a_function_call_alone_is_a_generation_over_at_once(duplex) -> None:
    """Nothing speech-shaped exists to end it, so it never waits on the gate."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    await _settle()
    call = llm.FunctionCall(call_id="c1", name="lookup", arguments="{}")
    fake.emit("function_call", call)

    assert session._burst is None
    assert len(generations) == 1
    ev = generations[0]
    assert [m async for m in ev.message_stream] == []
    assert [c async for c in ev.function_stream] == [call]

    # the idle stream keeps flowing and opens nothing of its own
    fake.push(0.001, count=5)
    await _settle()
    assert len(generations) == 1


async def test_reconnect_releases_the_burst_and_the_words_waiting_on_it(duplex) -> None:
    """A dropped connection never delivers the rest of what it was carrying."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    await _settle()
    fake.say("Half a", start_ms=2000, end_ms=2200)
    fake.say(" sentence", start_ms=4000, end_ms=4400)  # held for the next burst
    fake.push(0.3, count=1)
    await _settle()
    assert session._fragments

    fake.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
    await _settle()

    assert session._burst is None and not session._fragments
    # the gate closed with the burst, so the model's silence opens no generation of its own
    fake.push(0.001, count=3)
    await _settle()
    assert len(generations) == 1
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1)) == (4, "Half a")


async def test_the_callers_events_pass_through_and_leave_the_burst_alone(duplex) -> None:
    """The plugin detects the caller's turns; the adapter forwards them and never lets them cut
    the model's speech, since a full-duplex model speaks over the caller by design."""
    fake, session, generations = duplex
    events: list[str] = []
    for name in (
        "input_speech_started",
        "input_audio_transcription_completed",
        "input_speech_stopped",
    ):
        session.on(name, lambda ev, name=name: events.append(name))
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    await _settle()
    fake.heard("u1", "Wait")
    fake.push(0.3, count=3)
    await _settle()

    assert events == [
        "input_speech_started",
        "input_audio_transcription_completed",
        "input_speech_stopped",
    ]
    assert len(generations) == 1
    assert session._burst is not None


# the chat context


async def test_the_adapter_keeps_the_context_under_the_ids_the_framework_uses(duplex) -> None:
    """The framework names the user message after the transcription's item and the assistant
    message after the generation, so the adapter records both under those ids."""
    fake, session, generations = duplex
    fake.heard("u1", "What is the weather?")
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    await _settle()
    fake.say("Let me check.", start_ms=2000, end_ms=2400)
    fake.push(0.3, count=1)
    call = llm.FunctionCall(call_id="c1", name="lookup", arguments="{}")
    fake.emit("function_call", call)
    fake.push(0.001, count=8)
    await _settle()

    items = session.chat_ctx.items
    assert [type(i).__name__ for i in items] == ["ChatMessage", "ChatMessage", "FunctionCall"]
    user, assistant, recorded_call = items
    assert isinstance(user, llm.ChatMessage) and isinstance(assistant, llm.ChatMessage)
    assert (user.id, user.role, user.text_content) == ("u1", "user", "What is the weather?")
    assert user.transcript_confidence is not None
    assert (assistant.id, assistant.text_content) == (generations[0].response_id, "Let me check.")
    assert recorded_call is call


async def test_only_what_is_new_reaches_the_model(duplex, caplog) -> None:
    """A context update is diffed against the adapter's own record: the model's speech, which the
    framework hands back under the same ids, is never re-sent, and an edit is logged."""
    fake, session, _generations = duplex
    fake.heard("u1", "Hello")
    await _settle()

    chat_ctx = session.chat_ctx
    chat_ctx.add_message(role="user", content="typed by the app", id="typed_1")
    chat_ctx.items.append(llm.FunctionCallOutput(call_id="c1", output="rainy", is_error=False))
    await session.update_chat_ctx(chat_ctx)
    assert [i.id for i in fake.appended] == ["typed_1", chat_ctx.items[-1].id]

    chat_ctx = session.chat_ctx
    chat_ctx.remove("u1")
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await session.update_chat_ctx(chat_ctx)
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == [
        "duplex context is append-only; the model keeps what it has been told"
    ]
    assert len(fake.appended) == 2


async def test_startup_history_seeds_both_the_adapter_and_the_model(duplex) -> None:
    fake, session, _generations = duplex
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="a prior turn", id="m1")
    await session._update_session(chat_ctx=chat_ctx)

    assert [i.id for i in fake.appended] == ["m1"]
    assert [i.id for i in session.chat_ctx.items] == ["m1"]


# configuration and replies


async def test_configuration_is_handed_over_as_one_unit(duplex) -> None:
    """The plugin must not have to infer that the last individual update means "all of it"."""
    fake, session, _generations = duplex
    await session._update_session(instructions="be brief", tools=[])

    assert len(fake.config_batches) == 1
    instructions, _chat_ctx, tools = fake.config_batches[0]
    assert instructions == "be brief"
    assert tools == []


async def test_a_session_is_configured_once_the_adapter_hands_over_the_configuration(
    duplex,
) -> None:
    """A model whose configuration is immutable once started waits on this before connecting."""
    fake, session, _generations = duplex
    assert not fake._configured.is_set()
    await session._update_session(instructions="be brief")
    assert fake._configured.is_set()


async def test_a_half_applied_configuration_never_reaches_the_model(duplex) -> None:
    """An immutable model gets one chance at its configuration, so a partial one must not ship."""
    fake, session, _generations = duplex
    fake.fail_instructions = True
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="a prior turn", id="m1")

    with pytest.raises(llm.RealtimeError):
        await session._update_session(instructions="be brief", chat_ctx=chat_ctx, tools=[])

    # nothing past the step that failed is applied, and the session that cannot carry the
    # configuration is closed: the framework never closes a session whose start failed
    assert fake.appended == []
    assert fake.audio_ch.closed
    assert session._segment_atask.done()


async def test_closing_before_the_configuration_lands_releases_the_model(duplex) -> None:
    """A model waiting on the configuration before it connects must still be able to close."""
    fake, session, _generations = duplex
    assert not fake._configured.is_set()

    await session.aclose()

    assert fake._configured.is_set()


async def test_generate_reply_is_rejected_by_a_model_that_cannot_be_asked(duplex) -> None:
    """Whether the client may prompt a duplex model is the model's call, not the adapter's."""
    _fake, session, _generations = duplex
    with pytest.raises(llm.RealtimeError):
        await session.generate_reply()


def _askable() -> tuple[_FakeDuplexSession, _DuplexRealtimeSession]:
    """A session whose model can be asked to speak."""
    model = _FakeDuplexModel()
    model.askable = True
    session = llm.DuplexRealtimeAdapter(model).session()
    assert isinstance(session, _DuplexRealtimeSession)
    fake = model.session_obj
    assert fake is not None
    return fake, session


async def test_generate_reply_reaches_a_model_that_supports_it() -> None:
    fake, session = _askable()
    session.generate_reply(instructions="say hi")
    assert fake.replies_requested == ["say hi"]
    await session.aclose()


async def test_a_requested_reply_is_the_speech_that_follows_it() -> None:
    """The model answers on the same stream as everything else, so a burst is all there is."""
    fake, session = _askable()
    fut = session.generate_reply()
    fake.push(0.001, count=20)  # let the gate learn the model's floor
    fake.push(0.5, count=3)
    await _settle()

    generation = await asyncio.wait_for(fut, 1)
    assert generation.user_initiated  # or the framework schedules it a second time, on its own
    await session.aclose()


async def test_a_lone_function_call_is_not_the_reply_the_speech_after_it_is() -> None:
    """The model may delegate before it speaks; the handle must resolve on the speech, since the
    framework treats the resolved generation as the whole reply."""
    fake, session = _askable()
    fut = session.generate_reply()
    fake.push(0.001, count=20)
    await _settle()
    call = llm.FunctionCall(call_id="c1", name="lookup", arguments="{}")
    fake.emit("function_call", call)
    assert not fut.done()

    fake.push(0.5, count=3)
    fake.push(0.001, count=8)
    await _settle()
    generation = await asyncio.wait_for(fut, 1)
    assert generation.user_initiated
    assert [m.message_id async for m in generation.message_stream] == [generation.response_id]
    await session.aclose()


async def test_a_superseded_or_abandoned_ask_fails_rather_than_cancels() -> None:
    """The framework's reply task handles RealtimeError; a cancelled future would end it."""
    fake, session = _askable()
    first = session.generate_reply()
    second = session.generate_reply()
    with pytest.raises(llm.RealtimeError, match="superseded"):
        await first

    fake.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
    with pytest.raises(llm.RealtimeError, match="reconnected"):
        await second

    third = session.generate_reply()
    await session.aclose()
    with pytest.raises(llm.RealtimeError, match="closed"):
        await third


async def test_a_reply_the_model_never_gives_does_not_strand_the_caller(monkeypatch) -> None:
    """Asking a duplex model is a request; it stays free to say nothing at all."""
    monkeypatch.setattr("livekit.agents.llm.duplex_adapter._REPLY_TIMEOUT", 0.05)
    fake, session = _askable()
    fut = session.generate_reply()
    fake.push(0.001, count=20)  # silence: the model declined

    with pytest.raises(llm.RealtimeError):
        await asyncio.wait_for(fut, 1)
    await session.aclose()


async def test_a_duplex_model_is_wrapped_when_handed_to_a_session() -> None:
    """Callers pass the model itself; the adapter is the framework's business, not theirs."""
    from livekit.agents import Agent, AgentSession

    model = _FakeDuplexModel()
    session = AgentSession(llm=model)
    assert isinstance(session.llm, llm.DuplexRealtimeAdapter)
    assert session.llm.duplex_model is model

    agent = Agent(instructions="", llm=model)
    assert isinstance(agent.llm, llm.DuplexRealtimeAdapter)


async def test_duplex_session_reaches_the_plugin_past_the_adapter() -> None:
    """Provider-specific APIs live on the plugin's session, so apps must not stop at the adapter."""
    from livekit.agents import Agent, AgentSession

    model = _FakeDuplexModel()
    agent = Agent(instructions="")
    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        await session.start(agent)
        assert agent.duplex_session is model.session_obj


async def test_duplex_session_raises_for_a_model_that_is_not_duplex() -> None:
    from livekit.agents import Agent, AgentSession

    from .fake_realtime import FakeRealtimeModel

    agent = Agent(instructions="")
    async with AgentSession(llm=FakeRealtimeModel(), aec_warmup_duration=None) as session:
        await session.start(agent)
        with pytest.raises(RuntimeError, match="not running a DuplexModel"):
            _ = agent.duplex_session


async def test_a_failed_audio_stream_reports_an_unrecoverable_error(duplex) -> None:
    """The stream cannot be resumed, so a silent exit would leave the session permanently deaf."""

    class _Boom(Exception):
        pass

    class _FailingGate:
        def update(self, frame: rtc.AudioFrame) -> bool:
            raise _Boom

        def deactivate(self) -> None:
            pass

    fake, session, _generations = duplex
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    session._gate = _FailingGate()

    fake.audio_ch.send_nowait(llm.DuplexAudioFrame(frame=_frame(0.5)))
    for _ in range(20):
        await asyncio.sleep(0)

    assert [e.recoverable for e in errors] == [False]
    assert isinstance(errors[0].error, _Boom)
    assert errors[0].label == fake.duplex_model.label
