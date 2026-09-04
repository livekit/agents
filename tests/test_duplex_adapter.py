"""
Tests for the duplex adapter: a full-duplex model emits audio continuously and reports only
fragments of its transcript, so the framework gates the audio against the model's own noise floor,
cuts it into generations and attaches the words to the sound that carries them. Output the model
never transcribes still plays, it simply produces no chat item.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import llm
from livekit.agents.llm.duplex_adapter import AdaptiveNoiseGate, _DuplexRealtimeSession
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

    @property
    def model(self) -> str:
        return "fake-duplex"

    @property
    def provider(self) -> str:
        return "fake"

    def session(self, *, wait_for_config: bool = False) -> _FakeDuplexSession:
        self.session_obj = _FakeDuplexSession(self, wait_for_config=wait_for_config)
        return self.session_obj

    async def aclose(self) -> None:
        pass


class _FakeDuplexSession(llm.DuplexSession):
    def __init__(self, model: _FakeDuplexModel, *, wait_for_config: bool = False) -> None:
        super().__init__(model, wait_for_config=wait_for_config)
        self.audio_ch = aio.Chan[llm.DuplexAudioFrame]()
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext([])
        self.config_batches: list[tuple[object, object, object]] = []
        self.replies_requested: list[object] = []

    @property
    def audio_stream(self) -> aio.Chan[llm.DuplexAudioFrame]:
        return self.audio_ch

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    async def _update_instructions(self, instructions: str) -> None:
        pass

    async def _update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        pass

    async def aclose(self) -> None:
        if not self.audio_ch.closed:
            self.audio_ch.close()

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        if not self.capabilities.manual_response_creation:
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

    # -- test helpers ----------------------------------------------------------------------

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
            llm.DuplexTranscriptDelta(text=text, start_ms=start_ms, end_ms=end_ms),
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


# -- the gate ------------------------------------------------------------------------------


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


def test_gate_holds_through_hangover_then_closes() -> None:
    gate = AdaptiveNoiseGate(hangover=0.25)
    for _ in range(30):
        gate.update(_frame(0.001))
    assert gate.update(_frame(0.3))

    # two 100 ms quiet frames are inside the 250 ms hangover, the third crosses it
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


def test_gate_decides_on_audio_duration_not_on_frame_count() -> None:
    """The same sound streamed at two frame sizes is gated identically."""
    levels = [0.001] * 12 + [0.3] * 6 + [0.001] * 12
    decisions: list[list[bool]] = []
    for frame_ms in (20, FRAME_MS):
        gate = AdaptiveNoiseGate(window=1.0, hangover=0.45)
        states: list[bool] = []
        for level in levels:
            for _ in range(FRAME_MS // frame_ms):
                state = gate.update(_frame(level, duration_ms=frame_ms))
            states.append(state)
        decisions.append(states)

    assert decisions[0] == decisions[1]
    # the run exercises both edges rather than agreeing on a gate that never moved
    assert decisions[0][12] and not decisions[0][-1]


# -- the segmenter -------------------------------------------------------------------------


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
    fake.push(0.001, count=3)  # shorter than the gate's hangover
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
    fake.emit("input_speech_started", llm.InputSpeechStartedEvent())
    fake.emit(
        "input_audio_transcription_completed",
        llm.InputTranscriptionCompleted(item_id="u1", transcript="Wait", is_final=True),
    )
    fake.emit("input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False))
    fake.push(0.3, count=3)
    await _settle()

    assert events == [
        "input_speech_started",
        "input_audio_transcription_completed",
        "input_speech_stopped",
    ]
    assert len(generations) == 1
    assert session._burst is not None


# -- configuration and replies -------------------------------------------------------------


async def test_configuration_is_handed_over_as_one_unit(duplex) -> None:
    """The plugin must not have to infer that the last individual update means "all of it"."""
    fake, session, _generations = duplex
    await session._update_session(instructions="be brief", tools=[])

    assert len(fake.config_batches) == 1
    instructions, _chat_ctx, tools = fake.config_batches[0]
    assert instructions == "be brief"
    assert tools == []


async def test_adapter_promises_the_session_a_configuration() -> None:
    """Only the adapter knows AgentActivity will configure the session it just created."""
    model = _FakeDuplexModel()
    session = llm.DuplexRealtimeAdapter(model).session()
    promised = model.session_obj
    assert promised is not None and not promised._config_delivered.is_set()

    assert model.session()._config_delivered.is_set(), "a session built directly must not wait"
    await session.aclose()


async def test_generate_reply_is_rejected_by_a_model_that_cannot_be_asked(duplex) -> None:
    """Whether the client may prompt a duplex model is the model's call, not the adapter's."""
    _fake, session, _generations = duplex
    assert not session.realtime_model.capabilities.manual_response_creation
    with pytest.raises(llm.RealtimeError):
        await session.generate_reply()


def _askable() -> tuple[_FakeDuplexSession, _DuplexRealtimeSession]:
    """A session whose model can be asked to speak."""
    model = _FakeDuplexModel()
    model._capabilities.manual_response_creation = True
    session = llm.DuplexRealtimeAdapter(model).session()
    assert isinstance(session, _DuplexRealtimeSession)
    fake = model.session_obj
    assert fake is not None
    return fake, session


async def test_generate_reply_reaches_a_model_that_supports_it() -> None:
    fake, session = _askable()
    assert session.realtime_model.capabilities.manual_response_creation
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
