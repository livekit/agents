"""
Tests for the duplex adapter: a full-duplex model emits audio continuously, so the framework gates
it against the model's own noise floor and cuts it into turns. Output the model never transcribes
still plays — it simply produces no chat item.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import llm
from livekit.agents.llm.duplex_adapter import (
    _STALLED_TRANSCRIPT_TIMEOUT,
    AdaptiveNoiseGate,
    _DuplexRealtimeSession,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import aio

pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000
FRAME_MS = 100
# the adapter's liveness bound only releases a turn nothing else can close; tests that are not
# about it shrink it so the file does not spend its runtime asleep
STALLED = 0.1


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

    def session(self) -> _FakeDuplexSession:
        self.session_obj = _FakeDuplexSession(self)
        return self.session_obj

    async def aclose(self) -> None:
        pass


class _FakeDuplexSession(llm.DuplexSession):
    def __init__(self, model: _FakeDuplexModel) -> None:
        super().__init__(model)
        self.audio_ch = aio.Chan[llm.DuplexAudioFrame]()
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext([])
        self.model_ms = 0
        self.config_batches: list[tuple[object, object, object]] = []
        self.replies_requested: list[object] = []
        # the turn a protocol names as answering an ask, where it names one at all
        self.answering_turn: str | None = None

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
    ) -> asyncio.Future[str | None]:
        if not self.capabilities.manual_response_creation:
            return super()._generate_reply(
                instructions=instructions, tool_choice=tool_choice, tools=tools
            )
        self.replies_requested.append(instructions)
        fut: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        fut.set_result(self.answering_turn)
        return fut

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

    def push(self, level: float, *, count: int = 1, turn_id: str | None = None) -> None:
        for _ in range(count):
            self.audio_ch.send_nowait(
                llm.DuplexAudioFrame(frame=_frame(level), turn_id=turn_id, start_ms=self.model_ms)
            )
            self.model_ms += FRAME_MS


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


@pytest.fixture
async def duplex(request) -> tuple[_FakeDuplexSession, _DuplexRealtimeSession, list]:
    model = _FakeDuplexModel()
    adapter = llm.DuplexRealtimeAdapter(
        model, stalled_transcript_timeout=getattr(request, "param", STALLED)
    )
    session = adapter.session()
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
    """A backchannel: the model vocalises, nothing labels it, it still reaches the room."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3)
    fake.push(0.001, count=5)
    await _settle()

    assert len(generations) == 1
    frames, text = await _read(generations[0])
    assert frames >= 3
    assert text == ""


async def test_tagged_frames_split_on_turn_id(duplex) -> None:
    fake, _session, generations = duplex
    fake.push(0.3, count=3, turn_id="turn_a")
    await _settle()
    fake.push(0.3, count=3, turn_id="turn_b")
    await _settle()

    assert len(generations) == 2
    assert (await _read(generations[0]))[0] == 3


async def test_tagged_audio_is_forwarded_even_when_quiet(duplex) -> None:
    """A pause inside a turn is the model's pacing, not a boundary."""
    fake, _session, generations = duplex
    fake.push(0.3, count=2, turn_id="turn_a")
    fake.push(0.0, count=5, turn_id="turn_a")
    fake.push(0.3, count=2, turn_id="turn_a")
    await _settle()

    assert len(generations) == 1
    fake.audio_ch.close()
    await _settle()
    assert (await _read(generations[0]))[0] == 9


async def test_tail_after_turn_ended_stays_in_the_same_generation(duplex) -> None:
    """turn.done can lag the last audio, so the gate decides where the sound stopped."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)  # let the gate learn the model's floor
    fake.push(0.3, count=3, turn_id="turn_a")
    await _settle()
    fake.emit("turn_ended", llm.DuplexTurnEndedEvent(turn_id="turn_a"))
    fake.push(0.3, count=2)  # untagged tail, still audible
    await _settle()
    fake.push(0.001, count=5)  # now genuinely quiet
    await _settle()

    assert len(generations) == 1
    # 3 tagged + 2 tail + the four quiet frames that fall inside the 500 ms hangover
    assert (await asyncio.wait_for(_read(generations[0]), timeout=2))[0] == 9


async def test_turn_ended_while_audio_is_still_flowing_does_not_cut_it_short(duplex) -> None:
    """The boundary lags the sound, so the model calling a turn over never truncates playout."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    await _settle()
    fake.emit("turn_ended", llm.DuplexTurnEndedEvent(turn_id="turn_a"))
    await _settle()
    assert session._burst is not None

    fake.push(0.3, count=4, turn_id="turn_a")  # the model is audibly still speaking
    await _settle()
    assert len(generations) == 1
    fake.audio_ch.close()
    await _settle()
    assert (await _read(generations[0]))[0] == 7


@pytest.mark.parametrize("duplex", [_STALLED_TRANSCRIPT_TIMEOUT], indirect=True)
async def test_turn_ended_releases_a_burst_whose_transcript_never_caught_up(duplex) -> None:
    """Once the sound has stopped, the model calling the turn over settles it on its own."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text="Half a", turn_id="turn_a", start_ms=2000, end_ms=2200),
    )
    fake.push(0.001, count=5)
    await _settle()
    assert session._close_handle is not None  # the transcript is short of the audio

    fake.emit("turn_ended", llm.DuplexTurnEndedEvent(turn_id="turn_a"))
    await _settle()
    assert session._burst is None
    assert await asyncio.wait_for(_read(generations[0]), timeout=0.5) == (7, "Half a")


@pytest.mark.parametrize("duplex", [_STALLED_TRANSCRIPT_TIMEOUT], indirect=True)
async def test_a_transcript_that_catches_up_closes_the_burst_at_once(duplex) -> None:
    """A model that reports spans needs no timeout: audio it has transcribed is a finished turn."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    fake.push(0.001, count=5)  # the gate closes on the fifth, so the fourth ends the audio
    await _settle()
    assert session._close_handle is not None

    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(
            text="All done.", turn_id="turn_a", start_ms=2000, end_ms=fake.model_ms - FRAME_MS
        ),
    )
    await _settle()

    assert session._burst is None
    assert await asyncio.wait_for(_read(generations[0]), timeout=0.5) == (7, "All done.")


async def test_turn_started_adopts_a_burst_already_open(duplex) -> None:
    """Speech onset arrives before the turn that labels it; it must not become a second turn."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=2)
    await _settle()
    fake.emit("turn_started", llm.DuplexTurnStartedEvent(turn_id="turn_a"))
    fake.push(0.3, count=2, turn_id="turn_a")
    await _settle()

    assert len(generations) == 1


async def test_turn_started_for_another_turn_closes_the_burst(duplex) -> None:
    """When a model announces turns but never tags frames, the next turn is the only boundary."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=2)  # the first turn's onset, untagged
    await _settle()
    fake.emit("turn_started", llm.DuplexTurnStartedEvent(turn_id="turn_a"))
    fake.push(0.3, count=2)
    await _settle()
    assert len(generations) == 1

    fake.emit("turn_started", llm.DuplexTurnStartedEvent(turn_id="turn_b"))
    await _settle()
    assert session._burst is None

    fake.push(0.3, count=2)
    await _settle()
    assert len(generations) == 2
    assert (await _read(generations[0]))[0] == 4


async def test_transcript_is_timed_against_forwarded_audio(duplex) -> None:
    fake, _session, generations = duplex
    fake.push(0.3, count=3, turn_id="turn_a")
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text="hello", turn_id="turn_a", start_ms=0, end_ms=200),
    )
    await _settle()
    fake.audio_ch.close()
    await _settle()

    assert len(generations) == 1
    chunks: list[str] = []
    async for msg in generations[0].message_stream:
        async for _ in msg.audio_stream:
            pass
        async for chunk in msg.text_stream:
            chunks.append(chunk)

    assert "".join(chunks) == "hello"
    timed = chunks[0]
    assert isinstance(timed, TimedString)
    assert timed.start_time == pytest.approx(0.0)
    assert timed.end_time == pytest.approx(0.2)


async def test_function_call_opens_a_generation_when_none_is_open(duplex) -> None:
    fake, _session, generations = duplex
    call = llm.FunctionCall(call_id="c1", name="lookup", arguments="{}")
    fake.emit("function_call", call)
    await _settle()

    assert len(generations) == 1
    fake.audio_ch.close()
    await _settle()
    assert [c async for c in generations[0].function_stream] == [call]


async def test_transcript_fragments_never_split_a_burst(duplex) -> None:
    """Only audio defines boundaries.

    A model can label transcript fragments before it announces the turn they belong to, so
    consecutive fragments may carry different ids for one stretch of speech.
    """
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=2)
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text="Sure,", turn_id="item_1", start_ms=2000, end_ms=2200),
    )
    await _settle()
    fake.push(0.3, count=2)
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text=" I", turn_id="item_2", start_ms=2200, end_ms=2400),
    )
    await _settle()

    assert len(generations) == 1
    fake.audio_ch.close()
    await _settle()
    assert (await _read(generations[0]))[1] == "Sure, I"


async def test_a_trailing_transcript_lands_in_the_burst_the_gate_just_closed(duplex) -> None:
    """The model transcribes what it has just said, so the last fragment outlives the sound."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    speech_start = fake.model_ms
    fake.push(0.3, count=3, turn_id="turn_a")
    speech_end = fake.model_ms
    fake.push(0.001, count=5)  # the gate closes here, on the fifth quiet frame
    await _settle()
    assert len(generations) == 1

    # the watermark tracks the audio that carried sound, so the spans have to describe the speech
    mid = (speech_start + speech_end) // 2
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(
            text=" Alright", turn_id="turn_a", start_ms=speech_start, end_ms=mid
        ),
    )
    await asyncio.sleep(STALLED / 2)
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text=".", turn_id="turn_a", start_ms=mid, end_ms=speech_end),
    )
    await _settle()

    assert len(generations) == 1
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1))[1] == " Alright."


async def test_the_next_turn_does_not_inherit_the_previous_trailing_transcript(duplex) -> None:
    """The punctuation a turn ends on must not prefix the chat item of the turn after it."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    fake.push(0.001, count=5)
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(
            text=" Alright.", turn_id="turn_a", start_ms=2000, end_ms=fake.model_ms - FRAME_MS
        ),
    )
    await _settle()
    assert len(generations) == 1

    start_ms = fake.model_ms
    fake.push(0.3, count=3, turn_id="turn_b")
    await _settle()  # the transcript lags the speech it describes
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(
            text=" Once there", turn_id="turn_b", start_ms=start_ms, end_ms=fake.model_ms
        ),
    )
    await _settle()
    fake.audio_ch.close()
    await _settle()

    assert len(generations) == 2
    assert (await asyncio.wait_for(_read(generations[0]), timeout=2))[1] == " Alright."
    assert (await asyncio.wait_for(_read(generations[1]), timeout=2))[1] == " Once there"


async def test_a_turn_that_stops_being_transcribed_is_released_by_the_liveness_bound(
    duplex,
) -> None:
    """Nothing else can ever close a turn the model abandoned mid-transcript, so a bound must."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    fake.push(0.001, count=5)
    await _settle()
    assert len(generations) == 1
    assert session._close_handle is not None

    # 3 tagged frames plus the four quiet frames inside the gate's hangover
    assert await asyncio.wait_for(_read(generations[0]), timeout=2) == (7, "")
    assert session._burst is None


async def test_the_liveness_bound_is_armed_once_and_cleared_by_audio(duplex) -> None:
    """It bounds a stalled turn rather than pacing it, so a fragment must not push it out."""
    fake, session, _generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    fake.push(0.001, count=5)
    await _settle()
    assert session._close_handle is not None
    armed_at = session._close_handle.when()

    await asyncio.sleep(0.01)
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text="Half a", turn_id="turn_a", start_ms=2000, end_ms=2200),
    )
    await _settle()
    assert session._close_handle is not None
    assert session._close_handle.when() == armed_at

    fake.push(0.3, count=1, turn_id="turn_a")  # the turn resumes, nothing stays pending
    await _settle()
    assert session._close_handle is None


async def test_a_turn_announced_while_the_liveness_bound_runs_keeps_its_burst(duplex) -> None:
    """The announcement is the strongest sign yet that the burst is the model still speaking."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=2)  # onset, nothing labels it yet
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text=" Once", turn_id="item_1", start_ms=2000, end_ms=2100),
    )
    fake.push(0.001, count=5)
    await _settle()
    assert session._close_handle is not None

    fake.emit("turn_started", llm.DuplexTurnStartedEvent(turn_id="turn_a"))
    await _settle()
    assert session._close_handle is None
    assert len(generations) == 1


async def test_the_first_fragment_of_a_session_starts_the_turn_its_audio_completes(
    duplex, caplog
) -> None:
    """Nothing has played yet, so a fragment before any audio can only be a turn beginning."""
    fake, _session, generations = duplex
    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        fake.emit(
            "transcript_delta",
            llm.DuplexTranscriptDelta(text=" Sure.", start_ms=0, end_ms=200),
        )
        await _settle()
        fake.push(0.3, count=3, turn_id="turn_a")
        await _settle()

    assert [r.message for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert len(generations) == 1
    fake.audio_ch.close()
    await _settle()
    assert (await _read(generations[0])) == (3, " Sure.")


async def test_a_fragment_that_leads_its_audio_is_adopted_by_the_turn_it_describes(
    duplex, caplog
) -> None:
    """A transcript can beat its own audio to the wire, and that is an ordinary turn opening."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    fake.push(0.001, count=5)
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(
            text="All done.", turn_id="turn_a", start_ms=2000, end_ms=fake.model_ms - FRAME_MS
        ),
    )
    await _settle()
    assert session._burst is None

    start_ms = fake.model_ms  # the next turn reaches past everything already played
    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        fake.emit(
            "transcript_delta",
            llm.DuplexTranscriptDelta(text=" Sure.", start_ms=start_ms, end_ms=start_ms + 200),
        )
        await _settle()
        fake.push(0.3, count=3, turn_id="turn_b")
        await _settle()

    assert [r.message for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert len(generations) == 2
    fake.audio_ch.close()
    await _settle()
    assert (await _read(generations[1])) == (3, " Sure.")


async def test_a_fragment_that_outlived_its_audio_is_reported_and_never_adopted(
    duplex, caplog
) -> None:
    """Losing transcript is worse than an odd chat item, but it must not prefix the next turn."""
    fake, session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    fake.push(0.001, count=5)
    await _settle()
    audio_end = fake.model_ms - FRAME_MS
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(
            text="All done", turn_id="turn_a", start_ms=2000, end_ms=audio_end
        ),
    )
    await _settle()
    assert session._burst is None

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        # a fragment reaching no further than the audio already played describes speech gone by
        fake.emit(
            "transcript_delta",
            llm.DuplexTranscriptDelta(text=".", turn_id="turn_a", start_ms=2600, end_ms=audio_end),
        )
        await _settle()

    assert [r.message for r in caplog.records if r.levelno >= logging.ERROR] == [
        "duplex transcript outlived the audio it describes"
    ]
    assert len(generations) == 2

    fake.push(0.3, count=3, turn_id="turn_b")
    await _settle()
    fake.audio_ch.close()
    await _settle()

    assert len(generations) == 3
    assert (await asyncio.wait_for(_read(generations[1]), timeout=1)) == (0, ".")
    assert (await asyncio.wait_for(_read(generations[2]), timeout=1)) == (3, "")


@pytest.mark.parametrize("duplex", [_STALLED_TRANSCRIPT_TIMEOUT], indirect=True)
async def test_a_turn_announced_late_keeps_the_words_spoken_before_it(duplex) -> None:
    """A turn is announced ~450 ms after its first transcript; its onset is still the same turn."""
    fake, _session, generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=2)  # speech onset, before anything labels it
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(text=" Once upon", turn_id="item_1", start_ms=2000, end_ms=2100),
    )
    fake.push(0.001, count=5)  # the gate closes while the transcript is still short of the audio
    await _settle()
    assert len(generations) == 1

    await asyncio.sleep(0.45)
    fake.emit("turn_started", llm.DuplexTurnStartedEvent(turn_id="turn_a"))
    start_ms = fake.model_ms
    fake.push(0.3, count=2, turn_id="turn_a")
    await _settle()
    fake.emit(
        "transcript_delta",
        llm.DuplexTranscriptDelta(
            text=" a time", turn_id="turn_a", start_ms=start_ms, end_ms=fake.model_ms
        ),
    )
    await _settle()

    assert len(generations) == 1
    fake.audio_ch.close()
    await _settle()
    assert (await asyncio.wait_for(_read(generations[0]), timeout=1))[1] == " Once upon a time"


async def test_reconnect_releases_a_burst_held_by_an_open_turn(duplex) -> None:
    """A dropped connection never delivers turn.done, which would hold the burst open forever."""
    fake, _session, generations = duplex
    fake.emit("turn_started", llm.DuplexTurnStartedEvent(turn_id="turn_a"))
    fake.push(0.3, count=3, turn_id="turn_a")
    await _settle()
    assert len(generations) == 1

    fake.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
    await _settle()

    frames, _text = await asyncio.wait_for(_read(generations[0]), timeout=1)
    assert frames == 3


async def test_reconnect_and_shutdown_leave_no_close_pending(duplex) -> None:
    """A timer surviving either path would fire against a session that has moved on."""
    fake, session, _generations = duplex
    fake.push(0.001, count=20)
    fake.push(0.3, count=3, turn_id="turn_a")
    fake.push(0.001, count=5)
    await _settle()
    assert session._close_handle is not None

    fake.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
    await _settle()
    assert session._close_handle is None

    fake.push(0.3, count=3, turn_id="turn_b")
    fake.push(0.001, count=5)
    await _settle()
    assert session._close_handle is not None

    await session.aclose()
    assert session._close_handle is None


async def test_configuration_is_handed_over_as_one_unit(duplex) -> None:
    """The plugin must not have to infer that the last individual update means "all of it"."""
    fake, session, _generations = duplex
    await session._update_session(instructions="be brief", tools=[])

    assert len(fake.config_batches) == 1
    instructions, _chat_ctx, tools = fake.config_batches[0]
    assert instructions == "be brief"
    assert tools == []


async def test_generate_reply_is_rejected_by_a_model_that_cannot_be_asked(duplex) -> None:
    """Whether the client may prompt a duplex model is the model's call, not the adapter's."""
    _fake, session, _generations = duplex
    assert not session.realtime_model.capabilities.manual_response_creation
    with pytest.raises(llm.RealtimeError):
        await session.generate_reply()


async def test_generate_reply_reaches_a_model_that_supports_it() -> None:
    model = _FakeDuplexModel()
    model._capabilities.manual_response_creation = True
    session = llm.DuplexRealtimeAdapter(model).session()
    fake = model.session_obj
    assert fake is not None

    assert session.realtime_model.capabilities.manual_response_creation
    session.generate_reply(instructions="say hi")
    assert fake.replies_requested == ["say hi"]
    await session.aclose()


def _askable(
    answering_turn: str | None = None,
) -> tuple[_FakeDuplexSession, _DuplexRealtimeSession]:
    """A session whose model can be asked to speak, naming its answering turn or not."""
    model = _FakeDuplexModel()
    model._capabilities.manual_response_creation = True
    session = llm.DuplexRealtimeAdapter(model, stalled_transcript_timeout=STALLED).session()
    assert isinstance(session, _DuplexRealtimeSession)
    fake = model.session_obj
    assert fake is not None
    fake.answering_turn = answering_turn
    return fake, session


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


async def test_a_named_turn_is_not_claimed_by_a_burst_of_another() -> None:
    """A model asked mid-answer must not hand back the answer it was already giving."""
    fake, session = _askable(answering_turn="turn_reply")
    fut = session.generate_reply()
    fake.push(0.001, count=20)
    await _settle()

    fake.push(0.5, count=3, turn_id="turn_other")
    await _settle()
    assert not fut.done()

    fake.push(0.5, count=3, turn_id="turn_reply")
    await _settle()
    assert (await asyncio.wait_for(fut, 1)).user_initiated
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
