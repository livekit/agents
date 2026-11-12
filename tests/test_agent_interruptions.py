"""Interrupting, pausing and resuming agent speech, and how that interacts with user turns."""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, Mock

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    FlushSentinel,
    ModelSettings,
    RunContext,
    TurnHandlingOptions,
    function_tool,
)
from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FunctionToolCall,
    LLMStream,
    RealtimeError,
    Tool,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.agent_activity import AgentActivity, _PausedSpeechInfo
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
)
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.agents.voice.run_result import RunResult
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.agents.voice.turn import (
    TurnDetectionEvent,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_session import FakeActions, create_session, run_session
from .fake_vad import FakeVAD
from .test_agent_session import MyAgent, _close_test_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _end_of_turn(transcript: str = "", *, skip_reply: bool = False) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=skip_reply,
        new_transcript=transcript,
        transcript_confidence=1.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
    )


# --- SpeechHandle ---------------------------------------------------------------------


@pytest.mark.parametrize("settle", ["interrupted", "done"])
async def test_a_settled_protected_handle_ignores_interrupt(settle: str) -> None:
    handle = SpeechHandle.create(allow_interruptions=False)
    if settle == "interrupted":
        handle.interrupt(force=True)
    else:
        handle._mark_done()

    assert handle.interrupt() is handle  # must not raise
    assert handle.interrupted is (settle == "interrupted")
    handle._mark_done()


async def test_speech_handle_records_the_error_instead_of_raising() -> None:
    """Awaiting a failed handle never raises; the error is read with exception()."""
    handle = SpeechHandle.create()
    with pytest.raises(asyncio.InvalidStateError):
        handle.exception()

    handle._mark_done(error=RealtimeError("first"))
    # e.g. the task-level done callback marking the handle again
    handle._mark_done()
    handle._mark_done(error=RealtimeError("second"))

    assert await handle is handle
    await handle.wait_for_playout()
    exc = handle.exception()
    assert isinstance(exc, RealtimeError)
    assert str(exc) == "first"


async def test_speech_handle_without_error_has_no_exception() -> None:
    handle = SpeechHandle.create()
    handle._mark_done()

    await handle
    assert handle.exception() is None


async def test_run_result_propagates_speech_handle_error() -> None:
    run_result = RunResult[None](output_type=None)
    handle = SpeechHandle.create()
    run_result._watch_handle(handle)

    handle._mark_done(error=RealtimeError("generate_reply timed out."))

    with pytest.raises(RealtimeError, match="generate_reply timed out"):
        await run_result


# --- interrupt() with queued speeches that disallow interruptions -----------------------


@pytest.fixture
async def activity() -> AsyncIterator[AgentActivity]:
    activity = AgentActivity(MyAgent(), create_session(FakeActions()))
    activity._rt_session = Mock()
    try:
        yield activity
    finally:
        await _close_test_session(activity._session)


def _enqueue(activity: AgentActivity, *speeches: SpeechHandle, priority: int = 0) -> None:
    for speech in speeches:
        heapq.heappush(activity._speech_q, (-priority, time.perf_counter_ns(), speech))


async def test_interrupt_stops_the_queue_at_the_protected_speech(
    activity: AgentActivity, caplog: pytest.LogCaptureFixture
) -> None:
    current = SpeechHandle.create(allow_interruptions=True)
    first = SpeechHandle.create(allow_interruptions=True)
    protected = SpeechHandle.create(allow_interruptions=False)
    behind = SpeechHandle.create(allow_interruptions=True)
    activity._current_speech = current
    _enqueue(activity, first, protected, behind)

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        activity.interrupt()  # must not raise

    assert current.interrupted
    assert first.interrupted
    assert not protected.interrupted
    # no gap: what plays after the protected speech is untouched
    assert not behind.interrupted
    activity._rt_session.interrupt.assert_called_once()  # type: ignore[union-attr]
    assert any("force=True" in record.message for record in caplog.records)


async def test_a_protected_head_shields_the_whole_queue(activity: AgentActivity) -> None:
    speeches = [
        SpeechHandle.create(allow_interruptions=False),
        SpeechHandle.create(allow_interruptions=True),
        SpeechHandle.create(allow_interruptions=False),
    ]
    _enqueue(activity, *speeches)

    activity.interrupt()

    assert not any(s.interrupted for s in speeches)


async def test_a_cancelled_protected_speech_does_not_shield_the_queue(
    activity: AgentActivity,
) -> None:
    protected = SpeechHandle.create(allow_interruptions=False)
    behind = SpeechHandle.create(allow_interruptions=True)
    _enqueue(activity, protected, behind)
    # it will never play, so it cannot leave a gap behind it
    protected.interrupt(force=True)

    activity.interrupt()

    assert behind.interrupted


async def test_the_queue_is_walked_in_playout_order(activity: AgentActivity) -> None:
    low = SpeechHandle.create(allow_interruptions=True)
    urgent_protected = SpeechHandle.create(allow_interruptions=False)
    _enqueue(activity, low, priority=SpeechHandle.SPEECH_PRIORITY_NORMAL)
    _enqueue(activity, urgent_protected, priority=SpeechHandle.SPEECH_PRIORITY_HIGH)

    activity.interrupt()

    # the protected speech plays first, so the walk stops immediately
    assert not urgent_protected.interrupted
    assert not low.interrupted


async def test_a_protected_playing_speech_still_raises(activity: AgentActivity) -> None:
    activity._current_speech = SpeechHandle.create(allow_interruptions=False)

    with pytest.raises(RuntimeError):
        activity.interrupt()


async def test_force_interrupts_the_whole_chain(activity: AgentActivity) -> None:
    current = SpeechHandle.create(allow_interruptions=False)
    queued = SpeechHandle.create(allow_interruptions=False)
    behind = SpeechHandle.create(allow_interruptions=True)
    activity._current_speech = current
    _enqueue(activity, queued, behind)

    activity.interrupt(force=True)

    assert current.interrupted and queued.interrupted and behind.interrupted
    activity._rt_session.interrupt.assert_called_once()  # type: ignore[union-attr]


# --- a tool that disallows interruptions while the parent speech is paused --------------

TRANSFER_MESSAGE = "I'll connect you now."


class TransferAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Transfer callers when requested.")
        self.transfer_started = asyncio.Event()
        self.transfer_completed = asyncio.Event()
        self.paused_parent_state_before_disallow: str | None = None
        self.parent_state_after_disallow: str | None = None
        self.recognition_speaking_after_disallow: bool | None = None

    @function_tool
    async def transfer_to_live_agent(self, context: RunContext) -> None:
        """Transfer the caller to a live agent."""
        self.transfer_started.set()
        activity = context.session._activity
        assert activity is not None and activity._audio_recognition is not None
        paused_speech = activity._paused_speech
        self.paused_parent_state_before_disallow = (
            paused_speech.agent_state if paused_speech is not None else None
        )
        context.disallow_interruptions()
        self.parent_state_after_disallow = context.session.agent_state
        self.recognition_speaking_after_disallow = activity._audio_recognition._agent_speaking
        context.session.input.set_audio_enabled(False)
        await context.session.say(TRANSFER_MESSAGE, allow_interruptions=False)
        self.transfer_completed.set()


async def _run_until_transferred(session: AgentSession, agent: TransferAgent) -> None:
    run_task = asyncio.create_task(run_session(session, agent))
    try:
        await asyncio.wait_for(agent.transfer_started.wait(), timeout=7.0)
        await asyncio.wait_for(agent.transfer_completed.wait(), timeout=8.0)
    finally:
        if not run_task.done():
            await session.aclose()
        await run_task


async def test_uninterruptible_tool_speech_plays_after_paused_parent() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Connect me to an agent.")
    actions.add_llm(
        content="Happy to connect you.",
        tool_calls=[
            FunctionToolCall(name="transfer_to_live_agent", arguments="{}", call_id="transfer-1")
        ],
        ttft=0.1,
        duration=1.0,
    )
    # VAD pauses the parent after its first frame is forwarded but before the tool starts
    actions.add_tts(2.0, ttfb=0.1, duration=0.2)
    actions.add_user_speech(3.1, 5.0, "")
    actions.add_tts(1.0, input=TRANSFER_MESSAGE)

    session = create_session(actions, can_pause_audio=True)
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await _run_until_transferred(session, TransferAgent())

    assert [event.playback_position for event in playback_finished_events] == pytest.approx(
        [2.0, 1.0], abs=0.02
    )


async def test_uninterruptible_tool_restores_paused_parent_speaking_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedTransferAgent(TransferAgent):
        async def llm_node(
            self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
        ) -> AsyncIterable[ChatChunk | str | FlushSentinel]:
            yield "Happy to connect you."
            yield FlushSentinel()
            await asyncio.sleep(3.0)
            yield ChatChunk(
                id="transfer-response",
                delta=ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        FunctionToolCall(
                            name="transfer_to_live_agent", arguments="{}", call_id="transfer-1"
                        )
                    ],
                ),
            )

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Connect me to an agent.")
    # the first segment is playing when VAD pauses the parent, before the tool call arrives
    actions.add_tts(5.0, input="Happy to connect you.", ttfb=0.1, duration=0.2)
    actions.add_user_speech(4.0, 8.0, "")
    actions.add_tts(1.0, input=TRANSFER_MESSAGE)

    session = create_session(actions, can_pause_audio=True)
    agent = DelayedTransferAgent()
    audio_output = session.output.audio
    assert audio_output is not None
    original_resume = audio_output.resume
    restored_state_observed_during_resume = False

    def resume_with_state_assertion() -> None:
        nonlocal restored_state_observed_during_resume
        if agent.transfer_started.is_set() and agent.parent_state_after_disallow is None:
            activity = session._activity
            assert activity is not None and activity._audio_recognition is not None
            assert session.agent_state == "speaking"
            assert activity._audio_recognition._agent_speaking
            restored_state_observed_during_resume = True
        original_resume()

    monkeypatch.setattr(audio_output, "resume", resume_with_state_assertion)

    await _run_until_transferred(session, agent)

    assert agent.paused_parent_state_before_disallow == "speaking"
    assert restored_state_observed_during_resume
    assert agent.parent_state_after_disallow == "speaking"
    assert agent.recognition_speaking_after_disallow is True


# --- the false-interruption resume follows the end-of-turn decision ---------------------
# both are armed from the same VAD END_OF_SPEECH, but the turn commits at
# last_speaking_time + endpointing delay, so a max_delay decision lands after the resume timer

# scaled-down shipped defaults, keeping max_delay - vad_min_silence > timeout
VAD_MIN_SILENCE = 0.05
FALSE_INTERRUPTION_TIMEOUT = 0.3
MAX_DELAY = 0.5


def _recognition(hooks: AgentActivity, last_speaking_time: float) -> AudioRecognition:
    """AudioRecognition wired to drive one real eou bounce against ``hooks``."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._hooks = hooks
    ar._stt = None  # realtime model, no STT
    ar._audio_transcript = ""
    ar._turn_detection_mode = None
    ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)

    stream_mock = MagicMock(spec=_StreamingTurnDetectorStream)
    stream_mock.supports_language = AsyncMock(return_value=True)
    stream_mock.unlikely_threshold = AsyncMock(return_value=0.5)
    stream_mock.backchannel_threshold = AsyncMock(return_value=None)
    stream_mock.aclose = AsyncMock()
    stream_mock.prediction_timeout = 0.5
    ar._turn_detector_stream = stream_mock

    # the detector already answered: mid-utterance, so the bounce waits out max_delay
    fut: asyncio.Future[TurnDetectionEvent] = asyncio.Future()
    fut.set_result(
        TurnDetectionEvent(
            type="eot_prediction",
            end_of_turn_probability=0.1,
            last_speaking_time=last_speaking_time,
        )
    )
    ar._turn_detector_prediction_fut = fut
    ar._turn_detector_flushed = False
    ar._turn_detector_late_prediction_warned = False
    ar._agent_speaking = False
    ar._transcript_buffer = deque()
    ar._transcript_gate_active = False
    ar._interruption_enabled = False
    ar._interruption_ch = None
    ar._vad_base_turn_detection = True
    ar._backchannel_boundary = None
    ar._backchannel_boundary_timer = None
    ar._backchannel_boundary_callback = None
    ar._endpointing = MagicMock(min_delay=0.05, max_delay=MAX_DELAY)
    ar._ensure_user_turn_span = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(is_recording=MagicMock(return_value=False))
    )
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._eou_wait_span = None
    ar._eou_wait_started_at_ns = None
    ar._eou_wait_rearms = 0
    ar._eou_wait_floor_ns = None
    ar._eou_wait_not_committed = 0
    ar._user_turn_resumes = 0
    ar._eou_detection_span = None
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._final_transcript_confidence = []
    ar._stt_request_ids = []
    ar._stt_events = []
    ar._last_speaking_time = last_speaking_time
    ar._last_final_transcript_time = None
    ar._speech_start_time = None
    ar._vad_speech_started = False
    ar._end_of_turn_task = None
    ar._user_turn_committed = False
    ar._vad = None
    ar._last_language = None
    ar._last_emitted_prediction = None
    ar._turn_backchannel_over_agent = False
    ar._overlap_in_current_turn = False
    ar._turn_tracker = MagicMock()
    ar._closing = asyncio.Event()
    # only touched by _aclose
    ar._commit_user_turn_atask = None
    ar._stt_pipeline = None
    ar._stt_consumer_atask = None
    ar._vad_atask = None
    ar._interruption_atask = None
    ar._tasks = set()
    return ar


@pytest.fixture
def rt_session(monkeypatch: pytest.MonkeyPatch) -> AgentSession:
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")
    return AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=False)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad", interruption={"mode": "adaptive"}),
    )


def _paused_activity(session: AgentSession) -> tuple[AgentActivity, MagicMock]:
    activity = AgentActivity(Agent(instructions="test"), session)
    activity._scheduling_paused = False
    session.output.audio = FakeAudioOutput(can_pause=True)

    handle = MagicMock()
    handle.done.return_value = False
    handle.interrupted = False
    handle.allow_interruptions = True
    handle._agent_turn_context = None
    handle._generations = []
    activity._current_speech = handle
    activity._paused_speech = _PausedSpeechInfo(
        handle=handle, agent_state="speaking", timeout=FALSE_INTERRUPTION_TIMEOUT
    )
    return activity, handle


def _swallow_task(coro: Any, **kwargs: object) -> MagicMock:
    # stands in for _create_speech_task: the reply pipeline isn't under test here
    coro.close()
    return MagicMock()


def _record_resumes(session: AgentSession) -> list[float]:
    resumes: list[float] = []
    session.on("agent_false_interruption", lambda _: resumes.append(time.time()))
    return resumes


async def test_resume_waits_for_a_dropped_turn(rt_session: AgentSession) -> None:
    activity, _ = _paused_activity(rt_session)
    resumes = _record_resumes(rt_session)

    t0 = time.time()
    activity.on_end_of_speech(None)
    activity._audio_recognition = _recognition(activity, last_speaking_time=t0 - VAD_MIN_SILENCE)
    # a confirmed backchannel is what makes the turn drop rather than commit
    activity._audio_recognition._turn_backchannel_over_agent = True
    activity._audio_recognition._run_eou_detection(MagicMock(), trigger="vad")

    await asyncio.sleep(MAX_DELAY + 0.3)
    await rt_session.aclose()

    # the turn is dropped at max_delay - vad silence; resuming any earlier is the race
    assert [t - t0 for t in resumes] == [pytest.approx(MAX_DELAY - VAD_MIN_SILENCE, abs=0.1)]
    assert activity._paused_speech is None


async def test_committed_turn_suppresses_the_resume(rt_session: AgentSession) -> None:
    activity, _ = _paused_activity(rt_session)
    # a confirmed interruption: on_end_of_turn commits and the reply task interrupts the pause
    activity._interruption_detected = True
    activity._create_speech_task = _swallow_task  # type: ignore[method-assign, assignment]
    resumes = _record_resumes(rt_session)

    activity.on_end_of_speech(None)
    activity._audio_recognition = _recognition(
        activity, last_speaking_time=time.time() - VAD_MIN_SILENCE
    )
    activity._audio_recognition._run_eou_detection(MagicMock(), trigger="vad")

    await asyncio.sleep(MAX_DELAY + 0.3)
    await rt_session.aclose()

    assert resumes == []
    assert activity._false_interruption_pending is False
    assert activity._paused_speech is not None  # left for _cancel_speech_pause to interrupt


async def test_teardown_does_not_resume_a_deferred_pause(rt_session: AgentSession) -> None:
    # the close path releases the pause itself, so the resume must not fire mid-teardown
    activity, _ = _paused_activity(rt_session)
    resumes = _record_resumes(rt_session)

    activity.on_end_of_speech(None)
    activity._audio_recognition = _recognition(
        activity, last_speaking_time=time.time() - VAD_MIN_SILENCE
    )
    activity._audio_recognition._run_eou_detection(MagicMock(), trigger="vad")

    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.05)
    assert activity._false_interruption_pending is True

    await activity._audio_recognition._aclose()
    await asyncio.sleep(0.2)
    await rt_session.aclose()

    assert resumes == []


@pytest.mark.parametrize("cause", ["skipped_reply", "server_turn_detection"])
async def test_reply_that_returns_early_keeps_the_resume_armed(
    rt_session: AgentSession, cause: str
) -> None:
    # a skipped reply (AMD verdict, commit_user_turn on realtime) or server-side turn detection
    # returns before _cancel_speech_pause, so only the resume can un-pause the speech
    activity, _ = _paused_activity(rt_session)
    if cause == "skipped_reply":
        activity._interruption_detected = True
    else:
        activity._rt_turn_detection_enabled = True
    activity._create_speech_task = _swallow_task  # type: ignore[method-assign, assignment]
    resumes = _record_resumes(rt_session)

    activity.on_end_of_speech(None)
    assert activity.on_end_of_turn(_end_of_turn(skip_reply=cause == "skipped_reply")) is True

    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.2)
    await rt_session.aclose()

    assert len(resumes) == 1
    assert activity._paused_speech is None


async def test_resume_is_immediate_when_no_turn_decision_is_open(
    rt_session: AgentSession,
) -> None:
    # VAD-only noise on an stt pipeline never starts a bounce, so the timeout still rules
    activity, _ = _paused_activity(rt_session)
    resumes = _record_resumes(rt_session)

    t0 = time.time()
    activity.on_end_of_speech(None)

    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.2)
    await rt_session.aclose()

    assert [t - t0 for t in resumes] == [pytest.approx(FALSE_INTERRUPTION_TIMEOUT, abs=0.1)]


@pytest.mark.parametrize("interrupt", [True, False])
async def test_cancelling_a_pause_does_not_end_the_agent_turn(
    rt_session: AgentSession, interrupt: bool
) -> None:
    # interrupt=False is a handoff: the speech moves to the next activity
    activity, handle = _paused_activity(rt_session)
    activity._audio_recognition = MagicMock()

    await activity._cancel_speech_pause(interrupt=interrupt)
    await rt_session.aclose()

    assert handle.interrupt.call_count == int(interrupt)
    activity._audio_recognition._on_end_of_agent_speech.assert_not_called()


async def test_interruption_event_ends_the_agent_turn_once_when_pausing(
    rt_session: AgentSession,
) -> None:
    rt_session.options.interruption["resume_false_interruption"] = True
    rt_session.options.interruption["false_interruption_timeout"] = FALSE_INTERRUPTION_TIMEOUT
    activity, _ = _paused_activity(rt_session)
    activity._paused_speech = None
    activity._audio_recognition = MagicMock()

    activity.on_overlap_speech(
        OverlappingSpeechEvent(is_interruption=True, overlap_started_at=1.0, detected_at=2.0)
    )
    assert activity._paused_speech is not None
    await rt_session.aclose()

    activity._audio_recognition._on_end_of_agent_speech.assert_called_once_with(ended_at=ANY)


# --- a parked preemptive generation never blocks a handoff -----------------------------


class _GatedLLM(LLM):
    """Text-only LLM whose replies emit one token and then hold until ``gate`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    @property
    def model(self) -> str:
        return "gated"

    @property
    def provider(self) -> str:
        return "test"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        gate = self.gate

        class _GatedStream(LLMStream):
            async def _run(self) -> None:
                self._event_ch.send_nowait(
                    ChatChunk(id="chunk", delta=ChoiceDelta(role="assistant", content="one moment"))
                )
                await gate.wait()

        return _GatedStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2.0):
        while not predicate():
            await asyncio.sleep(0.01)


def _park_preemptive_reply(activity: AgentActivity) -> None:
    activity.on_preemptive_generation(
        _PreemptiveGenerationInfo(
            new_transcript="what is the rate",
            transcript_confidence=1.0,
            started_speaking_at=time.time(),
        )
    )
    assert activity._preemptive_generation is not None


async def _pause_for_handoff(session: AgentSession) -> None:
    await asyncio.wait_for(
        session._update_activity(
            Agent(instructions="hand off"),
            previous_activity="pause",
            blocked_tasks=[],
            wait_on_enter=False,
        ),
        timeout=2.0,
    )


async def test_handoff_after_a_turn_discarded_for_uninterruptible_speech() -> None:
    gated = _GatedLLM()
    session: AgentSession[None] = AgentSession(llm=gated)
    await session.start(agent=Agent(instructions="qualify the caller"))
    activity = session._activity
    assert activity is not None

    try:
        _park_preemptive_reply(activity)
        session.generate_reply(instructions="ask them to hold", allow_interruptions=False)
        await _wait_until(lambda: session.current_speech is not None)

        # the caller's turn ends while the uninterruptible message is still playing
        activity.on_end_of_turn(_end_of_turn("can you tell me the rate"))
        await _wait_until(
            lambda: (
                activity._user_turn_completed_atask is not None
                and activity._user_turn_completed_atask.done()
            )
        )

        gated.gate.set()
        await _wait_until(lambda: session.current_speech is None)

        await _pause_for_handoff(session)
        assert activity._preemptive_generation is None
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)


async def test_pause_clears_a_parked_preemptive_reply() -> None:
    session: AgentSession[None] = AgentSession(llm=_GatedLLM())
    await session.start(agent=Agent(instructions="qualify the caller"))
    activity = session._activity
    assert activity is not None

    try:
        _park_preemptive_reply(activity)
        await _pause_for_handoff(session)
        # the parked reply is gone, so a later resume cannot resurrect it
        assert activity._preemptive_generation is None
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)


async def test_reply_parked_while_pausing_does_not_block_the_handoff() -> None:
    """A reply parked from on_exit, after the handoff begins, is still dropped before the drain."""
    gated = _GatedLLM()
    session: AgentSession[None] = AgentSession(llm=gated)

    class _TalkingExit(Agent):
        async def on_exit(self) -> None:
            assert session._activity is not None
            _park_preemptive_reply(session._activity)

    await session.start(agent=_TalkingExit(instructions="qualify the caller"))
    activity = session._activity
    assert activity is not None

    try:
        await asyncio.wait_for(session._update_activity(Agent(instructions="hand off")), 2.0)
        assert activity._preemptive_generation is None
    finally:
        gated.gate.set()
        await asyncio.gather(session.aclose(), return_exceptions=True)


# --- cancelling a waiter never cancels the turn it waits on ----------------------------


async def test_cancelling_wait_for_idle_does_not_cancel_end_of_turn_task() -> None:
    activity = AgentActivity(Agent(instructions="test"), AgentSession())
    end_of_turn_task = asyncio.create_task(asyncio.Event().wait())
    activity._audio_recognition = cast(Any, SimpleNamespace(_end_of_turn_task=end_of_turn_task))

    idle_observer = asyncio.create_task(activity.wait_for_idle(wait_for_user=False))
    await asyncio.sleep(0)
    idle_observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await idle_observer

    assert not end_of_turn_task.cancelled()
    end_of_turn_task.cancel()
    await asyncio.gather(end_of_turn_task, return_exceptions=True)


async def test_cancelling_wait_for_idle_does_not_poison_user_turns() -> None:
    activity = AgentActivity(Agent(instructions="test"), AgentSession())
    activity._scheduling_paused = False
    activity._new_turns_blocked = False
    active_turn = asyncio.create_task(asyncio.Event().wait())
    activity._user_turn_completed_atask = active_turn

    idle_observer = asyncio.create_task(activity.wait_for_idle())
    await asyncio.sleep(0)
    idle_observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await idle_observer

    assert not active_turn.cancelled()
    active_turn.cancel()
    await asyncio.gather(active_turn, return_exceptions=True)

    assert activity.on_end_of_turn(_end_of_turn("turn 2"))
    successor = activity._user_turn_completed_atask
    assert successor is not None
    await successor
    assert not successor.cancelled()


async def test_cancelling_latest_user_turn_does_not_cancel_predecessor() -> None:
    activity = AgentActivity(Agent(instructions="test"), AgentSession())
    predecessor = asyncio.create_task(asyncio.Event().wait())
    latest_turn = asyncio.create_task(
        activity._user_turn_completed_task(predecessor, _end_of_turn("turn 2"))
    )
    await asyncio.sleep(0)
    latest_turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await latest_turn

    assert not predecessor.cancelled()
    predecessor.cancel()
    await asyncio.gather(predecessor, return_exceptions=True)


async def test_wait_for_idle_does_not_spin_through_speech_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sleep = asyncio.sleep
    yields = 0

    async def _counting_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        nonlocal yields
        if delay == 0 and asyncio.current_task() is idle_task:
            yields += 1
        return await real_sleep(delay, *args, **kwargs)

    async with AgentSession(min_consecutive_speech_delay=0.3) as session:
        await session.start(Agent(instructions="test"))
        session.say("first")
        session.say("second")
        await real_sleep(0.02)

        monkeypatch.setattr(asyncio, "sleep", _counting_sleep)
        idle_task = asyncio.create_task(session.wait_for_idle())
        await asyncio.wait_for(idle_task, timeout=5.0)

    # a few passes per speech; a spinning loop yields thousands of times through the delay
    assert yields < 20, f"wait_for_idle yielded {yields} times through the speech delay"
