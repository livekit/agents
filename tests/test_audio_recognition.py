from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.utils import aio
from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition, _STTPipeline
from livekit.agents.voice.turn import _StreamingTurnDetector

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _closable_recognition() -> AudioRecognition:
    ar = object.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._hooks = MagicMock()
    ar._closing = asyncio.Event()
    ar._tasks = set()
    ar._stt_consumer_atask = None
    ar._stt_pipeline = None
    ar._vad_atask = None
    ar._interruption_atask = None
    ar._turn_detector_stream = None
    ar._commit_user_turn_atask = None
    ar._end_of_turn_task = None
    ar._vad_ch = None
    ar._interruption_ch = None
    ar._audio_input_atask = None
    ar._backchannel_boundary_timer = None
    ar._AudioRecognition__stt_context = None
    ar._user_turn_span = None
    ar._stt_events = []
    ar._user_turn_start = None
    ar._eou_wait_span = None
    ar._eou_wait_started_at_ns = None
    ar._eou_wait_rearms = 0
    ar._eou_wait_floor_ns = None
    ar._eou_wait_not_committed = 0
    ar._user_turn_resumes = 0
    ar._eou_detection_span = None
    ar._transcription_timeout_handle = None
    return ar


async def test_aclose_handles_precancelled_tasks() -> None:
    ar = _closable_recognition()
    ar._commit_user_turn_atask = asyncio.create_task(asyncio.sleep(10))
    ar._end_of_turn_task = asyncio.create_task(asyncio.sleep(10))
    ar._commit_user_turn_atask.cancel()
    ar._end_of_turn_task.cancel()
    await asyncio.sleep(0)

    await ar._aclose()

    assert ar._closing.is_set()
    assert ar._commit_user_turn_atask.done()
    assert ar._end_of_turn_task.done()


@pytest.mark.parametrize("task_attr", ["_commit_user_turn_atask", "_end_of_turn_task"])
async def test_aclose_waits_for_pending_turn_task(task_attr: str) -> None:
    ar = _closable_recognition()
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    await asyncio.sleep(0)
    setattr(ar, task_attr, task)

    close_task = asyncio.create_task(ar._aclose())
    await asyncio.sleep(0)
    assert not close_task.done()
    assert not task.cancelled()

    release.set()
    await close_task
    assert task.done()


@pytest.mark.parametrize("task_attr", ["_commit_user_turn_atask", "_end_of_turn_task"])
async def test_aclose_propagates_cancellation_while_waiting_for_turn_task(task_attr: str) -> None:
    ar = _closable_recognition()
    task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    setattr(ar, task_attr, task)

    close_task = asyncio.create_task(ar._aclose())
    await asyncio.sleep(0)
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert task.cancelled()


@pytest.mark.parametrize(
    ("task_attr", "warning"),
    [
        ("_commit_user_turn_atask", "error while committing the final user turn on close"),
        ("_end_of_turn_task", "error while completing the final user turn on close"),
    ],
)
async def test_aclose_logs_failed_turn_task(
    task_attr: str, warning: str, caplog: pytest.LogCaptureFixture
) -> None:
    ar = _closable_recognition()

    async def failed_task() -> None:
        raise RuntimeError("turn task failed")

    setattr(ar, task_attr, asyncio.create_task(failed_task()))

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await ar._aclose()

    [record] = [r for r in caplog.records if r.getMessage() == f"{warning}: RuntimeError"]
    assert record.exc_info is None


@pytest.mark.parametrize(("is_recording", "expected_end_count"), [(True, 1), (False, 0)])
async def test_aclose_finalizes_user_turn_span(is_recording: bool, expected_end_count: int) -> None:
    ar = _closable_recognition()
    span = MagicMock()
    span.is_recording.return_value = is_recording
    ar._user_turn_span = span
    ar._user_turn_start = 123.0

    await ar._aclose()

    assert span.end.call_count == expected_end_count
    assert ar._user_turn_span is None
    assert ar._user_turn_start is None


async def test_aclose_ends_user_turn_span_when_teardown_raises() -> None:
    ar = _closable_recognition()
    span = MagicMock()
    span.is_recording.return_value = True
    ar._user_turn_span = span
    ar._user_turn_start = 123.0
    ar._transcription_timeout_handle = MagicMock()
    ar._stt_pipeline = MagicMock()
    ar._stt_pipeline.aclose = AsyncMock(side_effect=RuntimeError("vendor stream teardown failed"))
    timeout_handle = ar._transcription_timeout_handle

    with pytest.raises(RuntimeError, match="vendor stream teardown failed"):
        await ar._aclose()

    span.end.assert_called_once()
    timeout_handle.cancel.assert_called_once()
    assert ar._user_turn_span is None


def _frame(byte: int = 0x11, sample_rate: int = 16000) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes([byte, byte]) * 160,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=160,
    )


@pytest.fixture
def push_recognition() -> AudioRecognition:
    ar = object.__new__(AudioRecognition)
    ar._sample_rate = None
    ar._stt_pipeline = MagicMock()
    ar._stt_pipeline.input_started_at = None
    ar._vad_ch = MagicMock()
    ar._interruption_ch = MagicMock()
    ar._session = MagicMock()
    ar._turn_detector_stream = None
    return ar


@pytest.mark.parametrize("substitute", [False, True])
def test_push_audio_routes_frame_to_every_consumer(
    push_recognition: AudioRecognition, substitute: bool
) -> None:
    ar = push_recognition
    real = _frame(byte=0x11)
    silence = _frame(byte=0x00)

    ar._push_audio(real, stt_frame=silence if substitute else None)

    # only the STT path sees the substitute frame
    ar._stt_pipeline.audio_ch.send_nowait.assert_called_once_with(silence if substitute else real)
    ar._vad_ch.send_nowait.assert_called_once_with(real)
    ar._session.amd.push_audio.assert_called_once_with(real)
    ar._interruption_ch.send_nowait.assert_called_once_with(real)


def test_push_audio_skips_optional_consumers_when_unset(push_recognition: AudioRecognition) -> None:
    ar = push_recognition
    ar._stt_pipeline = None
    ar._vad_ch = None
    ar._interruption_ch = None
    ar._session.amd = None

    ar._push_audio(_frame())


def test_push_audio_records_sample_rate_and_input_start(
    push_recognition: AudioRecognition,
) -> None:
    push_recognition._push_audio(_frame(sample_rate=24000))

    assert push_recognition._sample_rate == 24000
    assert push_recognition._input_started_at is not None


def _activity(agent: Agent, stt: object, turn_detection: object = None) -> MagicMock:
    act = MagicMock(spec=AgentActivity)
    act.agent = agent
    act._audio_recognition = MagicMock()
    act._audio_recognition._detach_stt = AsyncMock(return_value=MagicMock())
    act._audio_recognition._detach_turn_detector = MagicMock(return_value=MagicMock())
    type(act).stt = PropertyMock(return_value=stt)
    act._turn_detection = turn_detection
    act._rt_session = None
    type(act).llm = PropertyMock(return_value=None)
    type(act).tools = PropertyMock(return_value=[])
    return act


class _CustomNodeAgent(Agent):
    def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings) -> None:
        return None


class _OtherCustomNodeAgent(Agent):
    def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings) -> None:
        return None


class _InheritedNodeAgent(_CustomNodeAgent):
    pass


class _OverriddenNodeAgent(_CustomNodeAgent):
    def stt_node(  # type: ignore[override]
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> None:
        return None


async def test_stt_pipeline_reused_for_same_stt_and_default_node() -> None:
    stt = MagicMock()
    old = _activity(Agent(instructions="a"), stt)

    resources = await AgentActivity._detach_reusable_resources(
        old, _activity(Agent(instructions="b"), stt)
    )

    assert resources.stt_pipeline is not None
    old._audio_recognition._detach_stt.assert_awaited_once()


_SHARED_STT = MagicMock()


@pytest.mark.parametrize(
    ("old_agent", "old_stt", "new_agent", "new_stt"),
    [
        pytest.param(Agent, MagicMock(), Agent, MagicMock(), id="different-stt"),
        pytest.param(Agent, None, Agent, _SHARED_STT, id="no-stt"),
        pytest.param(_CustomNodeAgent, _SHARED_STT, _OtherCustomNodeAgent, _SHARED_STT, id="nodes"),
        # a custom node is bound to the old agent and may touch its torn-down session
        pytest.param(
            _CustomNodeAgent, _SHARED_STT, _InheritedNodeAgent, _SHARED_STT, id="inherited-node"
        ),
        pytest.param(
            _CustomNodeAgent, _SHARED_STT, _OverriddenNodeAgent, _SHARED_STT, id="overridden-node"
        ),
    ],
)
async def test_stt_pipeline_not_reused(
    old_agent: type[Agent], old_stt: object, new_agent: type[Agent], new_stt: object
) -> None:
    resources = await AgentActivity._detach_reusable_resources(
        _activity(old_agent(instructions="a"), old_stt),
        _activity(new_agent(instructions="b"), new_stt),
    )
    assert resources.stt_pipeline is None


async def test_stt_pipeline_not_reused_without_audio_recognition() -> None:
    old = _activity(Agent(instructions="a"), _SHARED_STT)
    old._audio_recognition = None

    resources = await AgentActivity._detach_reusable_resources(
        old, _activity(Agent(instructions="b"), _SHARED_STT)
    )
    assert resources.stt_pipeline is None


@pytest.mark.parametrize("new_detector", ["same", "other", None])
async def test_turn_detector_stream_reused_only_for_same_detector(
    new_detector: str | None,
) -> None:
    detector = MagicMock(spec=_StreamingTurnDetector)
    old = _activity(Agent(instructions="a"), MagicMock(), turn_detection=detector)
    new = _activity(
        Agent(instructions="b"),
        MagicMock(),
        turn_detection={
            "same": detector,
            "other": MagicMock(spec=_StreamingTurnDetector),
            None: None,
        }[new_detector],
    )

    resources = await AgentActivity._detach_reusable_resources(old, new)

    reused = new_detector == "same"
    assert (resources.turn_detector_stream is not None) is reused
    assert old._audio_recognition._detach_turn_detector.called is reused


async def test_reused_stt_pipeline_keeps_input_anchor_and_rebinds_node() -> None:
    # stt end times are relative to the original stream start, so the anchor must carry over
    reused = object.__new__(_STTPipeline)
    reused.input_started_at = 1000.0
    reused._stt_node = MagicMock(name="previous_agent_node")
    ch = aio.Chan()  # type: ignore[var-annotated]
    ch.close()
    reused._event_ch = ch
    ar = object.__new__(AudioRecognition)
    ar._stt_consumer_atask = None
    ar._stt_pipeline = None
    ar._transcript_buffer = MagicMock()
    new_node = MagicMock(name="current_agent_node")

    ar._update_stt(new_node, pipeline=reused)

    assert ar._input_started_at == 1000.0
    # the previous agent's node is torn down; recreating through it would leave the agent deaf
    assert reused._stt_node is new_node
    if ar._stt_consumer_atask is not None:
        await ar._stt_consumer_atask


def _vad_event(type_: VADEventType, *, speech_duration: float = 0.0) -> VADEvent:
    return VADEvent(
        type=type_,
        samples_index=0,
        timestamp=time.time(),
        speech_duration=speech_duration,
        silence_duration=0.6 if type_ == VADEventType.END_OF_SPEECH else 0.0,
    )


def _vad_recognition() -> AudioRecognition:
    """An ``AudioRecognition`` with only the state its VAD events touch."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._speech_start_time = None
    ar._vad_speech_started = False
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._agent_speaking = False
    ar._turn_detector_stream = None
    ar._end_of_turn_task = None
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._eou_wait_span = None
    ar._eou_wait_started_at_ns = None
    ar._eou_wait_rearms = 0
    ar._eou_wait_floor_ns = None
    ar._eou_wait_not_committed = 0
    ar._user_turn_resumes = 0
    ar._eou_detection_span = None
    ar._user_turn_committed = False
    # no end-of-turn detection: only the VAD-driven state is under test
    ar._vad_base_turn_detection = False
    ar._turn_detection_mode = None
    ar._stt = None
    ar._stt_pipeline = None
    ar._stt_model = None
    ar._stt_provider = None
    ar._audio_transcript = ""
    ar._audio_interim_transcript = ""
    ar._last_speaking_time = None
    ar._transcription_timeout_handle = None
    ar._turn_speech_duration = 0.0
    ar._hooks = MagicMock()
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._session.amd = None
    ar._session._room_io = None
    ar._session.options.transcription_timeout = None
    return ar


async def test_user_turn_start_persists_across_vad_bursts_in_one_turn() -> None:
    """The turn starts at its first burst; only the per-burst speech start moves."""
    ar = _vad_recognition()

    first_sos_at = time.time()
    await ar._on_vad_event(_vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0))
    turn_start = ar._user_turn_start
    first_burst_start = ar._speech_start_time
    assert turn_start == pytest.approx(first_sos_at - 5.0, abs=0.01)

    await ar._on_vad_event(_vad_event(VADEventType.END_OF_SPEECH, speech_duration=5.0))
    await asyncio.sleep(0.05)
    await ar._on_vad_event(_vad_event(VADEventType.START_OF_SPEECH))

    assert ar._user_turn_start == turn_start
    assert first_burst_start is not None and ar._speech_start_time is not None
    assert ar._speech_start_time > first_burst_start


async def test_turn_detector_hears_the_user_turn_start_once() -> None:
    """The audio turn detector counts the turn length from ``_user_turn_start``, so it gets
    that value on the first start of speech and nothing on later bursts of the same turn."""
    ar = _vad_recognition()
    stream = MagicMock()
    ar._turn_detector_stream = stream

    await ar._on_vad_event(_vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0))
    await ar._on_vad_event(_vad_event(VADEventType.END_OF_SPEECH, speech_duration=5.0))
    await ar._on_vad_event(_vad_event(VADEventType.START_OF_SPEECH))

    stream.on_start_of_turn.assert_called_once_with(ar._user_turn_start)


async def test_turn_without_a_known_start_keeps_the_detector_count() -> None:
    """A turn opened by end of speech (its start of speech fell in the previous turn) has no
    known start: the turn detector keeps counting from its last flush instead of hearing a
    start of now."""
    ar = _vad_recognition()
    stream = MagicMock()
    ar._turn_detector_stream = stream

    await ar._on_vad_event(_vad_event(VADEventType.END_OF_SPEECH, speech_duration=3.0))

    assert ar._user_turn_start is not None
    stream.on_start_of_turn.assert_not_called()
