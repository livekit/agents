from __future__ import annotations

from collections.abc import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import aio
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition, _STTPipeline
from livekit.agents.voice.turn import _StreamingTurnDetector

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


def _make_activity(agent: Agent, stt: object, turn_detection: object = None) -> MagicMock:
    act = MagicMock(spec=AgentActivity)
    act.agent = agent
    act._audio_recognition = MagicMock()
    act._audio_recognition.detach_stt = AsyncMock(return_value=MagicMock())
    act._audio_recognition.detach_turn_detector = MagicMock(return_value=MagicMock())
    type(act).stt = PropertyMock(return_value=stt)
    # turn detector reuse checks read this; None disables the reuse branch
    act._turn_detection = turn_detection
    # rt session reuse checks need these
    act._rt_session = None
    type(act).llm = PropertyMock(return_value=None)
    type(act).tools = PropertyMock(return_value=[])
    return act


async def _detach_stt_if_reusable(old: MagicMock, new: MagicMock) -> object | None:
    """Call the real _detach_reusable_resources, return stt_pipeline."""
    resources = await AgentActivity._detach_reusable_resources(old, new)
    return resources.stt_pipeline


# ---------------------------------------------------------------------------
# STT pipeline reuse via _detach_reusable_resources
# ---------------------------------------------------------------------------


async def test_reusable_same_class_same_stt() -> None:
    """Two plain Agent instances sharing the same STT object → reusable."""
    shared_stt = MagicMock()
    old = _make_activity(Agent(instructions="a"), shared_stt)
    new = _make_activity(Agent(instructions="b"), shared_stt)

    result = await _detach_stt_if_reusable(old, new)
    assert result is not None  # detach_stt was called
    old._audio_recognition.detach_stt.assert_awaited_once()


async def test_not_reusable_different_stt_instance() -> None:
    """Different STT instances (different connections) → not reusable."""
    old = _make_activity(Agent(instructions="a"), MagicMock())
    new = _make_activity(Agent(instructions="b"), MagicMock())

    result = await _detach_stt_if_reusable(old, new)
    assert result is None


async def test_not_reusable_no_stt() -> None:
    """Either side missing STT → not reusable."""
    shared_stt = MagicMock()
    old = _make_activity(Agent(instructions="a"), None)
    new = _make_activity(Agent(instructions="b"), shared_stt)

    result = await _detach_stt_if_reusable(old, new)
    assert result is None


async def test_not_reusable_different_stt_node_override() -> None:
    """Both subclasses define their own stt_node override → not reusable."""
    shared_stt = MagicMock()

    class AgentA(Agent):
        def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> None:
            return None

    class AgentB(Agent):
        def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> None:
            return None

    old = _make_activity(AgentA(instructions="a"), shared_stt)
    new = _make_activity(AgentB(instructions="b"), shared_stt)

    result = await _detach_stt_if_reusable(old, new)
    assert result is None


async def test_not_reusable_subclass_inherits_custom_stt_node() -> None:
    """Both agents share a custom stt_node via inheritance → not reusable.

    The pipeline is bound to the old agent's `self`; a custom stt_node may access
    self.session/activity inside the yield loop, which raises after detach.
    Only the default Agent.stt_node is known to be safe to reuse.
    """
    shared_stt = MagicMock()

    class AgentA(Agent):
        def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> None:
            return None

    class AgentB(AgentA):
        pass  # inherits AgentA.stt_node without overriding

    assert AgentA.stt_node is AgentB.stt_node  # sanity check

    old = _make_activity(AgentA(instructions="a"), shared_stt)
    new = _make_activity(AgentB(instructions="b"), shared_stt)

    result = await _detach_stt_if_reusable(old, new)
    assert result is None


async def test_not_reusable_subclass_overrides_stt_node() -> None:
    """Old agent has stt_node; new agent's subclass overrides it differently → not reusable."""
    shared_stt = MagicMock()

    class AgentA(Agent):
        def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> None:
            return None

    class AgentB(AgentA):
        def stt_node(  # type: ignore[override]
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> None:
            return None

    old = _make_activity(AgentA(instructions="a"), shared_stt)
    new = _make_activity(AgentB(instructions="b"), shared_stt)

    result = await _detach_stt_if_reusable(old, new)
    assert result is None


async def test_not_reusable_no_audio_recognition() -> None:
    """Old activity has no audio recognition → not reusable."""
    shared_stt = MagicMock()
    old = _make_activity(Agent(instructions="a"), shared_stt)
    old._audio_recognition = None
    new = _make_activity(Agent(instructions="b"), shared_stt)

    result = await _detach_stt_if_reusable(old, new)
    assert result is None


# ---------------------------------------------------------------------------
# Turn detector stream reuse via _detach_reusable_resources
# ---------------------------------------------------------------------------


async def _detach_turn_detector_if_reusable(old: MagicMock, new: MagicMock) -> object | None:
    """Call the real _detach_reusable_resources, return turn_detector_stream."""
    resources = await AgentActivity._detach_reusable_resources(old, new)
    return resources.turn_detector_stream


async def test_turn_detector_reusable_same_instance() -> None:
    """Same TurnDetector instance carries over → live stream is detached for reuse."""
    shared_detector = MagicMock(spec=_StreamingTurnDetector)
    old = _make_activity(Agent(instructions="a"), MagicMock(), turn_detection=shared_detector)
    new = _make_activity(Agent(instructions="b"), MagicMock(), turn_detection=shared_detector)

    result = await _detach_turn_detector_if_reusable(old, new)
    assert result is not None
    old._audio_recognition.detach_turn_detector.assert_called_once()


async def test_turn_detector_not_reusable_different_instance() -> None:
    """Different detector instances → not reusable (old stream torn down normally)."""
    old = _make_activity(
        Agent(instructions="a"), MagicMock(), turn_detection=MagicMock(spec=_StreamingTurnDetector)
    )
    new = _make_activity(
        Agent(instructions="b"), MagicMock(), turn_detection=MagicMock(spec=_StreamingTurnDetector)
    )

    result = await _detach_turn_detector_if_reusable(old, new)
    assert result is None
    old._audio_recognition.detach_turn_detector.assert_not_called()


async def test_turn_detector_not_reusable_when_new_opts_out() -> None:
    """New agent resolves to no turn detection (e.g. realtime server-side) → not reusable."""
    shared_detector = MagicMock(spec=_StreamingTurnDetector)
    old = _make_activity(Agent(instructions="a"), MagicMock(), turn_detection=shared_detector)
    new = _make_activity(Agent(instructions="b"), MagicMock(), turn_detection=None)

    result = await _detach_turn_detector_if_reusable(old, new)
    assert result is None
    old._audio_recognition.detach_turn_detector.assert_not_called()


# ---------------------------------------------------------------------------
# Input-time anchor (end_time=0 wall clock) survives pipeline reuse
# ---------------------------------------------------------------------------


def _stub_recognition() -> AudioRecognition:
    ar = object.__new__(AudioRecognition)
    ar._stt_consumer_atask = None  # type: ignore[attr-defined]
    ar._stt_pipeline = None  # type: ignore[attr-defined]
    ar._transcript_buffer = MagicMock()  # type: ignore[attr-defined]
    ar._ignore_user_transcript_until = NOT_GIVEN  # type: ignore[attr-defined]
    return ar


async def test_input_anchor_preserved_when_pipeline_reused() -> None:
    """update_stt must not reset a reused pipeline's input anchor.

    The STT ``end_time`` clock is relative to the original stream start. Re-anchoring
    ``input_started_at`` to the handoff time would desync the two and push the derived
    speaking time minutes into the future (see the 68s end-of-turn stall).
    """
    reused = object.__new__(_STTPipeline)
    reused.input_started_at = 1000.0  # anchored during the previous activity
    ch = aio.Chan()  # type: ignore[var-annotated]
    ch.close()  # closed channel → the swapped-in consumer exits immediately
    reused._event_ch = ch  # type: ignore[attr-defined]

    ar = _stub_recognition()
    ar.update_stt(MagicMock(), pipeline=reused)

    assert ar._input_started_at == 1000.0  # carried over, not reset to None
    if ar._stt_consumer_atask is not None:
        await ar._stt_consumer_atask


def test_input_anchor_reads_through_to_pipeline() -> None:
    """The anchor lives on the pipeline so it travels with the stream across handoff."""
    ar = _stub_recognition()
    assert ar._input_started_at is None  # no pipeline attached yet

    pipeline = object.__new__(_STTPipeline)
    pipeline.input_started_at = 1234.5
    ar._stt_pipeline = pipeline  # type: ignore[attr-defined]

    assert ar._input_started_at == 1234.5  # read-only view of the pipeline's anchor
