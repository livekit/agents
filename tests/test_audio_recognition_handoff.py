from __future__ import annotations

from collections.abc import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.turn import _StreamingTurnDetector

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


def _make_activity(agent: Agent, stt: object, turn_detection: object = None) -> MagicMock:
    act = MagicMock(spec=AgentActivity)
    act.agent = agent
    act._audio_recognition = MagicMock()
    act._audio_recognition._detach_stt = AsyncMock(return_value=MagicMock())
    act._audio_recognition._detach_turn_detector = MagicMock(return_value=MagicMock())
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
    old._audio_recognition._detach_stt.assert_awaited_once()


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
    old._audio_recognition._detach_turn_detector.assert_called_once()


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
    old._audio_recognition._detach_turn_detector.assert_not_called()


async def test_turn_detector_not_reusable_when_new_opts_out() -> None:
    """New agent resolves to no turn detection (e.g. realtime server-side) → not reusable."""
    shared_detector = MagicMock(spec=_StreamingTurnDetector)
    old = _make_activity(Agent(instructions="a"), MagicMock(), turn_detection=shared_detector)
    new = _make_activity(Agent(instructions="b"), MagicMock(), turn_detection=None)

    result = await _detach_turn_detector_if_reusable(old, new)
    assert result is None
    old._audio_recognition._detach_turn_detector.assert_not_called()
