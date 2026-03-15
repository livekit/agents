from __future__ import annotations

from collections.abc import AsyncIterable
from unittest.mock import MagicMock, PropertyMock

from livekit import rtc
from livekit.agents import Agent
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition


def _make_activity(agent: Agent, stt: object) -> MagicMock:
    act = MagicMock(spec=AgentActivity)
    act.agent = agent
    act._audio_recognition = MagicMock(spec=AudioRecognition)
    type(act).stt = PropertyMock(return_value=stt)
    return act


def _reusable(old: MagicMock, new: MagicMock) -> bool:
    """Call the real _audio_recognition_reusable implementation, bypassing the mock."""
    return AgentActivity._audio_recognition_reusable(old, new)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _audio_recognition_reusable
# ---------------------------------------------------------------------------


def test_reusable_same_class_same_stt() -> None:
    """Two plain Agent instances sharing the same STT object → reusable."""
    shared_stt = MagicMock()
    old = _make_activity(Agent(instructions="a"), shared_stt)
    new = _make_activity(Agent(instructions="b"), shared_stt)

    assert _reusable(old, new) is True


def test_not_reusable_different_stt_instance() -> None:
    """Different STT instances (different connections) → not reusable."""
    old = _make_activity(Agent(instructions="a"), MagicMock())
    new = _make_activity(Agent(instructions="b"), MagicMock())

    assert _reusable(old, new) is False


def test_not_reusable_no_stt() -> None:
    """Either side missing STT → not reusable."""
    shared_stt = MagicMock()
    old = _make_activity(Agent(instructions="a"), None)
    new = _make_activity(Agent(instructions="b"), shared_stt)

    assert _reusable(old, new) is False


def test_not_reusable_different_stt_node_override() -> None:
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

    assert _reusable(old, new) is False


def test_reusable_subclass_inherits_stt_node() -> None:
    """Old agent overrides stt_node; new agent is a subclass that inherits it → reusable.

    B(A) does not override stt_node, so B.stt_node is A.stt_node (same object via MRO).
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

    assert _reusable(old, new) is True


def test_not_reusable_subclass_overrides_stt_node() -> None:
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

    assert _reusable(old, new) is False
