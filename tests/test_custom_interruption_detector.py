"""A user-supplied ``InterruptionDetector`` instance passed as ``interruption["mode"]``
replaces the built-in adaptive detector."""

from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions, inference
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.agent_activity import AgentActivity

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _CustomDetector(inference.InterruptionDetector):
    @property
    def model(self) -> str:
        return "custom"

    @property
    def provider(self) -> str:
        return "test"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> inference.InterruptionStream:
        raise NotImplementedError


def _session(mode: object, *, stt: bool = True) -> AgentSession:
    return AgentSession(
        llm=FakeLLM(fake_responses=[]),
        stt=FakeSTT() if stt else None,
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad", interruption={"mode": mode}),  # type: ignore[typeddict-item]
    )


def _agent(mode: object | None = None) -> Agent:
    if mode is None:
        return Agent(instructions="test")
    return Agent(
        instructions="test",
        turn_handling=TurnHandlingOptions(interruption={"mode": mode}),  # type: ignore[typeddict-item]
    )


async def test_session_detector_instance_is_used() -> None:
    detector = _CustomDetector()
    activity = AgentActivity(_agent(), _session(detector))

    assert activity._interruption_detector is detector
    assert activity._interruption_detection_enabled is True


async def test_agent_detector_instance_overrides_session_mode() -> None:
    detector = _CustomDetector()
    activity = AgentActivity(_agent(detector), _session("vad"))

    assert activity._interruption_detector is detector


async def test_agent_vad_mode_overrides_session_detector() -> None:
    activity = AgentActivity(_agent("vad"), _session(_CustomDetector()))

    assert activity._interruption_detector is None


async def test_detector_instance_still_requires_stt() -> None:
    activity = AgentActivity(_agent(), _session(_CustomDetector(), stt=False))

    assert activity._interruption_detector is None


def test_adaptive_detector_is_an_interruption_detector() -> None:
    assert issubclass(inference.AdaptiveInterruptionDetector, inference.InterruptionDetector)
