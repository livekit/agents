from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions
from livekit.agents.voice.agent_activity import AgentActivity

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


def _make_activity(session: AgentSession) -> AgentActivity:
    return AgentActivity(Agent(instructions="test"), session)


async def test_adaptive_interruption_enabled_for_realtime_without_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a realtime model with server-side turn detection off transcribes internally and
    # commits turns manually, so barge-in gatekeeps by withholding commit rather than
    # holding STT transcripts — no separate STT is required
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=False)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive"},
        ),
    )

    activity = _make_activity(session)

    assert activity._interruption_detection_enabled is True
    assert activity._interruption_detector is not None


async def test_adaptive_interruption_still_requires_stt_for_non_realtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the STT pipeline path is unchanged: without an aligned streaming STT there is
    # nothing to gatekeep, so adaptive interruption stays disabled
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = AgentSession(
        llm=FakeLLM(fake_responses=[]),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive"},
        ),
    )

    activity = _make_activity(session)

    assert activity._interruption_detection_enabled is False
    assert activity._interruption_detector is None


async def test_adaptive_interruption_disabled_for_realtime_with_server_turn_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # server-side turn detection creates turns automatically, so client-side barge-in
    # cannot take over — it stays disabled even for a realtime model
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(interruption={"mode": "adaptive"}),
    )

    activity = _make_activity(session)

    assert activity._interruption_detection_enabled is False
    assert activity._interruption_detector is None
