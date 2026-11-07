from __future__ import annotations

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    InterruptionOptions,
    TurnHandlingOptions,
    inference,
)
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.remote_session import _serialize_options

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


@pytest.fixture
def _no_env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
        "LIVEKIT_INFERENCE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def _build_session(interruption: InterruptionOptions) -> AgentSession:
    return AgentSession[None](
        vad=FakeVAD(fake_user_speeches=[]),
        stt=FakeSTT(),
        llm=FakeLLM(fake_responses=[]),
        tts=FakeTTS(fake_responses=[]),
        turn_handling=TurnHandlingOptions(turn_detection="stt", interruption=interruption),
        aec_warmup_duration=None,
    )


def test_interruption_options_credentials_reach_detector(_no_env_credentials: None) -> None:
    session = _build_session(
        InterruptionOptions(
            mode="adaptive",
            base_url="https://inference.example.com",
            api_key="opts-key",
            api_secret="opts-secret",
        )
    )
    activity = AgentActivity(Agent(instructions="test"), session)

    detector = activity._interruption_detector
    assert isinstance(detector, inference.AdaptiveInterruptionDetector)
    assert detector._opts.base_url == "https://inference.example.com"
    assert detector._opts.api_key == "opts-key"
    assert detector._opts.api_secret == "opts-secret"


def test_adaptive_without_credentials_falls_back_to_vad(_no_env_credentials: None) -> None:
    session = _build_session(InterruptionOptions(mode="adaptive"))
    activity = AgentActivity(Agent(instructions="test"), session)

    assert activity._interruption_detector is None


def test_remote_session_options_omit_credentials(_no_env_credentials: None) -> None:
    session = _build_session(
        InterruptionOptions(mode="adaptive", api_key="opts-key", api_secret="opts-secret")
    )

    serialized = _serialize_options(session.options)["interruption"]
    assert "opts-key" not in serialized
    assert "opts-secret" not in serialized
    assert "'mode': 'adaptive'" in serialized
