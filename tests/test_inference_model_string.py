"""Unit tests for resolving a ``llm=`` model string to an inference model.

Realtime (speech-to-speech) model strings resolve to
``inference.RealtimeModel``; everything else to ``inference.LLM``.
"""

from __future__ import annotations

import pytest

from livekit.agents import inference
from livekit.agents.voice import Agent, AgentSession

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def inference_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret-that-is-at-least-32-bytes")


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-realtime",
        "openai/gpt-realtime-mini",
        "openai/gpt-realtime-1.5",
        "openai/gpt-realtime-2",
        "openai/gpt-realtime-2.1",
        "openai/gpt-realtime-2.1-mini",
        "xai/grok-voice",
        "xai/grok-voice-latest",
        "xai/grok-voice-think-fast-2.0",
    ],
)
def test_realtime_model_strings(model: str) -> None:
    assert inference.is_realtime_model(model)

    resolved = inference.llm_from_model_string(model)
    assert isinstance(resolved, inference.RealtimeModel)
    assert resolved.model == model


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o",
        "google/gemini-2.5-flash",
        "zai/glm-5.1",
        # a realtime model has to be listed in RealtimeModels to be recognized
        "openai/gpt-realtime-not-released-yet",
    ],
)
def test_llm_model_strings(model: str) -> None:
    assert not inference.is_realtime_model(model)

    resolved = inference.llm_from_model_string(model)
    assert isinstance(resolved, inference.LLM)
    assert resolved.model == model


def test_agent_session_accepts_realtime_model_string() -> None:
    session = AgentSession(llm="openai/gpt-realtime")
    assert isinstance(session.llm, inference.RealtimeModel)


def test_agent_accepts_model_strings() -> None:
    agent = Agent(instructions="you are a helpful assistant", llm="openai/gpt-realtime")
    assert isinstance(agent.llm, inference.RealtimeModel)

    agent = Agent(instructions="you are a helpful assistant", llm="openai/gpt-4o")
    assert isinstance(agent.llm, inference.LLM)

    agent.update_options(llm="openai/gpt-realtime")
    assert isinstance(agent.llm, inference.RealtimeModel)
