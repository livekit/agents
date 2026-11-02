"""Unit tests for resolving a ``llm=`` model string to an inference model.

Realtime (speech-to-speech) model strings resolve to ``inference.RealtimeModel``,
duplex model strings to ``inference.DuplexModel``, everything else to ``inference.LLM``.
"""

from __future__ import annotations

import pytest

from livekit.agents import inference
from livekit.agents.llm import DuplexRealtimeAdapter
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
        "openai/gpt-realtime-2.1",
        "openai/gpt-realtime-2.1-mini",
        "google/gemini-3.8-live",
        "google/gemini-3.8-live-extended-thinking",
        "google/gemini-live",
        "xai/grok-voice",
        "xai/grok-voice-latest",
        "xai/grok-voice-3.0",
        # retired ids stay realtime rather than falling through to a text LLM
        "openai/gpt-realtime-2",
        "openai/gpt-realtime-1.5",
        "xai/grok-voice-think-fast-2.0",
    ],
)
def test_realtime_model_strings(model: str) -> None:
    assert inference.is_realtime_model(model)
    assert not inference.is_duplex_model(model)

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


def test_duplex_model_string() -> None:
    assert inference.is_duplex_model("openai/gpt-live-1")
    assert not inference.is_realtime_model("openai/gpt-live-1")

    resolved = inference.llm_from_model_string("openai/gpt-live-1")
    assert isinstance(resolved, inference.DuplexModel)
    assert resolved.model == "openai/gpt-live-1"


def test_agent_session_accepts_realtime_model_string() -> None:
    session = AgentSession(llm="openai/gpt-realtime")
    assert isinstance(session.llm, inference.RealtimeModel)


def test_agent_session_wraps_a_duplex_model_string() -> None:
    session = AgentSession(llm="openai/gpt-live-1")
    assert isinstance(session.llm, DuplexRealtimeAdapter)
    assert isinstance(session.llm._duplex_model, inference.DuplexModel)


def test_agent_accepts_model_strings() -> None:
    agent = Agent(instructions="you are a helpful assistant", llm="openai/gpt-realtime")
    assert isinstance(agent.llm, inference.RealtimeModel)

    agent = Agent(instructions="you are a helpful assistant", llm="openai/gpt-4o")
    assert isinstance(agent.llm, inference.LLM)

    agent.update_options(llm="openai/gpt-realtime")
    assert isinstance(agent.llm, inference.RealtimeModel)
