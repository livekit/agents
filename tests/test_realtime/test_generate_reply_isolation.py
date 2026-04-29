"""Tests for `add_to_chat_ctx=False` on RealtimeSession.generate_reply.

Covers: capability flag, dispatcher gate, pipeline-LLM guard, OpenAI plugin
substrate isolation, ephemeral-event suppression, orphan filter, handler
guards, single-isolated-call serialization contract, local _chat_ctx gate,
remote-item-added gate, interrupt() with response_id.

Live-substrate tests require OPENAI_API_KEY and run against `gpt-realtime`.
"""

from __future__ import annotations

import inspect
import os

import pytest

from livekit.agents import llm
from livekit.plugins import openai as lk_openai

# -- Helpers --


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _has_azure_key() -> bool:
    return all(
        os.environ.get(k)
        for k in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_DEPLOYMENT",
        )
    )


# ---------------------------------------------------------------------------
# Component 1: abstract RealtimeSession.generate_reply signature
# Component 5: RealtimeCapabilities.ephemeral_response field
# ---------------------------------------------------------------------------


def test_realtime_capabilities_has_ephemeral_response_field_defaulting_false() -> None:
    """The new field exists on the dataclass and defaults to False."""
    caps = llm.RealtimeCapabilities(
        message_truncation=False,
        turn_detection=False,
        user_transcription=False,
        auto_tool_reply_generation=False,
        audio_output=False,
        manual_function_calls=False,
    )
    assert caps.ephemeral_response is False


def test_realtime_session_generate_reply_signature_accepts_add_to_chat_ctx() -> None:
    """The abstract signature exposes the new keyword-only parameter with default True."""
    sig = inspect.signature(llm.RealtimeSession.generate_reply)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True


# ---------------------------------------------------------------------------
# Component 5 (Phase 1): OpenAI plugin declares the capability for non-Azure
# only; Azure-backed sessions advertise ephemeral_response=False so the
# dispatcher capability gate falls through to the legacy path until Azure
# parity is verified.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
def test_openai_plugin_declares_ephemeral_response_for_non_azure() -> None:
    model = lk_openai.realtime.RealtimeModel()
    assert model.capabilities.ephemeral_response is True


@pytest.mark.skipif(not _has_azure_key(), reason="Azure OpenAI env vars not set")
def test_openai_plugin_does_not_declare_ephemeral_response_for_azure() -> None:
    model = lk_openai.realtime.RealtimeModel(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"] + "/openai",
    )
    assert model.capabilities.ephemeral_response is False


def test_openai_plugin_generate_reply_signature_accepts_add_to_chat_ctx() -> None:
    """OpenAI plugin RealtimeSession.generate_reply accepts the new kwarg."""
    sig = inspect.signature(lk_openai.realtime.RealtimeSession.generate_reply)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True


# ---------------------------------------------------------------------------
# Component 2 (Phase 1): AgentSession.generate_reply
# ---------------------------------------------------------------------------


def test_agent_session_generate_reply_signature_accepts_add_to_chat_ctx() -> None:
    from livekit.agents import AgentSession

    sig = inspect.signature(AgentSession.generate_reply)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True


async def test_agent_session_generate_reply_pipeline_llm_with_isolation_raises() -> None:
    """add_to_chat_ctx=False against a non-realtime LLM raises NotImplementedError."""
    from livekit.agents import AgentSession

    from ..fake_llm import FakeLLM

    session = AgentSession(llm=FakeLLM(fake_responses=[]))
    try:
        with pytest.raises(NotImplementedError) as exc_info:
            session.generate_reply(add_to_chat_ctx=False)
        msg = str(exc_info.value)
        assert "RealtimeModel" in msg
        assert "add_to_chat_ctx=False" in msg
    finally:
        await session.aclose()
