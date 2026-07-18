"""agent_context carryover for LiveKit Inference STT.

Mirrors the AssemblyAI plugin's carryover behavior (see
tests/test_plugin_assemblyai_stt.py), gated to the AssemblyAI Universal-3 Pro
family models that support it, and applied through the inference ``extra`` path.
"""

from __future__ import annotations

import logging

import pytest

from livekit.agents.inference.stt import STT
from livekit.agents.llm import AgentHandoff, ChatMessage
from livekit.agents.voice.events import ConversationItemAddedEvent

pytestmark = pytest.mark.unit


def _make_stt(**kwargs) -> STT:
    defaults = {
        "model": "assemblyai/universal-3-5-pro",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": "https://example.livekit.cloud",
    }
    defaults.update(kwargs)
    return STT(**defaults)


def _assistant_item_event(text: str) -> ConversationItemAddedEvent:
    return ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=[text]))


# -- capability gating --


def test_carryover_defaults_on_for_u3_pro_family():
    """agent_context_carryover defaults to enabled on models that support it."""
    for model in ("assemblyai/u3-rt-pro", "assemblyai/universal-3-5-pro"):
        stt = _make_stt(model=model)
        assert stt.capabilities.chat_context is True


def test_carryover_defaults_off_for_unsupported_models_without_warning(caplog):
    """On non-U3-Pro models the default is silently off — no 'ignoring' warning."""
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(model="assemblyai/universal-streaming")

    assert stt.capabilities.chat_context is False
    assert "agent_context_carryover" not in caplog.text


def test_carryover_off_for_non_assemblyai_models():
    """Providers other than AssemblyAI don't carry agent_context."""
    stt = _make_stt(model="deepgram/nova-3")
    assert stt.capabilities.chat_context is False


def test_carryover_explicit_true_on_unsupported_model_warns(caplog):
    """Explicitly enabling carryover on an unsupported model warns and is ignored."""
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(model="assemblyai/universal-streaming", agent_context_carryover=True)

    assert stt.capabilities.chat_context is False
    assert "agent_context_carryover" in caplog.text


def test_carryover_explicit_false_disables():
    """agent_context_carryover=False opts out on a supported model."""
    stt = _make_stt(agent_context_carryover=False)
    assert stt.capabilities.chat_context is False


def test_carryover_disabled_by_previous_context_n_turns_zero():
    """previous_context_n_turns=0 (documented 'disable carryover') suppresses the default."""
    stt = _make_stt(extra_kwargs={"previous_context_n_turns": 0})
    assert stt.capabilities.chat_context is False


def test_carryover_explicit_true_wins_over_n_turns_zero():
    """An explicit agent_context_carryover=True overrides the n_turns=0 suppression."""
    stt = _make_stt(extra_kwargs={"previous_context_n_turns": 0}, agent_context_carryover=True)
    assert stt.capabilities.chat_context is True


# -- _push_conversation_item (native carryover sink) --


def test_carryover_forwards_short_reply_verbatim():
    """_push_conversation_item forwards assistant text within the cap unchanged."""
    stt = _make_stt()
    stt._push_conversation_item(_assistant_item_event("Your room is booked for Tuesday."))
    assert stt._opts.extra_kwargs.get("agent_context") == "Your room is booked for Tuesday."


def test_carryover_truncates_oversize_reply_keeping_tail():
    """_push_conversation_item truncates oversize replies to the last 1750 chars."""
    text = "a" * 2000 + "b" * 1750
    stt = _make_stt()
    stt._push_conversation_item(_assistant_item_event(text))
    assert stt._opts.extra_kwargs.get("agent_context") == "b" * 1750


def test_carryover_ignores_non_assistant_items():
    """User messages and textless assistant items are ignored."""
    stt = _make_stt()

    stt._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="user", content=["hi there"]))
    )
    assert stt._opts.extra_kwargs.get("agent_context") is None

    stt._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=[]))
    )
    assert stt._opts.extra_kwargs.get("agent_context") is None


def test_carryover_ignores_agent_handoff_items():
    """Non-message items (e.g. AgentHandoff) have `.type != 'message'` and are ignored."""
    stt = _make_stt()
    stt._push_conversation_item(
        ConversationItemAddedEvent(item=AgentHandoff(new_agent_id="agent-2"))
    )
    assert stt._opts.extra_kwargs.get("agent_context") is None


def test_agent_context_set_via_extra_kwargs():
    """An explicit agent_context in extra_kwargs is preserved (and later overwritten by carryover)."""
    stt = _make_stt(extra_kwargs={"agent_context": "The agent asked for a booking date."})
    assert stt._opts.extra_kwargs.get("agent_context") == "The agent asked for a booking date."

    stt._push_conversation_item(_assistant_item_event("And your zip code?"))
    assert stt._opts.extra_kwargs.get("agent_context") == "And your zip code?"
