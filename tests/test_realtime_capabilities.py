"""Unit tests for RealtimeCapabilities capability advertisement.

Tests that:
- OpenAI plugin declares both supports_say=True and ephemeral_say=True.
- Phonic plugin declares supports_say=True and ephemeral_say=False (default).

No network calls are made. API keys are set to dummy values via monkeypatch
to satisfy plugin constructors without triggering any remote connections.
"""

from __future__ import annotations

import inspect

import pytest


def test_openai_realtime_capability_advertises_ephemeral_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI plugin's RealtimeCapabilities declares ephemeral_say=True."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-not-real")

    from livekit.plugins import openai  # noqa: PLC0415

    model = openai.realtime.RealtimeModel()
    assert model.capabilities.ephemeral_say is True
    assert model.capabilities.supports_say is True


def test_phonic_realtime_capability_does_not_advertise_ephemeral_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phonic plugin's RealtimeCapabilities declares ephemeral_say=False.

    Phonic's substrate is strictly turn-based; it has no out-of-band
    response primitive equivalent to OpenAI's response.create(conversation: "none").
    """
    pytest.importorskip(
        "livekit.plugins.phonic",
        reason="livekit-plugins-phonic not installed in this environment",
    )
    monkeypatch.setenv("PHONIC_API_KEY", "phonic-test-fake")

    from livekit.plugins.phonic.realtime import (
        RealtimeModel as PhonicRealtimeModel,  # noqa: PLC0415
    )

    model = PhonicRealtimeModel()
    assert model.capabilities.ephemeral_say is False
    assert model.capabilities.supports_say is True


def test_phonic_say_signature_accepts_add_to_chat_ctx() -> None:
    """Phonic's say() signature accepts add_to_chat_ctx as a keyword-only kwarg.

    The dispatcher (AgentActivity.say) emits a DeprecationWarning before reaching
    Phonic's say() because Phonic declares ephemeral_say=False. The kwarg is
    accepted here for forward-compatible signature alignment with the abstract base.
    """
    pytest.importorskip(
        "livekit.plugins.phonic",
        reason="livekit-plugins-phonic not installed in this environment",
    )

    from livekit.plugins.phonic.realtime.realtime_model import (  # noqa: PLC0415
        RealtimeSession as PhonicRealtimeSession,
    )

    sig = inspect.signature(PhonicRealtimeSession.say)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.default is True
    assert param.kind is inspect.Parameter.KEYWORD_ONLY

    # Verify the full argument-binding rules accept the kwarg.
    _SELF_SENTINEL = object()
    bound = sig.bind(_SELF_SENTINEL, "hello", add_to_chat_ctx=False)
    assert bound.arguments["add_to_chat_ctx"] is False


def _make_openai_session_for_capture(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
):
    """Construct an OpenAI RealtimeSession bypassing network setup.

    Bare-construction pattern: create the instance via __new__ and seed only
    the attributes that say() reads. Replaces send_event with a list-append
    capture helper to record outbound events without sending. Avoids calling
    the heavyweight __init__ which spawns websocket tasks.

    Returns (session, captured_events_list) — the caller inspects the list to
    assert outbound event shape.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-not-real")
    from livekit.plugins.openai.realtime.realtime_model import RealtimeSession  # noqa: PLC0415

    session = RealtimeSession.__new__(RealtimeSession)
    session._response_created_futures = {}  # type: ignore[attr-defined]
    captured: list[object] = []
    session.send_event = captured.append  # type: ignore[method-assign]
    return session, captured


def test_openai_realtime_say_sends_response_create_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI say() sends a response.create event with assistant message + metadata."""
    session, captured = _make_openai_session_for_capture(monkeypatch)

    fut = session.say("the verification token alpha-bravo")

    assert len(captured) == 1
    sent_event = captured[0]
    assert sent_event.type == "response.create"  # type: ignore[attr-defined]
    assert sent_event.event_id.startswith("response_create_say_")  # type: ignore[attr-defined]
    assert sent_event.response.metadata["client_event_id"] == sent_event.event_id  # type: ignore[attr-defined]
    # Assistant message with text content
    assert sent_event.response.input is not None  # type: ignore[attr-defined]
    assert len(sent_event.response.input) == 1  # type: ignore[attr-defined]
    item = sent_event.response.input[0]  # type: ignore[attr-defined]
    assert item.type == "message"
    assert item.role == "assistant"
    assert len(item.content) == 1
    assert item.content[0].type == "output_text"
    assert item.content[0].text == "the verification token alpha-bravo"
    # Default add_to_chat_ctx=True does NOT set conversation
    assert sent_event.response.conversation is None  # type: ignore[attr-defined]

    # Cleanup the future to avoid pending task warnings
    fut.cancel()


def test_openai_realtime_say_isolation_sets_conversation_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI say(add_to_chat_ctx=False) sets conversation: 'none' on response.create."""
    session, captured = _make_openai_session_for_capture(monkeypatch)

    fut = session.say("purple-elephant-42", add_to_chat_ctx=False)

    sent_event = captured[0]
    assert sent_event.response.conversation == "none"  # type: ignore[attr-defined]

    # Cleanup
    fut.cancel()
