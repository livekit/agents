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
