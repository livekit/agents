"""Unit tests for the Ultravox ``/calls`` payload, focused on ``externalVoice``.

These drive ``RealtimeSession._build_call_payload`` directly through a
``_NoNetworkSession`` whose ``_main_task`` is a no-op, so no HTTP session or
WebSocket is created. That keeps the tests fast and deterministic while still
covering the built-in-voice vs. server-side-external-TTS branch.
"""

from __future__ import annotations

from typing import Any

import pytest

from livekit.plugins.ultravox.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
)

pytestmark = pytest.mark.unit


class _NoNetworkSession(RealtimeSession):
    """A session whose main task never touches the network."""

    async def _main_task(self) -> None:  # type: ignore[override]
        return None


def _make_session(**model_kwargs: Any) -> _NoNetworkSession:
    model_kwargs.setdefault("api_key", "test-key")
    model = RealtimeModel(**model_kwargs)
    return _NoNetworkSession(realtime_model=model)


# A MiniMax-style "generic" external voice, including the literal ``{text}``
# template token Ultravox substitutes per turn and a secret-bearing header.
_GENERIC_VOICE: dict[str, Any] = {
    "generic": {
        "url": "https://api.example.com/v1/t2a",
        "headers": {"Authorization": "Bearer __LK_SECRET__MINIMAX_API_KEY__"},
        "body": {"text": "{text}", "model": "speech-2.5-turbo-preview"},
        "responseSampleRate": 24000,
    }
}


async def test_voice_used_when_no_external_voice() -> None:
    payload = _make_session(voice="Aurora")._build_call_payload()

    assert payload["voice"] == "Aurora"
    assert "externalVoice" not in payload


async def test_external_voice_replaces_voice() -> None:
    payload = _make_session(external_voice=_GENERIC_VOICE)._build_call_payload()

    assert "voice" not in payload
    assert payload["externalVoice"] == _GENERIC_VOICE


async def test_external_voice_passes_through_verbatim() -> None:
    payload = _make_session(external_voice=_GENERIC_VOICE)._build_call_payload()

    external = payload["externalVoice"]
    # The blob is opaque to the plugin: every field, including the literal
    # ``{text}`` template token and the credential header, must survive untouched.
    assert external["generic"]["body"]["text"] == "{text}"
    assert (
        external["generic"]["headers"]["Authorization"] == "Bearer __LK_SECRET__MINIMAX_API_KEY__"
    )
    assert external is _GENERIC_VOICE


async def test_core_payload_unchanged_either_way() -> None:
    common = {"model": "fixie-ai/ultravox", "system_prompt": "Be helpful."}
    native = _make_session(voice="Mark", **common)._build_call_payload()
    external = _make_session(external_voice=_GENERIC_VOICE, **common)._build_call_payload()

    for payload in (native, external):
        assert payload["systemPrompt"] == "Be helpful."
        assert payload["model"] == "fixie-ai/ultravox"
        assert payload["medium"] == {
            "serverWebSocket": {
                "inputSampleRate": 16000,
                "outputSampleRate": 24000,
                "clientBufferSizeMs": 30000,
            }
        }
        assert payload["selectedTools"] == []

    # The two payloads differ only in the voice axis.
    native.pop("voice")
    external.pop("externalVoice")
    assert native == external
