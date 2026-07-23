"""Tests for Gradium STT plugin configuration and behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests._stt_send_drop import SEND_DROP_ERRORS, run_send_drop

pytestmark = pytest.mark.plugin("gradium")


def test_stt_default_language():
    """STT defaults to English."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key")

    assert stt._opts.language == "en"


def test_stt_custom_language():
    """STT accepts custom language."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr")

    assert stt._opts.language == "fr"


def test_stream_language_overrides_stream_options():
    """stream language overrides the copied stream config."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr", http_session=MagicMock())

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream(language="de")

    assert stream._opts.language == "de"
    assert stt._opts.language == "fr"


@pytest.mark.asyncio
async def test_speech_stream_sends_language_in_json_config():
    """SpeechStream sends language in setup json_config."""
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, LanguageCode
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    sent_messages: list[str] = []

    class FakeWebSocket:
        async def send_str(self, message: str) -> None:
            sent_messages.append(message)

    class FakeSession:
        async def ws_connect(self, url, headers):
            return FakeWebSocket()

    stt = STT(api_key="test-key")
    opts = STTOptions(language=LanguageCode("pt"), temperature=0.3)

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            model_endpoint="wss://api.gradium.ai/api/speech/asr",
            model_name="default",
            http_session=FakeSession(),
        )

    await stream._connect_ws()

    setup_msg = json.loads(sent_messages[0])
    assert setup_msg["json_config"] == {"language": "pt", "temp": 0.3}


# --- send-side WebSocket-drop acceptance matrix (issue #6473) -------------------------------
# Gradium is the reference provider; the same three tests run for the other six plugins via
# the shared harness (tests/_stt_send_drop.py), each swapping only `_build_stream`.

SETUP_SENDS = 1  # gradium sends one setup payload in _connect_ws before the audio loop


def _build_stream(session):
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, LanguageCode
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    stt = STT(api_key="test-key")
    return SpeechStream(
        stt=stt,
        opts=STTOptions(language=LanguageCode("en")),
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="test-key",
        model_endpoint="wss://api.gradium.ai/api/speech/asr",
        model_name="default",
        http_session=session,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", SEND_DROP_ERRORS)
async def test_send_drop_is_retryable(error_factory):
    """Every send-side failure surfaces as a retryable APIConnectionError, never a raw error.

    The base ``SpeechStream._main_task`` only reconnects on ``APIError``. ``send_bytes``/
    ``send_str`` on a peer-closed socket raises a raw ``aiohttp`` error that is *not* an
    ``APIError`` — so before the fix a send-side drop killed the session while an identical
    recv-side drop was retryable. This pins the send path to the same retryable behavior,
    which is what lets ``_main_task`` open a fresh socket and resume the transcript.
    """
    from livekit.agents import APIConnectionError, APIError

    exc, _ws = await run_send_drop(_build_stream, error_factory, setup_sends=SETUP_SENDS)

    assert isinstance(exc, APIConnectionError), (
        f"expected retryable APIConnectionError, got {exc!r}"
    )
    assert isinstance(exc, APIError)


@pytest.mark.asyncio
async def test_dropped_frame_attempted_once():
    """The in-flight frame is attempted exactly once — no inner send-retry loop, so a loss is
    observable rather than a silent duplicate."""
    exc, ws = await run_send_drop(
        _build_stream, SEND_DROP_ERRORS[0].values[0], setup_sends=SETUP_SENDS
    )

    assert ws.audio_sends == 1
    assert exc is not None


@pytest.mark.asyncio
async def test_drop_during_shutdown_does_not_reconnect():
    """A drop while the session is closing returns quietly instead of raising a retryable
    error (which would spin a reconnect loop)."""
    from livekit.agents import APIConnectionError

    exc, _ws = await run_send_drop(
        _build_stream,
        SEND_DROP_ERRORS[0].values[0],
        setup_sends=SETUP_SENDS,
        session_closed=True,
        timeout=1.5,
    )

    assert not isinstance(exc, APIConnectionError)
