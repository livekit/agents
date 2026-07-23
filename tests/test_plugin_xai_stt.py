"""Send-side WebSocket-drop acceptance matrix for xAI STT (issue #6473)."""

from __future__ import annotations

import pytest

from tests._stt_send_drop import SEND_DROP_ERRORS, run_send_drop

pytestmark = pytest.mark.plugin("xai")

SETUP_SENDS = 0  # xai sends no handshake in _connect_ws; the first send_task send is audio
# xai's send_task waits for a server "ready" (transcript.created) delivered over recv before
# it sends audio; feed one so the send path unblocks.
READY = ['{"type": "transcript.created"}']


def _build_stream(session):
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.xai import STT
    from livekit.plugins.xai.stt import SpeechStream

    stt = STT(api_key="test-key")
    return SpeechStream(
        stt=stt,
        opts=stt._opts,  # the STT's fully-built options (the dataclass has no defaults)
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="test-key",
        http_session=session,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", SEND_DROP_ERRORS)
async def test_send_drop_is_retryable(error_factory):
    """Each raw send-side failure surfaces as a retryable APIConnectionError, never a raw error."""
    from livekit.agents import APIConnectionError, APIError

    exc, _ws = await run_send_drop(
        _build_stream, error_factory, setup_sends=SETUP_SENDS, ready_messages=READY
    )

    assert isinstance(exc, APIConnectionError), (
        f"expected retryable APIConnectionError, got {exc!r}"
    )
    assert isinstance(exc, APIError)  # -> _main_task reconnects (bounded) -> transcript resumes


@pytest.mark.asyncio
async def test_dropped_frame_attempted_once():
    """The in-flight frame is attempted exactly once — no inner send-retry loop."""
    exc, ws = await run_send_drop(
        _build_stream, SEND_DROP_ERRORS[0].values[0], setup_sends=SETUP_SENDS, ready_messages=READY
    )

    assert ws.audio_sends == 1
    assert exc is not None


@pytest.mark.asyncio
async def test_drop_during_shutdown_does_not_reconnect():
    """A drop while the session is closing returns quietly instead of raising a retryable error."""
    from livekit.agents import APIConnectionError

    exc, _ws = await run_send_drop(
        _build_stream,
        SEND_DROP_ERRORS[0].values[0],
        setup_sends=SETUP_SENDS,
        ready_messages=READY,
        session_closed=True,
        timeout=1.5,
    )

    assert not isinstance(exc, APIConnectionError)
