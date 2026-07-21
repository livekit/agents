"""Send-side WebSocket-drop acceptance matrix for Smallest AI STT (issue #6473)."""

from __future__ import annotations

import pytest

from tests._stt_send_drop import SEND_DROP_ERRORS, run_send_drop

pytestmark = pytest.mark.plugin("smallestai")

SETUP_SENDS = 0


def _build_stream(session):
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.smallestai import STT
    from livekit.plugins.smallestai.stt import SpeechStream

    stt = STT(api_key="test-key")
    return SpeechStream(
        stt=stt,
        opts=stt._opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        http_session=session,  # smallestai SpeechStream takes no api_key
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", SEND_DROP_ERRORS)
async def test_send_drop_is_retryable(error_factory):
    from livekit.agents import APIConnectionError, APIError

    exc, _ws = await run_send_drop(_build_stream, error_factory, setup_sends=SETUP_SENDS)

    assert isinstance(exc, APIConnectionError), (
        f"expected retryable APIConnectionError, got {exc!r}"
    )
    assert isinstance(exc, APIError)


@pytest.mark.asyncio
async def test_dropped_frame_attempted_once():
    exc, ws = await run_send_drop(
        _build_stream, SEND_DROP_ERRORS[0].values[0], setup_sends=SETUP_SENDS
    )

    assert ws.audio_sends == 1
    assert exc is not None


@pytest.mark.asyncio
async def test_drop_during_shutdown_does_not_reconnect():
    from livekit.agents import APIConnectionError

    exc, _ws = await run_send_drop(
        _build_stream,
        SEND_DROP_ERRORS[0].values[0],
        setup_sends=SETUP_SENDS,
        session_closed=True,
        timeout=1.5,
    )

    assert not isinstance(exc, APIConnectionError)
