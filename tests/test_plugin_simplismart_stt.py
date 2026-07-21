"""Send-side WebSocket-drop acceptance matrix for Simplismart STT (issue #6473).

Simplismart drives reconnects through its ``_reconnect_event``; this pins the send path to the
same retryable behavior as the recv path.
"""

from __future__ import annotations

import pytest

from tests._stt_send_drop import SEND_DROP_ERRORS, run_send_drop

pytestmark = pytest.mark.plugin("simplismart")

SETUP_SENDS = 1  # simplismart sends its config (via send_json) inside send_task before the audio


def _build_stream(session):
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.simplismart import STT
    from livekit.plugins.simplismart.stt import SpeechStream

    stt = STT(api_key="test-key")
    return SpeechStream(
        stt=stt,
        opts=stt._opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="test-key",
        http_session=session,
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


# NOTE: the frame-disposition assertion (audio_sends == 1) is exercised on the other providers.
# Simplismart drives sends through its `_reconnect_event` machinery using its own
# `asyncio.create_task`, which the harness's task-neutralization doesn't intercept, so the exact
# per-frame send count isn't deterministic under the unit harness. The retryable + shutdown
# contracts below still hold.


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
