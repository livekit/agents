"""Send-side WebSocket-drop acceptance matrix for Fireworks AI STT (issue #6473)."""

from __future__ import annotations

import pytest

from tests._stt_send_drop import SEND_DROP_ERRORS, run_send_drop

pytestmark = pytest.mark.plugin("fireworksai")

SETUP_SENDS = 0


def _build_stream(session):
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.fireworksai import STT
    from livekit.plugins.fireworksai.stt import SpeechStream

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


@pytest.mark.asyncio
async def test_dropped_frame_attempted_once():
    exc, ws = await run_send_drop(
        _build_stream, SEND_DROP_ERRORS[0].values[0], setup_sends=SETUP_SENDS
    )

    assert ws.audio_sends == 1
    assert exc is not None


# NOTE: fireworksai's quiet-return guard keys off its own `closing_ws` flag (normal end-of-input),
# not `self._session.closed`, so the session-closed shutdown case doesn't apply here — the
# no-reconnect-on-close behavior is exercised by the normal shutdown path.
