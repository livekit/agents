"""Send-side WebSocket-drop acceptance matrix for Telnyx STT (issue #6473).

Telnyx opens its own socket through a ``SessionManager`` rather than taking ``http_session``
on the stream, so the dropping session is injected via ``STT(http_session=...)``.
"""

from __future__ import annotations

import pytest

from tests._stt_send_drop import SEND_DROP_ERRORS, run_send_drop

pytestmark = pytest.mark.plugin("telnyx")

SETUP_SENDS = 0  # the WAV header is telnyx's first send_task send; the drop lands there


def _build_stream(session):
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, LanguageCode
    from livekit.plugins.telnyx import STT
    from livekit.plugins.telnyx.stt import SpeechStream

    stt = STT(api_key="test-key", http_session=session)
    return SpeechStream(
        stt=stt,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        language=LanguageCode("en"),
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


# NOTE: telnyx's quiet-return guard keys off its own `closing_ws` flag (normal end-of-input),
# not `self._session.closed`, so the session-closed shutdown case doesn't apply here.
