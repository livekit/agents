"""Send-side WebSocket-drop acceptance matrix for SLNG STT (issue #6473).

SLNG deliberately raises ``APIStatusError`` (not ``APIConnectionError``) on a send-side drop so
it matches its recv-side failover path — both subclass ``APIError``, so ``_main_task`` retries
either. SLNG also reconnects *inside* ``_run`` (immediate same-endpoint retry, then multi-endpoint
failover), so the harness runs in ``drop_on_bytes`` mode: every reconnection's handshake
(``send_str``) succeeds and only the audio (``send_bytes``) drops.
"""

from __future__ import annotations

import pytest

from tests._stt_send_drop import SEND_DROP_ERRORS, run_send_drop

pytestmark = pytest.mark.plugin("slng")


def _build_stream(session):
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.slng import STT
    from livekit.plugins.slng.stt import SpeechStream

    stt = STT(api_key="test-key")
    return SpeechStream(
        stt=stt,
        opts=stt._opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="test-key",
        region_override_header=None,
        model_endpoints=["wss://fake.slng/asr"],
        models=[None],
        active_endpoint_index=0,
        model_options={},
        http_session=session,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", SEND_DROP_ERRORS)
async def test_send_drop_is_retryable(error_factory):
    """slng surfaces the drop as a retryable APIStatusError (matches its recv failover)."""
    from livekit.agents import APIError, APIStatusError

    exc, _ws = await run_send_drop(_build_stream, error_factory, drop_on_bytes=True)

    assert isinstance(exc, APIStatusError), f"expected retryable APIStatusError, got {exc!r}"
    assert isinstance(exc, APIError)  # -> _main_task reconnects (bounded)


@pytest.mark.asyncio
async def test_send_drop_is_attempted():
    """The audio send is attempted (and surfaced) rather than silently swallowed."""
    exc, ws = await run_send_drop(_build_stream, SEND_DROP_ERRORS[0].values[0], drop_on_bytes=True)

    assert (
        ws.audio_sends >= 1
    )  # slng advances a frame per same-endpoint retry (no duplicate replay)
    assert exc is not None


@pytest.mark.asyncio
async def test_drop_during_shutdown_does_not_reconnect():
    """A drop while the session is closing must not surface as a retryable error."""
    from livekit.agents import APIStatusError

    exc, _ws = await run_send_drop(
        _build_stream,
        SEND_DROP_ERRORS[0].values[0],
        drop_on_bytes=True,
        session_closed=True,
        timeout=1.5,
    )

    assert not isinstance(exc, APIStatusError)
