from __future__ import annotations

import pytest

from livekit.agents.llm import RealtimeError
from livekit.agents.voice.speech_handle import InputDetails, SpeechHandle

pytestmark = pytest.mark.unit


async def test_realtime_error_propagates_through_speech_handle() -> None:
    """When a RealtimeModel times out, the RealtimeError propagates through SpeechHandle.

    The voice pipeline (agent_activity._realtime_reply_task) calls
    ``speech_handle._mark_done(error=e)`` when it catches a RealtimeError from
    the underlying realtime model.  This test verifies that the error is
    properly surfaced through ``await handle``.

    Regression test for https://github.com/livekit/agents/issues/6224
    """
    handle = SpeechHandle(
        speech_id="test-realtime-error-propagation",
        allow_interruptions=False,
        input_details=InputDetails(modality="text"),
    )
    handle._mark_done(error=RealtimeError("generate_reply timed out."))

    with pytest.raises(RealtimeError, match="generate_reply timed out"):
        await handle


async def test_speech_handle_mark_done_no_error() -> None:
    """SpeechHandle._mark_done() without error completes normally (smoke test)."""
    handle = SpeechHandle(
        speech_id="test-mark-done-no-error",
        allow_interruptions=False,
        input_details=InputDetails(modality="text"),
    )
    handle._mark_done()
    # Should complete without raising
    result = await handle
    assert result is handle
