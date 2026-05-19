"""Tests for ``_CloudTransport`` (cloud WS body, driven by the unified
``_AudioTurnDetectorStreamImpl`` stream).

Uses the in-process ``FakeTurnDetectorWS`` to drive the transport
deterministically. Covers:

- Retry counter resets after a successful connect (so transient drops across
  the session lifetime don't accumulate toward ``max_retry``).
- All outbound messages are FIFO-ordered on the wire, even when control
  hooks fire synchronously between two awaited audio frames.
"""

from __future__ import annotations

from livekit import rtc
from livekit.agents._exceptions import APIConnectionError

from .fake_turn_detector_ws import (
    drain_send_queue,
    make_stream,
    wait_until_connected,
)


def _pcm_frame(samples: int = 320) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples,
    )


class TestCloudStreamRetry:
    """``_num_retries`` lifecycle across reconnects."""

    async def test_num_retries_resets_after_successful_connect(self) -> None:
        """Regression: after a successful WS connect, transient-failure
        counters must reset so transient drops across the session lifetime
        don't accumulate toward ``max_retry``."""
        stream, _fake_ws, transport = make_stream(
            connect_script=[APIConnectionError("transient", retryable=True), None],
            max_retry=3,
            retry_interval=0.0,
        )
        try:
            await wait_until_connected(transport)
            # Two attempts total: first raised (counter went 0 → 1), second
            # succeeded and must have reset to 0.
            assert transport._connect_calls == 2
            assert transport._num_retries == 0
        finally:
            await stream.aclose()


class TestCloudStreamSendOrdering:
    """FIFO delivery for the unified outbound channel."""

    async def test_inference_start_precedes_input_audio(self) -> None:
        """Regression: ``on_warmup_start`` (sync hook) used to schedule its
        send via ``asyncio.create_task``, which could land on the wire after
        an awaited ``input_audio`` send. With the unified channel, the
        sender drains FIFO so ``inference_start`` always reaches the wire
        first."""
        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            stream.warmup()
            stream.push_audio(_pcm_frame())
            await drain_send_queue(transport)

            kinds = [m.WhichOneof("message") for m in fake_ws.sent]
            inference_start_idx = kinds.index("inference_start")
            input_audio_idx = kinds.index("input_audio")
            assert inference_start_idx < input_audio_idx
        finally:
            await stream.aclose()

    async def test_inference_start_precedes_inference_stop(self) -> None:
        """Regression: two sync hooks back-to-back (``on_warmup_start`` then
        ``on_inference_stop``) used to race at the ``ws.send_bytes`` await
        because each ran in its own task. The unified channel serializes
        them in call order."""
        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            stream.warmup()
            stream.deactivate(trigger="vad sos")
            await drain_send_queue(transport)

            kinds = [m.WhichOneof("message") for m in fake_ws.sent]
            start_idx = kinds.index("inference_start")
            stop_idx = kinds.index("inference_stop")
            assert start_idx < stop_idx
        finally:
            await stream.aclose()
