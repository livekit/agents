"""The cloud turn-detector transport, driven over an in-process fake websocket."""

from __future__ import annotations

import pytest

from livekit import rtc
from livekit.agents._exceptions import APIConnectionError

from .fake_turn_detector_ws import (
    drain_send_queue,
    make_stream,
    wait_until_connected,
)

pytestmark = pytest.mark.audio_eot


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
        """A successful connect resets the transient-failure counter."""
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
        """run_inference reaches the wire after an earlier input_audio send."""
        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            stream.predict()
            stream.push_audio(_pcm_frame())
            await drain_send_queue(transport)

            kinds = [m.WhichOneof("message") for m in fake_ws.sent]
            inference_start_idx = kinds.index("inference_start")
            input_audio_idx = kinds.index("input_audio")
            assert inference_start_idx < input_audio_idx
        finally:
            await stream.aclose()

    async def test_consecutive_inference_starts_serialized(self) -> None:
        """Two back-to-back run_inference calls keep their order on the wire."""
        stream, fake_ws, transport = make_stream(connect_script=[None])
        try:
            await wait_until_connected(transport)
            stream.predict()
            first_id = stream._request_id
            stream.predict()
            second_id = stream._request_id
            await drain_send_queue(transport)

            start_ids = [
                m.inference_start.request_id
                for m in fake_ws.sent
                if m.WhichOneof("message") == "inference_start"
            ]
            assert start_ids == [first_id, second_id]
        finally:
            await stream.aclose()
