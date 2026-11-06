"""Tests for ``_CloudTransport`` (cloud WS body, driven by the unified
``_BaseStreamingTurnDetectorStream`` stream).

Uses the in-process ``FakeTurnDetectorWS`` to drive the transport
deterministically. Covers:

- Retry counter resets after a successful connect (so transient drops across
  the session lifetime don't accumulate toward ``max_retry``).
- All outbound messages are FIFO-ordered on the wire, even when control
  hooks fire synchronously between two awaited audio frames.
"""

from __future__ import annotations

import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from livekit import rtc
from livekit.agents._exceptions import APIConnectionError
from livekit.agents.metrics import EOTInferenceMetrics
from livekit.protocol.agent_pb.agent_inference import ServerMessage

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
        """Regression: ``run_inference`` (sync hook) used to schedule its
        send via ``asyncio.create_task``, which could land on the wire after
        an awaited ``input_audio`` send. With the unified channel, the
        sender drains FIFO so ``inference_start`` always reaches the wire
        first."""
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
        """Regression: two sync ``run_inference`` hooks back-to-back (a
        predict superseding another) used to race at the ``ws.send_bytes``
        await because each ran in its own task. The unified channel serializes
        them in call order."""
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


def _eot_prediction_message(
    request_id: str, *, sent_ms: int | None, probability: float = 0.9
) -> ServerMessage:
    """A server ``eot_prediction`` frame. When ``sent_ms`` is None the
    ``latest_client_created_at`` timestamp is left unset, exactly as the server
    may send it (an unset proto Timestamp reads back as epoch 0)."""
    msg = ServerMessage()
    msg.request_id = request_id
    prediction = msg.eot_prediction
    prediction.probability = probability
    if sent_ms is not None:
        prediction.inference_stats.latest_client_created_at.FromMilliseconds(sent_ms)
    return msg


def _emitted_eot_metrics(stream: object) -> list[EOTInferenceMetrics]:
    return [
        call.args[1]
        for call in stream._detector.emit.call_args_list  # type: ignore[attr-defined]
        if call.args and call.args[0] == "metrics_collected"
    ]


class TestCloudStreamDetectionDelay:
    """``detection_delay`` derivation from the server's client timestamp."""

    async def test_unset_client_timestamp_yields_null_detection_delay(self) -> None:
        """Regression: when the server omits ``latest_client_created_at`` the
        timestamp reads back as epoch 0, so ``now - 0`` used to report a
        detection_delay of decades. Both the resolved prediction and the emitted
        metric must report ``None`` (unknown) instead of a garbage duration."""
        stream, _fake_ws, transport = make_stream(connect_script=[None])
        try:
            fut = stream.predict()
            transport._process_message(_eot_prediction_message(stream._request_id, sent_ms=None))

            event = fut.result()
            assert event.detection_delay is None

            metrics = _emitted_eot_metrics(stream)
            assert metrics
            assert metrics[-1].detection_delay is None
        finally:
            await stream.aclose()

    async def test_known_client_timestamp_reports_real_detection_delay(self) -> None:
        """Control: with a real ``latest_client_created_at`` the delay is still a
        small positive number, so the guard does not change the correct case."""
        now = Timestamp()
        now.GetCurrentTime()
        stream, _fake_ws, transport = make_stream(connect_script=[None])
        try:
            fut = stream.predict()
            transport._process_message(
                _eot_prediction_message(stream._request_id, sent_ms=now.ToMilliseconds() - 50)
            )

            event = fut.result()
            assert event.detection_delay is not None
            assert 0.0 <= event.detection_delay < 60.0

            metrics = _emitted_eot_metrics(stream)
            assert metrics
            # The event and metric should expose the same computed delay.
            assert metrics[-1].detection_delay == event.detection_delay
            assert metrics[-1].detection_delay is not None
            assert 0.0 <= metrics[-1].detection_delay < 60.0
        finally:
            await stream.aclose()
