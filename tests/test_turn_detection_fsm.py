"""Inference-request lifecycle tests for ``_BaseStreamingTurnDetectorStream``.

The stream is a thin transport-facing surface: per-request state is one
(request_id, future) pair. ``predict`` starts a request and returns its
future, superseding any previous request; the transport's single prediction
completes the request by resolving the future; ``cancel_inference``/``flush`` close
a pending request, resolving its future with a default event so waiters never
see ``CancelledError``. All policy (when to start a request, await timeout,
turn commits) lives in ``AudioRecognition`` and is covered by
``test_audio_recognition_turn_detection``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents.inference.eot.base import (
    TurnDetectorOptions,
    _BaseStreamingTurnDetectorStream,
)
from livekit.agents.inference.eot.languages import ThresholdOptions
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.audio_eot


class _FakeTransport:
    """In-memory transport that records what the stream tells it to do."""

    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []
        self._stream: _BaseStreamingTurnDetectorStream | None = None

    def attach(self, stream: _BaseStreamingTurnDetectorStream) -> None:
        self._stream = stream

    async def run(self) -> None:
        assert self._stream is not None
        await self._stream._drain_audio_channel()

    def run_inference(self, request_id: str) -> None:
        self.events.append(("run_inference", request_id))

    def push_frame(self, frame: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def detach(self) -> None:
        pass


class _FakeBackend(_BaseStreamingTurnDetectorStream):
    """Stream + fake transport bundled for request-lifecycle testing,
    exposing the transport's recorded events."""

    def __init__(self, *, detector: Any, opts: TurnDetectorOptions) -> None:
        self._fake_transport = _FakeTransport()
        super().__init__(detector=detector, opts=opts, transport=self._fake_transport)

    @property
    def events(self) -> list[tuple[str, ...]]:
        return self._fake_transport.events

    def simulate_prediction(self, request_id: str, probability: float) -> None:
        """Mirror what a transport would do: hand the prediction to the stream."""
        self._resolve_prediction(request_id, probability)


def _make_opts(thresholds: dict[str, float] | None = None) -> TurnDetectorOptions:
    # Seed the resolved thresholds via a local-model dict override so ``lookup`` returns them.
    overrides = thresholds if thresholds is not None else NOT_GIVEN
    return TurnDetectorOptions(
        sample_rate=16000, thresholds=ThresholdOptions("turn-detector-v1-mini", overrides)
    )


def _make_stream(thresholds: dict[str, float] | None = None) -> _FakeBackend:
    return _FakeBackend(detector=MagicMock(), opts=_make_opts(thresholds))


class TestAudioTurnDetectionRequests:
    """Request lifecycle and hook dispatch for ``_BaseStreamingTurnDetectorStream``."""

    async def test_predict_starts_inference(self) -> None:
        s = _make_stream()
        try:
            fut = s.predict()
            assert s._request_id is not None
            assert not fut.done()
            assert s.events == [("run_inference", s._request_id)]
        finally:
            await s.aclose()

    async def test_predict_supersedes_previous_request(self) -> None:
        """A second predict closes the previous request (pending future
        resolves with the 0.0 default) and starts a fresh one."""
        s = _make_stream()
        try:
            old_fut = s.predict()
            old_id = s._request_id
            new_fut = s.predict()

            assert new_fut is not old_fut
            assert s._request_id != old_id
            assert old_fut.done()
            assert old_fut.result().end_of_turn_probability == 0.0
            assert sum(1 for e in s.events if e[0] == "run_inference") == 2
        finally:
            await s.aclose()

    async def test_cancel_inference_closes_request(self) -> None:
        s = _make_stream()
        try:
            fut = s.predict()
            s.cancel_inference()

            assert s._request_id is None
            assert fut.done()
            assert fut.result().end_of_turn_probability == 0.0
        finally:
            await s.aclose()

    async def test_cancel_inference_when_idle_is_noop(self) -> None:
        s = _make_stream()
        try:
            s.cancel_inference()
            assert s.events == []
        finally:
            await s.aclose()

    async def test_late_prediction_after_cancel_inference_dropped(self) -> None:
        """Regression: a late prediction for a closed request id must be
        dropped (request-id mismatch), not leak into the next request."""
        s = _make_stream()
        try:
            fut = s.predict()
            cancelled_request_id = s._request_id
            assert cancelled_request_id is not None

            s.cancel_inference()
            s.simulate_prediction(cancelled_request_id, probability=0.9)
            assert fut.result().end_of_turn_probability == 0.0  # cancel_inference default, not 0.9

            next_fut = s.predict()
            assert next_fut is not fut
            assert not next_fut.done()
            assert sum(1 for e in s.events if e[0] == "run_inference") == 2
        finally:
            await s.aclose()

    async def test_prediction_completes_request(self) -> None:
        """The transport's single prediction resolves the future and completes
        the request — nothing left to close on cancel_inference/flush."""
        s = _make_stream()
        try:
            fut = s.predict()
            request_id = s._request_id
            assert request_id is not None

            s.simulate_prediction(request_id, probability=0.3)
            assert fut.done()
            assert fut.result().end_of_turn_probability == 0.3
            assert s._request_id is None
        finally:
            await s.aclose()

    async def test_flush_closes_request(self) -> None:
        s = _make_stream()
        try:
            fut = s.predict()
            s.flush(reason="turn committed")
            assert s._request_id is None
            assert fut.result().end_of_turn_probability == 0.0
        finally:
            await s.aclose()

    async def test_flush_does_not_overwrite_resolved_prediction(self) -> None:
        s = _make_stream()
        try:
            fut = s.predict()
            request_id = s._request_id
            assert request_id is not None
            s.simulate_prediction(request_id, probability=0.7)

            s.flush(reason="turn committed")
            assert fut.result().end_of_turn_probability == 0.7
            assert s._request_id is None
        finally:
            await s.aclose()

    async def test_predict_after_end_input_returns_resolved_default(self) -> None:
        s = _make_stream()
        try:
            s.end_input()
            fut = s.predict()
            assert fut.done()
            assert fut.result().end_of_turn_probability == 1.0
            assert not any(e[0] == "run_inference" for e in s.events)
        finally:
            await s.aclose()

    async def test_aclose_resolves_pending_future(self) -> None:
        s = _make_stream()
        fut = s.predict()
        await s.aclose()
        assert fut.done()
        assert fut.result().end_of_turn_probability == 0.0

    async def test_timed_out_cancel_inference_local_model_no_fallback(self) -> None:
        """``timed_out=True`` only promotes the local fallback for the cloud
        model; the mini model just closes the request (the cloud case is
        covered in test_audio_turn_detector_fallback)."""
        s = _make_stream()
        try:
            fut = s.predict()
            s.cancel_inference(timed_out=True)
            assert fut.result().end_of_turn_probability == 0.0
            assert s.is_fallback is False
            assert s.model == "turn-detector-v1-mini"
        finally:
            await s.aclose()
