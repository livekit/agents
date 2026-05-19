"""FSM tests for ``_AudioTurnDetectorStream``.

Covers the warmup → activate → deactivate / flush lifecycle and the
regression cases:

- ``deactivate()`` from a pre-active state (the historical ``set_active(False)``
  during WARMING_UP) must stop the inference cleanly so a late prediction
  for the cancelled request isn't replayed on the next activate.
- ``predict_end_of_turn`` timeout must leave the FSM consistent so the next
  ``warmup()`` can proceed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.turn import (
    TurnDetectionEvent,
    TurnDetectorOptions,
    _AudioTurnDetectorStream,
    _Status,
)


class _FakeTransport:
    """In-memory transport that records what the stream tells it to do."""

    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []
        self._stream: _AudioTurnDetectorStream | None = None

    def bind(self, stream: _AudioTurnDetectorStream) -> None:
        self._stream = stream

    async def run(self) -> None:
        assert self._stream is not None
        await self._stream._drain_audio_channel()

    def transport_ready(self) -> bool:
        return True

    def start_inference(self, request_id: str) -> None:
        self.events.append(("start_inference", request_id))

    def stop_inference(self, *, reason: str | None) -> None:
        self.events.append(("stop_inference", reason or ""))

    async def push_frame(self, frame: Any) -> None:
        pass

    async def flush(self, sentinel: Any) -> None:
        pass

    def detach(self) -> None:
        pass


class _FakeBackend(_AudioTurnDetectorStream):
    """Stream + fake transport bundled for FSM testing. Wraps the events
    list from the transport plus an `emitted` list of outbound probabilities."""

    def __init__(self, *, detector: Any, opts: TurnDetectorOptions) -> None:
        self._fake_transport = _FakeTransport()
        self.emitted: list[float] = []
        super().__init__(detector=detector, opts=opts, transport=self._fake_transport)

    @property
    def events(self) -> list[tuple[str, ...]]:
        return self._fake_transport.events

    def _emit_event(self, event: TurnDetectionEvent) -> None:
        self.emitted.append(event.end_of_turn_probability)
        super()._emit_event(event)

    def simulate_prediction(self, request_id: str, probability: float) -> None:
        """Mirror what a transport would do: hand the prediction to the stream."""
        self._handle_prediction(request_id, probability)


def _make_opts() -> TurnDetectorOptions:
    return TurnDetectorOptions(
        sample_rate=16000,
        base_url="",
        api_key="",
        api_secret="",
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    )


def _make_stream() -> _FakeBackend:
    return _FakeBackend(detector=MagicMock(), opts=_make_opts())


class TestAudioTurnDetectionFSM:
    """State transitions and hook dispatch for ``_AudioTurnDetectorStream``."""

    async def test_warmup_starts_inference(self) -> None:
        s = _make_stream()
        try:
            fut = s.warmup()
            assert s._status == _Status.IDLE
            assert s.is_inference_running
            assert s._preemptive_request_id is not None
            assert not fut.done()
            assert s.events == [("start_inference", s._preemptive_request_id)]
        finally:
            await s.aclose()

    async def test_warmup_is_idempotent(self) -> None:
        s = _make_stream()
        try:
            s.warmup()
            first_id = s._preemptive_request_id
            s.warmup()
            assert s._preemptive_request_id == first_id
            assert sum(1 for e in s.events if e[0] == "start_inference") == 1
        finally:
            await s.aclose()

    async def test_activate_from_warmed_up(self) -> None:
        s = _make_stream()
        try:
            s.warmup()
            s.activate(trigger="vad eos")
            assert s._status == _Status.ACTIVE
            assert s.is_inference_running
        finally:
            await s.aclose()

    async def test_activate_without_warmup_auto_warmsup(self) -> None:
        s = _make_stream()
        try:
            s.activate(trigger="manual")
            assert s._status == _Status.ACTIVE
            warmups = [e for e in s.events if e[0] == "start_inference"]
            assert len(warmups) == 1
        finally:
            await s.aclose()

    async def test_deactivate_during_preemptive_phase_stops_inference(self) -> None:
        """Regression: previously ``set_active(False)`` no-op'd while warming
        up because ``is_active`` only matched ACTIVE; the warmup kept running
        and a late prediction got cached for replay on the next activate."""
        s = _make_stream()
        try:
            s.warmup()
            s.deactivate(trigger="vad sos")

            assert s._status == _Status.IDLE
            assert s._preemptive_request_id is None
            assert not s.is_inference_running
            assert ("stop_inference", "vad sos") in s.events
        finally:
            await s.aclose()

    async def test_late_prediction_after_deactivate_not_replayed(self) -> None:
        """Regression: a late prediction for the cancelled request id must be
        dropped, not stashed for replay by the next activate()."""
        s = _make_stream()
        try:
            s.warmup()
            cancelled_request_id = s._preemptive_request_id
            assert cancelled_request_id is not None

            s.deactivate(trigger="vad sos")

            s.simulate_prediction(cancelled_request_id, probability=0.9)
            assert s._preemptive_prediction is None

            s.warmup()
            s.activate(trigger="vad eos")
            assert s.emitted == []
        finally:
            await s.aclose()

    async def test_deactivate_when_idle_is_noop(self) -> None:
        s = _make_stream()
        try:
            s.deactivate(trigger="vad sos")
            assert s.events == []
            assert s._status == _Status.IDLE
        finally:
            await s.aclose()

    async def test_deactivate_during_warmup_resolves_future_with_zero(self) -> None:
        s = _make_stream()
        try:
            fut = s.warmup()
            s.deactivate()
            assert fut.done()
            assert fut.result() == 0.0
            assert s._status == _Status.IDLE
            assert s._preemptive_request_id is None
        finally:
            await s.aclose()

    async def test_predict_end_of_turn_timeout_leaves_fsm_consistent(self) -> None:
        """Regression: timeout previously cleared the fut/id but left status
        at ACTIVE, an inconsistent state that the next ``warmup()`` couldn't
        recover from."""
        s = _make_stream()
        try:
            prob = await s.predict_end_of_turn(timeout=0.01)
            assert prob == 1.0
            assert s._status == _Status.IDLE
            assert s._preemptive_request_id is None
            assert s._preemptive_request_fut is None
            assert ("stop_inference", "predict_end_of_turn timeout") in s.events
        finally:
            await s.aclose()

    async def test_predict_end_of_turn_timeout_allows_next_warmup(self) -> None:
        s = _make_stream()
        try:
            await s.predict_end_of_turn(timeout=0.01)
            fut = s.warmup()
            assert s._preemptive_request_id is not None
            assert not fut.done()
        finally:
            await s.aclose()

    async def test_flush_deactivates_and_emits_inference_stop(self) -> None:
        s = _make_stream()
        try:
            s.warmup()
            s.activate()
            s.flush(reason="turn committed")
            assert s._status == _Status.IDLE
            assert not s.is_inference_running
            assert ("stop_inference", "turn committed") in s.events
        finally:
            await s.aclose()

    async def test_activate_releases_preemptive_prediction(self) -> None:
        """Predictions arriving before activate are held and released on activate."""
        s = _make_stream()
        try:
            s.warmup()
            request_id = s._preemptive_request_id
            assert request_id is not None
            s.simulate_prediction(request_id, probability=0.7)
            # Held — not emitted yet.
            assert s.emitted == []
            assert s._preemptive_prediction is not None
            assert s._preemptive_prediction.end_of_turn_probability == 0.7

            s.activate()
            assert s.emitted == [0.7]
            assert s._preemptive_prediction is None
        finally:
            await s.aclose()
