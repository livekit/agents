"""Tests for the audio turn-detection FSM in ``_AudioTurnDetectorStream``.

Covers state transitions and hook dispatch, with regression coverage for the
``WARMING_UP`` / ``set_active(False)`` interaction that previously allowed an
in-flight warmup to leak a stale prediction into the next active window
(``is_active`` only matched ``ACTIVE``, so the deactivation path no-op'd on
WARMING_UP and the cached request id was never cleared).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.turn import (
    TurnDetectorOptions,
    _AudioTurnDetectorStream,
    _InferenceStatus,
)


class _FakeBackend(_AudioTurnDetectorStream):
    """In-memory backend that records hook events for FSM testing.

    Mirrors the cloud/local pattern: predictions arriving while the stream
    is not ACTIVE are stashed in ``_held_probability`` and replayed by
    ``_on_activate`` on the next activation; predictions whose request id
    no longer matches ``_active_request_id`` are discarded as stale.
    """

    def __init__(self, *, detector: Any, opts: TurnDetectorOptions) -> None:
        self.events: list[tuple[str, ...]] = []
        self.emitted: list[float] = []
        self._held_probability: float | None = None
        super().__init__(detector=detector, opts=opts)

    def _transport_ready(self) -> bool:
        return True

    async def _run_transport(self) -> None:
        await self._drain_audio_channel()

    def _on_warmup_start(self, request_id: str) -> None:
        self.events.append(("warmup_start", request_id))

    def _on_activate(self) -> None:
        self.events.append(("activate",))
        if self._held_probability is not None:
            prob = self._held_probability
            self._held_probability = None
            self._emit_prediction(prob)

    def _on_inference_stop(self, *, reason: str | None) -> None:
        self.events.append(("inference_stop", reason or ""))

    def _emit_prediction(self, probability: float, *, detection_delay: float | None = None) -> None:
        self.emitted.append(probability)
        super()._emit_prediction(probability, detection_delay=detection_delay)

    def simulate_prediction(self, request_id: str, probability: float) -> None:
        """Mirror the transport recv-loop: drop stale, hold or emit otherwise."""
        if request_id != self._active_request_id:
            return
        if self.is_active:
            self._emit_prediction(probability)
        else:
            self._held_probability = probability


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

    async def test_warmup_transitions_to_warming_up(self) -> None:
        s = _make_stream()
        try:
            fut = s.warmup()
            assert s._status == _InferenceStatus.WARMING_UP
            assert s._active_request_id is not None
            assert not fut.done()
            assert s.events == [("warmup_start", s._active_request_id)]
        finally:
            await s.aclose()

    async def test_warmup_is_idempotent(self) -> None:
        s = _make_stream()
        try:
            s.warmup()
            first_id = s._active_request_id
            s.warmup()
            assert s._active_request_id == first_id
            assert sum(1 for e in s.events if e[0] == "warmup_start") == 1
        finally:
            await s.aclose()

    async def test_set_active_true_activates_from_warming_up(self) -> None:
        s = _make_stream()
        try:
            s.warmup()
            s.set_active(True, trigger="vad eos")
            assert s._status == _InferenceStatus.ACTIVE
            assert ("activate",) in s.events
        finally:
            await s.aclose()

    async def test_set_active_true_from_deactivated_auto_warmsup(self) -> None:
        s = _make_stream()
        try:
            s.set_active(True, trigger="manual")
            assert s._status == _InferenceStatus.ACTIVE
            warmups = [e for e in s.events if e[0] == "warmup_start"]
            assert len(warmups) == 1
        finally:
            await s.aclose()

    async def test_set_active_false_during_warmup_stops_inference(self) -> None:
        """Regression: ``set_active(False)`` previously no-op'd in WARMING_UP
        because ``is_active`` only matched ACTIVE, leaving the warmup running."""
        s = _make_stream()
        try:
            s.warmup()
            s.set_active(False, trigger="vad sos")

            assert s._status == _InferenceStatus.DEACTIVATED
            assert s._active_request_id is None
            assert ("inference_stop", "vad sos") in s.events
        finally:
            await s.aclose()

    async def test_warmup_prediction_after_sos_is_not_replayed(self) -> None:
        """Regression: a late prediction for the cancelled warmup request id
        must be dropped, not stashed for replay by ``_on_activate`` on the
        next activation."""
        s = _make_stream()
        try:
            s.warmup()
            cancelled_request_id = s._active_request_id
            assert cancelled_request_id is not None

            s.set_active(False, trigger="vad sos")

            s.simulate_prediction(cancelled_request_id, probability=0.9)
            assert s._held_probability is None

            s.warmup()
            s.set_active(True, trigger="vad eos")
            assert s.emitted == []
        finally:
            await s.aclose()

    async def test_set_active_false_when_deactivated_is_noop(self) -> None:
        s = _make_stream()
        try:
            s.set_active(False, trigger="vad sos")
            assert s.events == []
            assert s._status == _InferenceStatus.DEACTIVATED
        finally:
            await s.aclose()

    async def test_stop_warmup_resolves_future_with_zero(self) -> None:
        s = _make_stream()
        try:
            fut = s.warmup()
            s.stop_warmup()
            assert fut.done()
            assert fut.result() == 0.0
            assert s._status == _InferenceStatus.DEACTIVATED
            assert s._active_request_id is None
        finally:
            await s.aclose()

    async def test_predict_end_of_turn_timeout_leaves_fsm_consistent(self) -> None:
        """Regression: timeout previously cleared ``_active_request_fut`` /
        ``_active_request_id`` but left ``_status`` at ACTIVE, an inconsistent
        state that the next ``warmup()`` could not recover from."""
        s = _make_stream()
        try:
            prob = await s.predict_end_of_turn(timeout=0.01)
            assert prob == 1.0
            assert s._status == _InferenceStatus.DEACTIVATED
            assert s._active_request_id is None
            assert s._active_request_fut is None
            assert ("inference_stop", "predict_end_of_turn timeout") in s.events
        finally:
            await s.aclose()

    async def test_predict_end_of_turn_timeout_allows_next_warmup(self) -> None:
        """Regression: post-timeout FSM must allow a fresh warmup(). Previously
        status stayed ACTIVE with ``_active_request_fut = None``, so the next
        ``warmup()`` short-circuited the ``_warmup()`` call and raised
        ``RuntimeError("eot detection warmup failed, no request future")``."""
        s = _make_stream()
        try:
            await s.predict_end_of_turn(timeout=0.01)
            fut = s.warmup()
            assert s._status == _InferenceStatus.WARMING_UP
            assert s._active_request_id is not None
            assert not fut.done()
        finally:
            await s.aclose()

    async def test_flush_transitions_to_flushed_and_emits_stop(self) -> None:
        s = _make_stream()
        try:
            s.warmup()
            s.set_active(True)
            s.flush(reason="turn committed")
            assert s._status == _InferenceStatus.FLUSHED
            assert ("inference_stop", "turn committed") in s.events
        finally:
            await s.aclose()
