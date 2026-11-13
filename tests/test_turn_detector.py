"""TurnDetector request lifecycle, auto-select, cloud-to-local fallback, thresholds, and what
the local transport hands the native model."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents._exceptions import APIConnectionError, APIError
from livekit.agents.inference.eot import TurnDetector
from livekit.agents.inference.eot.base import (
    TurnDetectorOptions,
    _BaseStreamingTurnDetectorStream,
    _StreamingTurnDetectionTransport,
)
from livekit.agents.inference.eot.languages import (
    LOCAL_LANGUAGES,
    ThresholdOptions,
    _normalize_overrides,
)
from livekit.agents.inference.eot.transports import EOT_MAX_SAMPLES, _LocalTransport
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .fake_turn_detector_ws import drain_audio_queue

pytestmark = pytest.mark.audio_eot

# Stand-in for the per-language defaults a 1.1.13 gateway returns in ``SessionCreated``.
SERVER_THRESHOLDS: dict[str, float] = {"en": 0.56, "ja": 0.37, "fr": 0.575}
SERVER_DEFAULT_THRESHOLD = 0.5


async def _wait_for(predicate: Any, *, ticks: int = 20) -> None:
    """Yield to the event loop until ``predicate()`` is true or we run out of ticks."""
    for _ in range(ticks):
        if predicate():
            return
        await asyncio.sleep(0)


@contextlib.contextmanager
def _clean_env(**overrides: str | None):
    """Patch env vars: keys with None get removed, others get set."""
    with patch.dict("os.environ", clear=False) as env:
        for k, v in overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        yield


class _ScriptedTransport:
    """Fake transport that structurally satisfies ``_StreamingTurnDetectionTransport``."""

    def __init__(
        self,
        *,
        run_behavior: str = "idle",
        run_exc: BaseException | None = None,
    ) -> None:
        self.run_behavior = run_behavior  # "idle" | "raise" | "return" | "drain"
        self.run_exc = run_exc
        self.session_id: str | None = None
        self.run_calls = 0
        self.stream_ref: Any = None
        self.events: list[tuple[str, Any]] = []

    def attach(self, stream: _BaseStreamingTurnDetectorStream) -> None:
        self.stream_ref = stream

    async def run(self) -> None:
        self.run_calls += 1
        if self.run_behavior == "raise":
            assert self.run_exc is not None
            raise self.run_exc
        if self.run_behavior == "return":
            return
        if self.run_behavior == "drain":
            # hand the queued audio and requests to this transport, as the real ones do
            await self.stream_ref._drain_audio_channel()
            return
        # idle — sleep until cancelled
        await asyncio.Future()

    def run_inference(self, request_id: str, *, chat_ctx: Any = None) -> None:
        self.events.append(("run_inference", request_id))

    def push_frame(self, frame: Any) -> None:
        self.events.append(("push_frame", frame))

    def on_start_of_turn(self, elapsed: float) -> None:
        self.events.append(("on_start_of_turn", elapsed))

    def flush(self) -> None:
        self.events.append(("flush", None))

    def detach(self) -> None:
        self.events.append(("detach", None))


def _make_opts(
    *,
    model: str = "turn-detector-v1",
    user_threshold: NotGivenOr[float | dict[str, float]] = NOT_GIVEN,
) -> TurnDetectorOptions:
    return TurnDetectorOptions(
        sample_rate=16000,
        thresholds=ThresholdOptions(model, user_threshold),  # type: ignore[arg-type]
    )


def _make_stream_with_transport(
    transport: _StreamingTurnDetectionTransport,
    *,
    model: str = "turn-detector-v1",
    user_threshold: NotGivenOr[float | dict[str, float]] = NOT_GIVEN,
) -> _BaseStreamingTurnDetectorStream:
    """Construct a stream wired to a scripted transport."""
    detector = MagicMock()
    detector.model = model
    detector.provider = "livekit"

    opts = _make_opts(model=model, user_threshold=user_threshold)
    detector._opts = opts

    return _BaseStreamingTurnDetectorStream(
        detector=detector,
        opts=opts,
        transport=transport,
        model=model,  # type: ignore[arg-type]
    )


def _inferences(transport: _ScriptedTransport) -> list[tuple[str, Any]]:
    return [e for e in transport.events if e[0] == "run_inference"]


def _audio_events(transport: _ScriptedTransport) -> list[tuple[str, Any]]:
    """What the drain loop handed the transport, in order; a frame shows as its length."""
    return [
        (name, value.samples_per_channel if name == "push_frame" else value)
        for name, value in transport.events
        if name != "detach"
    ]


def _frame(num_samples: int = 160, *, value: int = 0) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.full(num_samples, value, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=num_samples,
    )


_Stream = tuple[_BaseStreamingTurnDetectorStream, _ScriptedTransport]


@pytest.fixture
async def stream() -> Any:
    transport = _ScriptedTransport(run_behavior="drain")
    s = _make_stream_with_transport(transport, model="turn-detector-v1-mini")
    yield s, transport
    await s.aclose()


class TestStreamRequests:
    async def test_start_of_turn_reaches_transport_in_audio_order(self, stream: _Stream) -> None:
        """The turn length is measured when the start is queued, and it lands between the
        frames pushed before and after the call, however late the transport drains it."""
        s, transport = stream
        s.push_audio(_frame())
        with patch("livekit.agents.inference.eot.base.time.time", return_value=1000.25):
            s.on_start_of_turn(1000.0)
        s.push_audio(_frame())
        s.end_input()
        await s._task
        assert _audio_events(transport) == [
            ("push_frame", 160),
            ("on_start_of_turn", 0.25),
            ("push_frame", 160),
            ("flush", None),
        ]

    async def test_predict_starts_inference(self, stream: _Stream) -> None:
        s, transport = stream
        fut = s.predict()
        assert s._request_id is not None
        assert not fut.done()
        await drain_audio_queue(s)
        assert _inferences(transport) == [("run_inference", s._request_id)]

    async def test_predict_runs_after_the_audio_pushed_before_it(self, stream: _Stream) -> None:
        """predict queues behind the audio, so the transport sees every frame captured before
        the call first and counts it in the turn length."""
        s, transport = stream
        s.push_audio(_frame())
        s.push_audio(_frame())
        s.predict()
        s.push_audio(_frame())
        await drain_audio_queue(s)
        assert _audio_events(transport) == [
            ("push_frame", 160),
            ("push_frame", 160),
            ("run_inference", s._request_id),
            ("push_frame", 160),
        ]

    async def test_predict_links_request_to_inference_session(self, stream: _Stream) -> None:
        s, transport = stream
        transport.session_id = "inference_connection"
        s.predict()
        assert s._request_id is not None
        assert s._request_id.startswith("inference_connection_eot_")

    async def test_predict_supersedes_previous_request(self, stream: _Stream) -> None:
        """A second predict resolves the pending request with 0.0 and starts a fresh one."""
        s, transport = stream
        old_fut = s.predict()
        old_id = s._request_id
        new_fut = s.predict()

        assert new_fut is not old_fut
        assert s._request_id != old_id
        assert old_fut.done()
        assert old_fut.result().end_of_turn_probability == 0.0
        # the old request closed before the transport reached it, so only the new one runs
        await drain_audio_queue(s)
        assert _inferences(transport) == [("run_inference", s._request_id)]

    async def test_cancel_inference_closes_request(self, stream: _Stream) -> None:
        s, transport = stream
        fut = s.predict()
        s.cancel_inference()

        assert s._request_id is None
        assert fut.done()
        assert fut.result().end_of_turn_probability == 0.0

    async def test_cancel_inference_when_idle_is_noop(self, stream: _Stream) -> None:
        s, transport = stream
        s.cancel_inference()
        assert _inferences(transport) == []

    async def test_late_prediction_after_cancel_inference_dropped(self, stream: _Stream) -> None:
        """A late prediction for a closed request is dropped."""
        s, transport = stream
        fut = s.predict()
        cancelled_request_id = s._request_id
        assert cancelled_request_id is not None

        s.cancel_inference()
        s._resolve_prediction(cancelled_request_id, 0.9)
        assert fut.result().end_of_turn_probability == 0.0  # cancel_inference default, not 0.9

        next_fut = s.predict()
        assert next_fut is not fut
        assert not next_fut.done()
        await drain_audio_queue(s)
        assert _inferences(transport) == [("run_inference", s._request_id)]

    async def test_prediction_completes_request(self, stream: _Stream) -> None:
        """A prediction resolves the future and completes the request."""
        s, transport = stream
        fut = s.predict()
        request_id = s._request_id
        assert request_id is not None

        s._resolve_prediction(request_id, 0.3)
        assert fut.done()
        assert fut.result().end_of_turn_probability == 0.3
        assert s._request_id is None

    async def test_flush_closes_request(self, stream: _Stream) -> None:
        s, transport = stream
        fut = s.predict()
        s.flush(reason="turn committed")
        assert s._request_id is None
        assert fut.result().end_of_turn_probability == 0.0

    async def test_flush_does_not_overwrite_resolved_prediction(self, stream: _Stream) -> None:
        s, transport = stream
        fut = s.predict()
        request_id = s._request_id
        assert request_id is not None
        s._resolve_prediction(request_id, 0.7)

        s.flush(reason="turn committed")
        assert fut.result().end_of_turn_probability == 0.7
        assert s._request_id is None

    async def test_predict_after_end_input_returns_resolved_default(self, stream: _Stream) -> None:
        s, transport = stream
        s.end_input()
        fut = s.predict()
        assert fut.done()
        assert fut.result().end_of_turn_probability == 1.0
        assert not _inferences(transport)

    async def test_aclose_resolves_pending_future(self, stream: _Stream) -> None:
        s, transport = stream
        fut = s.predict()
        await s.aclose()
        assert fut.done()
        assert fut.result().end_of_turn_probability == 0.0

    async def test_timed_out_cancel_inference_local_model_no_fallback(
        self, stream: _Stream
    ) -> None:
        """A timed-out request on the mini model only closes the request."""
        s, transport = stream
        fut = s.predict()
        s.cancel_inference(timed_out=True)
        assert fut.result().end_of_turn_probability == 0.0
        assert s.is_fallback is False
        assert s.model == "turn-detector-v1-mini"


class TestAutoSelect:
    def test_auto_select_local_when_no_remote_eot_url(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = TurnDetector()
            assert detector.model == "turn-detector-v1-mini"

    def test_auto_select_cloud_when_remote_eot_url_set(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = TurnDetector()
            assert detector.model == "turn-detector-v1"

    def test_auto_select_downgrades_when_creds_missing(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY=None,
            LIVEKIT_API_SECRET=None,
            LIVEKIT_INFERENCE_API_KEY=None,
            LIVEKIT_INFERENCE_API_SECRET=None,
        ):
            detector = TurnDetector()
            # env said cloud, but creds absent → silent downgrade
            assert detector.model == "turn-detector-v1-mini"


class TestExplicitModelErrors:
    def test_explicit_cloud_missing_creds_raises(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL=None,
            LIVEKIT_API_KEY=None,
            LIVEKIT_API_SECRET=None,
            LIVEKIT_INFERENCE_API_KEY=None,
            LIVEKIT_INFERENCE_API_SECRET=None,
        ):
            with pytest.raises(ValueError):
                TurnDetector(version="v1")


class TestFallback:
    async def test_fallback_on_transport_error_emits_one(self) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            assert stream.model == "turn-detector-v1-mini"
            assert stream.is_fallback is True
            assert stream._warned_cloud_failure is True
            assert ("detach", None) in transport.events
            await stream.aclose()

    async def test_fallback_on_predict_timeout(self) -> None:
        """A timed-out cancel_inference swaps cloud to local."""
        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            fut = stream.predict()
            stream.cancel_inference(timed_out=True)
            assert fut.result().end_of_turn_probability == 0.0
            assert stream.model == "turn-detector-v1-mini"
            assert stream.is_fallback is True
            await stream.aclose()

    async def test_fallback_persists_across_turns(self) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            # Cloud transport ran exactly once; no resurrection.
            assert transport.run_calls == 1
            # Future turns can start inference without re-touching cloud.
            stream.predict()
            assert stream.model == "turn-detector-v1-mini"
            await stream.aclose()


class TestDetectorViewAfterFallback:
    async def test_detector_model_and_threshold_follow_fallback(self) -> None:
        """After fallback the detector reports the local model and rescaled thresholds."""
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = TurnDetector(unlikely_threshold=0.5)
            assert detector.model == "turn-detector-v1"
            # scalar override is resolvable pre-session via the catch-all
            assert await detector.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.5)

        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _BaseStreamingTurnDetectorStream(
                detector=detector,
                opts=detector._opts,
                transport=transport,
                model="turn-detector-v1",
            )
            # server defaults arrive, then the cloud session fails
            stream._opts.thresholds._update_defaults(
                dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD
            )
            stream._fallback_to_local(reason=APIConnectionError("boom"))
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")

            assert detector.model == "turn-detector-v1-mini"
            local_threshold = await detector.unlikely_threshold(LanguageCode("en"))
            expected = LOCAL_LANGUAGES["en"] * (0.5 / SERVER_THRESHOLDS["en"])
            assert local_threshold == pytest.approx(expected)

            await stream.aclose()


class TestLocalFailureRetry:
    async def test_local_failure_emits_default_and_retries_next_turn(self) -> None:
        """A failing local predict emits 1.0 and is retried on the next turn."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=RuntimeError("local boom"))
        stream = _make_stream_with_transport(transport, model="turn-detector-v1-mini")
        await _wait_for(lambda: stream._warned_local_failure)
        # Local failed; warning logged once; model stays turn-detector-v1-mini; no fallback flag.
        assert stream.model == "turn-detector-v1-mini"
        assert stream.is_fallback is False
        assert stream._warned_local_failure is True
        # Run a second cycle to confirm we'd accept another call (the
        # transport is still mounted; no swap occurred).
        assert stream._transport is transport
        await stream.aclose()

    async def test_run_loop_failure_keeps_serving_later_requests(self) -> None:
        """One failing call ends the transport's run loop. The stream starts it again, so the
        next request still reaches the transport instead of waiting in the queue."""

        class _FlakyTransport(_ScriptedTransport):
            def push_frame(self, frame: Any) -> None:
                super().push_frame(frame)
                if len([e for e in self.events if e[0] == "push_frame"]) == 1:
                    raise RuntimeError("local boom")

        transport = _FlakyTransport(run_behavior="drain")
        stream = _make_stream_with_transport(transport, model="turn-detector-v1-mini")
        try:
            stream.push_audio(_frame())
            await _wait_for(lambda: stream._warned_local_failure)

            stream.push_audio(_frame())
            stream.predict()
            await drain_audio_queue(stream)

            assert _inferences(transport) == [("run_inference", stream._request_id)]
            assert transport.run_calls == 2
        finally:
            await stream.aclose()

    async def test_failed_prediction_resolves_the_positive_default(self) -> None:
        class _FailingEOT:
            def predict(self, pcm: Any, turn_samples: int | None = None) -> float:
                raise RuntimeError("native boom")

        with patch("livekit.agents.inference.eot.transports._EOT", _FailingEOT):
            transport = _LocalTransport(opts=_make_opts(model="turn-detector-v1-mini"))
        stream = MagicMock()
        transport.attach(stream)
        transport.run_inference("request")
        await asyncio.gather(*transport._tasks)

        assert stream._resolve_prediction.call_args.args == ("request", 1.0)


class _RecordingEOT:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int | None]] = []

    def predict(self, pcm: np.ndarray, turn_samples: int | None = None) -> float:
        self.calls.append((pcm, turn_samples))
        return 0.5


class TestLocalTransportInput:
    """``_LocalTransport`` hands the native model the last ``EOT_MAX_SAMPLES`` of the user turn
    and the turn length in samples; both start at the turn start, or at the last flush."""

    @pytest.fixture
    def transport(self) -> _LocalTransport:
        with patch("livekit.agents.inference.eot.transports._EOT", _RecordingEOT):
            return _LocalTransport(opts=_make_opts(model="turn-detector-v1-mini"))

    async def _predict(self, transport: _LocalTransport) -> tuple[np.ndarray, int | None]:
        stream = MagicMock()
        transport.attach(stream)
        transport.run_inference("request")
        await asyncio.gather(*transport._tasks)
        stream._resolve_prediction.assert_called_once()
        calls: list[tuple[np.ndarray, int | None]] = transport._eot.calls  # type: ignore[attr-defined]
        return calls[-1]

    async def test_turn_samples_count_from_the_turn_start_past_the_window(
        self, transport: _LocalTransport
    ) -> None:
        transport.push_frame(_frame(8000))  # silence before the user speaks
        transport.push_frame(_frame(4000))
        transport.on_start_of_turn(0.25)
        for _ in range(2 * EOT_MAX_SAMPLES // 1600):
            transport.push_frame(_frame(1600))

        pcm, turn_samples = await self._predict(transport)

        assert len(pcm) == EOT_MAX_SAMPLES
        assert turn_samples == 4000 + 2 * EOT_MAX_SAMPLES // 1600 * 1600

    async def test_turn_start_drops_the_audio_before_it(self, transport: _LocalTransport) -> None:
        """A turn shorter than the window: the model reads the turn only, not the audio that
        filled the buffer before the user spoke."""
        transport.push_frame(_frame(EOT_MAX_SAMPLES, value=1))  # the agent speaks
        transport.push_frame(_frame(4000, value=2))
        transport.on_start_of_turn(0.25)
        transport.push_frame(_frame(1600, value=2))

        pcm, turn_samples = await self._predict(transport)

        assert len(pcm) == turn_samples == 5600
        assert (pcm == 2).all()

    async def test_flush_forgets_the_turn_start(self, transport: _LocalTransport) -> None:
        transport.push_frame(_frame(16000))
        transport.on_start_of_turn(0.1)
        transport.flush()
        transport.push_frame(_frame(6000))

        pcm, turn_samples = await self._predict(transport)

        assert len(pcm) == turn_samples == 6000

    async def test_turn_longer_than_the_audio_clamps_to_flush(
        self, transport: _LocalTransport
    ) -> None:
        transport.push_frame(_frame(1600))
        transport.on_start_of_turn(1.0)

        pcm, turn_samples = await self._predict(transport)

        assert len(pcm) == turn_samples == 1600


class TestWarningDedupe:
    async def test_warning_logged_once_per_session_cloud_to_local(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            # Trigger a second fallback path by calling the method directly.
            stream._fallback_to_local(reason=APIConnectionError("boom2"))
            # Only one warning across both invocations.
            cloud_warnings = [r for r in caplog.records if "cloud turn detector" in r.getMessage()]
            assert len(cloud_warnings) == 1
            await stream.aclose()

    async def test_warning_logged_once_per_session_local(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport, model="turn-detector-v1-mini")
        # Two local failures back to back.
        stream._on_local_failure(reason=RuntimeError("a"))
        stream._on_local_failure(reason=RuntimeError("b"))
        local_warnings = [
            r for r in caplog.records if "local audio turn detector" in r.getMessage()
        ]
        assert len(local_warnings) == 1
        await stream.aclose()


class TestResolveThresholds:
    """Cloud-override resolution against the server defaults, via ThresholdOptions."""

    @staticmethod
    def _cloud(overrides: Any = NOT_GIVEN) -> ThresholdOptions:
        # mirror the detector boundary: overrides are normalized before reaching ThresholdOptions
        opts = ThresholdOptions("turn-detector-v1", _normalize_overrides(overrides))
        opts._update_defaults(dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD)
        return opts

    def test_no_override_adopts_server_map_and_fallback(self) -> None:
        opts = self._cloud()
        assert opts.thresholds == SERVER_THRESHOLDS
        assert opts.default_threshold == pytest.approx(SERVER_DEFAULT_THRESHOLD)

    def test_scalar_override_replaces_with_empty_map(self) -> None:
        opts = self._cloud(0.8)
        # empty map → every language resolves through the scalar fallback
        assert opts.thresholds == {}
        assert opts.default_threshold == pytest.approx(0.8)

    def test_dict_override_layers_on_server_map(self) -> None:
        opts = self._cloud({"en": 0.7})
        assert opts.thresholds["en"] == pytest.approx(0.7)
        # unmapped languages keep the server values + server fallback
        assert opts.thresholds["ja"] == pytest.approx(SERVER_THRESHOLDS["ja"])
        assert opts.default_threshold == pytest.approx(SERVER_DEFAULT_THRESHOLD)

    def test_dict_keys_normalized(self) -> None:
        opts = self._cloud({"English": 0.7, "en-US": 0.7})
        assert opts.thresholds["en"] == pytest.approx(0.7)


class TestServerDefaults:
    async def test_cloud_thresholds_pending_before_session_created(self) -> None:
        """A cloud detector supports every language before SessionCreated arrives."""
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport)
        assert await stream.unlikely_threshold(LanguageCode("en")) is None
        assert await stream.supports_language(LanguageCode("en")) is True
        await stream.aclose()

    async def test_cloud_adopts_server_defaults(self) -> None:
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport)
        stream._opts.thresholds._update_defaults(dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD)
        assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(
            SERVER_THRESHOLDS["en"]
        )
        # language absent from the server map → catch-all default_threshold
        assert await stream.unlikely_threshold(LanguageCode("de")) == pytest.approx(
            SERVER_DEFAULT_THRESHOLD
        )
        # shared with the owning detector view too
        assert stream._detector._opts.thresholds.lookup(LanguageCode("en")) == pytest.approx(
            SERVER_THRESHOLDS["en"]
        )
        await stream.aclose()

    async def test_dict_override_layers_on_server_defaults(self) -> None:
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport, user_threshold={"en": 0.7, "ja": 0.2})
        stream._opts.thresholds._update_defaults(dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD)
        assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.7)
        assert await stream.unlikely_threshold(LanguageCode("ja")) == pytest.approx(0.2)
        # fr not overridden → server default for fr
        assert await stream.unlikely_threshold(LanguageCode("fr")) == pytest.approx(
            SERVER_THRESHOLDS["fr"]
        )
        await stream.aclose()

    async def test_degenerate_session_created_raises_without_override(self) -> None:
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport)
        with pytest.raises(APIError):
            stream._opts.thresholds._update_defaults({}, 0.0)
        await stream.aclose()

    async def test_degenerate_session_created_raises_even_with_override(self) -> None:
        # A degenerate server response always degrades to the local model (via APIError), even when
        # an override is set — the cloud session genuinely produced no usable defaults.
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport, user_threshold=0.8)
        with pytest.raises(APIError):
            stream._opts.thresholds._update_defaults({}, 0.0)
        await stream.aclose()


SERVER_BACKCHANNEL_THRESHOLDS: dict[str, float] = {"en": 0.62, "ja": 0.7}
SERVER_BACKCHANNEL_DEFAULT = 0.6


class TestBackchannelThresholds:
    """Backchannel thresholds come from the server and are disabled on mini and after fallback."""

    @staticmethod
    def _cloud() -> ThresholdOptions:
        opts = ThresholdOptions("turn-detector-v1")
        opts._update_defaults(
            dict(SERVER_THRESHOLDS),
            SERVER_DEFAULT_THRESHOLD,
            dict(SERVER_BACKCHANNEL_THRESHOLDS),
            SERVER_BACKCHANNEL_DEFAULT,
        )
        return opts

    def test_lookup_per_language_and_default(self) -> None:
        opts = self._cloud()
        assert opts.lookup_backchannel(LanguageCode("en")) == pytest.approx(
            SERVER_BACKCHANNEL_THRESHOLDS["en"]
        )
        # absent language → catch-all backchannel default
        assert opts.lookup_backchannel(LanguageCode("de")) == pytest.approx(
            SERVER_BACKCHANNEL_DEFAULT
        )
        # None language defaults to "en"
        assert opts.lookup_backchannel(None) == pytest.approx(SERVER_BACKCHANNEL_THRESHOLDS["en"])

    def test_disabled_when_server_omits_backchannel(self) -> None:
        opts = ThresholdOptions("turn-detector-v1")
        opts._update_defaults(dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD)
        assert opts.lookup_backchannel(LanguageCode("en")) is None

    def test_disabled_for_local_mini_model(self) -> None:
        opts = ThresholdOptions("turn-detector-v1-mini")
        assert opts.lookup_backchannel(LanguageCode("en")) is None

    def test_non_positive_threshold_treated_as_disabled(self) -> None:
        opts = ThresholdOptions("turn-detector-v1")
        opts._update_defaults(dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD, {"en": 0.0}, 0.6)
        # en explicitly 0 → disabled for en, but the positive default still applies elsewhere
        assert opts.lookup_backchannel(LanguageCode("en")) is None
        assert opts.lookup_backchannel(LanguageCode("de")) == pytest.approx(0.6)

    def test_cleared_on_local_fallback(self) -> None:
        opts = self._cloud()
        opts._to_local_fallback()
        assert opts.lookup_backchannel(LanguageCode("en")) is None


class TestResolveBackchannelThresholds:
    """User backchannel-threshold overrides layer on the server defaults."""

    @staticmethod
    def _cloud(overrides: Any = NOT_GIVEN) -> ThresholdOptions:
        opts = ThresholdOptions(
            "turn-detector-v1", backchannel_overrides=_normalize_overrides(overrides)
        )
        opts._update_defaults(
            dict(SERVER_THRESHOLDS),
            SERVER_DEFAULT_THRESHOLD,
            dict(SERVER_BACKCHANNEL_THRESHOLDS),
            SERVER_BACKCHANNEL_DEFAULT,
        )
        return opts

    def test_no_override_adopts_server_backchannel(self) -> None:
        opts = self._cloud()
        assert opts.lookup_backchannel(LanguageCode("en")) == pytest.approx(
            SERVER_BACKCHANNEL_THRESHOLDS["en"]
        )

    def test_scalar_override_applies_to_every_language(self) -> None:
        opts = self._cloud(0.8)
        assert opts.lookup_backchannel(LanguageCode("en")) == pytest.approx(0.8)
        assert opts.lookup_backchannel(LanguageCode("ja")) == pytest.approx(0.8)

    def test_dict_override_layers_on_server_map(self) -> None:
        opts = self._cloud({"en": 0.5})
        assert opts.lookup_backchannel(LanguageCode("en")) == pytest.approx(0.5)
        # unmapped languages keep the server values + server default
        assert opts.lookup_backchannel(LanguageCode("ja")) == pytest.approx(
            SERVER_BACKCHANNEL_THRESHOLDS["ja"]
        )
        assert opts.lookup_backchannel(LanguageCode("de")) == pytest.approx(
            SERVER_BACKCHANNEL_DEFAULT
        )

    def test_dict_keys_normalized(self) -> None:
        opts = self._cloud({"English": 0.5})
        assert opts.lookup_backchannel(LanguageCode("en")) == pytest.approx(0.5)

    def test_scalar_override_enables_before_server_defaults(self) -> None:
        # an explicit scalar override resolves up front, even though the server
        # backchannel defaults haven't arrived yet
        opts = ThresholdOptions("turn-detector-v1", backchannel_overrides=0.8)
        assert opts.lookup_backchannel(LanguageCode("en")) == pytest.approx(0.8)

    def test_update_backchannel_overrides_reresolves(self) -> None:
        opts = self._cloud()
        opts.update_backchannel_overrides(0.45)
        assert opts.lookup_backchannel(LanguageCode("ja")) == pytest.approx(0.45)


class TestOverrideWarning:
    def test_warning_on_construction_with_override(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            TurnDetector(unlikely_threshold=0.5)
        warnings = [
            r for r in caplog.records if "non-default turn detection threshold" in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_warning_on_construction_with_backchannel_override(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            TurnDetector(backchannel_threshold=0.7)
        warnings = [
            r for r in caplog.records if "non-default backchannel threshold" in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_no_warning_without_override(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            TurnDetector()
        warnings = [
            r for r in caplog.records if "non-default turn detection threshold" in r.getMessage()
        ]
        assert not warnings


class TestUpdateOptions:
    async def test_update_options_reresolves_active_cloud_stream(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = TurnDetector()

        # detector.stream() shares the detector's ThresholdOptions with the stream
        transport = _ScriptedTransport(run_behavior="idle")
        with patch("livekit.agents.inference.eot.detector._CloudTransport", return_value=transport):
            stream = detector.stream()
        stream._opts.thresholds._update_defaults(dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD)
        assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(
            SERVER_THRESHOLDS["en"]
        )

        detector.update_options(unlikely_threshold=0.7)
        # the shared resolver re-resolves against the cached server defaults; the stream sees it
        assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.7)
        warnings = [
            r for r in caplog.records if "non-default turn detection threshold" in r.getMessage()
        ]
        assert len(warnings) == 1
        await stream.aclose()

    async def test_update_options_local_model(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = TurnDetector()
            assert detector.model == "turn-detector-v1-mini"
            detector.update_options(unlikely_threshold=0.42)
            assert await detector.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.42)


class TestThresholdRescaleOnFallback:
    async def test_scalar_override_rescaled_against_server_on_fallback(self) -> None:
        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport, user_threshold=0.5)
            stream._opts.thresholds._update_defaults(
                dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD
            )
            stream._fallback_to_local(reason=APIConnectionError("boom"))
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            assert stream.is_fallback is True
            value = await stream.unlikely_threshold(LanguageCode("en"))
            expected = LOCAL_LANGUAGES["en"] * (0.5 / SERVER_THRESHOLDS["en"])
            assert value == pytest.approx(expected)
            await stream.aclose()

    async def test_no_override_fallback_uses_local_table(self) -> None:
        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            stream._opts.thresholds._update_defaults(
                dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD
            )
            stream._fallback_to_local(reason=APIConnectionError("boom"))
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            # ratio 1.0 → local table unchanged
            assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(
                LOCAL_LANGUAGES["en"]
            )
            await stream.aclose()

    async def test_dict_override_rescaled_per_language_on_fallback(self) -> None:
        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport, user_threshold={"en": 0.55, "ja": 0.25})
            stream._opts.thresholds._update_defaults(
                dict(SERVER_THRESHOLDS), SERVER_DEFAULT_THRESHOLD
            )
            stream._fallback_to_local(reason=APIConnectionError("boom"))
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            assert stream.is_fallback is True
            assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(
                LOCAL_LANGUAGES["en"] * (0.55 / SERVER_THRESHOLDS["en"])
            )
            assert await stream.unlikely_threshold(LanguageCode("ja")) == pytest.approx(
                LOCAL_LANGUAGES["ja"] * (0.25 / SERVER_THRESHOLDS["ja"])
            )
            # fr not in dict → server value as effective → plain local default
            assert await stream.unlikely_threshold(LanguageCode("fr")) == pytest.approx(
                LOCAL_LANGUAGES["fr"]
            )
            await stream.aclose()

    async def test_fallback_before_session_created_uses_local_materialize(self) -> None:
        """A cloud failure before SessionCreated uses the local table with overrides applied."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport, user_threshold=0.42)
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            assert stream.is_fallback is True
            # materialize_local_thresholds(0.42) → 0.42 for every local language
            assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.42)
            await stream.aclose()
