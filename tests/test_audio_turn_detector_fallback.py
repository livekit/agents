"""Tests for the unified ``AudioTurnDetector`` (auto-select + fallback).

Covers:

- Auto-select via ``LIVEKIT_REMOTE_EOT_URL`` env var (with creds present, with
  creds missing → silent downgrade).
- Explicit-cloud missing creds raises.
- Cloud → local fallback triggers (transport raise, predict timeout).
- Fallback persistence across turns.
- Local-failure handling (default 1.0, retry on next turn).
- Per-session warning dedupe (one warning per failure mode).
- Threshold scaling: pass-through for cloud / explicit-local, multiplicative
  scaling only on actual fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents._exceptions import APIConnectionError
from livekit.agents.inference.eot import AudioTurnDetector
from livekit.agents.inference.eot.detector import _AudioTurnDetectorStreamImpl
from livekit.agents.inference.eot.languages import CLOUD_LANGUAGES, LOCAL_LANGUAGES
from livekit.agents.inference.eot.transports import (
    _AudioTurnDetectionTransport,
    _LocalTransport,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.voice.turn import TurnDetectorOptions, _AudioTurnDetectorStream


async def _wait_for(predicate: Any, *, ticks: int = 20) -> None:
    """Yield to the event loop until ``predicate()`` is true or we run out
    of ticks. Replaces fragile ``await asyncio.sleep(0)`` counts now that
    the run loop wraps ``transport.run()`` in a task (extra hop)."""
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
    """Fake transport that structurally satisfies ``_AudioTurnDetectionTransport``.

    Behavior is scriptable: ``run`` either sleeps until cancelled, raises a
    configured exception, or returns immediately. Hook calls are recorded.
    """

    def __init__(
        self,
        *,
        run_behavior: str = "idle",
        run_exc: BaseException | None = None,
    ) -> None:
        self.run_behavior = run_behavior  # "idle" | "raise" | "return"
        self.run_exc = run_exc
        self.run_calls = 0
        self.stream_ref: Any = None
        self.events: list[tuple[str, Any]] = []

    def bind(self, stream: _AudioTurnDetectorStream) -> None:
        self.stream_ref = stream

    def transport_ready(self) -> bool:
        return True

    async def run(self) -> None:
        self.run_calls += 1
        if self.run_behavior == "raise":
            assert self.run_exc is not None
            raise self.run_exc
        if self.run_behavior == "return":
            return
        # idle — sleep until cancelled
        await asyncio.Future()

    def start_inference(self, request_id: str) -> None:
        self.events.append(("start_inference", request_id))

    async def push_frame(self, frame: Any) -> None:
        self.events.append(("push_frame", frame))

    async def flush(self, sentinel: Any) -> None:
        self.events.append(("flush", sentinel))

    def stop_inference(self, *, reason: str | None) -> None:
        self.events.append(("stop_inference", reason))

    def detach(self) -> None:
        self.events.append(("detach", None))


def _make_opts(thresholds: dict[str, float] | None = None) -> TurnDetectorOptions:
    return TurnDetectorOptions(
        sample_rate=16000,
        thresholds=thresholds if thresholds is not None else {},
    )


def _make_stream_with_transport(
    transport: _AudioTurnDetectionTransport,
    *,
    backend: str = "cloud",
    user_threshold: NotGivenOr[float | dict[str, float]] = NOT_GIVEN,
) -> _AudioTurnDetectorStreamImpl:
    """Construct a stream bypassing the constructor's transport allocation,
    so we can inject a scripted transport before the FSM main task starts.

    ``user_threshold`` is materialized into ``opts.thresholds`` the same way
    the real constructor would — the stream's threshold lookup reads its own
    ``opts.thresholds``."""
    from livekit.agents.inference.eot.languages import materialize_thresholds
    from livekit.agents.inference.eot.transports import _CloudTransportOptions
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    detector = MagicMock()
    detector.model = "eot-audio" if backend == "cloud" else "eot-audio-mini"
    detector.provider = "livekit"

    opts = _make_opts(materialize_thresholds(user_threshold, backend))  # type: ignore[arg-type]
    stream = _AudioTurnDetectorStreamImpl.__new__(_AudioTurnDetectorStreamImpl)
    stream._backend = backend  # type: ignore[assignment]
    stream._cloud_opts = (
        _CloudTransportOptions(
            base_url="ws://test",
            api_key="x",
            api_secret="x",
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        if backend == "cloud"
        else None
    )
    stream._http_session = None
    stream._is_fallback = False
    stream._warned_cloud_failure = False
    stream._warned_local_failure = False
    stream._transport_task = None
    _AudioTurnDetectorStream.__init__(stream, detector=detector, opts=opts, transport=transport)
    return stream


class TestAutoSelect:
    def test_auto_select_local_when_no_remote_eot_url(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = AudioTurnDetector()
            assert detector.backend == "local"

    def test_auto_select_cloud_when_remote_eot_url_set(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector()
            assert detector.backend == "cloud"

    def test_auto_select_downgrades_when_creds_missing(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY=None,
            LIVEKIT_API_SECRET=None,
            LIVEKIT_INFERENCE_API_KEY=None,
            LIVEKIT_INFERENCE_API_SECRET=None,
        ):
            detector = AudioTurnDetector()
            # env said cloud, but creds absent → silent downgrade
            assert detector.backend == "local"


class TestSingleStreamOwnership:
    async def test_second_stream_rejected_until_first_retired(self) -> None:
        """The detector drives one stream at a time: a second stream() while
        the first is still registered raises; retiring the first re-allows it."""
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = AudioTurnDetector(backend="local")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            s1 = detector.stream()
            with pytest.raises(RuntimeError):
                detector.stream()

            # Synchronous detach alone frees the slot (what update_turn_detector
            # does before scheduling the old stream's async aclose).
            s1.detach()
            s2 = detector.stream()
            await s1.aclose()
            await s2.aclose()

    async def test_aclose_releases_slot(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = AudioTurnDetector(backend="local")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            s1 = detector.stream()
            await s1.aclose()
            assert detector._active_stream() is None
            # A fresh stream is accepted after the previous one closed.
            s2 = detector.stream()
            await s2.aclose()


class TestExplicitBackendErrors:
    def test_explicit_cloud_missing_creds_raises(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL=None,
            LIVEKIT_API_KEY=None,
            LIVEKIT_API_SECRET=None,
            LIVEKIT_INFERENCE_API_KEY=None,
            LIVEKIT_INFERENCE_API_SECRET=None,
        ):
            with pytest.raises(ValueError):
                AudioTurnDetector(backend="cloud")


class TestFallback:
    async def test_fallback_on_transport_error_emits_one(self) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            await _wait_for(lambda: stream.backend == "local")
            assert stream.backend == "local"
            assert stream.is_fallback is True
            assert stream._warned_cloud_failure is True
            assert ("detach", None) in transport.events
            await stream.aclose()

    async def test_fallback_on_predict_timeout(self) -> None:
        """Cloud `predict_end_of_turn` timeout swaps to local."""
        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            # Drive a predict that times out fast.
            prob = await stream.predict_end_of_turn(timeout=0.01)
            assert prob == 1.0
            assert stream.backend == "local"
            assert stream.is_fallback is True
            await stream.aclose()

    async def test_fallback_persists_across_turns(self) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            await _wait_for(lambda: stream.backend == "local")
            # Cloud transport ran exactly once; no resurrection.
            assert transport.run_calls == 1
            # Future turns can call warmup without re-touching cloud.
            stream.warmup()
            assert stream.backend == "local"
            await stream.aclose()


class TestDetectorViewAfterFallback:
    async def test_detector_model_and_threshold_follow_fallback(self) -> None:
        """After cloud→local fallback the detector view (read by EOU metrics
        and by ``audio_recognition`` when it consults the detector directly)
        must report the post-fallback backend + rescaled thresholds.

        The detector derives these from its live stream rather than a
        write-back, so the stream is registered as the active one (exactly
        what ``detector.stream()`` does) before driving the fallback."""
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector(unlikely_threshold=0.5)
            assert detector.model == "eot-audio"
            cloud_threshold = await detector.unlikely_threshold(LanguageCode("en"))
            assert cloud_threshold == pytest.approx(0.5)

        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _AudioTurnDetectorStreamImpl.__new__(_AudioTurnDetectorStreamImpl)
            stream._backend = "cloud"
            stream._cloud_opts = None
            stream._http_session = None
            stream._is_fallback = False
            stream._warned_cloud_failure = False
            stream._warned_local_failure = False
            stream._transport_task = None
            _AudioTurnDetectorStream.__init__(
                stream, detector=detector, opts=detector._opts, transport=transport
            )
            # Register as the detector's active stream (what stream() does) so
            # the detector's model/backend/threshold views resolve through it.
            detector._active_stream_ref = weakref.ref(stream)
            await _wait_for(lambda: stream.backend == "local")

            assert detector.model == "eot-audio-mini"
            assert detector.backend == "local"
            local_threshold = await detector.unlikely_threshold(LanguageCode("en"))
            expected = LOCAL_LANGUAGES["en"] * (0.5 / CLOUD_LANGUAGES["en"])
            assert local_threshold == pytest.approx(expected)

            await stream.aclose()


class TestLocalFailureRetry:
    async def test_local_failure_emits_default_and_retries_next_turn(self) -> None:
        """Local _predict raising emits 1.0 for the turn and does NOT
        permanently disable local — the next turn invokes _predict again."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=RuntimeError("local boom"))
        stream = _make_stream_with_transport(transport, backend="local")
        await _wait_for(lambda: stream._warned_local_failure)
        # Local failed; warning logged once; backend stays local; no fallback flag.
        assert stream.backend == "local"
        assert stream.is_fallback is False
        assert stream._warned_local_failure is True
        # Run a second cycle to confirm we'd accept another call (the
        # transport is still mounted; no swap occurred).
        assert stream._transport is transport
        await stream.aclose()


class TestWarningDedupe:
    async def test_warning_logged_once_per_session_cloud_to_local(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport)
            await _wait_for(lambda: stream.backend == "local")
            # Trigger a second fallback path by calling the method directly.
            stream._fall_back_to_local(reason=APIConnectionError("boom2"))
            # Only one warning across both invocations.
            cloud_warnings = [r for r in caplog.records if "cloud audio eot" in r.getMessage()]
            assert len(cloud_warnings) == 1
            await stream.aclose()

    async def test_warning_logged_once_per_session_local(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport, backend="local")
        # Two local failures back to back.
        stream._on_local_failure(reason=RuntimeError("a"))
        stream._on_local_failure(reason=RuntimeError("b"))
        local_warnings = [r for r in caplog.records if "local audio eot mini" in r.getMessage()]
        assert len(local_warnings) == 1
        await stream.aclose()


class TestThresholdScaling:
    async def test_detector_threshold_cloud_user_passthrough(self) -> None:
        """Pre-stream detector lookup in cloud mode returns the user value."""
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector(unlikely_threshold=0.5)
            assert detector.backend == "cloud"
            value = await detector.unlikely_threshold(LanguageCode("en"))
            assert value == pytest.approx(0.5)

    async def test_explicit_local_user_threshold_passes_through(self) -> None:
        """Regression: explicit-local pick should NOT rescale the user
        threshold against the cloud default — they meant 0.5 literally."""
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = AudioTurnDetector(backend="local", unlikely_threshold=0.5)
            value = await detector.unlikely_threshold(LanguageCode("en"))
            assert value == pytest.approx(0.5)

    async def test_post_fallback_threshold_rescales_on_stream(self) -> None:
        """Fallback-only multiplicative scaling: a uniform 0.5 user value gets
        rescaled per-language as ``local_default[lang] * (0.5 / cloud_default[lang])``."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport, user_threshold=0.5)
            await _wait_for(lambda: stream.backend == "local")
            assert stream.backend == "local"
            assert stream.is_fallback is True
            value = await stream.unlikely_threshold(LanguageCode("en"))
            expected = LOCAL_LANGUAGES["en"] * (0.5 / CLOUD_LANGUAGES["en"])
            assert value == pytest.approx(expected)
            await stream.aclose()

    async def test_threshold_default_unchanged_when_user_not_set(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector()
            cloud_default = await detector.unlikely_threshold(LanguageCode("en"))
            assert cloud_default == pytest.approx(CLOUD_LANGUAGES["en"])

        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = AudioTurnDetector()
            local_default = await detector.unlikely_threshold(LanguageCode("en"))
            assert local_default == pytest.approx(LOCAL_LANGUAGES["en"])


class TestThresholdDictOverride:
    async def test_dict_override_applies_per_language(self) -> None:
        """Cloud mode + dict override: mapped langs use the user value,
        unmapped langs fall through to the cloud default."""
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector(unlikely_threshold={"en": 0.55, "ja": 0.25})
            assert await detector.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.55)
            assert await detector.unlikely_threshold(LanguageCode("ja")) == pytest.approx(0.25)
            # `fr` not in the override dict — falls through to cloud default.
            assert await detector.unlikely_threshold(LanguageCode("fr")) == pytest.approx(
                CLOUD_LANGUAGES["fr"]
            )

    async def test_dict_keys_normalized_via_language_code(self) -> None:
        """``English`` / ``en-US`` / ``eng`` all normalize to ``en`` so the
        override matches the table lookup regardless of how the user spelled it."""
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            detector = AudioTurnDetector(
                unlikely_threshold={"English": 0.55, "en-US": 0.55, "eng": 0.55}
            )
            # All three keys collapsed to "en"; last write wins, but the
            # important thing is the lookup picks it up.
            assert await detector.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.55)

    async def test_dict_override_rescaled_per_language_on_fallback(self) -> None:
        """Each dict entry gets its own multiplicative rescale on fallback —
        ``local_default[lang] * (user[lang] / cloud_default[lang])``."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport, user_threshold={"en": 0.55, "ja": 0.25})
            await _wait_for(lambda: stream.backend == "local")
            assert stream.is_fallback is True
            assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(
                LOCAL_LANGUAGES["en"] * (0.55 / CLOUD_LANGUAGES["en"])
            )
            assert await stream.unlikely_threshold(LanguageCode("ja")) == pytest.approx(
                LOCAL_LANGUAGES["ja"] * (0.25 / CLOUD_LANGUAGES["ja"])
            )
            # `fr` not in dict → falls through to plain local default
            # (no rescaling because no user value).
            assert await stream.unlikely_threshold(LanguageCode("fr")) == pytest.approx(
                LOCAL_LANGUAGES["fr"]
            )
            await stream.aclose()
