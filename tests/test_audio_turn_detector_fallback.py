"""Tests for the unified ``TurnDetector`` (auto-select + fallback + server defaults).

Covers:

- Auto-select via ``LIVEKIT_REMOTE_EOT_URL`` env var (with creds present, with
  creds missing → silent downgrade).
- Explicit-cloud missing creds raises.
- Cloud → local fallback triggers (transport raise, predict timeout).
- Fallback persistence across turns.
- Local-failure handling (default 1.0, retry on next turn).
- Per-session warning dedupe (one warning per failure mode).
- Server-provided default thresholds adopted from ``SessionCreated`` (protocol 1.1.13).
- Override resolution (scalar / dict / none) against the server defaults, the override warning,
  runtime ``update_options``, and the degenerate (no usable thresholds) → fallback path.
- Threshold rescaling against the server defaults on actual fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
from livekit.agents.inference.eot.transports import _LocalTransport
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr

pytestmark = pytest.mark.audio_eot

# Stand-in for the per-language defaults a 1.1.13 gateway returns in ``SessionCreated``.
SERVER_THRESHOLDS: dict[str, float] = {"en": 0.56, "ja": 0.37, "fr": 0.575}
SERVER_DEFAULT_THRESHOLD = 0.5


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
    """Fake transport that structurally satisfies ``_StreamingTurnDetectionTransport``.

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

    def attach(self, stream: _BaseStreamingTurnDetectorStream) -> None:
        self.stream_ref = stream

    async def run(self) -> None:
        self.run_calls += 1
        if self.run_behavior == "raise":
            assert self.run_exc is not None
            raise self.run_exc
        if self.run_behavior == "return":
            return
        # idle — sleep until cancelled
        await asyncio.Future()

    def run_inference(self, request_id: str) -> None:
        self.events.append(("run_inference", request_id))

    def push_frame(self, frame: Any) -> None:
        self.events.append(("push_frame", frame))

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
    """Construct a stream wired to a scripted transport.

    The cloud model starts with empty thresholds (its defaults arrive via ``SessionCreated`` —
    call ``stream._opts.thresholds._update_defaults`` to simulate that). The local mini model resolves its
    thresholds against ``LOCAL_LANGUAGES`` up front, matching the real constructor."""
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
        """A timed-out cancel_inference (AudioRecognition's predict timeout) swaps
        cloud to local."""
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
        """After cloud→local fallback the detector view (read by EOU metrics
        and by ``audio_recognition``) must report the post-fallback model +
        rescaled thresholds. The detector and stream share one ``ThresholdOptions``,
        so the fallback is visible to both without any copy-back."""
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
            stream._fall_back_to_local(reason=APIConnectionError("boom"))
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")

            assert detector.model == "turn-detector-v1-mini"
            local_threshold = await detector.unlikely_threshold(LanguageCode("en"))
            expected = LOCAL_LANGUAGES["en"] * (0.5 / SERVER_THRESHOLDS["en"])
            assert local_threshold == pytest.approx(expected)

            await stream.aclose()


class TestLocalFailureRetry:
    async def test_local_failure_emits_default_and_retries_next_turn(self) -> None:
        """Local _predict raising emits 1.0 for the turn and does NOT
        permanently disable local — the next turn invokes _predict again."""
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
            stream._fall_back_to_local(reason=APIConnectionError("boom2"))
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
        """A cloud detector has no per-language threshold until ``SessionCreated`` arrives, but
        reports the language as supported so the first turn isn't skipped."""
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
    """Backchannel thresholds: server-provided defaults, disabled on the mini model
    and after fallback (see TestResolveBackchannelThresholds for override layering)."""

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
    """User backchannel-threshold overrides layered against the server defaults,
    mirroring the EOT override resolution in TestResolveThresholds."""

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
            stream._fall_back_to_local(reason=APIConnectionError("boom"))
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
            stream._fall_back_to_local(reason=APIConnectionError("boom"))
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
            stream._fall_back_to_local(reason=APIConnectionError("boom"))
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
        """Cloud fails before any ``SessionCreated`` → no server map to rescale against, so the
        local table (with the override applied) is used directly."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
            stream = _make_stream_with_transport(transport, user_threshold=0.42)
            await _wait_for(lambda: stream.model == "turn-detector-v1-mini")
            assert stream.is_fallback is True
            # materialize_local_thresholds(0.42) → 0.42 for every local language
            assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.42)
            await stream.aclose()
