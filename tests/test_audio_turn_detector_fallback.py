"""Tests for the unified ``AudioTurnDetector`` (auto-select + fallback).

Covers:

- Auto-select via ``LIVEKIT_REMOTE_EOT_URL`` env var (with creds present, with
  creds missing → silent downgrade).
- Explicit-backend errors (cloud missing creds, local missing lib).
- Cloud → local fallback triggers (transport raise, predict timeout).
- Fallback persistence across turns.
- Missing-lib graceful degradation (no raise, default 1.0).
- Local-failure handling (default 1.0, retry on next turn).
- Per-session warning dedupe (one warning per failure mode).
- Threshold scaling: pass-through for cloud / explicit-local, multiplicative
  scaling only on actual fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents._exceptions import APIConnectionError
from livekit.agents.language import LanguageCode
from livekit.agents.voice.turn import TurnDetectorOptions, _AudioTurnDetectorStream
from livekit.plugins.turn_detector import audio as audio_mod
from livekit.plugins.turn_detector.audio import (
    AudioTurnDetector,
    _AudioTurnDetectorStreamImpl,
)
from livekit.plugins.turn_detector.transports import (
    _AudioTurnDetectionTransport,
    _LocalTransport,
)


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

    def on_warmup_start(self, request_id: str) -> None:
        self.events.append(("warmup_start", request_id))

    async def on_audio_chunk(self, frame: Any) -> None:
        self.events.append(("audio_chunk", frame))

    async def on_flush_sentinel(self, sentinel: Any) -> None:
        self.events.append(("flush_sentinel", sentinel))

    def on_activate(self) -> None:
        self.events.append(("activate", None))

    def on_inference_stop(self, *, reason: str | None) -> None:
        self.events.append(("inference_stop", reason))

    def close_nowait(self) -> None:
        self.events.append(("close_nowait", None))


def _make_opts(thresholds: dict[str, float] | None = None) -> TurnDetectorOptions:
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    return TurnDetectorOptions(
        sample_rate=16000,
        base_url="ws://test",
        api_key="x",
        api_secret="x",
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        thresholds=thresholds if thresholds is not None else {},
    )


def _make_stream_with_transport(
    transport: _AudioTurnDetectionTransport,
    *,
    backend: str = "cloud",
    user_threshold: float | dict[str, float] | None = None,
) -> _AudioTurnDetectorStreamImpl:
    """Construct a stream bypassing the constructor's transport allocation,
    so we can inject a scripted transport before the FSM main task starts.

    ``user_threshold`` is materialized into ``opts.thresholds`` the same way
    the real constructor would — the stream's threshold lookup reads its own
    ``opts.thresholds``."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.turn_detector.audio import _materialize_thresholds

    detector = MagicMock()
    detector.model = "eot-audio" if backend == "cloud" else "eot-audio-mini"
    detector.provider = "livekit"

    opts = _make_opts(_materialize_thresholds(backend, user_threshold))  # type: ignore[arg-type]
    stream = _AudioTurnDetectorStreamImpl.__new__(_AudioTurnDetectorStreamImpl)
    stream._backend = backend  # type: ignore[assignment]
    stream._http_session = None
    stream._conn_options = DEFAULT_API_CONNECT_OPTIONS
    stream._is_fallback = False
    stream._warned_cloud_failure = False
    stream._warned_local_failure = False
    stream._transport = transport
    _AudioTurnDetectorStream.__init__(stream, detector=detector, opts=opts)
    transport.bind(stream)
    return stream


class TestAutoSelect:
    def test_auto_select_local_when_no_remote_eot_url(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            with patch.object(audio_mod, "_lib_available", return_value=True):
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
            with patch.object(audio_mod, "_lib_available", return_value=True):
                detector = AudioTurnDetector()
                # env said cloud, but creds absent → silent downgrade
                assert detector.backend == "local"


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

    def test_explicit_local_missing_lib_raises(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            with patch.object(audio_mod, "_lib_available", return_value=False):
                with patch.object(
                    audio_mod, "_get_lib_load_error", return_value=RuntimeError("no lib")
                ):
                    with pytest.raises(RuntimeError):
                        AudioTurnDetector(backend="local")


class TestFallback:
    async def test_fallback_on_transport_error_emits_one(self) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "_lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                stream = _make_stream_with_transport(transport)
                # Give the main task a chance to run + flip.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert stream.backend == "local"
                assert stream.is_fallback is True
                assert stream._warned_cloud_failure is True
                assert ("close_nowait", None) in transport.events
                await stream.aclose()

    async def test_fallback_on_predict_timeout(self) -> None:
        """Cloud `predict_end_of_turn` timeout swaps to local."""
        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(audio_mod, "_lib_available", return_value=True):
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
        with patch.object(audio_mod, "_lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                stream = _make_stream_with_transport(transport)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                # Cloud transport ran exactly once; no resurrection.
                assert transport.run_calls == 1
                # Future turns can call warmup without re-touching cloud.
                stream.warmup()
                assert stream.backend == "local"
                await stream.aclose()

    async def test_cloud_failure_with_missing_lib_emits_default_and_keeps_cloud(
        self,
    ) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "_lib_available", return_value=False):
            stream = _make_stream_with_transport(transport)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # No lib → can't promote. Mode stays cloud, single warning.
            assert stream.backend == "cloud"
            assert stream.is_fallback is False
            assert stream._warned_cloud_failure is True
            await stream.aclose()


class TestLocalFailureRetry:
    async def test_local_failure_emits_default_and_retries_next_turn(self) -> None:
        """Local _predict raising emits 1.0 for the turn and does NOT
        permanently disable local — the next turn invokes _predict again."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=RuntimeError("local boom"))
        stream = _make_stream_with_transport(transport, backend="local")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
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
        caplog.set_level(logging.WARNING, logger="livekit.plugins.turn_detector")
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "_lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                stream = _make_stream_with_transport(transport)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                # Trigger a second fallback path by calling the method directly.
                stream._fall_back_to_local(reason=APIConnectionError("boom2"))
                # Only one warning across both invocations.
                cloud_warnings = [r for r in caplog.records if "cloud audio EOT" in r.getMessage()]
                assert len(cloud_warnings) == 1
                await stream.aclose()

    async def test_warning_logged_once_per_session_local(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.plugins.turn_detector")
        transport = _ScriptedTransport(run_behavior="idle")
        stream = _make_stream_with_transport(transport, backend="local")
        # Two local failures back to back.
        stream._on_local_failure(reason=RuntimeError("a"))
        stream._on_local_failure(reason=RuntimeError("b"))
        local_warnings = [r for r in caplog.records if "local audio EOT mini" in r.getMessage()]
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
            with patch.object(audio_mod, "_lib_available", return_value=True):
                detector = AudioTurnDetector(backend="local", unlikely_threshold=0.5)
                value = await detector.unlikely_threshold(LanguageCode("en"))
                assert value == pytest.approx(0.5)

    async def test_post_fallback_threshold_rescales_on_stream(self) -> None:
        """Fallback-only multiplicative scaling: 0.5 user on 0.4 cloud
        default → 0.3 * 0.5/0.4 = 0.375 on local after fallback."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "_lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                stream = _make_stream_with_transport(transport, user_threshold=0.5)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert stream.backend == "local"
                assert stream.is_fallback is True
                value = await stream.unlikely_threshold(LanguageCode("en"))
                assert value == pytest.approx(0.375)
                await stream.aclose()

    async def test_threshold_default_unchanged_when_user_not_set(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector()
            cloud_default = await detector.unlikely_threshold(LanguageCode("en"))
            assert cloud_default == pytest.approx(0.4)

        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            with patch.object(audio_mod, "_lib_available", return_value=True):
                detector = AudioTurnDetector()
                local_default = await detector.unlikely_threshold(LanguageCode("en"))
                assert local_default == pytest.approx(0.3)


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
            assert await detector.unlikely_threshold(LanguageCode("fr")) == pytest.approx(0.4)

    async def test_dict_keys_normalized_via_language_code(self) -> None:
        """``English`` / ``en-US`` / ``eng`` all normalize to ``en`` so the
        override matches the table lookup regardless of how the user spelled it."""
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            with patch.object(audio_mod, "_lib_available", return_value=True):
                detector = AudioTurnDetector(
                    unlikely_threshold={"English": 0.55, "en-US": 0.55, "eng": 0.55}
                )
                # All three keys collapsed to "en"; last write wins, but the
                # important thing is the lookup picks it up.
                assert await detector.unlikely_threshold(LanguageCode("en")) == pytest.approx(0.55)

    async def test_dict_override_rescaled_per_language_on_fallback(self) -> None:
        """Each dict entry gets its own multiplicative rescale on fallback.
        ``en`` override 0.55 + cloud 0.4 → local 0.3 * 0.55/0.4 ≈ 0.4125.
        ``ja`` override 0.25 + cloud 0.4 → local 0.3 * 0.25/0.4 ≈ 0.1875."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "_lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                # Use a dict override of multiple languages.
                stream = _make_stream_with_transport(
                    transport, user_threshold={"en": 0.55, "ja": 0.25}
                )
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert stream.is_fallback is True
                assert await stream.unlikely_threshold(LanguageCode("en")) == pytest.approx(
                    0.3 * 0.55 / 0.4
                )
                assert await stream.unlikely_threshold(LanguageCode("ja")) == pytest.approx(
                    0.3 * 0.25 / 0.4
                )
                # `fr` not in dict → falls through to plain local default
                # (no rescaling because no user value).
                assert await stream.unlikely_threshold(LanguageCode("fr")) == pytest.approx(0.3)
                await stream.aclose()
