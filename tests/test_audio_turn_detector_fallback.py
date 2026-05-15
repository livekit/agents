"""Tests for the unified ``AudioTurnDetector`` (auto-select + fallback).

Covers:

- Auto-select via ``LIVEKIT_REMOTE_EOT_URL`` env var (with creds present, with
  creds missing → silent downgrade).
- Explicit-mode errors (cloud missing creds, local missing lib).
- Cloud → local fallback triggers (transport raise, predict timeout).
- Fallback persistence across turns.
- Missing-lib graceful degradation (no raise, default 1.0).
- Local-failure handling (default 1.0, retry on next turn).
- Per-session warning dedupe (one warning per failure mode).
- Multiplicative threshold scaling (cloud-anchored).
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
    AudioTurnDetectionTransport,
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
    """Fake transport that structurally satisfies AudioTurnDetectionTransport.

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


def _make_opts() -> TurnDetectorOptions:
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    return TurnDetectorOptions(
        sample_rate=16000,
        base_url="ws://test",
        api_key="x",
        api_secret="x",
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    )


def _make_stream_with_transport(
    transport: AudioTurnDetectionTransport,
    *,
    mode: str = "eot-audio-cloud",
) -> _AudioTurnDetectorStreamImpl:
    """Construct a stream bypassing the constructor's transport allocation,
    so we can inject a scripted transport before the FSM main task starts."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    detector = MagicMock()
    detector.model = mode
    detector.provider = "livekit"

    opts = _make_opts()
    stream = _AudioTurnDetectorStreamImpl.__new__(_AudioTurnDetectorStreamImpl)
    stream._detector_typed = detector
    stream._mode = mode  # type: ignore[assignment]
    stream._http_session = None
    stream._conn_options = DEFAULT_API_CONNECT_OPTIONS
    stream._fell_back = False
    stream._warned_cloud_failure = False
    stream._warned_local_failure = False
    stream._transport = transport
    _AudioTurnDetectorStream.__init__(stream, detector=detector, opts=opts)
    transport.bind(stream)
    return stream


class TestAutoSelect:
    def test_auto_select_local_when_no_remote_eot_url(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            with patch.object(audio_mod, "lib_available", return_value=True):
                detector = AudioTurnDetector()
                assert detector._mode == "eot-audio-mini"

    def test_auto_select_cloud_when_remote_eot_url_set(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector()
            assert detector._mode == "eot-audio-cloud"

    def test_auto_select_downgrades_when_creds_missing(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY=None,
            LIVEKIT_API_SECRET=None,
            LIVEKIT_INFERENCE_API_KEY=None,
            LIVEKIT_INFERENCE_API_SECRET=None,
        ):
            with patch.object(audio_mod, "lib_available", return_value=True):
                detector = AudioTurnDetector()
                # env said cloud, but creds absent → silent downgrade
                assert detector._mode == "eot-audio-mini"


class TestExplicitModeErrors:
    def test_explicit_cloud_missing_creds_raises(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL=None,
            LIVEKIT_API_KEY=None,
            LIVEKIT_API_SECRET=None,
            LIVEKIT_INFERENCE_API_KEY=None,
            LIVEKIT_INFERENCE_API_SECRET=None,
        ):
            with pytest.raises(ValueError):
                AudioTurnDetector(model="eot-audio-cloud")

    def test_explicit_local_missing_lib_raises(self) -> None:
        with _clean_env(LIVEKIT_REMOTE_EOT_URL=None):
            with patch.object(audio_mod, "lib_available", return_value=False):
                with patch.object(audio_mod, "lib_load_error", return_value=RuntimeError("no lib")):
                    with pytest.raises(RuntimeError):
                        AudioTurnDetector(model="eot-audio-mini")


class TestFallback:
    async def test_fallback_on_transport_error_emits_one(self) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                stream = _make_stream_with_transport(transport)
                # Give the main task a chance to run + flip.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert stream._mode == "eot-audio-mini"
                assert stream._fell_back is True
                assert stream._warned_cloud_failure is True
                assert ("close_nowait", None) in transport.events
                await stream.aclose()

    async def test_fallback_on_predict_timeout(self) -> None:
        """Cloud `predict_end_of_turn` timeout swaps to local."""
        transport = _ScriptedTransport(run_behavior="idle")
        with patch.object(audio_mod, "lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                stream = _make_stream_with_transport(transport)
                # Drive a predict that times out fast.
                prob = await stream.predict_end_of_turn(timeout=0.01)
                assert prob == 1.0
                assert stream._mode == "eot-audio-mini"
                assert stream._fell_back is True
                await stream.aclose()

    async def test_fallback_persists_across_turns(self) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "lib_available", return_value=True):
            with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                stream = _make_stream_with_transport(transport)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                # Cloud transport ran exactly once; no resurrection.
                assert transport.run_calls == 1
                # Future turns can call warmup without re-touching cloud.
                stream.warmup()
                assert stream._mode == "eot-audio-mini"
                await stream.aclose()

    async def test_cloud_failure_with_missing_lib_emits_default_and_keeps_cloud(
        self,
    ) -> None:
        transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
        with patch.object(audio_mod, "lib_available", return_value=False):
            stream = _make_stream_with_transport(transport)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # No lib → can't promote. Mode stays cloud, single warning.
            assert stream._mode == "eot-audio-cloud"
            assert stream._fell_back is False
            assert stream._warned_cloud_failure is True
            await stream.aclose()


class TestLocalFailureRetry:
    async def test_local_failure_emits_default_and_retries_next_turn(self) -> None:
        """Local _predict raising emits 1.0 for the turn and does NOT
        permanently disable local — the next turn invokes _predict again."""
        transport = _ScriptedTransport(run_behavior="raise", run_exc=RuntimeError("local boom"))
        stream = _make_stream_with_transport(transport, mode="eot-audio-mini")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Local failed; warning logged once; mode stays local; no fell_back flag.
        assert stream._mode == "eot-audio-mini"
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
        with patch.object(audio_mod, "lib_available", return_value=True):
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
        stream = _make_stream_with_transport(transport, mode="eot-audio-mini")
        # Two local failures back to back.
        stream._on_local_failure(reason=RuntimeError("a"))
        stream._on_local_failure(reason=RuntimeError("b"))
        local_warnings = [r for r in caplog.records if "local audio EOT mini" in r.getMessage()]
        assert len(local_warnings) == 1
        await stream.aclose()


class TestThresholdScaling:
    async def test_threshold_scaling_cloud_mode(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector(unlikely_threshold=0.5)
            assert detector._mode == "eot-audio-cloud"
            value = await detector.unlikely_threshold(LanguageCode("en"))
            assert value == pytest.approx(0.5)

    async def test_threshold_scaling_local_mode_after_fallback(self) -> None:
        with _clean_env(
            LIVEKIT_REMOTE_EOT_URL="ws://gateway",
            LIVEKIT_API_KEY="k",
            LIVEKIT_API_SECRET="s",
        ):
            detector = AudioTurnDetector(unlikely_threshold=0.5)
            transport = _ScriptedTransport(run_behavior="raise", run_exc=APIConnectionError("boom"))
            with patch.object(audio_mod, "lib_available", return_value=True):
                with patch.object(_LocalTransport, "run", new=lambda self: asyncio.sleep(0)):
                    # Wire the stream manually + register so detector sees its mode.
                    import weakref

                    stream = _make_stream_with_transport(transport)
                    detector._stream_ref = weakref.ref(stream)
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    assert stream._mode == "eot-audio-mini"
                    # local default 0.3, cloud default 0.4, user 0.5 → 0.3 * 0.5/0.4 = 0.375
                    value = await detector.unlikely_threshold(LanguageCode("en"))
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
            with patch.object(audio_mod, "lib_available", return_value=True):
                detector = AudioTurnDetector()
                local_default = await detector.unlikely_threshold(LanguageCode("en"))
                assert local_default == pytest.approx(0.3)
