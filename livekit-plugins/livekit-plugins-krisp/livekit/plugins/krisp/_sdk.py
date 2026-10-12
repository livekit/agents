# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared license-mode plumbing for the public ``krisp_audio`` wheel.

Private to the plugin. Holds the process-wide SDK reference counter used by
every license-mode component (noise cancellation, turn taking, interruption
prediction), plus small helpers for building sessions and feeding them
fixed-size frames.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from threading import Lock
from typing import Any

import numpy as np

from livekit import rtc

from .log import logger

KRISP_SAMPLE_RATE = 16000
"""Sample rate the turn-taking / interruption sessions run at; input is resampled to it."""


def frame_samples(frame_duration_ms: int) -> int:
    return KRISP_SAMPLE_RATE * frame_duration_ms // 1000


class _KrispLicenseSDKManager:
    """Process-singleton ref counter for the public ``krisp_audio`` wheel.

    Krisp's ``globalInit`` / ``globalDestroy`` are process-global, so this
    manager keeps a single SDK alive across every license-mode session in the
    process. Lazy-imports ``krisp_audio`` on first acquire — never imported
    in cloud-only deployments.
    """

    _module: Any = None
    _sample_rates: dict[int, Any] | None = None
    _frame_durations: dict[int, Any] | None = None
    _reference_count: int = 0
    _lock = Lock()

    @classmethod
    def acquire(cls, license_key: str) -> Any:
        """Acquire a reference, returning the imported ``krisp_audio`` module."""
        with cls._lock:
            if cls._reference_count == 0:
                try:
                    import krisp_audio  # type: ignore[import-not-found]
                except ModuleNotFoundError as e:
                    raise RuntimeError(
                        "krisp-audio is not installed. Install Krisp's wheel "
                        "(`pip install krisp-audio`) or use the default "
                        "LiveKitCloudAuthProvider."
                    ) from e

                try:
                    krisp_audio.globalInit(
                        "",
                        license_key,
                        cls._licensing_error_callback,
                        cls._log_callback,
                        krisp_audio.LogLevel.Off,
                    )
                except TypeError:
                    # wheels built without licensing take no license key
                    krisp_audio.globalInit("", cls._log_callback, krisp_audio.LogLevel.Off)
                cls._module = krisp_audio
                cls._sample_rates = {
                    8000: krisp_audio.SamplingRate.Sr8000Hz,
                    16000: krisp_audio.SamplingRate.Sr16000Hz,
                    24000: krisp_audio.SamplingRate.Sr24000Hz,
                    32000: krisp_audio.SamplingRate.Sr32000Hz,
                    44100: krisp_audio.SamplingRate.Sr44100Hz,
                    48000: krisp_audio.SamplingRate.Sr48000Hz,
                }
                cls._frame_durations = {
                    10: krisp_audio.FrameDuration.Fd10ms,
                    15: krisp_audio.FrameDuration.Fd15ms,
                    20: krisp_audio.FrameDuration.Fd20ms,
                    30: krisp_audio.FrameDuration.Fd30ms,
                    32: krisp_audio.FrameDuration.Fd32ms,
                }
                version = krisp_audio.getVersion()
                logger.debug(
                    "Krisp Audio SDK (license) initialized - Version: %d.%d.%d",
                    version.major,
                    version.minor,
                    version.patch,
                )
            cls._reference_count += 1
            logger.debug("Krisp SDK (license) reference count: %d", cls._reference_count)
            return cls._module

    @classmethod
    def release(cls) -> None:
        """Release a reference, destroying the SDK on the last release."""
        with cls._lock:
            if cls._reference_count == 0:
                return
            cls._reference_count -= 1
            logger.debug("Krisp SDK (license) reference count: %d", cls._reference_count)
            if cls._reference_count == 0 and cls._module is not None:
                try:
                    cls._module.globalDestroy()
                    logger.debug("Krisp Audio SDK (license) destroyed")
                except Exception as e:
                    logger.error(f"Error during Krisp SDK cleanup: {e}")
                finally:
                    cls._module = None
                    cls._sample_rates = None
                    cls._frame_durations = None

    @classmethod
    def sample_rate_enum(cls, sample_rate: int) -> Any:
        assert cls._sample_rates is not None, "SDK not acquired"
        if sample_rate not in cls._sample_rates:
            supported = ", ".join(str(r) for r in sorted(cls._sample_rates))
            raise ValueError(
                f"Unsupported sample rate: {sample_rate} Hz. Supported rates: {supported} Hz"
            )
        return cls._sample_rates[sample_rate]

    @classmethod
    def frame_duration_enum(cls, frame_duration_ms: int) -> Any:
        assert cls._frame_durations is not None, "SDK not acquired"
        if frame_duration_ms not in cls._frame_durations:
            supported = ", ".join(str(d) for d in sorted(cls._frame_durations))
            raise ValueError(
                f"Unsupported frame duration: {frame_duration_ms} ms. "
                f"Supported durations: {supported} ms"
            )
        return cls._frame_durations[frame_duration_ms]

    @staticmethod
    def _log_callback(log_message: str, log_level: Any) -> None:
        logger.debug(f"[Krisp {log_level}] {log_message}")

    @staticmethod
    def _licensing_error_callback(error: Any, error_message: str) -> None:
        logger.error(f"[Krisp Licensing Error: {error}] {error_message}")


def resolve_model_path(model_path: str | None, env_var: str, what: str) -> str:
    """Resolve a ``.kef`` model path from an argument or environment variable."""
    resolved = model_path or os.getenv(env_var)
    if not resolved:
        raise ValueError(f"Krisp {what} model path is required. Pass it or set {env_var}.")
    if not resolved.endswith(".kef"):
        raise ValueError(f"Krisp {what} model must have .kef extension")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Krisp {what} model file not found: {resolved}")
    return resolved


def resolve_license_key(license_key: str | None) -> str:
    return license_key or os.getenv("KRISP_VIVA_SDK_LICENSE_KEY") or ""


class _SessionPair:
    """A Krisp VAD session and a detector session (TT or IP) fed the same frames.

    Owns one SDK reference for its lifetime. The VAD supplies the per-frame
    ``voiceDetected`` flag the detector needs. Sessions run on float32 frames of
    ``frame_duration_ms`` at :data:`KRISP_SAMPLE_RATE`.

    ``process`` is blocking (the binding releases the GIL), so callers run it
    in a worker thread. ``close`` waits for an in-flight ``process`` call to return
    before freeing the sessions, and ``process`` after ``close`` is a no-op, so a
    stream may close while its worker thread is still running.
    """

    def __init__(
        self,
        *,
        kind: str,
        license_key: str,
        model_path: str,
        vad_model_path: str,
        vad_threshold: float,
        frame_duration_ms: int,
    ) -> None:
        self._vad_threshold = vad_threshold
        self.frame_samples = frame_samples(frame_duration_ms)
        self._lock = Lock()  # serializes process() against close()

        module = _KrispLicenseSDKManager.acquire(license_key)
        self._released = False
        try:
            sample_rate = _KrispLicenseSDKManager.sample_rate_enum(KRISP_SAMPLE_RATE)
            frame_duration = _KrispLicenseSDKManager.frame_duration_enum(frame_duration_ms)

            # the session config holds a raw pointer to ModelInfo, keep both alive
            self._vad_model_info = module.ModelInfo()
            self._vad_model_info.path = vad_model_path
            vad_cfg = module.VadSessionConfig()
            vad_cfg.inputSampleRate = sample_rate
            vad_cfg.inputFrameDuration = frame_duration
            vad_cfg.modelInfo = self._vad_model_info
            self._vad = module.VadFloat.create(vad_cfg)

            self._model_info = module.ModelInfo()
            self._model_info.path = model_path
            if kind == "tt":
                cfg = module.TtSessionConfig()
                session_cls = module.TtFloat
            else:
                cfg = module.IpSessionConfig()
                session_cls = module.IpFloat
            cfg.inputSampleRate = sample_rate
            cfg.inputFrameDuration = frame_duration
            cfg.modelInfo = self._model_info
            self._session = session_cls.create(cfg)
        except Exception:
            self.close()
            raise

    def process(self, frame: np.ndarray, bot_speaking: bool) -> tuple[bool, float]:
        """Run one frame through the VAD and the detector; returns (voice, probability).

        Returns ``(False, 0.0)`` once the pair is closed.
        """
        with self._lock:
            if self._released:
                return False, 0.0
            voice = bool(self._vad.process(frame) >= self._vad_threshold)
            return voice, float(self._session.process(frame, voice, bot_speaking))

    def close(self) -> None:
        with self._lock:  # waits for an in-flight process() call
            if self._released:
                return
            self._released = True
            # drop every krisp_audio object: a closed pair can stay referenced (e.g. by a
            # job thread still alive at interpreter exit) and would otherwise outlive the SDK
            self._session = None
            self._vad = None
            self._model_info = None
            self._vad_model_info = None
        _KrispLicenseSDKManager.release()


async def open_in_thread(factory: Callable[[], _SessionPair]) -> _SessionPair:
    """Create a session pair in a worker thread, keeping model loading off the event loop.

    If the caller is cancelled while the models load, the pair is closed once ready so
    its SDK reference is not leaked.
    """
    fut = asyncio.ensure_future(asyncio.to_thread(factory))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:

        def _close_when_ready(f: asyncio.Future[_SessionPair]) -> None:
            if not f.cancelled() and f.exception() is None:
                f.result().close()

        fut.add_done_callback(_close_when_ready)
        raise


class _FrameChunker:
    """Resamples mono audio to :data:`KRISP_SAMPLE_RATE` and cuts it into fixed float32 frames."""

    def __init__(self, frame_samples: int) -> None:
        self._frame_samples = frame_samples
        self._buf = np.empty(0, dtype=np.float32)
        self._resampler: rtc.AudioResampler | None = None
        self._input_rate: int | None = None
        self._warned_channels = False

    def push(self, frame: rtc.AudioFrame) -> list[np.ndarray]:
        if frame.num_channels != 1:
            if not self._warned_channels:
                logger.warning(
                    "Krisp detector expects mono audio but got %d channels; frames are skipped.",
                    frame.num_channels,
                )
                self._warned_channels = True
            return []

        if frame.sample_rate != self._input_rate:
            # samples buffered at the previous rate must not be glued to the new audio
            self._buf = np.empty(0, dtype=np.float32)
            self._input_rate = frame.sample_rate
            self._resampler = (
                rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=KRISP_SAMPLE_RATE,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )
                if frame.sample_rate != KRISP_SAMPLE_RATE
                else None
            )

        resampled = self._resampler.push(frame) if self._resampler is not None else [frame]
        for f in resampled:
            samples = np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
            self._buf = np.concatenate((self._buf, samples))

        n = len(self._buf) // self._frame_samples
        if n == 0:
            return []
        consumed = n * self._frame_samples
        frames = list(self._buf[:consumed].reshape(n, self._frame_samples))
        self._buf = self._buf[consumed:].copy()
        return frames
