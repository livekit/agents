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

"""License-mode Krisp internals.

This module is private to the plugin. It wraps the public ``krisp_audio``
wheel and exposes a :class:`_LicenseKrispFrameProcessor` that
:class:`KrispVivaFilterFrameProcessor` instantiates when the user picks
:class:`KrispLicenseAuthProvider`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from livekit import rtc

from ._sdk import _KrispLicenseSDKManager
from .log import logger

__all__ = ["_KrispLicenseFrameProcessor", "_KrispLicenseSDKManager"]


class _KrispLicenseFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """License-mode FrameProcessor wrapping ``krisp_audio``.

    Internal implementation detail — users construct
    :class:`KrispVivaFilterFrameProcessor` and the facade picks this when the
    auth provider is :class:`KrispLicenseAuthProvider`.
    """

    def __init__(
        self,
        *,
        license_key: str,
        model_path: str,
        noise_suppression_level: int = 100,
        frame_duration_ms: int = 10,
        sample_rate: int | None = None,
    ) -> None:
        self._sdk_acquired = False
        self._filtering_enabled = True
        self._session: Any | None = None
        self._noise_suppression_level = noise_suppression_level
        self._sample_rate: int | None = None
        self._chunk_samples: int | None = None
        self._frame_duration_ms = frame_duration_ms
        self._model_path = model_path
        self._warned_channels = False

        # Adaptive frame-size buffering: krisp processes fixed-size chunks
        # (``frame_duration_ms`` worth of samples), but input frames may arrive
        # at any size. Incoming samples accumulate in ``_in_buf``, are processed
        # one whole chunk at a time, and the results queue in ``_out_buf``. Each
        # call emits whatever processed audio is ready (never zero-padded — see
        # ``_process``), so the output frame size floats and real audio is never
        # interrupted by injected silence.
        self._in_buf: np.ndarray = np.empty(0, dtype=np.int16)
        self._out_buf: np.ndarray = np.empty(0, dtype=np.int16)

        try:
            self._module = _KrispLicenseSDKManager.acquire(license_key)
            self._sdk_acquired = True
        except Exception as e:
            logger.error(f"Failed to acquire Krisp SDK: {e}")
            raise

        try:
            # Validate frame duration through the manager (raises if unsupported).
            _KrispLicenseSDKManager.frame_duration_enum(frame_duration_ms)

            # Pre-load the model now to fail fast on a bad license/model path.
            # The session is recreated automatically if the first frame arrives
            # at a different sample rate.
            init_sample_rate = sample_rate if sample_rate is not None else 16000
            self._create_session(init_sample_rate)
            logger.info(
                "Krisp frame processor initialized with %dHz session "
                "(adapts to the input sample rate and frame size)",
                init_sample_rate,
            )
        except Exception:
            if self._sdk_acquired:
                _KrispLicenseSDKManager.release()
                self._sdk_acquired = False
            raise

    def _create_session(self, sample_rate: int) -> None:
        """Create a new Krisp session for the given sample rate.

        Also recomputes the per-chunk sample count and resets the frame-size
        buffers, since both are tied to the sample rate.

        Args:
            sample_rate: The sample rate of the audio frames in Hz.
        """
        # If session already exists for this sample rate, don't recreate
        if self._session is not None and self._sample_rate == sample_rate:
            return

        logger.info("Creating Krisp session for sample rate: %dHz", sample_rate)

        model_info = self._module.ModelInfo()
        model_info.path = self._model_path

        nc_cfg = self._module.NcSessionConfig()
        nc_cfg.inputSampleRate = _KrispLicenseSDKManager.sample_rate_enum(sample_rate)
        nc_cfg.inputFrameDuration = _KrispLicenseSDKManager.frame_duration_enum(
            self._frame_duration_ms
        )
        nc_cfg.outputSampleRate = nc_cfg.inputSampleRate
        nc_cfg.modelInfo = model_info

        try:
            self._session = self._module.NcInt16.create(nc_cfg)
            self._sample_rate = sample_rate
            self._chunk_samples = int(sample_rate * self._frame_duration_ms / 1000)
            # The pending/processed buffers belong to the old rate; start fresh.
            self._in_buf = np.empty(0, dtype=np.int16)
            self._out_buf = np.empty(0, dtype=np.int16)
            logger.info("Krisp session created successfully")
        except Exception as e:
            logger.error(f"Failed to create Krisp session: {e}")
            raise

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._filtering_enabled:
            return frame

        if frame.num_channels != 1:
            # Krisp NC expects mono audio; pass multi-channel frames through.
            if not self._warned_channels:
                logger.warning(
                    "Krisp filter not applied: expected mono audio but got %d "
                    "channels; frames are passed through unprocessed.",
                    frame.num_channels,
                )
                self._warned_channels = True
            return frame

        # Adapt to the input sample rate, recreating the session on a change.
        if self._session is None or self._sample_rate != frame.sample_rate:
            self._create_session(frame.sample_rate)

        assert self._session is not None and self._chunk_samples is not None
        chunk = self._chunk_samples

        # Accumulate the incoming samples and process every whole chunk that is
        # now available (the input frame size need not match the chunk size).
        in_arr = np.frombuffer(frame.data, dtype=np.int16)
        self._in_buf = np.concatenate((self._in_buf, in_arr))

        n_chunks = len(self._in_buf) // chunk
        if n_chunks > 0:
            consumed = n_chunks * chunk
            pending = self._in_buf[:consumed]
            self._in_buf = self._in_buf[consumed:].copy()

            processed: list[np.ndarray] = []
            for i in range(n_chunks):
                chunk_in = pending[i * chunk : (i + 1) * chunk]
                try:
                    chunk_out = self._session.process(chunk_in, self._noise_suppression_level)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    chunk_out = chunk_in
                if chunk_out is None or len(chunk_out) != chunk:
                    logger.warning("Krisp returned unexpected output, using original audio")
                    chunk_out = chunk_in
                processed.append(np.asarray(chunk_out, dtype=np.int16))

            self._out_buf = np.concatenate([self._out_buf, *processed])

        # Emit the processed audio that is ready, keeping any surplus buffered
        # for the next call. We never zero-pad a deficit: doing so injects
        # silence *between* real audio (which recurs whenever the input frame
        # size isn't a multiple of the chunk size) and shows up as audible gaps
        # once frames are concatenated downstream. Instead the output frame size
        # floats — up to a full chunk of latency at the start (and an empty
        # frame if the buffer is momentarily starved), both of which downstream
        # handles fine — so real audio is never interrupted.
        n = frame.samples_per_channel
        k = min(n, len(self._out_buf))
        out = self._out_buf[:k]
        self._out_buf = self._out_buf[k:].copy()

        return rtc.AudioFrame(
            data=out.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=len(out),
        )

    @property
    def enabled(self) -> bool:
        return self._filtering_enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._filtering_enabled = value

    @property
    def noise_suppression_level(self) -> float:
        return self._noise_suppression_level

    @noise_suppression_level.setter
    def noise_suppression_level(self, value: float) -> None:
        # Applied on the next processed frame — the level is passed per-call to
        # the Krisp session, so no session recreation is needed.
        self._noise_suppression_level = int(max(0, min(100, value)))

    def _close(self) -> None:
        # _close runs on track transitions, not just final destruction. Drop
        # the session (and any buffered audio) here but keep the SDK reference
        # until __del__. A later frame recreates the session on demand.
        if self._session is not None:
            self._session = None
        self._in_buf = np.empty(0, dtype=np.int16)
        self._out_buf = np.empty(0, dtype=np.int16)
        logger.debug("Krisp frame processor session closed")

    def __del__(self) -> None:
        # During Python shutdown, the manager module may already be torn down.
        if _KrispLicenseSDKManager is None:
            return

        if getattr(self, "_sdk_acquired", False):
            try:
                if getattr(self, "_session", None) is not None:
                    self._session = None
                _KrispLicenseSDKManager.release()
                self._sdk_acquired = False
            except Exception:
                # Silently ignore errors during shutdown
                pass

    def __enter__(self) -> _KrispLicenseFrameProcessor:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        self._close()
        return False
