# Copyright 2023 LiveKit, Inc.
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

"""Krisp VIVA noise reduction audio filter for LiveKit Agents."""

from __future__ import annotations

import os
import warnings
from typing import Any, Literal

import numpy as np

from livekit import rtc

from .auth import (
    KrispAuthProvider,
    KrispBackend,
    KrispLicenseAuthProvider,
    LiveKitCloudAuthProvider,
)
from .krisp_instance import KrispSDKManager
from .log import logger

if not hasattr(rtc, "FrameProcessor"):
    raise ImportError(
        "FrameProcessor is not available in your livekit-rtc version. "
        "KrispVivaFilterFrameProcessor requires livekit-rtc >= 1.0.23 with FrameProcessor support. "
        "Please update livekit-rtc: pip install --upgrade 'livekit>=1.0.23'"
    )

_LEGACY_DEPRECATION_SHOWN = False


def _resolve_auth_provider(
    auth_provider: KrispAuthProvider | None,
    model_path: str | None,
) -> KrispAuthProvider:
    """Pick the auth provider, applying backwards-compat for legacy kwargs.

    Resolution rules:
    1. ``auth_provider`` given → use it.
    2. ``model_path`` given OR ``KRISP_VIVA_SDK_LICENSE_KEY`` env set →
       construct ``KrispLicenseAuthProvider`` (deprecation warning if
       ``model_path`` was passed explicitly).
    3. Otherwise → default ``LiveKitCloudAuthProvider``.
    """
    global _LEGACY_DEPRECATION_SHOWN

    if auth_provider is not None:
        if model_path is not None:
            raise ValueError(
                "Pass model_path on KrispLicenseAuthProvider(model_path=...) "
                "instead of on KrispVivaFilterFrameProcessor when an "
                "auth_provider is supplied."
            )
        return auth_provider

    if model_path is not None:
        if not _LEGACY_DEPRECATION_SHOWN:
            warnings.warn(
                "Passing `model_path` to KrispVivaFilterFrameProcessor is "
                "deprecated. Use `auth_provider=KrispLicenseAuthProvider("
                "model_path=...)` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            _LEGACY_DEPRECATION_SHOWN = True
        return KrispLicenseAuthProvider(model_path=model_path)

    if os.getenv("KRISP_VIVA_SDK_LICENSE_KEY"):
        return KrispLicenseAuthProvider()

    return LiveKitCloudAuthProvider()


class KrispVivaFilterFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """FrameProcessor for Krisp noise reduction.

    Example:
        ```python
        from livekit.agents import room_io
        from livekit.plugins import krisp

        # Default: uses LiveKit Cloud auth + bundled model.
        processor = krisp.KrispVivaFilterFrameProcessor()

        # Or, explicit Krisp-direct auth with a license + model file.
        processor = krisp.KrispVivaFilterFrameProcessor(
            auth_provider=krisp.KrispLicenseAuthProvider(
                license_key="...",
                model_path="/path/to/model.kef",
            ),
        )

        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    sample_rate=16000,
                    frame_size_ms=10,
                    noise_cancellation=processor,
                ),
            ),
        )
        ```
    """

    def __init__(
        self,
        *,
        auth_provider: KrispAuthProvider | None = None,
        model_path: str | None = None,
        noise_suppression_level: int = 100,
        frame_duration_ms: int = 10,
        sample_rate: int | None = None,
    ) -> None:
        """Initialize the Krisp frame processor.

        Args:
            auth_provider: Authentication provider. Defaults to
                :class:`LiveKitCloudAuthProvider` (LiveKit Cloud auth + bundled
                model). Pass :class:`KrispLicenseAuthProvider` to use a Krisp
                license key + ``.kef`` model file directly.
            model_path: **Deprecated.** Use
                ``auth_provider=KrispLicenseAuthProvider(model_path=...)``
                instead. Path to the Krisp model file (``.kef``).
            noise_suppression_level: Noise suppression level (0-100).
            frame_duration_ms: Frame duration in milliseconds (10, 15, 20, 30
                or 32).
            sample_rate: Sample rate in Hz. Defaults to 16000.
        """
        provider = _resolve_auth_provider(auth_provider, model_path)

        self._provider = provider
        self._sdk_acquired = False
        self._filtering_enabled = True
        self._session: Any | None = None
        self._noise_suppression_level = noise_suppression_level
        self._sample_rate: int | None = None
        self._frame_duration_ms = frame_duration_ms
        self._desired_sample_rate = sample_rate if sample_rate is not None else 16000
        self._backend: KrispBackend | None = None
        self._model_path: str | None = (
            provider.model_path if isinstance(provider, KrispLicenseAuthProvider) else None
        )

        try:
            self._backend = KrispSDKManager.acquire(provider)
            self._sdk_acquired = True
        except Exception as e:
            logger.error(f"Failed to acquire Krisp SDK: {e}")
            raise

        try:
            if frame_duration_ms not in self._backend.frame_durations:
                raise ValueError(
                    f"Unsupported frame duration: {frame_duration_ms}ms. "
                    f"Supported durations: {sorted(self._backend.frame_durations)}"
                )

            if self._backend.ready:
                # License backend: globalInit already happened, pre-load the
                # session so the model is hot when the first frame arrives.
                self._create_session(self._desired_sample_rate)
                logger.info(
                    "Krisp frame processor initialized with %dHz session "
                    "(model pre-loaded, will recreate session if different sample rate)",
                    self._desired_sample_rate,
                )
            else:
                # Cloud backend: real init + session creation are deferred
                # until the framework hands us credentials via
                # _on_credentials_updated.
                logger.debug(
                    "Krisp frame processor initialized; deferring session "
                    "creation until credentials arrive"
                )
        except Exception:
            if self._sdk_acquired:
                KrispSDKManager.release()
                self._sdk_acquired = False
            raise

    def _create_session(self, sample_rate: int) -> None:
        if self._session is not None and self._sample_rate == sample_rate:
            return

        assert self._backend is not None
        logger.info("Creating Krisp session for sample rate: %dHz", sample_rate)
        try:
            self._session = self._backend.create_session(
                sample_rate,
                self._frame_duration_ms,
                model_path=self._model_path,
            )
            self._sample_rate = sample_rate
            logger.info("Krisp session created successfully")
        except Exception as e:
            logger.error(f"Failed to create Krisp session: {e}")
            raise

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._filtering_enabled:
            return frame

        if self._session is None or self._sample_rate != frame.sample_rate:
            raise ValueError(f"Session not created or sample rate mismatch: {frame.sample_rate}Hz")

        expected_samples = int((frame.sample_rate * self._frame_duration_ms) / 1000)
        if frame.samples_per_channel != expected_samples:
            raise ValueError(
                f"Frame size mismatch: expected {expected_samples} samples "
                f"({self._frame_duration_ms}ms @ {frame.sample_rate}Hz), "
                f"got {frame.samples_per_channel} samples"
            )

        audio_samples = np.frombuffer(frame.data, dtype=np.int16)

        try:
            filtered_samples = self._session.process(audio_samples, self._noise_suppression_level)

            if filtered_samples is None or len(filtered_samples) == 0:
                logger.warning("Krisp returned empty output, using original audio")
                filtered_samples = audio_samples
            elif len(filtered_samples) != len(audio_samples):
                logger.warning(
                    f"Krisp output size mismatch: expected {len(audio_samples)}, "
                    f"got {len(filtered_samples)}, using original audio"
                )
                filtered_samples = audio_samples

            return rtc.AudioFrame(
                data=filtered_samples.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=len(filtered_samples),
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        return self._process(frame)

    def _on_stream_info_updated(
        self,
        *,
        room_name: str,
        participant_identity: str,
        publication_sid: str,
    ) -> None:
        if self._backend is None:
            return
        self._provider.update_stream_info(
            self._backend,
            room_name=room_name,
            participant_identity=participant_identity,
            publication_sid=publication_sid,
        )

    def _on_credentials_updated(self, *, token: str, url: str) -> None:
        if self._backend is None:
            return
        try:
            self._provider.update_credentials(self._backend, token=token, url=url)
        except Exception as e:
            logger.error(f"Krisp update_credentials failed: {e}")
            raise

        # Cloud backend: now that the underlying SDK is initialized, create
        # the session if we haven't already.
        if self._session is None and self._backend.ready:
            self._create_session(self._desired_sample_rate)

    def enable(self) -> None:
        self._filtering_enabled = True

    def disable(self) -> None:
        self._filtering_enabled = False

    @property
    def enabled(self) -> bool:
        return self._filtering_enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._filtering_enabled = value

    @property
    def is_enabled(self) -> bool:
        return self._filtering_enabled

    def _close(self) -> None:
        # _close runs on track transitions, not just final destruction. Drop
        # the session here but keep the SDK reference until __del__.
        if self._session is not None:
            self._session = None
        logger.debug("Krisp frame processor session closed")

    def close(self) -> None:
        self._close()

    def __del__(self) -> None:
        # During Python shutdown, the manager module may already be torn down.
        if KrispSDKManager is None:
            return

        if getattr(self, "_sdk_acquired", False):
            try:
                if getattr(self, "_session", None) is not None:
                    self._session = None
                KrispSDKManager.release()
                self._sdk_acquired = False
            except Exception:
                pass

    def __enter__(self) -> KrispVivaFilterFrameProcessor:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        self.close()
        return False
