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

"""Krisp SDK instance manager for LiveKit Agents.

This module provides a singleton manager for the Krisp VIVA SDK with reference
counting, ensuring proper initialization and cleanup when multiple components
(filters) use the SDK.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

from .log import logger

try:
    import krisp_audio  # type: ignore[import-not-found]

    KRISP_AUDIO_AVAILABLE = True

    # Mapping of sample rates (Hz) to Krisp SDK SamplingRate enums
    KRISP_SAMPLE_RATES = {
        8000: krisp_audio.SamplingRate.Sr8000Hz,
        16000: krisp_audio.SamplingRate.Sr16000Hz,
        24000: krisp_audio.SamplingRate.Sr24000Hz,
        32000: krisp_audio.SamplingRate.Sr32000Hz,
        44100: krisp_audio.SamplingRate.Sr44100Hz,
        48000: krisp_audio.SamplingRate.Sr48000Hz,
    }

    KRISP_FRAME_DURATIONS = {
        10: krisp_audio.FrameDuration.Fd10ms,
        15: krisp_audio.FrameDuration.Fd15ms,
        20: krisp_audio.FrameDuration.Fd20ms,
        30: krisp_audio.FrameDuration.Fd30ms,
        32: krisp_audio.FrameDuration.Fd32ms,
    }
except ModuleNotFoundError:
    KRISP_AUDIO_AVAILABLE = False
    KRISP_SAMPLE_RATES = {}
    KRISP_FRAME_DURATIONS = {}
    logger.warning(
        "krisp-audio package not found. "
        "Install it to use Krisp SDK features: pip install krisp-audio"
    )


def int_to_krisp_frame_duration(frame_duration_ms: int) -> Any:
    if frame_duration_ms not in KRISP_FRAME_DURATIONS:
        supported_durations = ", ".join(
            str(duration) for duration in sorted(KRISP_FRAME_DURATIONS.keys())
        )
        raise ValueError(
            f"Unsupported frame duration: {frame_duration_ms} ms. "
            f"Supported durations: {supported_durations} ms"
        )
    return KRISP_FRAME_DURATIONS[frame_duration_ms]


def int_to_krisp_sample_rate(sample_rate: int) -> Any:
    if sample_rate not in KRISP_SAMPLE_RATES:
        supported_rates = ", ".join(str(rate) for rate in sorted(KRISP_SAMPLE_RATES.keys()))
        raise ValueError(
            f"Unsupported sample rate: {sample_rate} Hz. Supported rates: {supported_rates} Hz"
        )
    return KRISP_SAMPLE_RATES[sample_rate]


class KrispSDKManager:
    """Singleton manager for Krisp VIVA SDK with reference counting.

    This manager ensures the Krisp SDK is initialized only once and properly
    cleaned up when all components are done using it. It uses reference counting
    to track active users (filters).

    Thread-safe implementation using a lock for all operations.

    The license key should be provided via the KRISP_VIVA_SDK_LICENSE_KEY environment variable.
    """

    _initialized = False
    _lock = Lock()
    _reference_count = 0

    @staticmethod
    def _log_callback(log_message: str, log_level: Any) -> None:
        """Thread-safe callback for Krisp SDK logging."""
        logger.debug(f"[Krisp {log_level}] {log_message}")

    @staticmethod
    def licensing_error_callback(error: Any, error_message: str) -> None:
        logger.error(f"[Krisp Licensing Error: {error}] {error_message}")

    @classmethod
    def _get_license_key(cls) -> str:
        """Get the license key from the KRISP_VIVA_SDK_LICENSE_KEY environment variable."""
        return os.getenv("KRISP_VIVA_SDK_LICENSE_KEY", "")

    @classmethod
    def acquire(cls) -> None:
        """Acquire a reference to the SDK (initializes if needed).

        Call this when creating a filter instance.
        The SDK will be initialized on the first call.

        Raises:
            Exception: If SDK initialization fails (propagated from krisp_audio)
        """
        with cls._lock:
            # Initialize SDK on first acquire
            if cls._reference_count == 0:
                try:
                    license_key = cls._get_license_key()
                    krisp_audio.globalInit(
                        "",
                        license_key,
                        cls.licensing_error_callback,
                        cls._log_callback,
                        krisp_audio.LogLevel.Off,
                    )
                    cls._initialized = True

                    version = krisp_audio.getVersion()
                    logger.debug(
                        f"Krisp Audio SDK initialized - "
                        f"Version: {version.major}.{version.minor}.{version.patch}"
                    )

                except Exception as e:
                    cls._initialized = False
                    logger.error(f"Krisp SDK initialization failed: {e}")
                    raise

            cls._reference_count += 1
            logger.debug(f"Krisp SDK reference count: {cls._reference_count}")

    @classmethod
    def release(cls) -> None:
        """Release a reference to the SDK (destroys if last reference).

        Call this when destroying a filter instance.
        The SDK will be cleaned up when the last reference is released.
        """
        with cls._lock:
            if cls._reference_count > 0:
                cls._reference_count -= 1
                logger.debug(f"Krisp SDK reference count: {cls._reference_count}")

                # Destroy SDK when last reference is released
                if cls._reference_count == 0 and cls._initialized:
                    try:
                        krisp_audio.globalDestroy()
                        cls._initialized = False
                        logger.debug("Krisp Audio SDK destroyed (all references released)")
                    except Exception as e:
                        logger.error(f"Error during Krisp SDK cleanup: {e}")
                        cls._initialized = False

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the SDK is currently initialized.

        Returns:
            True if SDK is initialized, False otherwise.
        """
        with cls._lock:
            return cls._initialized

    @classmethod
    def get_reference_count(cls) -> int:
        """Get the current reference count.

        Returns:
            Number of active references to the SDK.
        """
        with cls._lock:
            return cls._reference_count
