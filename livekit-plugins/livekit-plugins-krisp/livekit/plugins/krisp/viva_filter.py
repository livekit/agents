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

"""Krisp VIVA noise reduction audio filter for LiveKit Agents.

This module provides an audio filter implementation using Krisp VIVA SDK
for real-time noise suppression in LiveKit voice agents.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

from livekit import rtc

from .krisp_instance import (
    KRISP_FRAME_DURATIONS,
    KrispSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
)
from .log import logger

# Check if FrameProcessor is available (requires livekit-rtc >= 1.0.23 with PR #4145)
if not hasattr(rtc, "FrameProcessor"):
    raise ImportError(
        "FrameProcessor is not available in your livekit-rtc version. "
        "KrispVivaFilterFrameProcessor requires livekit-rtc >= 1.0.23 with FrameProcessor support. "
        "Please update livekit-rtc: pip install --upgrade 'livekit>=1.0.23'"
    )

try:
    import krisp_audio  # type: ignore[import-not-found]

    KRISP_AUDIO_AVAILABLE = True
except ModuleNotFoundError:
    KRISP_AUDIO_AVAILABLE = False
    logger.warning(
        "krisp-audio package not found. "
        "Install it to use Krisp noise reduction: pip install krisp-audio"
    )


class KrispVivaFilterFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """FrameProcessor implementation for Krisp noise reduction.

    This class implements the FrameProcessor interface from livekit-rtc,
    allowing it to be used directly with the noise_cancellation parameter
    in AudioInputOptions or RoomInputOptions.

    Example:
        ```python
        from livekit.agents import room_io
        from livekit.plugins import krisp

        # Create frame processor
        processor = krisp.KrispVivaFilterFrameProcessor(
            noise_suppression_level=100,
            frame_duration_ms=10,
        )

        # Use it directly in AudioInputOptions
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
        model_path: str | None = None,
        noise_suppression_level: int = 100,
        frame_duration_ms: int = 10,
        sample_rate: int | None = None,
    ) -> None:
        """Initialize the Krisp frame processor.

        Args:
            model_path: Path to the Krisp model file (.kef extension).
                If None, uses KRISP_VIVA_FILTER_MODEL_PATH environment variable.
            noise_suppression_level: Noise suppression level (0-100, default: 100).
            frame_duration_ms: Frame duration in milliseconds (10, 15, 20, 30, or 32, default: 10).
            sample_rate: sample rate in Hz. If None, default to 16000 Hz.

        Raises:
            RuntimeError: If krisp-audio package is not installed.
            ValueError: If model_path is not provided and KRISP_VIVA_FILTER_MODEL_PATH is not set,
                or if frame_duration_ms is not supported.
            Exception: If model file doesn't have .kef extension.
            FileNotFoundError: If model file doesn't exist.
        """
        # Check if krisp-audio is available
        if not KRISP_AUDIO_AVAILABLE:
            raise RuntimeError("krisp-audio package is not installed.")

        # Initialize state variables first
        self._sdk_acquired = False
        self._filtering_enabled = True
        self._session: Any | None = None
        self._noise_suppression_level = noise_suppression_level
        self._sample_rate: int | None = None
        self._frame_duration_ms = frame_duration_ms

        # Acquire SDK reference (initializes on first call)
        try:
            KrispSDKManager.acquire()
            self._sdk_acquired = True
        except Exception as e:
            logger.error(f"Failed to acquire Krisp SDK: {e}")
            raise RuntimeError(f"Failed to acquire Krisp SDK: {e}") from e

        try:
            # Set model path, checking environment if not specified
            self._model_path = model_path or os.getenv("KRISP_VIVA_FILTER_MODEL_PATH")
            if not self._model_path:
                logger.error(
                    "Model path is not provided and KRISP_VIVA_FILTER_MODEL_PATH is not set."
                )
                raise ValueError("Model path for KrispVivaFilterFrameProcessor must be provided.")

            if not self._model_path.endswith(".kef"):
                raise Exception("Model is expected with .kef extension")

            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            # Validate frame duration
            if frame_duration_ms not in KRISP_FRAME_DURATIONS:
                raise ValueError(
                    f"Unsupported frame duration: {frame_duration_ms}ms. "
                    f"Supported durations: {list(KRISP_FRAME_DURATIONS.keys())}"
                )

            # Always create session to pre-load the model
            # Use provided sample rate, or default to 16kHz (most common)
            init_sample_rate = sample_rate if sample_rate is not None else 16000
            self._create_session(init_sample_rate)
            logger.info(
                f"Krisp frame processor initialized with {init_sample_rate}Hz session "
                f"(model pre-loaded, will recreate session if different sample rate)"
            )
        except Exception:
            # If initialization fails after acquiring SDK, release it
            if self._sdk_acquired:
                KrispSDKManager.release()
                self._sdk_acquired = False
            raise

    def _create_session(self, sample_rate: int) -> None:
        """Create a new Krisp session with the correct sample rate.

        Args:
            sample_rate: The sample rate of the audio frames in Hz.
        """
        # If session already exists for this sample rate, don't recreate
        if self._session is not None and self._sample_rate == sample_rate:
            return

        logger.info(f"Creating Krisp session for sample rate: {sample_rate}Hz")

        model_info = krisp_audio.ModelInfo()
        model_info.path = self._model_path

        nc_cfg = krisp_audio.NcSessionConfig()
        nc_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
        nc_cfg.inputFrameDuration = int_to_krisp_frame_duration(self._frame_duration_ms)
        nc_cfg.outputSampleRate = nc_cfg.inputSampleRate
        nc_cfg.modelInfo = model_info

        try:
            self._session = krisp_audio.NcInt16.create(nc_cfg)
            self._sample_rate = sample_rate

            logger.info("✅ Krisp session created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create Krisp session: {e}")
            raise

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """Process an audio frame with Krisp noise reduction.

        This is the method required by the FrameProcessor interface.

        Args:
            frame: Input audio frame. Must contain exactly the number of samples
                   matching the configured frame_duration_ms at the frame's sample_rate.
                   For example: 10ms @ 16kHz = 160 samples, 20ms @ 32kHz = 640 samples.

        Returns:
            Filtered audio frame with noise reduction applied.
            If filtering is disabled, returns the original frame.

        Raises:
            ValueError: If frame size doesn't match the expected frame duration.
        """
        if not self._filtering_enabled:
            return frame

        if self._session is None or self._sample_rate != frame.sample_rate:
            raise ValueError(f"Session not created or sample rate mismatch: {frame.sample_rate}Hz")

        # Verify frame size matches expected duration
        expected_samples = int((frame.sample_rate * self._frame_duration_ms) / 1000)
        if frame.samples_per_channel != expected_samples:
            raise ValueError(
                f"Frame size mismatch: expected {expected_samples} samples "
                f"({self._frame_duration_ms}ms @ {frame.sample_rate}Hz), "
                f"got {frame.samples_per_channel} samples"
            )

        # Convert frame to numpy array
        audio_samples = np.frombuffer(frame.data, dtype=np.int16)

        try:
            # Process through Krisp
            filtered_samples = self._session.process(audio_samples, self._noise_suppression_level)

            # Validate output
            if filtered_samples is None or len(filtered_samples) == 0:
                logger.warning("Krisp returned empty output, using original audio")
                filtered_samples = audio_samples
            elif len(filtered_samples) != len(audio_samples):
                logger.warning(
                    f"Krisp output size mismatch: expected {len(audio_samples)}, "
                    f"got {len(filtered_samples)}, using original audio"
                )
                filtered_samples = audio_samples

            # Return filtered frame
            return rtc.AudioFrame(
                data=filtered_samples.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=len(filtered_samples),
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Return original frame on error
            return frame

    def process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """Public method that calls _process (for backward compatibility)."""
        return self._process(frame)

    def enable(self) -> None:
        """Enable noise filtering."""
        self._filtering_enabled = True

    def disable(self) -> None:
        """Disable noise filtering (audio will pass through unmodified)."""
        self._filtering_enabled = False

    @property
    def enabled(self) -> bool:
        """Check if filtering is currently enabled (required by FrameProcessor interface)."""
        return self._filtering_enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set filtering enabled state (required by FrameProcessor interface)."""
        self._filtering_enabled = value

    @property
    def is_enabled(self) -> bool:
        """Check if filtering is currently enabled (backward compatibility)."""
        return self._filtering_enabled

    def _close(self) -> None:
        """Clean up processor session resources (required by FrameProcessor interface).

        Note: This method is called during track transitions (when streams are closed/reopened),
        not just when the processor is destroyed. Therefore, we only clean up the session here,
        not the SDK reference. The SDK will be released in __del__ when the processor is
        actually being destroyed (at the end of the call).
        """
        if self._session is not None:
            self._session = None

        logger.debug("Krisp frame processor session closed")

    def close(self) -> None:
        """Clean up processor session resources (public method for backward compatibility)."""
        self._close()

    def __del__(self) -> None:
        """Destructor to ensure cleanup of session resources.

        Note: During Python shutdown, we avoid calling C extensions to prevent GIL errors.
        Always call close() explicitly for proper cleanup.
        """
        # Check if we're in Python shutdown (modules being cleaned up)
        # If KrispSDKManager is None, we're in shutdown - don't do anything
        if KrispSDKManager is None:
            return

        if getattr(self, "_sdk_acquired", False):
            try:
                if getattr(self, "_session", None) is not None:
                    self._session = None
                # Release SDK reference only if we still have it
                KrispSDKManager.release()
                self._sdk_acquired = False
            except Exception:
                # Silently ignore errors during shutdown
                pass

    def __enter__(self) -> "KrispVivaFilterFrameProcessor":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        """Context manager exit - clean up session."""
        self.close()
        return False
