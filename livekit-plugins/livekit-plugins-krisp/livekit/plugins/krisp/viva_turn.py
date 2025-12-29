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

"""Krisp VIVA turn detection for LiveKit Agents.

This module provides audio-based turn detection using Krisp's VIVA SDK that works
at the audio level, complementing text-based turn detection approaches.

Note: This uses a different model than KrispVivaFilter. Set the model path via
KRISP_VIVA_TURN_MODEL_PATH environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from livekit import rtc
from livekit.agents import llm

from .krisp_instance import KrispSDKManager, int_to_krisp_frame_duration, int_to_krisp_sample_rate
from .log import logger

try:
    import krisp_audio
    KRISP_AUDIO_AVAILABLE = True
except ModuleNotFoundError:
    KRISP_AUDIO_AVAILABLE = False
    logger.warning(
        "krisp-audio package not found. "
        "Install it to use Krisp turn detection: pip install krisp-audio"
    )


@dataclass
class TurnState:
    """State tracked during turn detection."""

    audio_buffer: bytearray
    speech_triggered: bool
    last_probability: float | None
    frame_probabilities: list[float]


class KrispVivaTurn:
    """Audio-based turn detection using Krisp VIVA SDK.

    Unlike text-based turn detectors, this works directly on audio frames using
    Krisp's turn detection (Tt) API. It's designed to be used alongside VAD for
    optimal turn detection.

    Supported sample rates: 8000, 16000, 24000, 32000, 44100, 48000 Hz
    Supported frame durations: 10, 15, 20, 30, 32 ms

    Example:
        ```python
        from livekit.plugins import krisp
        
        turn_detector = krisp.KrispVivaTurn(threshold=0.6)
        
        session = AgentSession(
            turn_detection=turn_detector,
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            ...
        )
        ```
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        threshold: float = 0.5,
        frame_duration_ms: int = 20,
        sample_rate: int | None = None,
    ) -> None:
        """Initialize Krisp VIVA turn detector.

        Args:
            model_path: Path to turn detection model (.kef file).
                If None, uses KRISP_VIVA_TURN_MODEL_PATH environment variable.
            threshold: Turn completion threshold (0.0-1.0). Higher values require
                more confidence. Default: 0.5
            frame_duration_ms: Frame size in milliseconds. Must be one of:
                10, 15, 20, 30, 32. Default: 20
            sample_rate: Optional sample rate to pre-initialize. If provided, the
                model will be loaded immediately. If None, loaded on first audio.

        Raises:
            RuntimeError: If krisp-audio package is not installed.
            ValueError: If model_path not provided and env var not set, or if
                frame_duration_ms is invalid.
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If Krisp SDK initialization fails.
        """

        # Check if krisp-audio is available
        if not KRISP_AUDIO_AVAILABLE:
            raise RuntimeError(
                "krisp-audio package is not installed. "
                "Install it with: pip install krisp-audio"
            )

        # Initialize state variables first
        self._sdk_acquired = False
        self._threshold = threshold
        self._frame_duration_ms = frame_duration_ms
        self._tt_session: Any | None = None
        self._pre_load_turn_session: Any | None = None
        self._sample_rate: int | None = None
        self._samples_per_frame: int | None = None

        # Per-session state (reset on clear)
        self._state = TurnState(
            audio_buffer=bytearray(),
            speech_triggered=False,
            last_probability=None,
            frame_probabilities=[],
        )

        # Acquire SDK reference (initializes on first call)
        try:
            KrispSDKManager.acquire()
            self._sdk_acquired = True
        except Exception as e:
            logger.error(f"Failed to acquire Krisp SDK: {e}")
            raise

        try:
            # Get and validate model path
            self._model_path = model_path or os.getenv("KRISP_VIVA_TURN_MODEL_PATH")
            if not self._model_path:
                raise ValueError(
                    "Model path must be provided via model_path parameter or "
                    "KRISP_VIVA_TURN_MODEL_PATH environment variable"
                )

            if not self._model_path.endswith(".kef"):
                raise ValueError("Model file must have .kef extension")

            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            # Pre-load model if sample rate provided (or default to 16kHz)
            pre_load_sample_rate = sample_rate if sample_rate is not None else 16000
            self._pre_load_turn_session = self._create_session(pre_load_sample_rate)
            self._tt_session = self._pre_load_turn_session
            logger.info(
                f"Krisp VIVA turn detector initialized with {pre_load_sample_rate}Hz "
                f"({frame_duration_ms}ms frames, threshold={threshold})"
            )
        except Exception:
            # If initialization fails after acquiring SDK, release it
            if self._sdk_acquired:
                KrispSDKManager.release()
                self._sdk_acquired = False
            raise


    def _create_session(self, sample_rate: int) -> Any:
        """Create or recreate Krisp turn detection session.
        
        Returns:
            krisp_audio.TtFloat instance
        """
        if self._sample_rate == sample_rate and self._pre_load_turn_session is not None:
            return self._pre_load_turn_session

        try:
            model_info = krisp_audio.ModelInfo()
            model_info.path = self._model_path

            tt_cfg = krisp_audio.TtSessionConfig()
            tt_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
            tt_cfg.inputFrameDuration = int_to_krisp_frame_duration(self._frame_duration_ms)
            tt_cfg.modelInfo = model_info

            turn_session = krisp_audio.TtFloat.create(tt_cfg)

            # Update state when creating a new session
            self._sample_rate = sample_rate
            self._samples_per_frame = int((sample_rate * self._frame_duration_ms) / 1000)

            logger.debug(f"Created Krisp turn session for {sample_rate}Hz ({self._samples_per_frame} samples/frame)")

            return turn_session

        except Exception as e:
            logger.error(f"Failed to create Krisp session: {e}", exc_info=True)
            raise

    def process_audio(self, frame: rtc.AudioFrame, *, is_speech: bool = False) -> float:
        """Process an audio frame and return turn-end probability.

        Args:
            frame: Audio frame to process.
            is_speech: Whether the frame contains speech (from VAD).

        Returns:
            Turn-end probability (0.0-1.0), or -1.0 if model not ready yet.
        """
        # Create session if needed
        if self._tt_session is None or self._sample_rate != frame.sample_rate:
            self._tt_session = self._create_session(frame.sample_rate)

        if self._samples_per_frame is None:
            return -1.0

        # Add audio to buffer
        self._state.audio_buffer.extend(frame.data)

        # Clear previous frame probabilities
        self._state.frame_probabilities = []

        # Process complete frames
        total_samples = len(self._state.audio_buffer) // 2  # 2 bytes per int16
        num_complete_frames = total_samples // self._samples_per_frame

        if num_complete_frames == 0:
            return -1.0

        # Extract audio to process
        bytes_to_process = num_complete_frames * self._samples_per_frame * 2
        audio_data = bytes(self._state.audio_buffer[:bytes_to_process])
        self._state.audio_buffer = self._state.audio_buffer[bytes_to_process:]

        # Convert to float32
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        frames = audio_float32.reshape(-1, self._samples_per_frame)

        max_prob = -1.0

        try:
            for audio_frame in frames:
                # Track speech state
                if is_speech:
                    if not self._state.speech_triggered:
                        logger.debug("Speech detected, turn analysis started")
                    self._state.speech_triggered = True

                # Process frame
                prob = self._tt_session.process(audio_frame.tolist())

                # Skip negative values (model warmup)
                if prob < 0:
                    continue

                # Store probability
                self._state.last_probability = prob
                self._state.frame_probabilities.append(prob)
                max_prob = max(max_prob, prob)

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return -1.0

        return max_prob

    def clear(self) -> None:
        """Reset turn detection state."""
        self._state = TurnState(
            audio_buffer=bytearray(),
            speech_triggered=False,
            last_probability=None,
            frame_probabilities=[],
        )

    @property
    def model(self) -> str:
        """Model identifier."""
        return "krisp-viva-turn"

    @property
    def provider(self) -> str:
        """Provider name."""
        return "krisp"

    @property
    def threshold(self) -> float:
        """Turn probability threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set turn probability threshold."""
        self._threshold = value

    @property
    def last_probability(self) -> float | None:
        """Last computed turn probability."""
        return self._state.last_probability

    @property
    def frame_probabilities(self) -> list[float]:
        """All frame probabilities from last processing."""
        return self._state.frame_probabilities

    @property
    def speech_triggered(self) -> bool:
        """Whether speech has been detected."""
        return self._state.speech_triggered

    # Protocol methods for compatibility with text-based turn detectors

    async def supports_language(self, language: str | None) -> bool:
        """Check if language is supported.

        Currently only English is supported.
        
        Args:
            language: Language code (e.g., "en", "en-US") or None for default
            
        Returns:
            True if language is English or None, False otherwise
        """
        if language is None:
            return True  # Default to English

        # Check if language starts with "en" (covers "en", "en-US", "en-GB", etc.)
        return language.lower().startswith("en")

    async def unlikely_threshold(self, language: str | None) -> float | None:
        """Get unlikely threshold for language.

        Returns the configured threshold for all languages.
        """
        return self._threshold

    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext,
        *,
        timeout: float | None = None,
    ) -> float:
        """Predict end-of-turn probability.

        Note: This method is for compatibility with text-based turn detectors.
        For audio-based detection, use `process_audio()` instead.

        Returns the last computed probability, or 0.0 if no audio processed yet.
        """
        return self._state.last_probability if self._state.last_probability is not None else 0.0

    def close(self) -> None:
        """Clean up resources."""
        self._tt_session = None
        self._pre_load_turn_session = None
        self._state.audio_buffer.clear()

        # Release SDK reference (only if not already released)
        if getattr(self, "_sdk_acquired", False):
            KrispSDKManager.release()
            self._sdk_acquired = False

        logger.debug("Krisp VIVA turn detector closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        """Destructor to ensure cleanup of session resources.
        
        Note: During Python shutdown, we avoid calling C extensions to prevent GIL errors.
        Always call close() explicitly for proper cleanup.
        """
        # Check if we're in Python shutdown (modules being cleaned up)
        # If KrispSDKManager is None, we're in shutdown - don't do anything
        if KrispSDKManager is None:
            return

        # Use getattr for safe access during shutdown
        if getattr(self, "_sdk_acquired", False):
            try:
                # Clean up session first
                if getattr(self, "_tt_session", None) is not None:
                    self._tt_session = None
                if getattr(self, "_pre_load_turn_session", None) is not None:
                    self._pre_load_turn_session = None

                KrispSDKManager.release()
                self._sdk_acquired = False
            except Exception:
                # Silently ignore all errors during shutdown
                pass

