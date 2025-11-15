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

"""Audio-based turn detector using ONNX model for end-of-turn detection.

This module provides an audio-based turn detector that analyzes raw audio frames
to predict end-of-turn, complementing or replacing text-based turn detection.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from livekit import rtc
from livekit.agents import llm, utils

from .log import logger

try:
    import onnxruntime as ort
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AudioTurnDetector, you need to `pip install onnxruntime`."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class AudioTurnPrediction:
    """Result from audio turn detection."""

    probability: float
    """Probability that this is an end-of-turn (0.0 to 1.0)."""

    prediction: int
    """Binary prediction: 1 for end-of-turn, 0 for incomplete."""

    audio_duration: float
    """Duration of audio analyzed in seconds."""

    inference_duration: float
    """Time taken for inference in seconds."""


class AudioTurnDetector:
    """Audio-based turn detector using ONNX model.

    This detector analyzes raw audio frames to predict whether the user has
    finished their turn, without relying on text transcription.
    """

    def __init__(
        self,
        *,
        model_path: str,
        feature_type: Literal["whisper", "raw"] = "whisper",
        max_audio_seconds: int = 8,
        sample_rate: int = 16000,
        activation_threshold: float = 0.5,
        cpu_count: int = 1,
        unlikely_threshold: float | None = None,
    ) -> None:
        """Initialize the audio turn detector.

        Args:
            model_path: Path to the ONNX model file.
            feature_type: Type of audio features to extract ("whisper" or "raw").
            max_audio_seconds: Maximum audio duration to analyze (will truncate or pad).
            sample_rate: Expected audio sample rate (default: 16000 Hz).
            activation_threshold: Threshold for binary prediction (default: 0.5).
            cpu_count: Number of CPU threads for inference (default: 1).
            unlikely_threshold: Custom threshold for "unlikely to be end-of-turn".
                If probability < unlikely_threshold, use max endpointing delay.
        """
        self._model_path = model_path
        self._feature_type = feature_type
        self._max_audio_seconds = max_audio_seconds
        self._sample_rate = sample_rate
        self._activation_threshold = activation_threshold
        self._unlikely_threshold = unlikely_threshold
        self._cpu_count = cpu_count

        # Configure ONNX Runtime session
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = cpu_count
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info(f"Loading audio turn detector from {model_path}")
        self._session = ort.InferenceSession(model_path, sess_options=so)

        # Initialize feature extractor if using Whisper features
        self._feature_extractor = None
        if feature_type == "whisper":
            try:
                from transformers import WhisperFeatureExtractor

                self._feature_extractor = WhisperFeatureExtractor(
                    chunk_length=max_audio_seconds
                )
                logger.debug("Initialized Whisper feature extractor")
            except ImportError:
                logger.error(
                    "Whisper feature extractor requires transformers library. "
                    "Install with: pip install transformers"
                )
                raise

        # Thread pool for async inference
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info("Audio turn detector loaded successfully")

    @property
    def model(self) -> str:
        return "audio-turn-detector"

    @property
    def provider(self) -> str:
        return "onnx"

    async def unlikely_threshold(self, language: str | None) -> float | None:
        """Get the unlikely threshold for a given language.

        For audio-based detection, language may not be relevant.
        Returns the configured unlikely_threshold or None.
        """
        return self._unlikely_threshold

    async def supports_language(self, language: str | None) -> bool:
        """Check if the detector supports a given language.

        Audio-based detectors typically work across languages.
        """
        return True  # Audio features are generally language-agnostic

    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext | None = None,
        *,
        audio_frames: list[rtc.AudioFrame] | None = None,
        timeout: float | None = 3.0,
    ) -> float:
        """Predict end-of-turn probability from audio frames.

        This method is called by AudioRecognition when VAD detects END_OF_SPEECH.

        Args:
            chat_ctx: Chat context (not used for audio-based detection, for compatibility).
            audio_frames: List of audio frames to analyze (from VAD event).
            timeout: Maximum time to wait for inference (default: 3 seconds).

        Returns:
            Probability that this is an end-of-turn (0.0 to 1.0).
        """
        if not audio_frames:
            logger.warning("No audio frames provided for turn detection")
            return 0.0

        # Combine all frames into a single audio array
        combined_frame = utils.combine_frames(audio_frames)

        # Convert to numpy array and normalize
        audio_data = np.frombuffer(combined_frame.data, dtype=np.int16)
        audio_array = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        # Run prediction with timeout
        try:
            result = await asyncio.wait_for(
                self._predict_endpoint(audio_array, combined_frame.sample_rate),
                timeout=timeout,
            )
            return result.probability
        except asyncio.TimeoutError:
            logger.error(f"Audio turn detection timed out after {timeout}s")
            return 0.0

    async def predict_from_frames(
        self,
        audio_frames: list[rtc.AudioFrame],
    ) -> AudioTurnPrediction:
        """Predict end-of-turn from audio frames (direct API).

        Args:
            audio_frames: List of audio frames to analyze.

        Returns:
            AudioTurnPrediction with probability, prediction, and metadata.
        """
        if not audio_frames:
            return AudioTurnPrediction(
                probability=0.0,
                prediction=0,
                audio_duration=0.0,
                inference_duration=0.0,
            )

        combined_frame = utils.combine_frames(audio_frames)
        audio_data = np.frombuffer(combined_frame.data, dtype=np.int16)
        audio_array = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        return await self._predict_endpoint(audio_array, combined_frame.sample_rate)

    async def _predict_endpoint(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> AudioTurnPrediction:
        """Run inference on audio array (async wrapper).

        Args:
            audio_array: Normalized float32 audio array.
            sample_rate: Sample rate of the audio.

        Returns:
            AudioTurnPrediction with results.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor, self._predict_sync, audio_array, sample_rate
        )
        return result

    def _predict_sync(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> AudioTurnPrediction:
        """Synchronous inference (runs in thread pool).

        Args:
            audio_array: Normalized float32 audio array.
            sample_rate: Sample rate of the audio.

        Returns:
            AudioTurnPrediction with results.
        """
        start_time = time.perf_counter()

        # Resample if needed
        if sample_rate != self._sample_rate:
            logger.warning(
                f"Resampling from {sample_rate} Hz to {self._sample_rate} Hz"
            )
            # Simple resampling (for production, use proper resampling library)
            ratio = self._sample_rate / sample_rate
            new_length = int(len(audio_array) * ratio)
            audio_array = np.interp(
                np.linspace(0, len(audio_array) - 1, new_length),
                np.arange(len(audio_array)),
                audio_array,
            )

        original_duration = len(audio_array) / self._sample_rate

        # Truncate to last N seconds or pad to N seconds
        audio_array = self._truncate_or_pad_audio(audio_array)

        # Extract features based on feature type
        if self._feature_type == "whisper" and self._feature_extractor:
            input_features = self._extract_whisper_features(audio_array)
        else:
            # Raw waveform input
            input_features = audio_array.reshape(1, -1).astype(np.float32)

        # Run ONNX inference
        try:
            outputs = self._session.run(None, {"input_features": input_features})
            probability = float(outputs[0][0].item())
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            probability = 0.0

        inference_duration = time.perf_counter() - start_time

        prediction = 1 if probability > self._activation_threshold else 0

        return AudioTurnPrediction(
            probability=probability,
            prediction=prediction,
            audio_duration=original_duration,
            inference_duration=inference_duration,
        )

    def _truncate_or_pad_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Truncate audio to last N seconds or pad with zeros to N seconds.

        Args:
            audio_array: Input audio array.

        Returns:
            Audio array of exactly max_audio_seconds duration.
        """
        max_samples = self._max_audio_seconds * self._sample_rate

        if len(audio_array) > max_samples:
            # Keep only the last N seconds
            return audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            # Pad with zeros at the beginning
            padding = max_samples - len(audio_array)
            return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)

        return audio_array

    def _extract_whisper_features(self, audio_array: np.ndarray) -> np.ndarray:
        """Extract Whisper mel-spectrogram features.

        Args:
            audio_array: Normalized float32 audio array.

        Returns:
            Feature array ready for ONNX input.
        """
        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=self._sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_audio_seconds * self._sample_rate,
            truncation=True,
            do_normalize=True,
        )

        # Extract features and ensure correct shape for ONNX
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

        return input_features

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
