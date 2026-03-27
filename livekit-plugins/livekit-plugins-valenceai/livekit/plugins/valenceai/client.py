# Copyright 2025 LiveKit, Inc.
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

"""WebSocket client for Valence AI streaming emotion detection API."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Literal

import numpy as np

try:
    import socketio  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "python-socketio package is required. Install it with: pip install python-socketio[asyncio_client] aiohttp"
    ) from e

from .log import logger

EmotionModel = Literal["4emotions", "7emotions"]


class ValenceWebSocketClient:
    """WebSocket client for Valence AI streaming emotion detection API.

    This client maintains a persistent WebSocket connection to the Valence API
    and continuously streams audio for real-time emotion detection.

    Supports two modes:
    - **Continuous streaming** (preferred): Call `start_streaming()`, then
      `send_audio_chunk()` for each frame. Predictions arrive asynchronously
      and are stored with timestamps. Use `get_latest_emotion()` or
      `get_emotion_for_timerange()` to retrieve predictions without blocking.
    - **Batch** (legacy): Call `process_audio()` with a full audio buffer
      and wait for the prediction synchronously.

    Args:
        api_key: Your Valence AI API key.
        server_url: The Valence API server URL.
        model: Emotion model to use - "4emotions" or "7emotions".
    """

    def __init__(
        self,
        api_key: str,
        server_url: str = "https://api.getvalenceai.com",
        model: EmotionModel = "4emotions",
    ) -> None:
        self._api_key = api_key
        self._server_url = server_url
        self._model = model
        self._sio = socketio.AsyncClient()
        self._session_id: str | None = None
        self._latest_emotion: dict = {
            "dominant": "neutral",
            "confidence": 0.0,
            "all_emotions": {},
        }
        # Legacy batch mode support
        self._prediction_event: asyncio.Event | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3

        # Continuous streaming state
        self._emotion_history: list[dict] = []
        self._emotion_history_lock = asyncio.Lock()
        self._streaming = False
        self._total_audio_sent_ms: float = 0.0
        self._chunks_sent: int = 0

        self._setup_handlers()

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        return bool(self._sio.connected)

    @property
    def session_id(self) -> str | None:
        """Return the current session ID."""
        return self._session_id

    @property
    def latest_emotion(self) -> dict:
        """Return the latest emotion prediction."""
        return self._latest_emotion

    def _setup_handlers(self) -> None:
        """Set up Socket.IO event handlers."""

        @self._sio.on("connect")  # type: ignore[untyped-decorator]
        async def on_connect() -> None:
            logger.debug(f"Connected to Valence API at {self._server_url}")

        @self._sio.on("connected")  # type: ignore[untyped-decorator]
        async def on_connected(data: dict) -> None:
            self._session_id = data.get("session_id")
            logger.info(f"Valence session established: {self._session_id}")

        @self._sio.on("prediction")  # type: ignore[untyped-decorator]
        async def on_prediction(prediction: dict) -> None:
            emotion_entry = {
                "dominant": prediction.get("main_emotion", "neutral"),
                "confidence": prediction.get("confidence", 0.0),
                "all_emotions": prediction.get("all_predictions", {}),
                "timestamp_ms": self._total_audio_sent_ms,
            }

            # Update latest emotion for backward compat
            self._latest_emotion = {
                "dominant": emotion_entry["dominant"],
                "confidence": emotion_entry["confidence"],
                "all_emotions": emotion_entry["all_emotions"],
            }

            # Store in timestamped history for continuous streaming mode
            async with self._emotion_history_lock:
                self._emotion_history.append(emotion_entry)
                # Cap at 20 entries (~100s of predictions at 5s intervals)
                if len(self._emotion_history) > 20:
                    self._emotion_history = self._emotion_history[-20:]

                history_size = len(self._emotion_history)
                prev_entry = self._emotion_history[-2] if history_size > 1 else None

            logger.debug(
                f"Emotion prediction at {self._total_audio_sent_ms:.0f}ms: "
                f"{emotion_entry['dominant']} ({emotion_entry['confidence']:.1%})"
            )

            # Log prediction gap for performance monitoring
            if prev_entry is not None:
                gap_ms = emotion_entry["timestamp_ms"] - prev_entry["timestamp_ms"]
                logger.info(
                    f"[PERF] VALENCE | prediction_gap={gap_ms:.0f}ms "
                    f"emotion={emotion_entry['dominant']} "
                    f"confidence={emotion_entry['confidence']:.1%} "
                    f"history_size={history_size}"
                )
            else:
                logger.info(
                    f"[PERF] VALENCE | first_prediction "
                    f"at_audio={emotion_entry['timestamp_ms']:.0f}ms "
                    f"emotion={emotion_entry['dominant']} "
                    f"confidence={emotion_entry['confidence']:.1%}"
                )

            # Wake up legacy batch waiters
            if self._prediction_event:
                self._prediction_event.set()

        @self._sio.on("error")  # type: ignore[untyped-decorator]
        async def on_error(error: dict) -> None:
            logger.error(f"Valence API error: {error}")

        @self._sio.on("disconnect")  # type: ignore[untyped-decorator]
        async def on_disconnect() -> None:
            logger.debug("Disconnected from Valence API")

    async def connect(self) -> None:
        """Connect to Valence WebSocket server with retry logic.

        Raises:
            Exception: If connection fails after max retry attempts.
        """
        if self._sio.connected:
            logger.debug("Already connected to Valence API")
            return

        last_error: Exception | None = None
        for attempt in range(self._max_reconnect_attempts):
            try:
                await self._sio.connect(
                    self._server_url,
                    auth={"api_key": self._api_key},
                    transports=["websocket", "polling"],
                )
                self._reconnect_attempts = 0
                logger.info("Valence WebSocket connection established")
                return
            except Exception as e:
                last_error = e
                self._reconnect_attempts = attempt + 1
                if attempt < self._max_reconnect_attempts - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)

        logger.error(f"Connection failed after {self._max_reconnect_attempts} attempts")
        if last_error:
            raise last_error

    async def disconnect(self) -> None:
        """Disconnect from Valence WebSocket server."""
        if self._sio.connected:
            await self._sio.disconnect()
            logger.debug("Disconnected from Valence API")

    # ── Continuous streaming mode ────────────────────────────────────────

    async def start_streaming(self) -> None:
        """Begin a continuous streaming session. Resets emotion history."""
        self._emotion_history = []
        self._streaming = True
        self._total_audio_sent_ms = 0.0
        self._chunks_sent = 0
        logger.debug("Started continuous emotion streaming session")

    async def stop_streaming(self) -> None:
        """End the continuous streaming session."""
        self._streaming = False
        logger.debug("Stopped continuous emotion streaming session")

    async def send_audio_chunk(
        self,
        audio_data: bytes,
        sample_rate: int,
        samples_per_channel: int,
    ) -> None:
        """Send a single audio chunk to Valence API without waiting for a response.

        This should be called for every audio frame as it arrives, enabling
        the server-side buffer to accumulate enough audio for predictions.

        Args:
            audio_data: Raw PCM audio bytes (int16).
            sample_rate: Sample rate of the audio.
            samples_per_channel: Number of samples per channel in this chunk.
        """
        if not self._sio.connected or not self._streaming:
            return

        try:
            samples = np.frombuffer(audio_data, dtype=np.int16)
            base64_audio = base64.b64encode(samples.tobytes()).decode("utf-8")

            frame_duration_ms = (samples_per_channel / sample_rate) * 1000
            self._total_audio_sent_ms += frame_duration_ms
            self._chunks_sent += 1

            message = {
                "service": "emotion",
                "action": "prediction",
                "model": self._model,
                "sample_rate": sample_rate,
                "payload": base64_audio,
            }

            await self._sio.emit("message", json.dumps(message))

            # Log streaming health every ~5s (250 frames at 20ms each)
            if self._chunks_sent % 250 == 0:
                logger.info(
                    f"[PERF] VALENCE | streaming_active "
                    f"chunks_sent={self._chunks_sent} "
                    f"audio_sent={self._total_audio_sent_ms:.0f}ms"
                )
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")

    async def get_latest_emotion(self) -> dict:
        """Return the most recent emotion prediction without blocking.

        Returns:
            dict with 'dominant', 'confidence', and 'all_emotions' keys.
        """
        async with self._emotion_history_lock:
            if self._emotion_history:
                entry = self._emotion_history[-1]
                return {
                    "dominant": entry["dominant"],
                    "confidence": entry["confidence"],
                    "all_emotions": entry["all_emotions"],
                }
        return {"dominant": "neutral", "confidence": 0.0, "all_emotions": {}}

    async def get_emotion_for_timerange(
        self,
        start_ms: float,
        end_ms: float,
    ) -> dict:
        """Return the emotion prediction closest to a given audio time range.

        Finds the prediction whose audio timestamp is closest to the midpoint
        of the requested range. Falls back to the latest available prediction.

        Args:
            start_ms: Start of the time range in milliseconds.
            end_ms: End of the time range in milliseconds.

        Returns:
            dict with 'dominant', 'confidence', and 'all_emotions' keys.
        """
        async with self._emotion_history_lock:
            if not self._emotion_history:
                return {"dominant": "neutral", "confidence": 0.0, "all_emotions": {}}

            midpoint = (start_ms + end_ms) / 2.0
            closest = min(
                self._emotion_history,
                key=lambda e: abs(e["timestamp_ms"] - midpoint),
            )
            return {
                "dominant": closest["dominant"],
                "confidence": closest["confidence"],
                "all_emotions": closest["all_emotions"],
            }

    # ── Legacy batch mode ────────────────────────────────────────────────

    async def process_audio(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = 48000,
        timeout: float = 7.0,
    ) -> dict:
        """Send audio samples to Valence API and wait for a prediction.

        This is the legacy batch method. Prefer continuous streaming mode
        (start_streaming / send_audio_chunk / get_latest_emotion) for
        real-time use cases.

        Args:
            audio_samples: numpy array of int16 PCM audio samples.
            sample_rate: Sample rate of the audio (default 48000 for LiveKit).
            timeout: Max seconds to wait for prediction response.

        Returns:
            dict: Emotion prediction containing 'dominant', 'confidence',
                  and 'all_emotions' keys.
        """
        if not self._sio.connected:
            logger.warning("Not connected to Valence API, skipping audio processing")
            return self._latest_emotion

        try:
            # Ensure audio is int16
            if audio_samples.dtype != np.int16:
                if audio_samples.dtype in [np.float32, np.float64]:
                    audio_samples = (audio_samples * 32767).astype(np.int16)
                else:
                    audio_samples = audio_samples.astype(np.int16)

            # Convert to base64
            audio_bytes = audio_samples.tobytes()
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            # Calculate audio duration for logging
            duration_ms = (len(audio_samples) / sample_rate) * 1000

            # Send to Valence API
            message = {
                "service": "emotion",
                "action": "prediction",
                "model": self._model,
                "sample_rate": sample_rate,
                "payload": base64_audio,
            }

            # Create event to wait for prediction
            self._prediction_event = asyncio.Event()

            logger.debug(f"Sending audio: {len(audio_samples)} samples ({duration_ms:.0f}ms)")
            await self._sio.emit("message", json.dumps(message))

            # Wait for prediction response with timeout
            try:
                await asyncio.wait_for(self._prediction_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Prediction timeout after {timeout}s, "
                    f"using last known emotion: {self._latest_emotion['dominant']}"
                )
            finally:
                self._prediction_event = None

        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)

        return self._latest_emotion
