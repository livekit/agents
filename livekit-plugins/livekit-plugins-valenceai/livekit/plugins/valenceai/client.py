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
    import socketio
except ImportError as e:
    raise ImportError(
        "python-socketio package is required. Install it with: pip install python-socketio[asyncio_client] aiohttp"
    ) from e

from .log import logger

EmotionModel = Literal["4emotions", "7emotions"]


class ValenceWebSocketClient:
    """WebSocket client for Valence AI streaming emotion detection API.

    This client maintains a persistent WebSocket connection to the Valence API
    and processes audio chunks for real-time emotion detection.

    Args:
        api_key: Your Valence AI API key.
        server_url: The Valence API server URL.
        model: Emotion model to use - "4emotions" or "7emotions".
    """

    def __init__(
        self,
        api_key: str,
        server_url: str = "https://qa.getvalenceai.com",
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
        self._prediction_event: asyncio.Event | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._setup_handlers()

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        return self._sio.connected

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

        @self._sio.on("connect")
        async def on_connect() -> None:
            logger.debug(f"Connected to Valence API at {self._server_url}")

        @self._sio.on("connected")
        async def on_connected(data: dict) -> None:
            self._session_id = data.get("session_id")
            logger.info(f"Valence session established: {self._session_id}")

        @self._sio.on("prediction")
        async def on_prediction(prediction: dict) -> None:
            self._latest_emotion = {
                "dominant": prediction.get("main_emotion", "neutral"),
                "confidence": prediction.get("confidence", 0.0),
                "all_emotions": prediction.get("all_predictions", {}),
            }
            logger.debug(
                f"Emotion prediction: {self._latest_emotion['dominant']} "
                f"({self._latest_emotion['confidence']:.1%})"
            )
            if self._prediction_event:
                self._prediction_event.set()

        @self._sio.on("error")
        async def on_error(error: dict) -> None:
            logger.error(f"Valence API error: {error}")

        @self._sio.on("disconnect")
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

    async def process_audio(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = 48000,
        timeout: float = 2.0,
    ) -> dict:
        """Send audio samples to Valence API for emotion detection.

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
