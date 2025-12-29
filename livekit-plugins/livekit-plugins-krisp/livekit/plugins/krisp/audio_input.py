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

"""Krisp-filtered AudioInput wrapper for LiveKit Agents.

This module provides an AudioInput wrapper that applies Krisp noise cancellation
to incoming audio before it reaches STT/VAD components in the agent pipeline.
"""

from __future__ import annotations

from livekit import rtc
from livekit.agents.voice import io

from .log import logger
from .viva_filter import KrispVivaFilter


class KrispAudioInput(io.AudioInput):
    """AudioInput wrapper that applies Krisp noise cancellation to incoming audio.
    
    This class wraps an existing AudioInput (typically from RoomIO) and applies
    Krisp VIVA noise cancellation to each audio frame before passing it downstream
    to STT, VAD, or other audio processing components.
    
    The audio pipeline becomes:
        Room → RoomIO → KrispAudioInput (NC applied here) → VAD/STT → LLM
    
    Example:
        ```python
        from livekit.agents import AgentSession, room_io
        from livekit.plugins.krisp import KrispAudioInput
        
        session = AgentSession(...)
        
        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    sample_rate=16000,
                    frame_size_ms=10,
                ),
            ),
        )
        
        # Wrap the audio input with Krisp filtering
        if session.input.audio:
            session.input.audio = KrispAudioInput(
                source=session.input.audio,
                noise_suppression_level=100,
                frame_duration_ms=10,
            )
            session.input.audio.on_attached()
        ```
    """

    def __init__(
        self,
        source: io.AudioInput,
        *,
        model_path: str | None = None,
        noise_suppression_level: int = 100,
        frame_duration_ms: int = 10,
        sample_rate: int | None = None,
    ) -> None:
        """Initialize the Krisp-filtered audio input.
        
        Args:
            source: The upstream AudioInput to wrap (e.g., from RoomIO).
            model_path: Path to the Krisp model file (.kef extension).
                If None, uses KRISP_VIVA_FILTER_MODEL_PATH environment variable.
            noise_suppression_level: Noise suppression level (0-100, default: 100).
            frame_duration_ms: Frame duration in milliseconds (10, 15, 20, 30, or 32).
                Must match the frame_size_ms in AudioInputOptions.
            sample_rate: Optional sample rate in Hz. If provided, the Krisp session
                will be created immediately. If None, it will be created on the first frame.
        
        Raises:
            ValueError: If model_path is not provided and KRISP_VIVA_FILTER_MODEL_PATH is not set,
                or if frame_duration_ms is not supported.
            Exception: If model file doesn't have .kef extension.
            FileNotFoundError: If model file doesn't exist.
        """
        super().__init__(label="KrispNC", source=source)
        
        self._filter = KrispVivaFilter(
            model_path=model_path,
            noise_suppression_level=noise_suppression_level,
            frame_duration_ms=frame_duration_ms,
            sample_rate=sample_rate,
        )
        
        logger.info(
            f"KrispAudioInput initialized: suppression={noise_suppression_level}, "
            f"frame_duration={frame_duration_ms}ms"
        )
    
    async def __anext__(self) -> rtc.AudioFrame:
        """Get next audio frame from source and apply Krisp noise cancellation.
        
        Returns:
            Filtered audio frame with noise reduction applied.
        
        Raises:
            ValueError: If frame size doesn't match the expected frame duration.
        """
        # Get frame from upstream source (RoomIO)
        frame = await self.source.__anext__()
        
        # Apply Krisp noise cancellation
        filtered_frame = await self._filter.filter(frame)
        
        return filtered_frame
    
    def on_detached(self) -> None:
        """Clean up Krisp filter resources when input is detached."""
        logger.info("KrispAudioInput detached, cleaning up filter resources")
        self._filter.close()
        super().on_detached()
    
    def enable_filtering(self) -> None:
        """Enable Krisp noise filtering."""
        self._filter.enable()
        logger.info("Krisp filtering enabled")
    
    def disable_filtering(self) -> None:
        """Disable Krisp noise filtering (audio passes through unmodified)."""
        self._filter.disable()
        logger.info("Krisp filtering disabled")
    
    @property
    def is_filtering_enabled(self) -> bool:
        """Check if Krisp filtering is currently enabled."""
        return self._filter.is_enabled

