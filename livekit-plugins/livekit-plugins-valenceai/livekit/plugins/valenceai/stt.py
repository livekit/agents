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

"""Valence AI Emotion-Aware STT Plugin for LiveKit Agents.

This plugin wraps an underlying STT provider (e.g., Deepgram) and enriches
transcriptions with emotion tags from Valence AI.

Output format:
    [Neutral] Hi there. [Angry] This is frustrating! [Sad] I'm so disappointed.

Example:
    from livekit.plugins import valenceai, deepgram

    emotion_stt = valenceai.STT(
        underlying_stt=deepgram.STT(),
        api_key="your-valence-api-key",
    )

    session = AgentSession(stt=emotion_stt, ...)
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

import numpy as np

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, stt

from .client import ValenceWebSocketClient
from .log import logger

EmotionModel = Literal["4emotions", "7emotions"]


class STT(stt.STT):
    """Emotion-aware STT that combines an underlying STT with Valence AI emotion detection.

    This STT wrapper intercepts audio, sends it to both the underlying STT provider
    and the Valence AI emotion detection API, then enriches the transcriptions with
    emotion tags in the format: [Emotion] transcribed text

    Args:
        underlying_stt: The base STT provider (e.g., Deepgram, AssemblyAI).
        api_key: Valence AI API key (defaults to VALENCE_API_KEY env var).
        server_url: Valence API server URL.
        model: Emotion model - "4emotions" or "7emotions".
        min_confidence: Minimum confidence threshold for emotion tags (0.0-1.0).

    Example:
        from livekit.plugins import valenceai, deepgram

        stt = valenceai.STT(
            underlying_stt=deepgram.STT(),
            model="4emotions",
            min_confidence=0.3,
        )
    """

    # Maximum audio frames to buffer (prevents memory leak if no FINAL_TRANSCRIPT arrives)
    # At 48kHz with 20ms frames, 500 frames = ~10 seconds of audio
    MAX_BUFFER_FRAMES = 500

    def __init__(
        self,
        *,
        underlying_stt: stt.STT,
        api_key: str | None = None,
        server_url: str = "https://qa.getvalenceai.com",
        model: EmotionModel = "4emotions",
        min_confidence: float = 0.0,
    ) -> None:
        # Copy capabilities from underlying STT
        super().__init__(capabilities=underlying_stt.capabilities)

        self._underlying_stt = underlying_stt
        self._api_key = api_key or os.getenv("VALENCE_API_KEY")
        self._server_url = server_url
        self._model = model
        self._min_confidence = min_confidence

        # Valence client will be initialized on first use
        self._valence_client: ValenceWebSocketClient | None = None
        self._valence_connected = False

    async def _ensure_valence_connected(self) -> bool:
        """Ensure Valence client is connected, reconnecting if necessary.

        Returns:
            bool: True if connected, False otherwise.
        """
        # Check if we have a client and it's actually connected
        if self._valence_client and self._valence_client.is_connected:
            return True

        # Reset state if we had a stale connection
        if self._valence_client and not self._valence_client.is_connected:
            logger.warning("Valence connection was lost, attempting to reconnect...")
            self._valence_connected = False

        if not self._api_key:
            logger.warning("No Valence API key provided, emotions will be disabled")
            return False

        try:
            # Create new client if needed
            if not self._valence_client:
                self._valence_client = ValenceWebSocketClient(
                    api_key=self._api_key,
                    server_url=self._server_url,
                    model=self._model,
                )
            await self._valence_client.connect()
            self._valence_connected = True
            logger.info("Connected to Valence AI emotion detection API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Valence: {e}")
            return False

    @property
    def model(self) -> str:
        """Return the combined model name."""
        return f"valence+{self._underlying_stt.model}"

    @property
    def provider(self) -> str:
        """Return the combined provider name."""
        return f"valence+{self._underlying_stt.provider}"

    def stream(
        self,
        *,
        language: str = "",
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> EmotionAwareRecognizeStream:
        """Create a streaming recognition session with emotion awareness.

        Args:
            language: Language code for speech recognition.
            conn_options: API connection options.

        Returns:
            EmotionAwareRecognizeStream: A streaming recognition session.
        """
        return EmotionAwareRecognizeStream(
            stt_instance=self,
            underlying_stt=self._underlying_stt,
            min_confidence=self._min_confidence,
            language=language,
            conn_options=conn_options,
        )

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: str,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Recognize speech from an audio buffer with emotion awareness.

        Args:
            buffer: Audio buffer to recognize.
            language: Language code.
            conn_options: API connection options.

        Returns:
            SpeechEvent: Recognition result with emotion-enriched text.
        """
        await self._ensure_valence_connected()

        # Get transcription from underlying STT
        result = await self._underlying_stt.recognize(
            buffer, language=language, conn_options=conn_options
        )

        # If we have audio and Valence is connected, detect emotion
        if self._valence_client and self._valence_connected:
            try:
                # Convert buffer to numpy array for Valence
                samples = np.frombuffer(buffer.data, dtype=np.int16)
                emotions = await self._valence_client.process_audio(
                    samples, sample_rate=buffer.sample_rate
                )

                # Enrich the transcription with emotion
                if result.alternatives:
                    emotion = emotions.get("dominant", "neutral")
                    confidence = emotions.get("confidence", 0.0)

                    if confidence >= self._min_confidence:
                        emotion_tag = f"[{emotion.capitalize()}]"
                        new_alternatives = []
                        for alt in result.alternatives:
                            new_alternatives.append(
                                stt.SpeechData(
                                    language=alt.language,
                                    text=f"{emotion_tag} {alt.text}",
                                    start_time=alt.start_time,
                                    end_time=alt.end_time,
                                    confidence=alt.confidence,
                                    speaker_id=alt.speaker_id,
                                    is_primary_speaker=alt.is_primary_speaker,
                                )
                            )
                        result = stt.SpeechEvent(
                            type=result.type,
                            request_id=result.request_id,
                            alternatives=new_alternatives,
                            recognition_usage=result.recognition_usage,
                        )

            except Exception as e:
                logger.error(f"Error detecting emotion: {e}")

        return result

    async def aclose(self) -> None:
        """Close the STT and cleanup resources."""
        if self._valence_client:
            await self._valence_client.disconnect()
        await self._underlying_stt.aclose()


class EmotionAwareRecognizeStream(stt.RecognizeStream):
    """Streaming recognition with emotion awareness.

    This stream wraps an underlying STT stream and enriches transcriptions
    with emotion tags based on the audio being processed.

    Architecture: Audio is buffered continuously. When a FINAL_TRANSCRIPT arrives,
    emotion detection runs on the buffered audio BEFORE the transcript is forwarded.
    This ensures the emotion tag accurately reflects the audio that produced the transcript.
    """

    def __init__(
        self,
        *,
        stt_instance: STT,
        underlying_stt: stt.STT,
        min_confidence: float,
        language: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt_instance, conn_options=conn_options)

        self._parent_stt = stt_instance
        self._underlying_stt = underlying_stt
        self._min_confidence = min_confidence
        self._language = language

        # Current emotion state (updated when transcripts arrive)
        self._current_emotion = "neutral"
        self._current_confidence = 0.0

        # Buffer for emotion detection - stores audio frames until transcript arrives
        self._audio_buffer: list[rtc.AudioFrame] = []

    async def _run(self) -> None:
        """Main processing loop."""
        logger.debug("Starting emotion-aware streaming recognition")

        # Ensure Valence is connected
        valence_connected = await self._parent_stt._ensure_valence_connected()
        logger.debug(f"Valence connected: {valence_connected}")

        # Create underlying stream
        underlying_stream = self._underlying_stt.stream(
            language=self._language,
            conn_options=self._conn_options,
        )

        frame_count = 0

        async def forward_audio() -> None:
            """Forward audio frames to underlying stream (non-blocking)."""
            nonlocal frame_count
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    logger.debug(f"Flush received after {frame_count} frames")
                    underlying_stream.flush()
                else:
                    frame: rtc.AudioFrame = item
                    frame_count += 1

                    # Forward to underlying STT immediately
                    underlying_stream.push_frame(frame)

                    # Buffer audio for emotion detection when transcript arrives
                    self._audio_buffer.append(frame)

                    # Enforce buffer size limit to prevent memory leak
                    if len(self._audio_buffer) > STT.MAX_BUFFER_FRAMES:
                        excess = len(self._audio_buffer) - STT.MAX_BUFFER_FRAMES
                        self._audio_buffer = self._audio_buffer[excess:]
                        logger.debug(f"Buffer trimmed, removed {excess} old frames")

            logger.debug(f"Input ended. Total frames: {frame_count}")
            underlying_stream.end_input()

        async def receive_events() -> None:
            """Receive events from underlying stream and enrich with emotions."""
            async for event in underlying_stream:
                # For FINAL transcripts, detect emotion FIRST (blocking), then enrich
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    await self._detect_emotion_from_buffer()
                    # Clear buffer after final transcript - this audio segment is complete
                    self._audio_buffer.clear()

                # Enrich transcriptions with emotion tags
                enriched_event = self._enrich_with_emotion(event)
                self._event_ch.send_nowait(enriched_event)

        # Run both tasks concurrently
        forward_task = asyncio.create_task(forward_audio())
        receive_task = asyncio.create_task(receive_events())

        try:
            await asyncio.gather(forward_task, receive_task)
        finally:
            await underlying_stream.aclose()

    async def _detect_emotion_from_buffer(self) -> None:
        """Detect emotion from buffered audio frames (blocking call)."""
        valence_client = self._parent_stt._valence_client
        if not valence_client:
            logger.debug("No Valence client available for emotion detection")
            return

        if not self._audio_buffer:
            logger.debug("Audio buffer is empty, skipping emotion detection")
            return

        try:
            # Combine buffered frames into a single audio chunk
            all_samples: list[np.ndarray] = []
            sample_rate = 48000

            for frame in self._audio_buffer:
                samples = np.frombuffer(frame.data, dtype=np.int16)
                all_samples.append(samples)
                sample_rate = frame.sample_rate

            if not all_samples:
                logger.debug("No samples extracted from buffer")
                return

            combined = np.concatenate(all_samples)

            # Skip if too little audio (~33ms at 48kHz)
            if len(combined) < 1600:
                logger.debug(
                    f"Audio too short: {len(combined)} samples "
                    f"({len(combined) / sample_rate * 1000:.0f}ms)"
                )
                return

            # Get emotion from Valence (this blocks until we get the response)
            duration_ms = len(combined) / sample_rate * 1000
            logger.debug(f"Detecting emotion from {len(combined)} samples ({duration_ms:.0f}ms)")

            emotions = await valence_client.process_audio(combined, sample_rate)

            new_emotion = emotions.get("dominant", "neutral")
            new_confidence = emotions.get("confidence", 0.0)
            all_emotions = emotions.get("all_emotions", {})

            logger.debug(f"Emotion: {new_emotion} ({new_confidence:.1%}) | all: {all_emotions}")

            if new_confidence >= self._min_confidence:
                old_emotion = self._current_emotion
                self._current_emotion = new_emotion
                self._current_confidence = new_confidence
                if old_emotion != new_emotion:
                    logger.info(f"Emotion changed: {old_emotion} -> {new_emotion}")

        except Exception as e:
            logger.error(f"Error detecting emotion: {e}", exc_info=True)

    def _enrich_with_emotion(self, event: stt.SpeechEvent) -> stt.SpeechEvent:
        """Add emotion tags to speech event transcriptions.

        Args:
            event: The original speech event.

        Returns:
            SpeechEvent: Event with emotion-enriched text.
        """
        # Only enrich final and interim transcripts
        if event.type not in (
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            stt.SpeechEventType.INTERIM_TRANSCRIPT,
        ):
            return event

        # Create new alternatives with emotion tags
        new_alternatives = []
        for alt in event.alternatives:
            if alt.text.strip():
                emotion_tag = f"[{self._current_emotion.capitalize()}]"
                new_text = f"{emotion_tag} {alt.text}"
                logger.debug(
                    f"Enriching {event.type.name}: '{alt.text[:30]}' -> [{self._current_emotion}]"
                )
                new_alternatives.append(
                    stt.SpeechData(
                        language=alt.language,
                        text=new_text,
                        start_time=alt.start_time,
                        end_time=alt.end_time,
                        confidence=alt.confidence,
                        speaker_id=alt.speaker_id,
                        is_primary_speaker=alt.is_primary_speaker,
                    )
                )
            else:
                new_alternatives.append(alt)

        return stt.SpeechEvent(
            type=event.type,
            request_id=event.request_id,
            alternatives=new_alternatives,
            recognition_usage=event.recognition_usage,
        )
