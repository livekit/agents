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
transcriptions with emotion tags from Valence AI on a per-sentence basis.

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
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, stt

from .client import ValenceWebSocketClient
from .log import logger

EmotionModel = Literal["4emotions", "7emotions"]

# Sentence boundary pattern - splits on . ! ? followed by space or end
SENTENCE_PATTERN = re.compile(r'([.!?]+)(?:\s+|$)')


@dataclass
class TimestampedFrame:
    """Audio frame with timestamp information."""
    frame: rtc.AudioFrame
    start_time_ms: float  # Start time in milliseconds
    end_time_ms: float    # End time in milliseconds


class STT(stt.STT):
    """Emotion-aware STT that combines an underlying STT with Valence AI emotion detection.

    This STT wrapper intercepts audio, sends it to both the underlying STT provider
    and the Valence AI emotion detection API, then enriches the transcriptions with
    emotion tags on a per-sentence basis.

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

        # If we have audio and Valence is connected, detect emotion per sentence
        if self._valence_client and self._valence_connected:
            try:
                # Convert buffer to numpy array for Valence
                samples = np.frombuffer(buffer.data, dtype=np.int16)

                # Enrich the transcription with emotions per sentence
                if result.alternatives:
                    new_alternatives = []
                    for alt in result.alternatives:
                        enriched_text = await self._enrich_text_with_emotions(
                            alt.text, samples, buffer.sample_rate
                        )
                        new_alternatives.append(
                            stt.SpeechData(
                                language=alt.language,
                                text=enriched_text,
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

    async def _enrich_text_with_emotions(
        self,
        text: str,
        samples: np.ndarray,
        sample_rate: int,
    ) -> str:
        """Enrich text with per-sentence emotion tags.

        Args:
            text: The transcribed text.
            samples: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Text with emotion tags for each sentence.
        """
        if not text.strip():
            return text

        sentences = split_into_sentences(text)
        if not sentences:
            return text

        # If only one sentence, detect emotion for the whole audio
        if len(sentences) == 1:
            emotions = await self._valence_client.process_audio(samples, sample_rate)
            emotion = emotions.get("dominant", "neutral")
            confidence = emotions.get("confidence", 0.0)
            if confidence >= self._min_confidence:
                return f"[{emotion.capitalize()}] {sentences[0]}"
            return sentences[0]

        # Multiple sentences - divide audio proportionally by character count
        total_chars = sum(len(s) for s in sentences)
        total_samples = len(samples)

        enriched_parts = []
        sample_offset = 0

        for sentence in sentences:
            # Calculate audio segment for this sentence based on character proportion
            char_ratio = len(sentence) / total_chars
            segment_samples = int(total_samples * char_ratio)

            # Extract audio segment
            segment_end = min(sample_offset + segment_samples, total_samples)
            audio_segment = samples[sample_offset:segment_end]
            sample_offset = segment_end

            # Detect emotion for this segment
            if len(audio_segment) >= 1600:  # Minimum ~33ms at 48kHz
                try:
                    emotions = await self._valence_client.process_audio(
                        audio_segment, sample_rate
                    )
                    emotion = emotions.get("dominant", "neutral")
                    confidence = emotions.get("confidence", 0.0)

                    if confidence >= self._min_confidence:
                        enriched_parts.append(f"[{emotion.capitalize()}] {sentence}")
                    else:
                        enriched_parts.append(f"[Neutral] {sentence}")
                except Exception as e:
                    logger.error(f"Error detecting emotion for sentence: {e}")
                    enriched_parts.append(f"[Neutral] {sentence}")
            else:
                enriched_parts.append(f"[Neutral] {sentence}")

        return " ".join(enriched_parts)

    async def aclose(self) -> None:
        """Close the STT and cleanup resources."""
        if self._valence_client:
            await self._valence_client.disconnect()
        await self._underlying_stt.aclose()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: The text to split.

    Returns:
        List of sentences.
    """
    if not text.strip():
        return []

    # Split on sentence boundaries
    parts = SENTENCE_PATTERN.split(text)

    sentences = []
    current = ""

    for i, part in enumerate(parts):
        if not part:
            continue
        # Check if this part is punctuation
        if SENTENCE_PATTERN.match(part + " "):
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current += part

    # Add any remaining text
    if current.strip():
        sentences.append(current.strip())

    return sentences


class EmotionAwareRecognizeStream(stt.RecognizeStream):
    """Streaming recognition with per-sentence emotion awareness.

    This stream wraps an underlying STT stream and enriches transcriptions
    with emotion tags for each sentence based on the corresponding audio segment.

    Architecture: Audio is buffered with timestamps. When a FINAL_TRANSCRIPT arrives,
    the text is split into sentences and emotion detection runs on each sentence's
    corresponding audio segment.
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

        # Buffer for emotion detection - stores audio frames with timestamps
        self._audio_buffer: list[TimestampedFrame] = []
        self._buffer_start_time_ms: float = 0.0
        self._current_time_ms: float = 0.0

    async def _run(self) -> None:
        """Main processing loop."""
        logger.debug("Starting emotion-aware streaming recognition (per-sentence)")

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

                    # Buffer audio with timestamp for emotion detection
                    frame_duration_ms = (frame.samples_per_channel / frame.sample_rate) * 1000
                    timestamped_frame = TimestampedFrame(
                        frame=frame,
                        start_time_ms=self._current_time_ms,
                        end_time_ms=self._current_time_ms + frame_duration_ms,
                    )
                    self._audio_buffer.append(timestamped_frame)
                    self._current_time_ms += frame_duration_ms

                    # Enforce buffer size limit to prevent memory leak
                    if len(self._audio_buffer) > STT.MAX_BUFFER_FRAMES:
                        excess = len(self._audio_buffer) - STT.MAX_BUFFER_FRAMES
                        self._audio_buffer = self._audio_buffer[excess:]
                        if self._audio_buffer:
                            self._buffer_start_time_ms = self._audio_buffer[0].start_time_ms
                        logger.debug(f"Buffer trimmed, removed {excess} old frames")

            logger.debug(f"Input ended. Total frames: {frame_count}")
            underlying_stream.end_input()

        async def receive_events() -> None:
            """Receive events from underlying stream and enrich with per-sentence emotions."""
            async for event in underlying_stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    # Enrich final transcript with per-sentence emotions
                    enriched_event = await self._enrich_final_transcript(event)
                    self._event_ch.send_nowait(enriched_event)
                    # Clear buffer after processing final transcript
                    self._audio_buffer.clear()
                    self._buffer_start_time_ms = self._current_time_ms
                elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    # For interim transcripts, just pass through without emotion
                    # (we'll add emotions on the final)
                    self._event_ch.send_nowait(event)
                else:
                    # Pass through other events unchanged
                    self._event_ch.send_nowait(event)

        # Run both tasks concurrently
        forward_task = asyncio.create_task(forward_audio())
        receive_task = asyncio.create_task(receive_events())

        try:
            await asyncio.gather(forward_task, receive_task)
        finally:
            await underlying_stream.aclose()

    async def _enrich_final_transcript(self, event: stt.SpeechEvent) -> stt.SpeechEvent:
        """Enrich a final transcript with per-sentence emotion tags.

        Args:
            event: The final transcript event.

        Returns:
            Event with emotion-enriched text for each sentence.
        """
        valence_client = self._parent_stt._valence_client
        if not valence_client or not self._audio_buffer:
            return event

        # Get combined audio from buffer
        all_samples: list[np.ndarray] = []
        sample_rate = 48000

        for tf in self._audio_buffer:
            samples = np.frombuffer(tf.frame.data, dtype=np.int16)
            all_samples.append(samples)
            sample_rate = tf.frame.sample_rate

        if not all_samples:
            return event

        combined_samples = np.concatenate(all_samples)
        total_duration_ms = self._current_time_ms - self._buffer_start_time_ms

        logger.debug(
            f"Processing final transcript with {len(combined_samples)} samples "
            f"({total_duration_ms:.0f}ms)"
        )

        new_alternatives = []
        for alt in event.alternatives:
            if alt.text.strip():
                enriched_text = await self._enrich_text_per_sentence(
                    alt.text,
                    combined_samples,
                    sample_rate,
                    valence_client,
                )
                logger.debug(f"Enriched: '{alt.text[:50]}...' -> '{enriched_text[:80]}...'")
                new_alternatives.append(
                    stt.SpeechData(
                        language=alt.language,
                        text=enriched_text,
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

    async def _enrich_text_per_sentence(
        self,
        text: str,
        samples: np.ndarray,
        sample_rate: int,
        valence_client: ValenceWebSocketClient,
    ) -> str:
        """Enrich text with emotion tags for each sentence.

        Args:
            text: The transcribed text.
            samples: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.
            valence_client: The Valence WebSocket client.

        Returns:
            Text with emotion tags for each sentence.
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return text

        logger.debug(f"Split into {len(sentences)} sentences: {sentences}")

        # If only one sentence, detect emotion for the whole audio
        if len(sentences) == 1:
            if len(samples) >= 1600:
                try:
                    emotions = await valence_client.process_audio(samples, sample_rate)
                    emotion = emotions.get("dominant", "neutral")
                    confidence = emotions.get("confidence", 0.0)
                    logger.debug(f"Single sentence emotion: {emotion} ({confidence:.1%})")
                    if confidence >= self._min_confidence:
                        return f"[{emotion.capitalize()}] {sentences[0]}"
                except Exception as e:
                    logger.error(f"Error detecting emotion: {e}")
            return f"[Neutral] {sentences[0]}"

        # Multiple sentences - divide audio proportionally by character count
        total_chars = sum(len(s) for s in sentences)
        total_samples = len(samples)

        enriched_parts = []
        sample_offset = 0

        for i, sentence in enumerate(sentences):
            # Calculate audio segment for this sentence based on character proportion
            char_ratio = len(sentence) / total_chars
            segment_samples = int(total_samples * char_ratio)

            # For the last sentence, take all remaining samples
            if i == len(sentences) - 1:
                audio_segment = samples[sample_offset:]
            else:
                segment_end = min(sample_offset + segment_samples, total_samples)
                audio_segment = samples[sample_offset:segment_end]
                sample_offset = segment_end

            # Detect emotion for this segment
            emotion = "neutral"
            if len(audio_segment) >= 1600:  # Minimum ~33ms at 48kHz
                try:
                    emotions = await valence_client.process_audio(audio_segment, sample_rate)
                    detected_emotion = emotions.get("dominant", "neutral")
                    confidence = emotions.get("confidence", 0.0)
                    logger.debug(
                        f"Sentence {i+1}/{len(sentences)} '{sentence[:30]}...' -> "
                        f"{detected_emotion} ({confidence:.1%})"
                    )
                    if confidence >= self._min_confidence:
                        emotion = detected_emotion
                except Exception as e:
                    logger.error(f"Error detecting emotion for sentence {i+1}: {e}")

            enriched_parts.append(f"[{emotion.capitalize()}] {sentence}")

        return " ".join(enriched_parts)
