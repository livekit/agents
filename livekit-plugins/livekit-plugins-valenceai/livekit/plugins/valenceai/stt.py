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

Audio is streamed continuously to the Valence API, which produces emotion
predictions every ~5 seconds. When a FINAL_TRANSCRIPT arrives from the
underlying STT, the text is enriched with the closest available emotion
prediction — no blocking wait required.

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
import time
from typing import Literal

import numpy as np

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from .client import ValenceWebSocketClient
from .log import logger

EmotionModel = Literal["4emotions", "7emotions"]

# Sentence boundary pattern - splits on . ! ? followed by space or end
SENTENCE_PATTERN = re.compile(r"([.!?]+)(?:\s+|$)")


class STT(stt.STT):
    """Emotion-aware STT that combines an underlying STT with Valence AI emotion detection.

    This STT wrapper streams audio continuously to the Valence AI API and
    enriches transcriptions with emotion tags on a per-sentence basis using
    the latest available prediction — never blocking the STT pipeline.

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
        language: NotGivenOr[str] = NOT_GIVEN,
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
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Recognize speech from an audio buffer with emotion awareness.

        Uses the legacy batch process_audio() method with a 7s timeout.

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
                # Combine frames if buffer is a list of AudioFrames
                combined = rtc.combine_audio_frames(buffer) if isinstance(buffer, list) else buffer
                samples = np.frombuffer(combined.data, dtype=np.int16)

                # Enrich the transcription with emotions per sentence
                if result.alternatives:
                    new_alternatives = []
                    for alt in result.alternatives:
                        enriched_text = await self._enrich_text_with_emotions(
                            alt.text, samples, combined.sample_rate
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
        """Enrich text with per-sentence emotion tags (legacy batch path)."""
        if not text.strip():
            return text

        sentences = split_into_sentences(text)
        if not sentences:
            return text

        # If only one sentence, detect emotion for the whole audio
        if len(sentences) == 1:
            assert self._valence_client is not None
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
            char_ratio = len(sentence) / total_chars
            segment_samples = int(total_samples * char_ratio)

            segment_end = min(sample_offset + segment_samples, total_samples)
            audio_segment = samples[sample_offset:segment_end]
            sample_offset = segment_end

            if len(audio_segment) >= 1600:  # Minimum ~33ms at 48kHz
                try:
                    assert self._valence_client is not None
                    emotions = await self._valence_client.process_audio(audio_segment, sample_rate)
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

    for _i, part in enumerate(parts):
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
    """Streaming recognition with continuous emotion detection.

    Audio frames are streamed to both the underlying STT and the Valence API
    simultaneously. Predictions arrive asynchronously every ~5 seconds and are
    stored with timestamps. When a FINAL_TRANSCRIPT arrives, the text is
    enriched with the closest available emotion prediction — instantly, with
    no blocking wait.
    """

    def __init__(
        self,
        *,
        stt_instance: STT,
        underlying_stt: stt.STT,
        min_confidence: float,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(stt=stt_instance, conn_options=conn_options)

        self._parent_stt = stt_instance
        self._underlying_stt = underlying_stt
        self._min_confidence = min_confidence
        self._language = language

        # Audio position tracking for timestamp correlation
        self._current_audio_position_ms: float = 0.0
        self._last_final_transcript_ms: float = 0.0

    async def _run(self) -> None:
        """Main processing loop with continuous Valence streaming."""
        logger.debug("Starting emotion-aware streaming recognition (continuous)")

        # Ensure Valence is connected
        valence_connected = await self._parent_stt._ensure_valence_connected()
        valence_client = self._parent_stt._valence_client
        logger.debug(f"Valence connected: {valence_connected}")

        # Start continuous streaming to Valence
        if valence_client and valence_connected:
            await valence_client.start_streaming()

        # Create underlying stream
        underlying_stream = self._underlying_stt.stream(
            language=self._language,
            conn_options=self._conn_options,
        )

        frame_count = 0

        _background_tasks: set[asyncio.Task[None]] = set()

        async def forward_audio() -> None:
            """Forward audio frames to underlying STT and stream to Valence."""
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

                    # Track audio position
                    frame_duration_ms = (frame.samples_per_channel / frame.sample_rate) * 1000
                    self._current_audio_position_ms += frame_duration_ms

                    # Stream to Valence API continuously (fire-and-forget)
                    if valence_client and valence_connected:
                        task = asyncio.create_task(
                            valence_client.send_audio_chunk(
                                audio_data=bytes(frame.data),
                                sample_rate=frame.sample_rate,
                                samples_per_channel=frame.samples_per_channel,
                            )
                        )
                        _background_tasks.add(task)
                        task.add_done_callback(_background_tasks.discard)

            logger.debug(f"Input ended. Total frames: {frame_count}")
            underlying_stream.end_input()

        async def receive_events() -> None:
            """Receive events from underlying stream and enrich with emotions."""
            async for event in underlying_stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    # Enrich with the latest available emotion (non-blocking)
                    enriched_event = await self._enrich_final_transcript(event, valence_client)
                    self._event_ch.send_nowait(enriched_event)
                    self._last_final_transcript_ms = self._current_audio_position_ms
                elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    self._event_ch.send_nowait(event)
                else:
                    self._event_ch.send_nowait(event)

        # Run both tasks concurrently
        forward_task = asyncio.create_task(forward_audio())
        receive_task = asyncio.create_task(receive_events())

        try:
            await asyncio.gather(forward_task, receive_task)
        finally:
            for task in _background_tasks:
                task.cancel()
            if valence_client and valence_connected:
                await valence_client.stop_streaming()
            await underlying_stream.aclose()

    async def _enrich_final_transcript(
        self,
        event: stt.SpeechEvent,
        valence_client: ValenceWebSocketClient | None,
    ) -> stt.SpeechEvent:
        """Enrich a final transcript using available emotion predictions.

        This method never blocks waiting for new predictions. It uses whatever
        emotion data has already been received from the continuous stream.
        """
        if not valence_client:
            return event

        t0 = time.perf_counter()

        new_alternatives = []
        for alt in event.alternatives:
            if alt.text.strip():
                enriched_text = await self._enrich_text(alt.text, valence_client)
                logger.debug(f"Enriched: '{alt.text[:50]}' -> '{enriched_text[:80]}'")
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

        elapsed_ms = (time.perf_counter() - t0) * 1000
        history_len = (
            len(valence_client._emotion_history)
            if hasattr(valence_client, "_emotion_history")
            else 0
        )
        logger.info(
            f"[PERF] EMOTION | enrichment={elapsed_ms:.1f}ms "
            f"predictions_available={history_len} "
            f"audio_position={self._current_audio_position_ms:.0f}ms"
        )

        return stt.SpeechEvent(
            type=event.type,
            request_id=event.request_id,
            alternatives=new_alternatives,
            recognition_usage=event.recognition_usage,
        )

    async def _enrich_text(
        self,
        text: str,
        valence_client: ValenceWebSocketClient,
    ) -> str:
        """Enrich text with emotion tags using cached predictions.

        Uses timestamp correlation to match emotion predictions to the
        audio time range of this transcript segment. Never blocks.
        """
        sentences = split_into_sentences(text)
        if not sentences:
            return text

        # Time range for this transcript
        transcript_start_ms = self._last_final_transcript_ms
        transcript_end_ms = self._current_audio_position_ms

        if len(sentences) == 1:
            emotion_data = await valence_client.get_emotion_for_timerange(
                transcript_start_ms, transcript_end_ms
            )
            emotion = emotion_data.get("dominant", "neutral")
            confidence = emotion_data.get("confidence", 0.0)
            logger.info(
                f"[PERF] EMOTION | text='{text[:40]}' emotion={emotion} "
                f"confidence={confidence:.1%} "
                f"from_prediction_at={emotion_data.get('timestamp_ms', 0):.0f}ms "
                f"transcript_range=[{transcript_start_ms:.0f}-{transcript_end_ms:.0f}ms]"
            )
            if confidence >= self._min_confidence:
                return f"[{emotion.capitalize()}] {sentences[0]}"
            return f"[Neutral] {sentences[0]}"

        # Multiple sentences: split time range proportionally by character count
        total_chars = sum(len(s) for s in sentences)
        total_duration_ms = transcript_end_ms - transcript_start_ms

        enriched_parts = []
        time_offset_ms = transcript_start_ms

        for sentence in sentences:
            char_ratio = len(sentence) / total_chars
            sentence_duration_ms = total_duration_ms * char_ratio
            sentence_end_ms = time_offset_ms + sentence_duration_ms

            emotion_data = await valence_client.get_emotion_for_timerange(
                time_offset_ms, sentence_end_ms
            )
            emotion = emotion_data.get("dominant", "neutral")
            confidence = emotion_data.get("confidence", 0.0)

            if confidence >= self._min_confidence:
                enriched_parts.append(f"[{emotion.capitalize()}] {sentence}")
            else:
                enriched_parts.append(f"[Neutral] {sentence}")

            time_offset_ms = sentence_end_ms

        return " ".join(enriched_parts)
