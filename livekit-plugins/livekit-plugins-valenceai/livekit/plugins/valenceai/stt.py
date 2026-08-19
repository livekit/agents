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
        server_url: str = "https://api.getvalenceai.com",
        model: EmotionModel = "4emotions",
        min_confidence: float = 0.0,
    ) -> None:
        # Copy capabilities from underlying STT
        super().__init__(capabilities=underlying_stt.capabilities)

        self._underlying_stt = underlying_stt
        key = api_key or os.getenv("VALENCE_API_KEY")
        if not key:
            raise ValueError(
                "Valence API key is required. Provide it via the 'api_key' parameter "
                "or set the VALENCE_API_KEY environment variable."
            )
        self._api_key: str = key
        self._server_url = server_url
        self._model = model
        self._min_confidence = min_confidence

    def _create_client(self) -> ValenceWebSocketClient:
        """Create a new Valence client.

        Each stream owns its own client so that
        concurrent streams never share connection state: one Valence connection
        maps to one server-side audio session, and the per-session emotion
        history and audio clock must not be reset by another stream starting.
        """
        return ValenceWebSocketClient(
            api_key=self._api_key,
            server_url=self._server_url,
            model=self._model,
        )

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
        """Recognize speech from an audio buffer.

        Batch recognition delegates to the underlying STT without emotion
        enrichment: the Valence API only produces a prediction after ~5s of
        accumulated audio, so per-utterance batch requests would either stall
        the response or time out. Emotion tags are added on the streaming
        path (see stream()).

        Args:
            buffer: Audio buffer to recognize.
            language: Language code.
            conn_options: API connection options.

        Returns:
            SpeechEvent: Recognition result from the underlying STT.
        """
        return await self._underlying_stt.recognize(
            buffer, language=language, conn_options=conn_options
        )

    async def aclose(self) -> None:
        """Close the STT and cleanup resources."""
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

        # Each stream owns its own client: one Valence connection maps to one
        # server-side audio session, so concurrent streams must never share
        # a connection or its emotion history. If Valence is unavailable the
        # stream degrades to plain transcription.
        valence_client = self._parent_stt._create_client()
        valence_active = True

        frame_count = 0

        # Frames for Valence go through a single ordered sender task: one
        # writer per socket guarantees chunks arrive in order, and a bounded
        # queue caps memory. Emotion detection is best-effort, so when the
        # queue is full we drop the frame rather than backpressure the STT
        # pipeline.
        send_queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue(maxsize=100)

        async def valence_sender() -> None:
            """Connect to Valence, then send queued frames in order.

            Connecting here — instead of before the underlying stream starts —
            keeps transcription independent of the emotion service: a slow or
            failed Valence connection never delays speech recognition. Frames
            arriving meanwhile buffer in the bounded queue.
            """
            nonlocal valence_active
            try:
                await valence_client.connect()
                await valence_client.start_streaming()
                logger.info("Connected to Valence AI emotion detection API")
            except Exception as e:
                logger.error(f"Failed to connect to Valence, continuing without emotions: {e}")
                valence_active = False
                return
            while True:
                queued = await send_queue.get()
                if queued is None:
                    break
                await valence_client.send_audio_chunk(
                    audio_data=bytes(queued.data),
                    sample_rate=queued.sample_rate,
                    samples_per_channel=queued.samples_per_channel,
                )

        # Everything after client creation runs under one try/finally so the
        # Valence connection can never outlive the stream, no matter where
        # setup fails or is cancelled.
        underlying_stream: stt.RecognizeStream | None = None
        try:
            stream = self._underlying_stt.stream(
                language=self._language,
                conn_options=self._conn_options,
            )
            underlying_stream = stream

            async def forward_audio() -> None:
                """Forward audio frames to underlying STT and stream to Valence."""
                nonlocal frame_count
                async for item in self._input_ch:
                    if isinstance(item, self._FlushSentinel):
                        logger.debug(f"Flush received after {frame_count} frames")
                        stream.flush()
                    else:
                        frame: rtc.AudioFrame = item
                        frame_count += 1

                        # Forward to underlying STT immediately
                        stream.push_frame(frame)

                        # Track audio position
                        frame_duration_ms = (frame.samples_per_channel / frame.sample_rate) * 1000
                        self._current_audio_position_ms += frame_duration_ms

                        # Hand the frame to the ordered Valence sender
                        if valence_active:
                            try:
                                send_queue.put_nowait(frame)
                            except asyncio.QueueFull:
                                logger.warning("Valence send queue full, dropping frame")

                logger.debug(f"Input ended. Total frames: {frame_count}")
                stream.end_input()

            async def receive_events() -> None:
                """Receive events from underlying stream and enrich with emotions."""
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        # Enrich with the latest available emotion (non-blocking);
                        # skip enrichment while Valence is not connected
                        enriched_event = await self._enrich_final_transcript(
                            event,
                            valence_client if valence_client.is_connected else None,
                        )
                        self._event_ch.send_nowait(enriched_event)
                        self._last_final_transcript_ms = self._current_audio_position_ms
                    else:
                        self._event_ch.send_nowait(event)

            # Run the workers concurrently
            forward_task = asyncio.create_task(forward_audio())
            receive_task = asyncio.create_task(receive_events())
            sender_task = asyncio.create_task(valence_sender())

            try:
                await asyncio.gather(forward_task, receive_task)
            finally:
                # gather propagates the first exception without cancelling the
                # siblings, which would otherwise keep consuming audio frames
                for task in (forward_task, receive_task, sender_task):
                    task.cancel()
                await asyncio.gather(
                    forward_task, receive_task, sender_task, return_exceptions=True
                )
        finally:
            await valence_client.stop_streaming()
            await valence_client.disconnect()
            if underlying_stream is not None:
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
