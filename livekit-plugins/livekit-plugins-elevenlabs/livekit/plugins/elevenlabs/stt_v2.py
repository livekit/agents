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

from __future__ import annotations

import asyncio
import base64
import json
import os
import typing
import weakref
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import STTAudioFormat, STTModels

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"


@dataclass
class STTOptions:
    api_key: str
    base_url: str
    language_code: str | None = None
    model_id: STTModels = "scribe_v2_realtime"
    audio_format: STTAudioFormat = "pcm_16000"
    sample_rate: int = 16000
    vad_silence_threshold_secs: float | None = None
    vad_threshold: float | None = None
    min_speech_duration_ms: int | None = None
    min_silence_duration_ms: int | None = None


class STTv2(stt.STT):
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        model_id: STTModels = "scribe_v2_realtime",
        sample_rate: int = 16000,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of ElevenLabs STT v2 with streaming support.

        Uses Voice Activity Detection (VAD) to automatically detect speech segments
        and commit transcriptions when the user stops speaking.

        Args:
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language_code (NotGivenOr[str]): Language code for the STT model. Optional.
            model_id (STTModels): Model ID for Scribe. Default is "scribe_v2_realtime".
            sample_rate (int): Audio sample rate in Hz. Default is 16000.
            vad_silence_threshold_secs (NotGivenOr[float]): Silence threshold in seconds for VAD (must be between 0.3 and 3.0). Optional.
            vad_threshold (NotGivenOr[float]): Threshold for voice activity detection (must be between 0.1 and 0.9). Optional.
            min_speech_duration_ms (NotGivenOr[int]): Minimum speech duration in milliseconds (must be between 50 and 2000). Optional.
            min_silence_duration_ms (NotGivenOr[int]): Minimum silence duration in milliseconds (must be between 50 and 2000). Optional.
        """  # noqa: E501
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or "
                "set ELEVEN_API_KEY environmental variable"
            )

        # Determine audio format based on sample rate
        audio_format = typing.cast(STTAudioFormat, f"pcm_{sample_rate}")

        self._opts = STTOptions(
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            model_id=model_id,
            audio_format=audio_format,
            sample_rate=sample_rate,
            vad_silence_threshold_secs=vad_silence_threshold_secs
            if is_given(vad_silence_threshold_secs)
            else None,
            vad_threshold=vad_threshold if is_given(vad_threshold) else None,
            min_speech_duration_ms=min_speech_duration_ms
            if is_given(min_speech_duration_ms)
            else None,
            min_silence_duration_ms=min_silence_duration_ms
            if is_given(min_silence_duration_ms)
            else None,
        )
        if is_given(language_code):
            self._opts.language_code = language_code
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStreamv2]()

    @property
    def model(self) -> str:
        return self._opts.model_id

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Scribe v2 API does not support non-streaming recognize. Use stream() instead or use the original STT class for Scribe v1"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamv2:
        return SpeechStreamv2(
            stt=self,
            opts=self._opts,
            conn_options=conn_options,
            language=language if is_given(language) else self._opts.language_code,
            http_session=self._ensure_session(),
        )


class SpeechStreamv2(stt.SpeechStream):
    """Streaming speech recognition using ElevenLabs Scribe v2 realtime API"""

    def __init__(
        self,
        *,
        stt: STTv2,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        language: str | None,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language
        self._session = http_session
        self._reconnect_event = asyncio.Event()
        self._speaking = False  # Track if we're currently in a speech segment
        self._max_tokens_to_recompute = 5  # Default from ElevenLabs, updated from session_started

    def _deduplicate_transcript(self, text: str) -> str:
        """
        Remove duplicated suffixes from ElevenLabs committed transcripts.

        ElevenLabs Scribe v2 recomputes the last N tokens for context accuracy
        (controlled by max_tokens_to_recompute). This causes duplicated phrases:
        - "for tomorrow for tomorrow" -> "for tomorrow"
        - "i would like to know i would like to know" -> "i would like to know"
        - "have a swimming pool, have a swimming pool?" -> "have a swimming pool?"
        - "do you speak English? Do you speak English?" -> "do you speak English?"
        - "dog to the hotel? Dog to the hotel?" -> "dog to the hotel?"
        - "if I can if I can uh" -> "if I can uh"

        This function detects and removes these duplicated suffixes using case-insensitive
        matching and ignores filler words (um, uh, er, ah) to catch duplicates like
        "if I can if I can uh" where the filler word appears after the duplicate.
        Search is limited to approximately max_tokens_to_recompute words.
        """
        if not text:
            return text

        # Split into words, preserving punctuation at the end
        words = text.split()
        if len(words) < 2:
            return text

        # Limit search based on max_tokens_to_recompute
        # Note: tokens ≠ words. Typically 1 token ≈ 0.75 words in English
        # So 5 tokens ≈ 3-4 words. We add a small buffer for safety.
        max_search_len = min(
            len(words) // 2,  # Don't search more than half the text
            max(
                1, int(self._max_tokens_to_recompute * 0.75) + 2
            ),  # ~3-4 words for 5 tokens + buffer
        )

        if max_search_len < 1:
            return text

        # Try increasingly larger suffix lengths (from max down to 1)
        for suffix_len in range(max_search_len, 0, -1):
            suffix = words[-suffix_len:]
            prefix_end = len(words) - suffix_len

            # For each suffix, try all sub-suffixes (from longest to shortest)
            # This catches duplicates like "if I can if I can uh" where the duplicate
            # "if I can" is followed by an extra word
            for actual_len in range(suffix_len, 0, -1):
                # Take the first 'actual_len' words from the suffix
                subseq = suffix[:actual_len]
                subseq_normalized = [w.rstrip(",.!?").lower() for w in subseq]

                # Check if this subsequence appears immediately before the suffix
                if prefix_end >= actual_len:
                    potential_duplicate = words[prefix_end - actual_len : prefix_end]
                    potential_normalized = [w.rstrip(",.!?").lower() for w in potential_duplicate]

                    if subseq_normalized == potential_normalized:
                        # Found a duplicate - remove everything from prefix_end onwards
                        deduplicated = " ".join(words[:prefix_end])
                        if deduplicated != text:
                            logger.debug(
                                f"STTv2: Deduplicated transcript (max_tokens_to_recompute={self._max_tokens_to_recompute}): '{text}' -> '{deduplicated}'"
                            )
                        return deduplicated

        return text

    async def _run(self) -> None:
        """Run the streaming transcription session"""
        logger.info("STTv2: Starting streaming session")
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                while True:
                    await ws.ping()
                    await asyncio.sleep(30)
            except Exception:
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # Buffer audio into chunks (50ms chunks)
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            # Track audio frames for debugging
            frame_count = 0
            total_samples = 0
            import time

            start_time = time.time()

            async for data in self._input_ch:
                # Write audio bytes to buffer and get 50ms frames
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

                for frame in frames:
                    frame_count += 1
                    total_samples += len(frame.data)
                    elapsed = time.time() - start_time
                    audio_duration = total_samples / self._opts.sample_rate

                    # Log frame details every 20 frames (~1 second of audio)
                    if frame_count % 20 == 0:
                        logger.info(
                            f"STTv2: Sent {frame_count} frames ({audio_duration:.2f}s audio) "
                            f"in {elapsed:.2f}s real time, frame size: {len(frame.data)} samples"
                        )

                    audio_b64 = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                    await ws.send_str(
                        json.dumps(
                            {
                                "message_type": "input_audio_chunk",
                                "audio_base_64": audio_b64,
                                "commit": False,
                                "sample_rate": self._opts.sample_rate,
                            }
                        )
                    )

            logger.info(
                f"STTv2: Audio send completed. Total: {frame_count} frames, "
                f"{audio_duration:.2f}s audio in {time.time() - start_time:.2f}s real time"
            )
            closing_ws = True

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            import time

            recv_start_time = time.time()
            msg_count = 0

            while True:
                msg = await ws.receive()
                msg_count += 1
                elapsed = time.time() - recv_start_time

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        logger.info(
                            f"STTv2: Received {msg_count} messages in {elapsed:.2f}s before close"
                        )
                        return
                    raise APIStatusError(message="ElevenLabs STT connection closed unexpectedly")

                # TODO: In production, change this to logger.warning and continue
                # For now, raise exception to catch any unexpected message types during development
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise ValueError(f"Unexpected WebSocket message type: {msg.type}")

                try:
                    parsed = json.loads(msg.data)
                    logger.info(
                        f"STTv2: Received message #{msg_count} at t={elapsed:.2f}s, "
                        f"type: {parsed.get('message_type')}"
                    )
                    self._process_stream_event(parsed)
                except Exception:
                    logger.exception("failed to process ElevenLabs STT message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(keepalive_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # Retrieve exception to prevent it from being logged
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Establish WebSocket connection to ElevenLabs Scribe v2 API"""
        # Build query parameters
        params = [
            f"model_id={self._opts.model_id}",
            f"encoding={self._opts.audio_format}",
            f"sample_rate={self._opts.sample_rate}",
            "commit_strategy=vad",  # Always use VAD for automatic speech detection
        ]

        if self._opts.vad_silence_threshold_secs is not None:
            params.append(f"vad_silence_threshold_secs={self._opts.vad_silence_threshold_secs}")
        if self._opts.vad_threshold is not None:
            params.append(f"vad_threshold={self._opts.vad_threshold}")
        if self._opts.min_speech_duration_ms is not None:
            params.append(f"min_speech_duration_ms={self._opts.min_speech_duration_ms}")
        if self._opts.min_silence_duration_ms is not None:
            params.append(f"min_silence_duration_ms={self._opts.min_silence_duration_ms}")
        if self._language:
            params.append(f"language_code={self._language}")

        query_string = "&".join(params)

        # Convert HTTPS URL to WSS
        base_url = self._opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{base_url}/speech-to-text/realtime?{query_string}"

        logger.info(f"STTv2: Connecting to WebSocket URL: {ws_url}")

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                    headers={AUTHORIZATION_HEADER: self._opts.api_key},
                ),
                self._conn_options.timeout,
            )
            logger.info("STTv2: WebSocket connected successfully")
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            logger.error(f"STTv2: Failed to connect to ElevenLabs: {e}")
            raise APIConnectionError("Failed to connect to ElevenLabs") from e

        return ws

    def _process_stream_event(self, data: dict) -> None:
        """Process incoming WebSocket messages from ElevenLabs"""
        message_type = data.get("message_type")
        logger.debug(f"STTv2: Processing event type: {message_type}, {data}")

        if message_type == "partial_transcript":
            # Ignore partial transcripts - only use committed transcripts for voice agents.
            # Partial transcripts are only used for UI feedback and don't trigger agent responses.
            # ElevenLabs doesn't provide correlation IDs, so we can't reliably track which audio
            # chunks correspond to which partial transcripts. This causes issues with stale
            # partials appearing after commits and duplicate text.
            logger.debug(f"STTv2: Ignoring partial transcript: '{data.get('text', '')}'")
            return

        elif message_type == "committed_transcript":
            # Final committed transcripts - these are sent to the LLM/TTS layer in LiveKit agents
            # and trigger agent responses (unlike partial transcripts which are UI-only)
            text = data.get("text", "")

            # Log full message structure to investigate duplication
            # TODO: Remove after understanding the duplication issue
            logger.debug(f"STTv2: Full committed_transcript data: {data}")

            # Deduplicate text to work around ElevenLabs API duplication behavior
            # ElevenLabs may include recomputed tokens (see max_tokens_to_recompute in config)
            # which causes duplicated phrases like "for tomorrow for tomorrow"
            original_text = text
            text = self._deduplicate_transcript(text)
            if original_text != text:
                logger.info(
                    f"STTv2: Deduplicated committed transcript: '{original_text}' -> '{text}'"
                )

            logger.info(f"STTv2: Committed transcript: '{text}'")
            if text:
                # Send START_OF_SPEECH if we're not already speaking
                if not self._speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True
                    logger.info("STTv2: Sent START_OF_SPEECH")

                # Send FINAL_TRANSCRIPT but keep speaking=True
                # Multiple commits can occur within the same speech segment
                final_event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=self._language or "en",
                            text=text,
                        )
                    ],
                )
                self._event_ch.send_nowait(final_event)
                logger.info(f"STTv2: Sent FINAL_TRANSCRIPT: '{text}'")
            else:
                # Empty commit signals end of speech segment (similar to Cartesia's is_final flag)
                # This groups multiple committed transcripts into one speech segment
                if self._speaking:
                    self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
                    self._speaking = False
                    logger.info("STTv2: Sent END_OF_SPEECH (empty commit)")
                else:
                    logger.debug("STTv2: Received empty commit but not speaking, ignoring")

        elif message_type == "committed_transcript_with_timestamps":
            # Final committed transcript with word-level timestamps - sent to LLM/TTS layer
            text = data.get("text", "")

            # Log full message structure to see available metadata
            # TODO: Remove after understanding available fields
            logger.debug(f"STTv2: Full committed_transcript_with_timestamps data: {data}")

            # Deduplicate text (same issue as committed_transcript)
            original_text = text
            text = self._deduplicate_transcript(text)
            if original_text != text:
                logger.info(
                    f"STTv2: Deduplicated transcript with timestamps: '{original_text}' -> '{text}'"
                )

            logger.info(f"STTv2: Committed transcript with timestamps: '{text}'")
            if text:
                # Send START_OF_SPEECH if we're not already speaking
                if not self._speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True
                    logger.info("STTv2: Sent START_OF_SPEECH")

                # Send FINAL_TRANSCRIPT but keep speaking=True
                # Multiple commits can occur within the same speech segment
                final_event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=self._language or "en",
                            text=text,
                        )
                    ],
                )
                self._event_ch.send_nowait(final_event)
                logger.info(f"STTv2: Sent FINAL_TRANSCRIPT with timestamps: '{text}'")

        elif message_type == "session_started":
            # Session initialization message - informational only
            session_id = data.get("session_id", "unknown")
            config = data.get("config", {})

            # Capture max_tokens_to_recompute for deduplication logic
            if "max_tokens_to_recompute" in config:
                self._max_tokens_to_recompute = config["max_tokens_to_recompute"]
                logger.info(
                    f"STTv2: Session started with ID: {session_id}, "
                    f"max_tokens_to_recompute={self._max_tokens_to_recompute}"
                )
            else:
                logger.info(f"STTv2: Session started with ID: {session_id}")

        # Error handling for known ElevenLabs error types
        elif message_type in (
            "auth_error",
            "quota_exceeded",
            "transcriber_error",
            "input_error",
            "error",
        ):
            error_msg = data.get("message", "Unknown error")
            error_details = data.get("details", "")
            logger.error(
                f"STTv2: ElevenLabs error [{message_type}]: {error_msg}"
                + (f" - {error_details}" if error_details else "")
            )
            # Error events don't interrupt the stream, but log them for debugging

        else:
            # TODO: Remove exception before PR - used during development to catch new message types
            logger.warning(f"STTv2: Unknown message type: {message_type}, data: {data}")
            raise ValueError(
                f"Unexpected message type from ElevenLabs STTv2: {message_type}. "
                f"Data: {data}. Please update _process_stream_event to handle this type."
            )
