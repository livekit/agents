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
import dataclasses
import json
import logging
import os
import weakref
from dataclasses import dataclass
from urllib.parse import urlencode

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
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import STTAudioFormat, STTModels

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
WS_BASE_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
AUTHORIZATION_HEADER = "xi-api-key"


@dataclass
class STTOptions:
    model_id: STTModels | str
    language_code: str | None
    audio_format: STTAudioFormat
    commit_strategy: str
    include_timestamps: bool
    vad_silence_threshold_secs: float
    vad_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    endpoint_url: str


class STTv2(stt.STT):
    def __init__(
        self,
        *,
        model_id: STTModels | str = "scribe_v2_realtime",
        language_code: NotGivenOr[str] = NOT_GIVEN,
        audio_format: STTAudioFormat = "pcm_16000",
        commit_strategy: Literal["vad", "manual"] = "vad",
        include_timestamps: bool = False,
        vad_silence_threshold_secs: float = 1.5,
        vad_threshold: float = 0.4,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = WS_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of ElevenLabs STT v2 (Scribe v2 Realtime).

        Args:
            model_id: The ElevenLabs model to use. Defaults to "scribe_v2_realtime".
            language_code: Language code for transcription (ISO-639-1/3). Auto-detect if not specified.
            audio_format: Audio format to use. Defaults to "pcm_16000". Supported formats:
                         pcm_8000, pcm_16000, pcm_22050, pcm_24000, pcm_44100, pcm_48000, ulaw_8000
            commit_strategy: Strategy for committing transcripts. Options:
                           - "vad": Automatic segmentation using Voice Activity Detection
                           - "manual": Explicit control over segment commits
            include_timestamps: Whether to include word-level timestamps in transcripts. Defaults to False.
                              When True, provides start/end times for each word in final transcripts.
                              Useful for subtitle generation, word highlighting, and audio-text alignment.
            vad_silence_threshold_secs: Silence duration (in seconds) to trigger commit (0.3-3.0).
                                      Only applies when commit_strategy="vad".
            vad_threshold: VAD sensitivity (0.1-0.9). Higher = more sensitive to speech.
                         Only applies when commit_strategy="vad".
            min_speech_duration_ms: Minimum speech duration in milliseconds (50-2000).
            min_silence_duration_ms: Minimum silence duration in milliseconds (50-2000).
            api_key: ElevenLabs API key. Can be set via argument or ELEVEN_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession for connection pooling.

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Example:
            >>> from livekit.plugins.elevenlabs import STTv2
            >>> stt = STTv2(
            ...     language_code="en",
            ...     commit_strategy="vad",
            ...     vad_threshold=0.4,
            ... )

        Note:
            - Scribe v2 Realtime supports 90+ languages with auto-detection
            - Built-in VAD provides automatic speech segmentation
            - Ultra-low latency: ~150ms for real-time transcription
            - Requires ElevenLabs API key with STT permission enabled
        """

        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or "
                "set ELEVEN_API_KEY environmental variable"
            )

        self._api_key = elevenlabs_api_key
        self._opts = STTOptions(
            model_id=model_id,
            language_code=language_code if is_given(language_code) else None,
            audio_format=audio_format,
            commit_strategy=commit_strategy,
            include_timestamps=include_timestamps,
            vad_silence_threshold_secs=vad_silence_threshold_secs,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            endpoint_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStreamv2]()

    def _get_sample_rate(self) -> int:
        """Extract sample rate from audio_format string (e.g., 'pcm_16000' -> 16000)"""
        try:
            # Audio format is like "pcm_16000" or "ulaw_8000"
            parts = self._opts.audio_format.split("_")
            if len(parts) >= 2:
                return int(parts[-1])
            # Default to 16000 if parsing fails
            return 16000
        except (ValueError, IndexError):
            logger.warning(
                f"Could not parse sample rate from '{self._opts.audio_format}', defaulting to 16000"
            )
            return 16000

    def _build_websocket_url(self, opts: STTOptions) -> str:
        """Build WebSocket URL with query parameters using proper URL encoding

        Args:
            opts: STTOptions to use for building the URL (may have session overrides)
        """
        params: dict[str, str] = {
            "model_id": opts.model_id,
            "audio_format": opts.audio_format,
            "commit_strategy": opts.commit_strategy,
        }

        # Add optional parameters
        if opts.language_code:
            params["language_code"] = opts.language_code

        if opts.include_timestamps:
            params["include_timestamps"] = "true"

        if opts.commit_strategy == "vad":
            params["vad_silence_threshold_secs"] = str(opts.vad_silence_threshold_secs)
            params["vad_threshold"] = str(opts.vad_threshold)
            params["min_speech_duration_ms"] = str(opts.min_speech_duration_ms)
            params["min_silence_duration_ms"] = str(opts.min_silence_duration_ms)

        return f"{WS_BASE_URL}?{urlencode(params)}"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Non-streaming recognition fallback.

        Note:
            Scribe v2 Realtime is optimized for streaming. For best performance,
            use stream() method instead. This fallback uses the REST API with scribe_v1.
        """
        raise NotImplementedError(
            "Scribe v2 Realtime is designed for streaming only. "
            "Use stream() method or use STT (scribe_v1) for non-streaming recognition."
        )

    @property
    def model(self) -> str:
        return self._opts.model_id

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for WebSocket connections"""
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        """Create a copy of options with session overrides applied.

        Args:
            language: Override language code for this session.

        Returns:
            STTOptions: Merged options with session-specific overrides.
        """
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language_code = language

        return config

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamv2:
        """Create a streaming STT session using WebSocket.

        Args:
            language: Override language code for this stream. If not specified,
                     uses the language_code from initialization or auto-detection.
            conn_options: Connection options for retry and timeout configuration.

        Returns:
            SpeechStreamv2: A streaming STT session for real-time transcription.

        Example:
            >>> stream = stt.stream()
            >>> async for event in stream:
            ...     if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            ...         print(f"Final: {event.alternatives[0].text}")
        """
        config = self._sanitize_options(language=language)
        stream = SpeechStreamv2(
            stt=self,
            opts=config,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self._ensure_session(),
            base_url=self._opts.endpoint_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model_id: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        commit_strategy: NotGivenOr[str] = NOT_GIVEN,
        include_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update STT options and propagate to active streams.

        This will trigger reconnection of all active streams with new parameters.

        Args:
            model_id: ElevenLabs model to use (e.g., "scribe_v2_realtime")
            language_code: Language code (ISO-639-1/3) or None for auto-detection
            commit_strategy: "vad" (automatic) or "manual" (explicit control)
            include_timestamps: Whether to include word-level timestamps
            vad_silence_threshold_secs: Silence duration to trigger commit (0.3-3.0)
            vad_threshold: VAD sensitivity (0.1-0.9, higher = more sensitive)
            min_speech_duration_ms: Minimum speech duration (50-2000ms)
            min_silence_duration_ms: Minimum silence duration (50-2000ms)
            endpoint_url: WebSocket endpoint URL
        """
        if is_given(model_id):
            self._opts.model_id = model_id
        if is_given(language_code):
            self._opts.language_code = language_code
        if is_given(commit_strategy):
            self._opts.commit_strategy = commit_strategy
        if is_given(include_timestamps):
            self._opts.include_timestamps = include_timestamps
        if is_given(vad_silence_threshold_secs):
            self._opts.vad_silence_threshold_secs = vad_silence_threshold_secs
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url

        # Propagate to all active streams
        for stream in self._streams:
            stream.update_options(
                model_id=model_id,
                language_code=language_code,
                commit_strategy=commit_strategy,
                include_timestamps=include_timestamps,
                vad_silence_threshold_secs=vad_silence_threshold_secs,
                vad_threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                endpoint_url=endpoint_url,
            )


class SpeechStreamv2(stt.RecognizeStream):
    """Streaming STT session for ElevenLabs Scribe v2 Realtime using WebSocket

    Note: Uses RecognizeStream (canonical name) instead of SpeechStream (deprecated alias).
    This is the modern, future-proof base class for STT streaming.
    """

    def __init__(
        self,
        *,
        stt: STTv2,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        base_url: str,
    ):
        # Get sample rate from audio format for proper audio resampling
        sample_rate = stt._get_sample_rate()
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._stt_v2 = stt  # Type-safe reference to STTv2
        self._opts = opts  # Stream-specific options (may have session overrides)
        self._api_key = api_key
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session = http_session

        # Reconnection and state tracking
        self._reconnect_event = asyncio.Event()
        self._session_id = ""  # Track session ID for event correlation
        self._speaking = False  # Track if user is currently speaking
        self._speech_duration = 0.0  # Track audio duration for usage reporting

    def update_options(
        self,
        *,
        model_id: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        commit_strategy: NotGivenOr[str] = NOT_GIVEN,
        include_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update stream options and trigger reconnection.

        This triggers a reconnection to apply the new settings.

        Args:
            model_id: ElevenLabs model to use (e.g., "scribe_v2_realtime")
            language_code: Language code (ISO-639-1/3) or None for auto-detection
            commit_strategy: "vad" (automatic) or "manual" (explicit control)
            include_timestamps: Whether to include word-level timestamps
            vad_silence_threshold_secs: Silence duration to trigger commit (0.3-3.0)
            vad_threshold: VAD sensitivity (0.1-0.9, higher = more sensitive)
            min_speech_duration_ms: Minimum speech duration (50-2000ms)
            min_silence_duration_ms: Minimum silence duration (50-2000ms)
            endpoint_url: WebSocket endpoint URL
        """
        if is_given(model_id):
            self._opts.model_id = model_id
        if is_given(language_code):
            self._opts.language_code = language_code
        if is_given(commit_strategy):
            self._opts.commit_strategy = commit_strategy
        if is_given(include_timestamps):
            self._opts.include_timestamps = include_timestamps
        if is_given(vad_silence_threshold_secs):
            self._opts.vad_silence_threshold_secs = vad_silence_threshold_secs
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url

        # Trigger reconnection with new options
        self._reconnect_event.set()

    def _send_transcript_event(
        self,
        event_type: stt.SpeechEventType,
        text: str,
        *,
        timestamps: list[dict] | None = None,
    ) -> None:
        """Helper to send transcript events with consistent structure.

        Args:
            event_type: Type of speech event (INTERIM_TRANSCRIPT, FINAL_TRANSCRIPT, etc.)
            text: Transcribed text
            timestamps: Optional list of word timestamps from ElevenLabs
        """
        # Build speech data with explicit fields for type safety
        start_time = 0.0
        end_time = 0.0
        if timestamps and len(timestamps) > 0:
            start_time = timestamps[0]["start"]
            end_time = timestamps[-1]["end"]

        event = stt.SpeechEvent(
            type=event_type,
            request_id=self._session_id,
            alternatives=[
                stt.SpeechData(
                    language=self._opts.language_code or "en",
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.0,  # ElevenLabs doesn't provide confidence scores
                )
            ],
        )
        self._event_ch.send_nowait(event)

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Establish WebSocket connection to ElevenLabs"""
        # Build WebSocket URL with stream-specific options (including any session overrides)
        url = self._stt_v2._build_websocket_url(self._opts)
        headers = {AUTHORIZATION_HEADER: self._api_key}

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    url,
                    headers=headers,
                    heartbeat=30.0,  # Send heartbeat every 30 seconds to keep connection alive
                ),
                timeout=self._conn_options.timeout,
            )
            logger.debug("Established new ElevenLabs STT WebSocket connection")
            return ws
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to ElevenLabs") from e

    async def _run(self) -> None:
        """Main WebSocket connection and message handling loop with reconnection support"""
        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:  # Reconnection loop
            closing_ws = False

            @utils.log_exceptions(logger=logger)
            async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
                """Send audio frames to ElevenLabs"""
                nonlocal closing_ws

                # Buffer audio into 50ms chunks for network efficiency
                sample_rate = self._stt_v2._get_sample_rate()
                samples_50ms = sample_rate // 20
                audio_bstream = utils.audio.AudioByteStream(
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=samples_50ms,
                )

                frame_count = 0
                has_ended = False
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []

                    if isinstance(data, rtc.AudioFrame):
                        # Write to buffer and get 50ms chunks
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        # Flush any remaining audio
                        frames.extend(audio_bstream.flush())
                        has_ended = True

                        # Handle manual commit if needed
                        if self._opts.commit_strategy == "manual":
                            await self._send_commit(ws)

                    # Send all buffered 50ms chunks
                    for frame in frames:
                        frame_count += 1
                        if frame_count == 1:
                            logger.debug("Audio streaming started")

                        # Track audio duration for usage reporting
                        self._speech_duration += frame.duration

                        # Send audio frame
                        await self._send_audio_frame(frame, ws)

                        if has_ended:
                            # Flush audio duration tracking on last frame
                            has_ended = False

                # Signal we're done sending
                closing_ws = True
                # ElevenLabs handles WebSocket protocol close frames (sent in finally block)
                # No need for application-level close message

            @utils.log_exceptions(logger=logger)
            async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
                """Receive and process messages from ElevenLabs"""
                nonlocal closing_ws

                while True:
                    msg = await ws.receive()

                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        # Check if this is an expected close
                        if closing_ws or (self._session and self._session.closed):  # noqa: B023
                            return

                        # Unexpected close, trigger reconnection
                        raise APIStatusError(message="ElevenLabs connection closed unexpectedly")

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning(f"unexpected message type: {msg.type}")
                        continue

                    try:
                        self._process_stream_event(json.loads(msg.data))
                    except Exception:
                        logger.exception("failed to process ElevenLabs message")

            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    # Check if we should exit or reconnect
                    if wait_reconnect_task not in done:
                        break  # Normal exit

                    # Reconnect requested
                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # Retrieve the exception
            finally:
                if ws is not None:
                    await ws.close()

    def _process_stream_event(self, data: dict) -> None:
        """Handle incoming WebSocket messages"""
        message_type = data.get("message_type")

        if message_type == "session_started":
            if session_id := data.get("session_id"):
                self._session_id = session_id
            logger.debug("ElevenLabs session started - ready to receive audio")

        elif message_type == "partial_transcript":
            # Interim result
            text = data.get("text", "")
            if text:
                # Emit START_OF_SPEECH on first partial transcript
                if not self._speaking:
                    start_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH,
                        request_id=self._session_id,
                    )
                    self._event_ch.send_nowait(start_event)
                    self._speaking = True

                # Emit interim transcript
                self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, text)

        elif message_type == "committed_transcript":
            # Final result (without timestamps)
            text = data.get("text", "")
            if text:
                # Emit final transcript
                self._send_transcript_event(stt.SpeechEventType.FINAL_TRANSCRIPT, text)

                # Emit END_OF_SPEECH
                if self._speaking:
                    end_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        request_id=self._session_id,
                    )
                    self._event_ch.send_nowait(end_event)
                    self._speaking = False

                # Emit RECOGNITION_USAGE for billing/analytics
                if self._speech_duration > 0:
                    usage_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._session_id,
                        alternatives=[],
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=self._speech_duration
                        ),
                    )
                    self._event_ch.send_nowait(usage_event)
                    self._speech_duration = 0

        elif message_type == "committed_transcript_with_timestamps":
            # Final result with word-level timestamps
            text = data.get("text", "")
            words = data.get("words", [])

            if text:
                # Emit final transcript with utterance-level timing
                self._send_transcript_event(
                    stt.SpeechEventType.FINAL_TRANSCRIPT, text, timestamps=words
                )

                # Log word-level details if available (for debugging/advanced use)
                if words and logger.isEnabledFor(logging.DEBUG):
                    duration = words[-1].get("end", 0.0) - words[0].get("start", 0.0)
                    logger.debug(
                        f"Word-level timestamps: {len(words)} words, duration: {duration:.2f}s"
                    )

                # Emit END_OF_SPEECH
                if self._speaking:
                    end_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        request_id=self._session_id,
                    )
                    self._event_ch.send_nowait(end_event)
                    self._speaking = False

                # Emit RECOGNITION_USAGE for billing/analytics
                if self._speech_duration > 0:
                    usage_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._session_id,
                        alternatives=[],
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=self._speech_duration
                        ),
                    )
                    self._event_ch.send_nowait(usage_event)
                    self._speech_duration = 0

        elif message_type == "auth_error":
            error_msg = data.get("error", "Authentication failed")
            logger.error(f"ElevenLabs authentication error: {error_msg}", extra={"data": data})
            logger.error("Please check your ELEVEN_API_KEY and ensure STT permission is enabled")
            raise APIStatusError(message=f"Authentication failed: {error_msg}")

        elif message_type == "input_error":
            error_msg = data.get("message", "Invalid input data")
            error_details = data.get("details", {})
            logger.error(
                f"ElevenLabs input error: {error_msg}",
                extra={"details": error_details, "data": data},
            )
            raise APIStatusError(message=f"Input error: {error_msg}")

        elif message_type == "quota_exceeded":
            error_msg = data.get("message", "API quota exceeded")
            logger.error(f"ElevenLabs quota exceeded: {error_msg}", extra={"data": data})
            logger.error("Please check your ElevenLabs account quota and billing")
            raise APIStatusError(message=f"Quota exceeded: {error_msg}", status_code=429)

        elif message_type == "transcriber_error":
            error_msg = data.get("message", "Transcription error")
            logger.error(f"ElevenLabs transcriber error: {error_msg}", extra={"data": data})
            raise APIStatusError(message=f"Transcriber error: {error_msg}")

        elif message_type == "error":
            error_msg = data.get("message", "Unknown error")
            error_code = data.get("code", -1)
            logger.error(
                f"ElevenLabs API error: {error_msg}", extra={"code": error_code, "data": data}
            )
            raise APIStatusError(message=error_msg, status_code=error_code)

        else:
            # Log unknown message types for debugging
            logger.debug(f"Unknown message type: {message_type}", extra={"data": data})

    async def _send_audio_frame(
        self, frame: rtc.AudioFrame, ws: aiohttp.ClientWebSocketResponse
    ) -> None:
        """Send audio frame to WebSocket"""
        try:
            # Convert frame to bytes and base64 encode for JSON transport
            audio_data = frame.data.tobytes()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": audio_base64,
                "commit": False,  # Let VAD handle commits if using VAD strategy
                "sample_rate": frame.sample_rate,
            }

            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Error sending audio frame: {e}", exc_info=True)

    async def _send_commit(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send commit signal for manual commit strategy"""
        try:
            commit_msg = {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": self._stt_v2._get_sample_rate(),
            }
            await ws.send_json(commit_msg)
        except Exception as e:
            logger.error(f"Error sending commit: {e}", exc_info=True)
