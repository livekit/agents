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

"""Speech-to-Text implementation for SimpliSmart

This module provides an STT implementation that uses the SimpliSmart API.
"""

import asyncio
import base64
import enum
import json
import os
import weakref
from typing import Any, Literal

import aiohttp
from pydantic import BaseModel

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, rtc
from livekit.agents.utils.misc import is_given

from .log import logger


class ConnectionState(enum.Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class SimplismartSTTOptions(BaseModel):
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    without_timestamps: bool = True
    vad_model: Literal["silero", "frame"] = "frame"
    vad_filter: bool = True
    model: str | None = "openai/whisper-large-v3-turbo"
    word_timestamps: bool = False
    vad_onset: float | None = 0.5
    vad_offset: float | None = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = 30
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
    diarization: bool = False
    initial_prompt: str | None = None
    hotwords: str | None = None
    num_speakers: int = 0
    compression_ratio_threshold: float | None = 2.4
    beam_size: int = 4
    temperature: float = 0.0
    multilingual: bool = False
    max_tokens: float | None = 400
    log_prob_threshold: float | None = -1.0
    length_penalty: int = 1
    repetition_penalty: float = 1.01
    suppress_tokens: list[int] = [-1]
    strict_hallucination_reduction: bool = False
    streaming_url: str | None = None


class STT(stt.STT):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        streaming_url: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | SimplismartSTTOptions | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ):

        assert (
            base_url is not None or streaming_url is not None
        ), "base_url or streaming_url are required"
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True if streaming_url is not None else False,
                interim_results=False,
                aligned_transcript="word",
            )
        )

        self._api_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not self._api_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        if params is None:
            params = SimplismartSTTOptions()

        if isinstance(params, SimplismartSTTOptions):
            self._opts = params
            self._model = params.model
        else:
            self._opts = SimplismartSTTOptions(**params)
        self._opts.streaming_url = streaming_url
        self._base_url = base_url
        self._streaming_url = streaming_url
        self._logger = logger.getChild(self.__class__.__name__)
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def provider(self) -> str:
        return "Simplismart"

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
        language = (
            self._opts.language if isinstance(language, type(NOT_GIVEN)) else language
        )
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = self._opts.model_dump()

        payload["audio_data"] = audio_b64
        payload["language"] = language
        payload["model"] = self._model

        try:
            async with self._ensure_session().post(
                self._base_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    self._logger.error(
                        f"Simplismart API error: {res.status} - {error_text}"
                    )
                    raise APIStatusError(
                        message=f"Simplismart API Error: {error_text}",
                        status_code=res.status,
                    )

                response_json = await res.json()

                detected_language = response_json["info"]["language"]

                start_time = response_json["timestamps"][0][0]
                end_time = response_json["timestamps"][-1][1]
                request_id = response_json.get("request_id", "")
                text = "".join(response_json["transcription"])

                alternatives = [
                    stt.SpeechData(
                        language=detected_language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=1.0,
                    ),
                ]

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=alternatives,
                )
        except asyncio.TimeoutError as e:
            self._logger.error(f"Simplismart API timeout: {e}")
            raise APITimeoutError("Simplismart API request timed out") from e
        except aiohttp.ClientError as e:
            self._logger.error(f"Simplismart API client error: {e}")
            raise APIConnectionError(f"Simplismart API connection error: {e}") from e
        except Exception as e:
            self._logger.error(f"Error during Simplismart STT processing: {e}")
            raise APIConnectionError(f"Unexpected error in Simplismart STT: {e}") from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> "SpeechStream":
        """Create a streaming transcription session."""
        opts_language = language if is_given(language) else self._opts.language
        opts_model = model if is_given(model) else self._opts.model

        if not isinstance(opts_language, str):
            opts_language = self._opts.language
        if not isinstance(opts_model, str):
            opts_model = self._opts.model

        # Create options for the stream
        stream_opts = SimplismartSTTOptions(
            language=opts_language, model=opts_model, streaming_url=self._streaming_url
        )

        # Create a fresh session for this stream to avoid conflicts
        stream_session = aiohttp.ClientSession()

        if self._api_key is None:
            raise ValueError("API key cannot be None")
        stream = SpeechStream(
            stt=self,
            opts=stream_opts,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=stream_session,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
    """Simplismart streaming speech-to-text implementation."""

    _CHUNK_DURATION_MS = 50
    _SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        stt: STT,
        opts: SimplismartSTTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        self._opts = opts
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=self._SAMPLE_RATE
        )
        self._api_key = api_key
        self._session = http_session
        self._logger = logger.getChild(self.__class__.__name__)
        self._reconnect_event = asyncio.Event()

        # Connection state management
        self._connection_state = ConnectionState.DISCONNECTED
        self._connection_lock = asyncio.Lock()
        self._session_id = id(self)

        # Add flush mechanism
        self._ws: aiohttp.ClientWebSocketResponse | None = (
            None  # Store WebSocket reference for flush
        )
        self._should_flush = False  # Flag to trigger flush

        # Task management for cleanup
        self._audio_task: asyncio.Task | None = None
        self._message_task: asyncio.Task | None = None
        self._chunk_size = max(
            int(self._SAMPLE_RATE * self._CHUNK_DURATION_MS / 1000),
            1,
        )

    async def aclose(self) -> None:
        """Close the stream and clean up resources."""
        self._logger.debug(
            "Starting stream cleanup", extra={"session_id": self._session_id}
        )

        async with self._connection_lock:
            self._connection_state = ConnectionState.DISCONNECTED

        # Cancel running tasks first
        tasks_to_cancel = []
        if self._audio_task and not self._audio_task.done():
            tasks_to_cancel.append(self._audio_task)
        if self._message_task and not self._message_task.done():
            tasks_to_cancel.append(self._message_task)

        if tasks_to_cancel:
            try:
                await utils.aio.cancel_and_wait(*tasks_to_cancel)
            except Exception as e:
                self._logger.warning(
                    f"Error cancelling tasks: {e}",
                    extra={"session_id": self._session_id},
                )

        # Close WebSocket
        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
                self._logger.debug(
                    "WebSocket closed", extra={"session_id": self._session_id}
                )
        except Exception as e:
            self._logger.warning(
                f"Error closing WebSocket: {e}", extra={"session_id": self._session_id}
            )
        finally:
            self._ws = None

        # Call parent cleanup
        try:
            await super().aclose()
        except Exception as e:
            self._logger.warning(
                f"Error in parent cleanup: {e}", extra={"session_id": self._session_id}
            )

        # Close session last
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                self._logger.debug(
                    "HTTP session closed", extra={"session_id": self._session_id}
                )
        except Exception as e:
            self._logger.warning(
                f"Error closing session: {e}", extra={"session_id": self._session_id}
            )
        finally:
            # Clear reference to help with garbage collection
            pass  # Session reference will be cleared when object is destroyed

    async def _send_initial_config(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send initial configuration message with language for Simplismart models."""
        try:
            config_message = {"language": self._opts.language}
            await ws.send_json(config_message)
            self._logger.info(
                "Sent initial config for Simplismart model",
                extra={"session_id": self._session_id, "language": self._opts.language},
            )
        except Exception as e:
            self._logger.error(
                f"Failed to send initial configuration: {e}",
                extra={"session_id": self._session_id},
                exc_info=True,
            )
            raise APIConnectionError(f"Failed to send initial config: {e}") from e

    async def _run(self) -> None:
        """Main streaming loop with WebSocket connection."""
        num_retries = 0
        max_retries = getattr(self._conn_options, "max_retry_count", 3)

        while num_retries <= max_retries:
            try:
                await self._run_connection()
                break  # Success, exit retry loop

            except (
                aiohttp.ClientConnectorError,
                asyncio.TimeoutError,
            ) as e:  # TODO: Check if retry should happen for every Exception type
                if num_retries == max_retries:
                    async with self._connection_lock:
                        self._connection_state = ConnectionState.FAILED
                    raise APIConnectionError(
                        f"Failed to connect to STT WebSocket after {max_retries} attempts"
                    ) from e

                # Exponential backoff with jitter, max 30 seconds
                retry_interval = min(2**num_retries + (num_retries * 0.1), 30)
                async with self._connection_lock:
                    self._connection_state = ConnectionState.RECONNECTING

                self._logger.warning(
                    f"Connection failed, retrying in {retry_interval:.1f}s",
                    extra={
                        "session_id": self._session_id,
                        "attempt": num_retries + 1,
                        "max_retries": max_retries + 1,
                        "error": str(e),
                    },
                )
                await asyncio.sleep(retry_interval)
                num_retries += 1

            except Exception as e:
                async with self._connection_lock:
                    self._connection_state = ConnectionState.FAILED
                self._logger.error(
                    f"Unrecoverable error in WebSocket connection: {e}",
                    extra={"session_id": self._session_id},
                    exc_info=True,
                )
                raise APIConnectionError(f"WebSocket connection failed: {e}") from e

    async def _run_connection(self) -> None:
        """Run a single WebSocket connection attempt."""
        # Check if session is still valid
        if self._session.closed:
            raise APIConnectionError(
                "Session is closed, cannot establish WebSocket connection"
            )

        async with self._connection_lock:
            self._connection_state = ConnectionState.CONNECTING

        # Build WebSocket URL with parameters
        if self._opts.streaming_url is None:
            raise ValueError("streaming_url cannot be None")
        ws_url = self._opts.streaming_url

        # Connect to WebSocket with proper authentication
        headers = {"api-subscription-key": self._api_key}

        self._logger.info(
            "Connecting to STT WebSocket",
            extra={"session_id": self._session_id, "url": ws_url},
        )

        ws = await asyncio.wait_for(
            self._session.ws_connect(ws_url, headers=headers),
            self._conn_options.timeout,
        )

        # Store WebSocket reference for cleanup - ensure it's always cleaned up
        self._ws = ws

        async with self._connection_lock:
            self._connection_state = ConnectionState.CONNECTED

        self._logger.info(
            "WebSocket connected successfully", extra={"session_id": self._session_id}
        )

        # Send initial configuration message for Simplismart models
        if self._opts.language:
            await self._send_initial_config(ws)

        # Create tasks for audio processing and message handling
        self._audio_task = asyncio.create_task(self._process_audio(ws))
        self._message_task = asyncio.create_task(self._process_messages(ws))

        # Wait for both tasks to complete or reconnection event
        tasks = [self._audio_task, self._message_task]
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())

        try:
            done, pending = await asyncio.wait(
                tasks + [reconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Check if reconnection was requested
            if reconnect_task in done:
                self._logger.info(
                    "Reconnection requested, closing current connection",
                    extra={"session_id": self._session_id},
                )
                self._reconnect_event.clear()
                return

            # Cancel remaining tasks using LiveKit's utility
            if pending:
                await utils.aio.cancel_and_wait(*pending)

            # Check for exceptions in completed tasks
            for task in done:
                if task != reconnect_task:
                    exc = task.exception()
                    if exc is not None:
                        if isinstance(exc, BaseException):
                            raise exc
                        else:
                            raise RuntimeError(
                                f"Task failed with non-BaseException: {exc}"
                            )

        finally:
            # Clean up tasks
            all_tasks = tasks + [reconnect_task]
            await utils.aio.cancel_and_wait(*all_tasks)

            # Close WebSocket
            try:
                if ws and not ws.closed:
                    await ws.close()
            except Exception as e:
                self._logger.warning(
                    f"Error closing WebSocket: {e}",
                    extra={"session_id": self._session_id},
                )

    @utils.log_exceptions(logger=logger)
    async def _process_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Process audio frames and send them in chunks."""

        import numpy as np

        # Audio buffering for chunked sending
        audio_buffer: list[np.int16] = []
        chunk_size = self._chunk_size  # Derived from selected sample rate
        chunks_sent = 0

        self._logger.debug(
            "Starting audio processing",
            extra={"session_id": self._session_id, "chunk_size": chunk_size},
        )

        try:
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    try:
                        # Convert audio frame to Int16 data
                        audio_data = frame.data.tobytes()
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        audio_buffer.extend(audio_array)

                        # Check if we have enough data for a chunk
                        while len(audio_buffer) >= chunk_size:
                            # Convert to Int16Array
                            chunk_data = np.array(
                                audio_buffer[:chunk_size], dtype=np.int16
                            )
                            await ws.send_bytes(chunk_data.tobytes())
                            chunks_sent += 1

                            # Remove sent data from buffer
                            audio_buffer = audio_buffer[chunk_size:]

                            # Log progress periodically
                            if chunks_sent % 100 == 0:
                                self._logger.debug(
                                    f"Sent {chunks_sent} audio chunks",
                                    extra={"session_id": self._session_id},
                                )

                    except Exception as e:
                        self._logger.error(
                            f"Error processing audio frame: {e}",
                            extra={"session_id": self._session_id},
                            exc_info=True,
                        )
                        raise

                elif isinstance(frame, self._FlushSentinel):
                    # LiveKit VAD FlushSentinel - handles stream termination
                    self._logger.debug(
                        "Received FlushSentinel, sending end of stream",
                        extra={"session_id": self._session_id},
                    )
                    await ws.send_str(self._end_of_stream_msg)
                    break

                # Check if Simplismart VAD triggered flush
                if self._should_flush:
                    self._logger.debug(
                        "VAD triggered flush, sending flush message",
                        extra={"session_id": self._session_id},
                    )
                    flush_message = {"type": "flush"}
                    await ws.send_str(json.dumps(flush_message))
                    self._should_flush = False  # Reset flag

        except Exception as e:
            self._logger.error(
                f"Error in audio processing: {e}",
                extra={"session_id": self._session_id, "chunks_sent": chunks_sent},
                exc_info=True,
            )
            raise
        finally:
            self._logger.debug(
                f"Audio processing completed, sent {chunks_sent} chunks",
                extra={"session_id": self._session_id},
            )

    @utils.log_exceptions(logger=logger)
    async def _process_messages(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Process incoming messages from the WebSocket."""
        self._logger.info(
            "Starting message processing",
            extra={"session_id": self._session_id, "ws_closed": ws.closed},
        )

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    try:
                        data = msg.data.decode("utf-8")
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        self._logger.warning(
                            "Invalid JSON received from WebSocket",
                            extra={
                                "session_id": self._session_id,
                                "raw_data": msg.data,
                                "error": str(e),
                            },
                        )
                        continue  # Skip malformed message
                    except Exception as e:
                        self._logger.error(
                            "Error processing WebSocket message",
                            extra={"session_id": self._session_id, "error": str(e)},
                            exc_info=True,
                        )
                        # Re-raise unexpected errors as they might indicate serious issues
                        raise APIStatusError(f"Message processing error: {e}") from e

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    error_msg = f"WebSocket error: {ws.exception()}"
                    self._logger.error(
                        error_msg, extra={"session_id": self._session_id}
                    )
                    raise APIConnectionError(error_msg)

                else:
                    self._logger.debug(
                        f"Unknown WebSocket message type: {msg.type}",
                        extra={"session_id": self._session_id},
                    )

        except Exception as e:
            self._logger.error(
                f"Error in message processing loop: {e}",
                extra={"session_id": self._session_id},
                exc_info=True,
            )
            raise

    async def _handle_message(self, data: str) -> None:
        """Handle different types of messages from Simplismart streaming API."""
        try:
            await self._handle_transcript_data(data)

        except KeyError as e:
            self._logger.warning(
                f"Missing required field in message: {e}",
                extra={"session_id": self._session_id, "data": data},
            )
        except Exception as e:
            self._logger.error(
                f"Unexpected error handling message: {e}",
                extra={"session_id": self._session_id, "data": data},
                exc_info=True,
            )
            raise APIStatusError(f"Message processing error: {e}") from e

    async def _handle_transcript_data(self, data: str) -> None:
        """Handle transcription result messages."""
        transcript_text = data
        request_id = self._session_id

        try:
            # Create usage event with proper metrics extraction
            metrics = {}
            request_data = {
                "original_id": request_id,
                "processing_latency": metrics.get("processing_latency", 0.0),
            }
            usage_event = stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=json.dumps(request_data),
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=metrics.get("audio_duration", 0.0),
                ),
            )
            self._event_ch.send_nowait(usage_event)

            # Create speech data
            speech_data = stt.SpeechData(
                language=self._opts.language,
                text=transcript_text,
            )

            # Create final transcript event with request_id
            speech_event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[speech_data],
            )
            self._event_ch.send_nowait(speech_event)

            self._logger.debug(
                "Transcript processed successfully",
                extra={
                    "session_id": self._session_id,
                    "text_length": len(transcript_text),
                    "language": self._opts.language,
                    "request_id": request_id,
                    "confidence": speech_data.confidence,
                },
            )

        except Exception as e:
            self._logger.error(
                f"Error processing transcript data: {e}",
                extra={
                    "session_id": self._session_id,
                    "transcript_text": transcript_text,
                },
                exc_info=True,
            )
            raise
