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

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
    vad,
)
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from .log import logger
from .models import SonioxModels

# Base URL for Soniox Speech-to-Text API.
BASE_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# WebSocket messages and tokens.
KEEPALIVE_MESSAGE = '{"type": "keepalive"}'
FINALIZE_MESSAGE = '{"type": "finalize"}'
END_TOKEN = "<end>"
FINALIZED_TOKEN = "<fin>"


def is_end_token(token: dict) -> bool:
    """Return True if the given token marks an end or finalized event."""
    return token.get("text") in (END_TOKEN, FINALIZED_TOKEN)


@dataclass
class STTOptions:
    """Configuration options for Soniox Speech-to-Text service."""

    model: SonioxModels | None = "stt-rt-preview"
    language_hints: list[str] | None = None
    context: str | None = None

    num_channels: int = 1
    sample_rate: int = 16000
    auto_finalize_delay_ms: int = 3000

    enable_non_final_tokens: bool = True
    max_non_final_tokens_duration_ms: int = 800

    enable_endpoint_detection: bool = True

    client_reference_id: str | None = None


class STT(stt.STT):
    """Speech-to-Text service using Soniox Speech-to-Text API.

    This service connects to Soniox Speech-to-Text API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

    For complete API documentation, see: https://soniox.com/docs/speech-to-text/api-reference/websocket-api
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        vad: vad.VAD | None = None,
        params: STTOptions | None = None,
    ):
        """Initialize instance of Soniox Speech-to-Text API service.

        Args:
            api_key: Soniox API key, if not provided, will look for SONIOX_API_KEY env variable.
            base_url: Base URL for Soniox Speech-to-Text API, default to BASE_URL defined in this
                module.
            http_session: Optional aiohttp.ClientSession to use for requests.
            vad: If passed, enable Voice Activity Detection (VAD) for audio frames.
            params: Additional configuration parameters, such as model, language hints, context and
                speaker diarization.
        """
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        self._api_key = api_key or os.getenv("SONIOX_API_KEY")
        self._base_url = base_url
        self._http_session = http_session
        self._vad_stream = vad.stream() if vad else None
        self._params = params or STTOptions()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Raise error since single-frame recognition is not supported
        by Soniox Speech-to-Text API."""
        raise NotImplementedError(
            "Soniox Speech-to-Text API does not support single frame recognition"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Return a new LiveKit streaming speech-to-text session."""
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Set up state and queues for a WebSocket-based transcription stream."""
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._stt = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_event = asyncio.Event()

        self.audio_queue = asyncio.Queue()

        self._last_tokens_received: float | None = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession for WebSocket connections."""
        if not self._stt._http_session:
            self._stt._http_session = utils.http_context.http_session()

        return self._stt._http_session

    async def _connect_ws(self):
        """Open a WebSocket connection to the Soniox Speech-to-Text API and send the
        initial configuration."""
        # Create initial config object.
        config = {
            "api_key": self._stt._api_key,
            "model": self._stt._params.model,
            "audio_format": "pcm_s16le",
            "num_channels": self._stt._params.num_channels or 1,
            "enable_endpoint_detection": self._stt._params.enable_endpoint_detection,
            "sample_rate": self._stt._params.sample_rate,
            "language_hints": self._stt._params.language_hints,
            "context": self._stt._params.context,
            "enable_non_final_tokens": self._stt._params.enable_non_final_tokens,
            "max_non_final_tokens_duration_ms": self._stt._params.max_non_final_tokens_duration_ms,
            "client_reference_id": self._stt._params.client_reference_id,
        }
        # Connect to the Soniox Speech-to-Text API.
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(self._stt._base_url),
            timeout=self._conn_options.timeout,
        )
        # Set initial configuration message.
        await ws.send_str(json.dumps(config))
        logger.debug("Soniox Speech-to-Text API connection established!")
        return ws

    async def _run(self) -> None:
        """Manage connection lifecycle, spawning tasks and handling reconnection."""
        while True:
            try:
                ws = await self._connect_ws()
                self._ws = ws
                # Create task for audio processing, voice turn detection and message handling.
                tasks = [
                    asyncio.create_task(self._prepare_audio_task()),
                    asyncio.create_task(self._handle_vad_task()),
                    asyncio.create_task(self._send_audio_task()),
                    asyncio.create_task(self._recv_messages_task()),
                    asyncio.create_task(self._keepalive_task()),
                    asyncio.create_task(self._finalize_if_no_tokens_task()),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
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
            # Handle errors.
            except asyncio.TimeoutError as e:
                logger.error(
                    f"Timeout during Soniox Speech-to-Text API connection/initialization: {e}"
                )
                raise APITimeoutError(
                    "Timeout connecting to or initializing Soniox Speech-to-Text API session"
                ) from e

            except aiohttp.ClientResponseError as e:
                logger.error(
                    "Soniox Speech-to-Text API status error during session init:"
                    + f"{e.status} {e.message}"
                )
                raise APIStatusError(
                    message=e.message, status_code=e.status, request_id=None, body=None
                ) from e

            except aiohttp.ClientError as e:
                logger.error(f"Soniox Speech-to-Text API connection error: {e}")
                raise APIConnectionError(f"Soniox Speech-to-Text API connection error: {e}") from e

            except Exception as e:
                logger.exception(f"Unexpected error occurred: {e}")
                raise APIConnectionError(f"An unexpected error occurred: {e}") from e
            # Close the WebSocket connection on finish.
            finally:
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None

    async def _keepalive_task(self):
        """Periodically send keepalive messages (while no audio is being sent)
        to maintain the WebSocket connection."""
        try:
            while self._ws:
                await self._ws.send_str(KEEPALIVE_MESSAGE)
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error while sending keep alive message: {e}")

    async def _prepare_audio_task(self):
        """Read audio frames, process VAD, and enqueue PCM data for sending."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        async for data in self._input_ch:
            if self._stt._vad_stream:
                # If VAD is enabled, push the audio frame to the VAD stream.
                if isinstance(data, self._FlushSentinel):
                    self._stt._vad_stream.flush()
                else:
                    self._stt._vad_stream.push_frame(data)

            if isinstance(data, rtc.AudioFrame):
                # Combine audio frames to get a single frame with all raw PCM data.
                combined_frame = rtc.combine_audio_frames(data)

                # Get the raw bytes from the combined frame.
                pcm_data = combined_frame.data.tobytes()
                self.audio_queue.put_nowait(pcm_data)

    async def _send_audio_task(self):
        """Take queued audio data and transmit it over the WebSocket."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        while self._ws:
            try:
                data = await self.audio_queue.get()

                if isinstance(data, bytes):
                    await self._ws.send_bytes(data)
                else:
                    await self._ws.send_str(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error while sending audio data: {e}")
                break

    async def _handle_vad_task(self):
        """Listen for VAD events to trigger finalize or keepalive messages."""
        if not self._stt._vad_stream:
            logger.debug("VAD stream is not enabled, skipping VAD task")
            return

        async for event in self._stt._vad_stream:
            if event.type == vad.VADEventType.END_OF_SPEECH:
                self.audio_queue.put_nowait(FINALIZE_MESSAGE)

    async def _recv_messages_task(self):
        """Receive transcription messages, handle tokens, errors, and dispatch events."""

        # Transcription frame will be only sent after we get the "endpoint" event.
        final_transcript_buffer = ""

        def send_endpoint_transcript():
            nonlocal final_transcript_buffer
            if final_transcript_buffer:
                event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text=final_transcript_buffer, language=None)],
                )
                self._event_ch.send_nowait(event)
                final_transcript_buffer = ""

        # Method handles receiving messages from the Soniox Speech-to-Text API.
        while self._ws:
            try:
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            content = json.loads(msg.data)
                            tokens = content["tokens"]

                            if tokens:
                                if len(tokens) == 1 and tokens[0]["text"] == FINALIZED_TOKEN:
                                    # Ignore finalized token, prevent auto finalize cycle.
                                    pass
                                else:
                                    # Got at least one token, reset the auto finalize delay.
                                    self._last_tokens_received = time.time()

                            # We will only send the final tokens after we get the "endpoint" event.
                            non_final_transcription = ""

                            for token in tokens:
                                if token["is_final"]:
                                    if is_end_token(token):
                                        # Found an endpoint, tokens until here will be sent as
                                        # transcript, the rest will be sent as interim tokens
                                        # (even final tokens).
                                        send_endpoint_transcript()
                                    else:
                                        final_transcript_buffer += token["text"]
                                else:
                                    non_final_transcription += token["text"]

                            if final_transcript_buffer or non_final_transcription:
                                event = stt.SpeechEvent(
                                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=final_transcript_buffer + non_final_transcription,
                                            language=None,
                                        )
                                    ],
                                )
                                self._event_ch.send_nowait(event)

                            error_code = content.get("error_code")
                            error_message = content.get("error_message")

                            if error_code or error_message:
                                # In case of error, still send the final transcript.
                                send_endpoint_transcript()
                                logger.error(f"WebSocket error: {error_code} - {error_message}")

                            finished = content.get("finished")

                            if finished:
                                # When finished, still send the final transcript.
                                send_endpoint_transcript()
                                logger.debug("Transcription finished")

                        except Exception as e:
                            logger.exception(f"Error processing message: {e}")
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break
                    else:
                        logger.warning(
                            f"Unexpected message type from Soniox Speech-to-Text API: {msg.type}"
                        )
            except aiohttp.ClientError as e:
                logger.error(f"WebSocket error while receiving: {e}")
            except Exception as e:
                logger.error(f"Unexpected error while receiving messages: {e}")

    async def _finalize_if_no_tokens_task(self):
        """Call finalize if no new tokens are received for a configured duration."""
        if not self._ws or self._stt._params.auto_finalize_delay_ms is None:
            return

        try:
            while True:
                await asyncio.sleep(0.5)

                if not self._ws:
                    break

                # Check if we have anything in queue to send.
                if not self.audio_queue.empty():
                    continue

                # Check if enough time has passed since the last tokens were received.
                if self._last_tokens_received:
                    last_token_age_ms = (time.time() - self._last_tokens_received) * 1000

                    if last_token_age_ms > self._stt._params.auto_finalize_delay_ms:
                        # No new tokens received for a while, finalize the transcription.
                        logger.debug("No pending frames, sending finalize message")
                        self.audio_queue.put_nowait(FINALIZE_MESSAGE)
                        self._last_tokens_received = None
        except aiohttp.ClientConnectionError:
            # Expected when closing the connection.
            pass
        except Exception as e:
            logger.error(f"Error in _finalize_if_no_tokens_task_handler: {e}")
