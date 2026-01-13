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
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from .log import logger

# Base URL for Soniox Speech-to-Text API.
BASE_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# WebSocket messages and tokens.
KEEPALIVE_MESSAGE = '{"type": "keepalive"}'
END_TOKEN = "<end>"
FINALIZED_TOKEN = "<fin>"


def is_end_token(token: dict) -> bool:
    """Return True if the given token marks an end or finalized event."""
    return token.get("text") in (END_TOKEN, FINALIZED_TOKEN)


@dataclass
class ContextGeneralItem:
    key: str
    value: str


@dataclass
class ContextTranslationTerm:
    source: str
    target: str


@dataclass
class ContextObject:
    """Context object for models with context_version 2, for Soniox stt-rt-v3-preview and higher.

    Learn more about context in the documentation:
    https://soniox.com/docs/stt/concepts/context
    """

    general: list[ContextGeneralItem] | None = None
    text: str | None = None
    terms: list[str] | None = None
    translation_terms: list[ContextTranslationTerm] | None = None


@dataclass
class STTOptions:
    """Configuration options for Soniox Speech-to-Text service."""

    model: str = "stt-rt-v3"

    language_hints: list[str] | None = None
    language_hints_strict: bool = False
    context: ContextObject | str | None = None

    num_channels: int = 1
    sample_rate: int = 16000

    enable_speaker_diarization: bool = False
    enable_language_identification: bool = True

    client_reference_id: str | None = None


class STT(stt.STT):
    """Speech-to-Text service using Soniox Speech-to-Text API.

    This service connects to Soniox Speech-to-Text API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

    For complete API documentation, see: https://soniox.com/docs/stt/api-reference/websocket-api
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        params: STTOptions | None = None,
    ):
        """Initialize instance of Soniox Speech-to-Text API service.

        Args:
            api_key: Soniox API key, if not provided, will look for SONIOX_API_KEY env variable.
            base_url: Base URL for Soniox Speech-to-Text API, default to BASE_URL defined in this
                module.
            http_session: Optional aiohttp.ClientSession to use for requests.
            params: Additional configuration parameters, such as model, language hints, context and
                speaker diarization.
        """
        params = params or STTOptions()
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
                diarization=params.enable_speaker_diarization,
            )
        )

        self._api_key = api_key or os.getenv("SONIOX_API_KEY")
        if not self._api_key:
            raise ValueError("Soniox API key is required. Set SONIOX_API_KEY or pass api_key")
        self._base_url = base_url
        self._http_session = http_session
        self._params = params

    @property
    def model(self) -> str:
        return self._params.model

    @property
    def provider(self) -> str:
        return "Soniox"

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
        self._stt: STT = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_event = asyncio.Event()

        self.audio_queue: asyncio.Queue[bytes | str] = asyncio.Queue()

        self._reported_duration_ms = 0

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession for WebSocket connections."""
        if not self._stt._http_session:
            self._stt._http_session = utils.http_context.http_session()

        return self._stt._http_session

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection to the Soniox Speech-to-Text API and send the
        initial configuration."""
        context_raw = self._stt._params.context
        context_value: dict[str, Any] | str | None
        if isinstance(context_raw, ContextObject):
            context_value = asdict(context_raw)
        else:
            context_value = context_raw

        # Create initial config object.
        config: dict[str, Any] = {
            "api_key": self._stt._api_key,
            "model": self._stt._params.model,
            "audio_format": "pcm_s16le",
            "num_channels": self._stt._params.num_channels or 1,
            "enable_endpoint_detection": True,
            "sample_rate": self._stt._params.sample_rate,
            "language_hints": self._stt._params.language_hints,
            "language_hints_strict": self._stt._params.language_hints_strict,
            "context": context_value,
            "enable_speaker_diarization": self._stt._params.enable_speaker_diarization,
            "enable_language_identification": self._stt._params.enable_language_identification,
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

        # Reset duration tracking on new connection
        self._reported_duration_ms = 0
        return ws

    def _report_processed_audio_duration(self, total_audio_proc_ms: float) -> None:
        """Report the total audio duration processed by the STT engine."""
        to_report_ms = total_audio_proc_ms - self._reported_duration_ms
        if to_report_ms <= 0:
            return

        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(
                audio_duration=to_report_ms / 1000,
            ),
        )
        self._event_ch.send_nowait(usage_event)
        self._reported_duration_ms = int(total_audio_proc_ms)

    async def _run(self) -> None:
        """Manage connection lifecycle, spawning tasks and handling reconnection."""
        while True:
            try:
                ws = await self._connect_ws()
                self._ws = ws
                # Create task for audio processing, voice turn detection and message handling.
                tasks: list[asyncio.Task[None]] = [
                    asyncio.create_task(self._prepare_audio_task()),
                    asyncio.create_task(self._send_audio_task()),
                    asyncio.create_task(self._recv_messages_task()),
                    asyncio.create_task(self._keepalive_task()),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                tasks_group: asyncio.Future[Any] = asyncio.gather(*tasks)
                try:
                    done, _ = await asyncio.wait(
                        [tasks_group, wait_reconnect_task],
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
                    tasks_group.exception()

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

    async def _keepalive_task(self) -> None:
        """Periodically send keepalive messages (while no audio is being sent)
        to maintain the WebSocket connection."""
        try:
            while self._ws:
                await self._ws.send_str(KEEPALIVE_MESSAGE)
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error while sending keep alive message: {e}")

    async def _prepare_audio_task(self) -> None:
        """Read audio frames and enqueue PCM data for sending."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                # Get the raw bytes from the audio frame.
                pcm_data = data.data.tobytes()
                self.audio_queue.put_nowait(pcm_data)

    async def _send_audio_task(self) -> None:
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
                raise
            except Exception as e:
                logger.error(f"Error while sending audio data: {e}")
                break

    async def _recv_messages_task(self) -> None:
        """Receive transcription messages, handle tokens, errors, and dispatch events."""

        # Transcription frame will be only sent after we get the "endpoint" event.
        final_transcript_buffer = ""
        # Language code sent by Soniox if language detection is enabled (e.g. "en", "de", "fr")
        final_transcript_language: str = ""
        final_speaker_id: str | None = None

        is_speaking = False

        def send_endpoint_transcript() -> None:
            nonlocal \
                final_transcript_buffer, \
                final_transcript_language, \
                final_speaker_id, \
                is_speaking
            if final_transcript_buffer:
                event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text=final_transcript_buffer,
                            language=final_transcript_language,
                            speaker_id=final_speaker_id,
                        )
                    ],
                )
                self._event_ch.send_nowait(event)

                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

                # Reset buffers.
                final_transcript_buffer = ""
                final_transcript_language = ""
                final_speaker_id = None

                # Reset speaking state, so the next transcript will send START_OF_SPEECH again.
                is_speaking = False

        # Method handles receiving messages from the Soniox Speech-to-Text API.
        while self._ws:
            try:
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            content = json.loads(msg.data)
                            tokens = content["tokens"]

                            # We will only send the final tokens after we get the "endpoint" event.
                            non_final_transcription = ""
                            non_final_transcription_language: str = ""
                            non_final_speaker_id: str | None = None

                            total_audio_proc_ms = content.get("total_audio_proc_ms", 0)

                            for token in tokens:
                                if token["is_final"]:
                                    if is_end_token(token):
                                        # Found an endpoint, tokens until here will be sent as
                                        # transcript, the rest will be sent as interim tokens
                                        # (even final tokens).
                                        send_endpoint_transcript()
                                        self._report_processed_audio_duration(total_audio_proc_ms)
                                    else:
                                        final_transcript_buffer += token["text"]

                                        # Soniox provides language for each token,
                                        # LiveKit requires only a single language for the entire transcription chunk.
                                        # Current heuristic is to take the first language we see.
                                        if token.get("language") and not final_transcript_language:
                                            final_transcript_language = token.get("language")

                                        if "speaker" in token and final_speaker_id is None:
                                            final_speaker_id = str(token["speaker"])
                                else:
                                    non_final_transcription += token["text"]
                                    if (
                                        token.get("language")
                                        and not non_final_transcription_language
                                    ):
                                        non_final_transcription_language = token.get("language")

                                    if "speaker" in token and non_final_speaker_id is None:
                                        non_final_speaker_id = str(token["speaker"])

                            if final_transcript_buffer or non_final_transcription:
                                if not is_speaking:
                                    # Send START_OF_SPEECH if this is the first transcript.
                                    is_speaking = True
                                    self._event_ch.send_nowait(
                                        stt.SpeechEvent(
                                            type=SpeechEventType.START_OF_SPEECH,
                                        )
                                    )

                                event = stt.SpeechEvent(
                                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=final_transcript_buffer + non_final_transcription,
                                            language=(
                                                final_transcript_language
                                                if final_transcript_language
                                                else non_final_transcription_language
                                            ),
                                            speaker_id=(
                                                final_speaker_id
                                                if final_speaker_id is not None
                                                else non_final_speaker_id
                                            ),
                                        )
                                    ],
                                )
                                self._event_ch.send_nowait(event)

                            error_code = content.get("error_code")
                            error_message = content.get("error_message")

                            if error_code or error_message:
                                # In case of error, still send the final transcript.
                                send_endpoint_transcript()
                                self._report_processed_audio_duration(total_audio_proc_ms)
                                logger.error(f"WebSocket error: {error_code} - {error_message}")

                            finished = content.get("finished")

                            if finished:
                                # When finished, still send the final transcript.
                                send_endpoint_transcript()
                                self._report_processed_audio_duration(total_audio_proc_ms)
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
