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

        # final tokens are accumulated across messages until an endpoint is detected.
        final = _TokenAccumulator()
        is_speaking = False

        def send_endpoint_transcript() -> None:
            nonlocal is_speaking
            if final.text:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[final.to_speech_data()],
                    )
                )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                    )
                )

                # Reset buffers.
                final.reset()

                # Reset speaking state, so the next transcript will send START_OF_SPEECH again.
                is_speaking = False

        # Method handles receiving messages from the Soniox Speech-to-Text API.
        while self._ws:
            try:
                async for msg in self._ws:
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning(
                            f"Unexpected message type from Soniox Speech-to-Text API: {msg.type}"
                        )
                        continue

                    try:
                        content = json.loads(msg.data)
                        tokens = content["tokens"]

                        non_final = _TokenAccumulator()
                        total_audio_proc_ms = content.get("total_audio_proc_ms", 0)

                        # 1) process tokens: accumulate final/non-final,
                        #    flush immediately on endpoint tokens.
                        for token in tokens:
                            if token["is_final"]:
                                if is_end_token(token):
                                    send_endpoint_transcript()
                                    self._report_processed_audio_duration(
                                        total_audio_proc_ms,
                                    )
                                else:
                                    final.update(token)
                            else:
                                non_final.update(token)

                        # 2) emit START_OF_SPEECH + interim for remaining content.
                        if final.text or non_final.text:
                            if not is_speaking:
                                is_speaking = True
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                                )
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[final.merged_speech_data(non_final)],
                                )
                            )

                        # 3) on error or finish, flush any remaining final tokens.
                        if (
                            content.get("finished")
                            or content.get("error_code")
                            or content.get("error_message")
                        ):
                            send_endpoint_transcript()
                            self._report_processed_audio_duration(total_audio_proc_ms)

                        if content.get("error_code") or content.get("error_message"):
                            logger.error(
                                f"WebSocket error: {content.get('error_code')}"
                                f" - {content.get('error_message')}"
                            )

                        if content.get("finished"):
                            logger.debug("Transcription finished")

                    except Exception as e:
                        logger.exception(f"Error processing message: {e}")

            except aiohttp.ClientError as e:
                logger.error(f"WebSocket error while receiving: {e}")
            except Exception as e:
                logger.error(f"Unexpected error while receiving messages: {e}")


class _TokenAccumulator:
    """Accumulates token metadata (text, language, speaker, timing, confidence).

    Tokens are assumed to arrive in chronological order, so start_time is taken
    from the first token and end_time is continuously overwritten by the latest.
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.language: str = ""
        self.speaker_id: str | None = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0

    def update(self, token: dict[str, Any]) -> None:
        self.text += token["text"]
        if token.get("language") and not self.language:
            self.language = token["language"]
        if "speaker" in token and self.speaker_id is None:
            self.speaker_id = str(token["speaker"])
        if "start_ms" in token and self.start_time == 0.0:
            self.start_time = float(token["start_ms"])
        if "end_ms" in token:
            self.end_time = float(token["end_ms"])
        if "confidence" in token:
            self._confidence_sum += token["confidence"]
            self._confidence_count += 1

    @property
    def confidence(self) -> float:
        if self._confidence_count == 0:
            return 0.0
        return self._confidence_sum / self._confidence_count

    def reset(self) -> None:
        self.text = ""
        self.language = ""
        self.speaker_id = None
        self.start_time = 0.0
        self.end_time = 0.0
        self._confidence_sum = 0.0
        self._confidence_count = 0

    def to_speech_data(self) -> stt.SpeechData:
        return stt.SpeechData(
            text=self.text,
            language=self.language,
            speaker_id=self.speaker_id,
            start_time=self.start_time / 1000,
            end_time=self.end_time / 1000,
            confidence=self.confidence,
        )

    def merged_speech_data(self, other: _TokenAccumulator) -> stt.SpeechData:
        """Build a SpeechData combining self (final) with other (non-final)."""
        candidates = [t for t in (self.start_time, other.start_time) if t > 0.0]
        start = min(candidates) if candidates else 0.0
        end = max(self.end_time, other.end_time)
        total_count = self._confidence_count + other._confidence_count
        total_sum = self._confidence_sum + other._confidence_sum
        return stt.SpeechData(
            text=self.text + other.text,
            language=self.language if self.language else other.language,
            speaker_id=self.speaker_id if self.speaker_id is not None else other.speaker_id,
            start_time=start / 1000,
            end_time=end / 1000,
            confidence=total_sum / total_count if total_count > 0 else 0.0,
        )
