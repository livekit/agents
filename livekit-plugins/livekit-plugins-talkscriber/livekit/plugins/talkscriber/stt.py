# Copyright 2024 LiveKit, Inc.
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
import uuid
from dataclasses import dataclass

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.utils import AudioBuffer

from .log import logger

# Talkscriber STT WebSocket API endpoint
# Support environment variables for flexible deployment
# Default to Talkscriber API server as per reference implementation
STT_SERVER_HOST = os.environ.get("STT_SERVER_HOST", "api.talkscriber.com")
STT_SERVER_PORT = int(os.environ.get("STT_SERVER_PORT", "9090"))
STT_SERVER_USE_SSL = os.environ.get("STT_SERVER_USE_SSL", "true").lower() == "true"

# Build URL based on environment
_protocol = "wss" if STT_SERVER_USE_SSL else "ws"
BASE_URL = f"{_protocol}://{STT_SERVER_HOST}:{STT_SERVER_PORT}"


@dataclass
class STTOptions:
    language: str | None
    interim_results: bool
    sample_rate: int
    num_channels: int
    multilingual: bool
    translate: bool
    api_key: str
    enable_turn_detection: bool
    turn_detection_timeout: float


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        interim_results: bool = True,
        sample_rate: int = 16000,
        model: str = "general",
        multilingual: bool = False,
        translate: bool = True,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = BASE_URL,
        enable_turn_detection: bool = True,
        turn_detection_timeout: float = 0.6,
    ) -> None:
        """Create a new instance of Talkscriber STT.

        Args:
            language: The language code for recognition. Defaults to "en".
            interim_results: Whether to return interim (non-final) transcription results. Defaults to True.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            model: The model to use for transcription. Defaults to "general".
            multilingual: Enable multilingual mode. Defaults to False.
            translate: Enable translation. Defaults to True.
            api_key: Your Talkscriber API key. If not provided, will look for TALKSCRIBER_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for Talkscriber API. Defaults to WebSocket endpoint.
            enable_turn_detection: Enable smart turn detection using ML model for better endpoint detection. Defaults to True.
            turn_detection_timeout: Timeout threshold for end-of-speech detection in seconds (fallback when ML model confidence is low). Defaults to 0.6.

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting
            the TALKSCRIBER_API_KEY environmental variable.
        """

        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=interim_results)
        )
        self._base_url = base_url

        api_key = api_key or os.environ.get("TALKSCRIBER_API_KEY")
        if api_key is None:
            raise ValueError("Talkscriber API key is required")

        self._api_key = api_key
        self._http_session = http_session

        self._opts = STTOptions(
            language=language,
            interim_results=interim_results,
            sample_rate=sample_rate,
            num_channels=1,
            multilingual=multilingual,
            translate=translate,
            api_key=api_key,
            enable_turn_detection=enable_turn_detection,
            turn_detection_timeout=turn_detection_timeout,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """
        Non-streaming recognition (placeholder implementation).
        """
        raise NotImplementedError("Non-streaming recognition not implemented for Talkscriber")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            opts=self._sanitize_options(language=language),
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self._ensure_session(),
            base_url=self._base_url,
        )

    def update_options(
        self,
        *,
        language: str | None = None,
        interim_results: bool | None = None,
        sample_rate: int | None = None,
        multilingual: bool | None = None,
        translate: bool | None = None,
        enable_turn_detection: bool | None = None,
        turn_detection_timeout: float | None = None,
    ):
        if language is not None:
            self._opts.language = language
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if multilingual is not None:
            self._opts.multilingual = multilingual
        if translate is not None:
            self._opts.translate = translate
        if enable_turn_detection is not None:
            self._opts.enable_turn_detection = enable_turn_detection
        if turn_detection_timeout is not None:
            self._opts.turn_detection_timeout = turn_detection_timeout

    def _sanitize_options(self, *, language: str | None = None) -> STTOptions:
        if language is None:
            language = self._opts.language

        return STTOptions(
            language=language,
            interim_results=self._opts.interim_results,
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            multilingual=self._opts.multilingual,
            translate=self._opts.translate,
            api_key=self._opts.api_key,
            enable_turn_detection=self._opts.enable_turn_detection,
            turn_detection_timeout=self._opts.turn_detection_timeout,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._base_url = base_url
        self._speaking = False
        self._session_id = str(uuid.uuid4())
        self._is_recording = False
        self._reconnect_event = asyncio.Event()
        self._finalized_segments = set()  # Track segments that were already sent as FINAL
        self._last_pending_segment = None  # Track the last non-finalized segment

    def update_options(
        self,
        *,
        language: str | None = None,
        interim_results: bool | None = None,
        sample_rate: int | None = None,
        multilingual: bool | None = None,
        translate: bool | None = None,
        enable_turn_detection: bool | None = None,
        turn_detection_timeout: float | None = None,
    ):
        if language is not None:
            self._opts.language = language
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if multilingual is not None:
            self._opts.multilingual = multilingual
        if translate is not None:
            self._opts.translate = translate
        if enable_turn_detection is not None:
            self._opts.enable_turn_detection = enable_turn_detection
        if turn_detection_timeout is not None:
            self._opts.turn_detection_timeout = turn_detection_timeout

        self._reconnect_event.set()

    async def _run(self) -> None:
        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            # Wait for server to be ready
            while not self._is_recording:
                await asyncio.sleep(0.1)
                if ws.closed:
                    return

            logger.info("Server ready, starting audio stream")

            # Process audio in chunks - send as raw binary data
            samples_100ms = self._opts.sample_rate // 10  # 100ms chunks
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_100ms,
            )

            try:
                async for data in self._input_ch:
                    if ws.closed:
                        logger.debug("WebSocket closed, stopping send task")
                        break

                    frames: list[rtc.AudioFrame] = []

                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())

                    for frame in frames:
                        if ws.closed:
                            break

                        try:
                            # Convert int16 to float32 audio data like the official client
                            audio_int16 = np.frombuffer(frame.data.tobytes(), dtype=np.int16)
                            audio_float32 = audio_int16.astype(np.float32) / 32768.0

                            # Send raw binary audio data directly
                            await ws.send_bytes(audio_float32.tobytes())

                        except Exception as e:
                            if "closed" in str(e).lower():
                                logger.debug(f"Connection closed while sending audio: {e}")
                                break
                            else:
                                logger.warning(f"Failed to send audio data: {e}")
                                break

            except Exception as e:
                logger.error(f"Error in send_task: {e}")

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            while not ws.closed:
                try:
                    msg = await ws.receive()

                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        logger.warning("Talkscriber connection closed")
                        return

                    if msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        return

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning("unexpected Talkscriber message type %s", msg.type)
                        continue

                    try:
                        self._process_stream_event(json.loads(msg.data))
                    except Exception:
                        logger.exception("failed to process Talkscriber message")

                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.warning(f"Error receiving message: {e}")
                    return

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, pending = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if wait_reconnect_task in done:
                        logger.info("Reconnecting to Talkscriber due to options update")
                        self._reconnect_event.clear()
                        for task in tasks:
                            task.cancel()
                        if ws and not ws.closed:
                            await ws.close()
                        continue

                    # Check if any task completed successfully
                    for task in done:
                        if task.exception():
                            logger.warning(f"Task failed: {task.exception()}")
                            break
                    else:
                        # All tasks completed without exception
                        return

                except APIConnectionError as e:
                    logger.warning("Connection error with Talkscriber: %s", e)
                    await asyncio.sleep(2)
                    continue

                except Exception as e:
                    logger.exception("Unexpected error in Talkscriber stream: %s", e)
                    await asyncio.sleep(2)
                    continue

                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    if ws and not ws.closed:
                        await ws.close()

            except Exception as e:
                logger.exception("Failed to connect to Talkscriber: %s", e)
                await asyncio.sleep(5)
                continue

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Establish WebSocket connection to Talkscriber."""

        try:
            # Connect to WebSocket (no headers needed for auth)
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._base_url,
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(total=30),
                ),
                self._conn_options.timeout,
            )

            logger.info("WebSocket connected to Talkscriber")

            # Send authentication and configuration as JSON message (like official client)
            task = "translate" if self._opts.translate else "transcribe"
            auth_message = {
                "uid": self._session_id,
                "multilingual": self._opts.multilingual,
                "language": self._opts.language,
                "task": task,
                "auth": self._api_key,
                "enable_turn_detection": self._opts.enable_turn_detection,
                "turn_detection_timeout": self._opts.turn_detection_timeout,
            }

            await ws.send_str(json.dumps(auth_message))
            logger.info(f"Sent authentication to Talkscriber: {auth_message}")

            # Wait a moment for initial response to check for immediate errors
            try:
                initial_response = await asyncio.wait_for(ws.receive(), timeout=2.0)
                if initial_response.type == aiohttp.WSMsgType.TEXT:
                    response_data = json.loads(initial_response.data)
                    logger.info(f"Received initial response: {response_data}")

                    # Check for authentication errors
                    if "error" in response_data:
                        error_msg = response_data.get("error", "Authentication failed")
                        raise APIConnectionError(f"Talkscriber authentication error: {error_msg}")

                    # Process the first message
                    self._process_stream_event(response_data)
                elif initial_response.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    raise APIConnectionError(
                        "Talkscriber connection closed immediately after authentication"
                    )
            except asyncio.TimeoutError:
                # No immediate response is fine, server might be processing
                logger.debug("No immediate response from Talkscriber, continuing...")

            return ws

        except Exception as e:
            logger.error(f"Failed to connect to Talkscriber: {e}")
            raise APIConnectionError(f"Failed to connect to Talkscriber: {e}") from e

    def _process_stream_event(self, data: dict) -> None:
        """Process incoming messages from Talkscriber WebSocket."""

        logger.debug(f"Received message from Talkscriber: {data}")

        # Handle authentication errors
        if "error" in data:
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Talkscriber error: {error_msg}")
            raise APIStatusError(message=f"Talkscriber error: {error_msg}")

        # Handle unauthorized access
        if data.get("message") == "UNAUTHORIZED":
            logger.error("Talkscriber authentication failed - invalid API key")
            raise APIStatusError(
                message="Talkscriber authentication failed: Invalid API key. Please check your TALKSCRIBER_API_KEY environment variable."
            )

        # Handle server ready message
        if data.get("message") == "SERVER_READY":
            logger.info("Talkscriber server is ready")
            self._is_recording = True
            return

        # Handle wait status
        if data.get("status") == "WAIT":
            wait_time = data.get("info", 0)
            logger.warning(f"Talkscriber server busy. Estimated wait: {round(wait_time)} minutes.")
            return

        # Handle disconnect message
        if data.get("message") == "DISCONNECT":
            logger.info("Talkscriber server initiated disconnect")
            self._is_recording = False
            return

        # Handle language detection
        if "language" in data:
            detected_language = data.get("detected_language")
            language_confidence = data.get("language_confidence")
            logger.info(
                f"Detected language {detected_language} with confidence {language_confidence}"
            )
            return

        # Handle transcription segments
        if "segments" in data:
            segments = data["segments"]
            if not segments:
                return

            # Talkscriber sends ALL segments (cumulative), process each one
            has_any_finalized = False

            # Check if the last pending segment disappeared (Talkscriber dropped it without finalizing)
            # This happens when turn detection doesn't mark EOS=True quickly enough
            # BUT: Don't finalize immediately if there's still an ongoing (non-EOS) segment,
            # as Talkscriber often revises interim transcriptions
            if self._last_pending_segment:
                last_pending_text = self._last_pending_segment.get("text", "")
                last_pending_id = last_pending_text

                # Check if this segment is still in the current message
                segment_still_present = any(
                    seg.get("text", "") == last_pending_text for seg in segments
                )

                # Check if there's still an ongoing non-finalized segment (might be a revision)
                has_ongoing_segment = any(not seg.get("EOS", False) for seg in segments)

                # Only finalize if:
                # 1. Segment disappeared AND not already finalized
                # 2. AND there's no ongoing segment (if there is, it might be a revision)
                if (
                    not segment_still_present
                    and last_pending_id not in self._finalized_segments
                    and not has_ongoing_segment
                ):
                    # The segment disappeared without being finalized - finalize it now
                    logger.info(f"Finalizing disappeared segment: {last_pending_text[:50]}...")
                    speech_data = [
                        stt.SpeechData(
                            language=self._opts.language or "en",
                            start_time=self._last_pending_segment.get("start", 0),
                            end_time=self._last_pending_segment.get("end", 0),
                            confidence=self._last_pending_segment.get("confidence", 1.0),
                            text=last_pending_text,
                        )
                    ]
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=speech_data,
                    )
                    self._event_ch.send_nowait(final_event)
                    self._finalized_segments.add(last_pending_id)
                    has_any_finalized = True

                    # Send END_OF_SPEECH since the segment was lost
                    if self._speaking:
                        logger.info("End of speech detected - pending segment was dropped")
                        self._speaking = False
                        end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        self._event_ch.send_nowait(end_event)

                    self._last_pending_segment = None

            # Now process all segments in the current message
            current_pending_segment = None

            for segment in segments:
                text = segment.get("text", "")
                if not text:
                    continue

                is_eos = segment.get("EOS", False)
                # Use text as identifier (simple but effective for most cases)
                segment_id = text

                if not self._speaking:
                    self._speaking = True
                    start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    self._event_ch.send_nowait(start_event)

                # Create speech data
                speech_data = [
                    stt.SpeechData(
                        language=self._opts.language or "en",
                        start_time=segment.get("start", 0),
                        end_time=segment.get("end", 0),
                        confidence=segment.get("confidence", 1.0),
                        text=text,
                    )
                ]

                if is_eos and segment_id not in self._finalized_segments:
                    # This segment is complete and not yet sent as FINAL
                    logger.info(f"Sending final transcript for segment (EOS=True): {text[:50]}...")
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=speech_data,
                    )
                    self._event_ch.send_nowait(final_event)
                    self._finalized_segments.add(segment_id)
                    has_any_finalized = True
                elif not is_eos:
                    # Send as INTERIM transcript for partial results
                    if self._opts.interim_results:
                        interim_event = stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=speech_data,
                        )
                        self._event_ch.send_nowait(interim_event)
                    # Track this as a pending segment
                    current_pending_segment = segment

            # Update the last pending segment
            self._last_pending_segment = current_pending_segment

            # Send END_OF_SPEECH when the last segment has EOS and we finalized something
            last_segment = segments[-1] if segments else None
            if last_segment and last_segment.get("EOS", False) and has_any_finalized:
                logger.info("End of speech detected - all segments finalized")
                if self._speaking:
                    self._speaking = False
                    end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    self._event_ch.send_nowait(end_event)
                self._last_pending_segment = None  # Clear pending segment
                # NOTE: Do NOT clear _finalized_segments here!
                # Talkscriber sends cumulative segments across multiple messages,
                # so we need to keep tracking finalized segments throughout the entire session
                # to prevent sending duplicates in subsequent utterances.
