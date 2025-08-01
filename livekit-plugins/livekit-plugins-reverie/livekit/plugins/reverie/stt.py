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
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

BASE_URL = "wss://revapi.reverieinc.com/stream"


@dataclass
class STTOptions:
    language: str
    domain: str
    sample_rate: int
    num_channels: int
    continuous: bool
    timeout: float
    silence: float
    format: str
    logging: str
    punctuate: bool


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "hi_en",
        domain: str = "generic",
        sample_rate: int = 16000,
        continuous: bool = True,
        timeout: float = 15.0,
        silence: float = 0.5,
        format: str = "16k_int16",
        logging: str = "true",
        punctuate: bool = False,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        """Create a new instance of Reverie STT.

        Args:
            language: The language code for recognition (e.g., "hi_en" for Hindi-English). Defaults to "hi_en".
            domain: The domain context for transcription (e.g., "generic"). Defaults to "generic".
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            continuous: Whether to use continuous mode for streaming. Defaults to True.
            timeout: Connection timeout in seconds (max 180). Defaults to 15.0.
            silence: Auto-disconnect after silence in seconds (max 30). Defaults to 0.5.
            format: Audio format specification. Defaults to "16k_int16".
            logging: Logging mode - "true", "no_audio", "no_transcript", or "false". Defaults to "true".
            punctuate: Whether to add punctuation and capitalization. Defaults to False.
            api_key: Your Reverie API key. If not provided, will look for REVERIE_API_KEY environment variable.
            app_id: Your Reverie App ID. If not provided, will look for REVERIE_APP_ID environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for Reverie API. Defaults to "wss://revapi.reverieinc.com/stream".

        Raises:
            ValueError: If no API key or App ID is provided or found in environment variables.

        Note:
            The api_key and app_id must be set either through the constructor arguments or by setting
            the REVERIE_API_KEY and REVERIE_APP_ID environmental variables.
            Timeout must be between 0 and 180 seconds.
            Silence must be between 0 and 30 seconds.
            Punctuation is only supported for English and Hindi languages.
        """

        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._base_url = base_url

        reverie_api_key = api_key if is_given(api_key) else os.environ.get("REVERIE_API_KEY")
        if not reverie_api_key:
            raise ValueError("Reverie API key is required")
        self._api_key = reverie_api_key

        reverie_app_id = app_id if is_given(app_id) else os.environ.get("REVERIE_APP_ID")
        if not reverie_app_id:
            raise ValueError("Reverie App ID is required")
        self._app_id = reverie_app_id

        # Validate parameters
        if not 0 <= timeout <= 180:
            raise ValueError("Timeout must be between 0 and 180 seconds")
        if not 0 <= silence <= 30:
            raise ValueError("Silence must be between 0 and 30 seconds")
        if logging not in ("true", "no_audio", "no_transcript", "false"):
            raise ValueError("Logging must be one of: 'true', 'no_audio', 'no_transcript', 'false'")

        self._opts = STTOptions(
            language=language,
            domain=domain,
            sample_rate=sample_rate,
            num_channels=1,
            continuous=continuous,
            timeout=timeout,
            silence=silence,
            format=format,
            logging=logging,
            punctuate=punctuate,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

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
        raise NotImplementedError("Azure STT does not support single frame recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            app_id=self._app_id,
            http_session=self._ensure_session(),
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        config = STTOptions(
            language=language if is_given(language) else self._opts.language,
            domain=self._opts.domain,
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            continuous=self._opts.continuous,
            timeout=self._opts.timeout,
            silence=self._opts.silence,
            format=self._opts.format,
            logging=self._opts.logging,
            punctuate=self._opts.punctuate,
        )
        return config


class SpeechStream(stt.SpeechStream):
    _EOF_MSG: str = "--EOF--"

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        app_id: str,
        http_session: aiohttp.ClientSession,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._app_id = app_id
        self._session = http_session
        self._base_url = base_url
        self._speaking = False

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # Send audio to Reverie in chunks
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    try:
                        await ws.send_bytes(frame.data.tobytes())
                    except (aiohttp.ClientConnectionError, ConnectionResetError) as e:
                        if closing_ws:
                            return
                        logger.error("error sending audio data: %s", e)
                        raise APIConnectionError("failed to send audio data") from e

                    if has_ended:
                        try:
                            await ws.send_str(SpeechStream._EOF_MSG)
                        except (aiohttp.ClientConnectionError, ConnectionResetError):
                            # Connection might be closed, that's ok for EOF
                            pass
                        has_ended = False

            # Tell Reverie we are done sending audio
            closing_ws = True
            try:
                await ws.send_str(SpeechStream._EOF_MSG)
            except (aiohttp.ClientConnectionError, ConnectionResetError):
                # Connection might be closed, that's ok for final EOF
                pass

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                try:
                    msg = await ws.receive()
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    if closing_ws or self._session.closed:
                        return
                    logger.error("error receiving message from reverie: %s", e)
                    raise APIConnectionError("failed to receive message from reverie") from e

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return

                    raise APIStatusError(message="reverie connection closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"websocket error: {ws.exception()}")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected reverie message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process reverie message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        try:
            ws = await self._connect_ws()
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                # Clean up tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
        finally:
            if ws is not None and not ws.closed:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        params = {
            "apikey": self._api_key,
            "appid": self._app_id,
            "appname": "stt_stream",
            "src_lang": self._opts.language,
            "domain": self._opts.domain,
            "continuous": "true" if self._opts.continuous else "false",
            "timeout": str(self._opts.timeout),
            "silence": str(self._opts.silence),
            "format": self._opts.format,
            "logging": self._opts.logging,
            "punctuate": "true" if self._opts.punctuate else "false",
        }

        url = f"{self._base_url}?{urlencode(params)}"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(url),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to reverie") from e
        return ws

    def _process_stream_event(self, data: dict) -> None:
        if not data.get("success", False):
            logger.warning("received error from reverie: %s", data)
            return

        text = data.get("text", "").strip()
        confidence = data.get("confidence", 0.0)
        request_id = data.get("id", "")
        cause = data.get("cause", "")

        if cause == "ready":
            # Connection established, no transcription yet
            return

        if text:
            # We have some transcription
            if not self._speaking:
                self._speaking = True
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)

            alternatives = [
                stt.SpeechData(
                    language=self._opts.language,
                    start_time=0,  # Reverie doesn't provide timing info
                    end_time=0,
                    confidence=confidence,
                    text=text,
                )
            ]

            # Only consider it final when cause="silence detected"
            if cause == "silence detected":
                final_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(final_event)

                # End of speech after silence detection
                if self._speaking:
                    self._speaking = False
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
            else:
                # All other cases are interim results
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)

        # Handle EOF without text (connection ending)
        elif cause == "EOF received" and self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
