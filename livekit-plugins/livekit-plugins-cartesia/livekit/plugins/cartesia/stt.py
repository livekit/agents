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
import json
import os
import uuid
import weakref
from dataclasses import dataclass
from enum import Enum

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import STTEncoding, STTLanguages, STTModels

API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "Cartesia-Version"
API_VERSION = "2025-04-16"

# Audio energy threshold for speech detection
MAGIC_NUMBER_THRESHOLD = 0.004**2


class AudioEnergyFilter:
    """Local voice activity detection based on audio energy levels."""

    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(self, *, min_silence: float = 1.5, rms_threshold: float = MAGIC_NUMBER_THRESHOLD):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> State:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                # keep speaking during cooldown
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING

        return self._state


@dataclass
class STTOptions:
    model: STTModels | str
    language: STTLanguages | str | None
    encoding: STTEncoding
    sample_rate: int
    api_key: str
    base_url: str
    energy_filter: AudioEnergyFilter | bool

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        # If base_url already has a protocol, replace it, otherwise add wss://
        if self.base_url.startswith(("http://", "https://")):
            return f"{self.base_url.replace('http', 'ws', 1)}{path}"
        else:
            return f"wss://{self.base_url}{path}"


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "ink-whisper",
        language: STTLanguages | str = "en",
        encoding: STTEncoding = "pcm_s16le",
        sample_rate: int = 16000,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "https://api.cartesia.ai",
        energy_filter: AudioEnergyFilter | bool = False,
    ) -> None:
        """
        Create a new instance of Cartesia STT.

        Args:
            model: The Cartesia STT model to use. Defaults to "ink-whisper".
            language: The language code for recognition. Defaults to "en".
            encoding: The audio encoding format. Defaults to "pcm_s16le".
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            api_key: The Cartesia API key. If not provided, it will be read from
                the CARTESIA_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for the Cartesia API.
                Defaults to "https://api.cartesia.ai".
            energy_filter: The energy filter to use for local voice activity
                detection. Defaults to False.

        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))

        cartesia_api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY must be set")

        self._opts = STTOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            api_key=cartesia_api_key,
            base_url=base_url,
            energy_filter=AudioEnergyFilter() if energy_filter is True else energy_filter,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Create a streaming transcription session."""
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            opts=config,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Update STT configuration options."""
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language

        # Update all active streams
        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
            )

    def _sanitize_options(
        self, *, language: NotGivenOr[STTLanguages | str] = NOT_GIVEN
    ) -> STTOptions:
        """Create a sanitized copy of options with language override if provided."""
        config = STTOptions(
            model=self._opts.model,
            language=self._opts.language,
            encoding=self._opts.encoding,
            sample_rate=self._opts.sample_rate,
            api_key=self._opts.api_key,
            base_url=self._opts.base_url,
            energy_filter=self._opts.energy_filter,
        )

        if is_given(language):
            config.language = language

        return config


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = stt._ensure_session()
        self._request_id = str(uuid.uuid4())
        self._reconnect_event = asyncio.Event()
        self._speaking = False

        # Set up audio energy filter for local VAD
        self._audio_energy_filter: AudioEnergyFilter | None = None
        if opts.energy_filter:
            if isinstance(opts.energy_filter, AudioEnergyFilter):
                self._audio_energy_filter = opts.energy_filter
            else:
                self._audio_energy_filter = AudioEnergyFilter()

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Update streaming transcription options."""
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language

        self._reconnect_event.set()

    def _check_energy_state(self, frame: rtc.AudioFrame) -> AudioEnergyFilter.State:
        """Check the energy state of an audio frame for voice activity detection."""
        if self._audio_energy_filter:
            return self._audio_energy_filter.update(frame)
        return AudioEnergyFilter.State.SPEAKING

    async def _run(self) -> None:
        """Main loop for streaming transcription."""
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

            # Forward audio to Cartesia in chunks
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            last_frame: rtc.AudioFrame | None = None
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    state = self._check_energy_state(data)
                    if state in (
                        AudioEnergyFilter.State.START,
                        AudioEnergyFilter.State.SPEAKING,
                    ):
                        # Send buffered silence frame if we have one
                        if last_frame:
                            frames.extend(audio_bstream.write(last_frame.data.tobytes()))
                            last_frame = None
                        frames.extend(audio_bstream.write(data.data.tobytes()))

                        # Emit START_OF_SPEECH event if we just started speaking
                        if state == AudioEnergyFilter.State.START and not self._speaking:
                            self._speaking = True
                            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                            self._event_ch.send_nowait(start_event)

                    elif state == AudioEnergyFilter.State.END:
                        # Flush remaining audio and mark as ended
                        frames.extend(audio_bstream.flush())
                        has_ended = True
                    elif state == AudioEnergyFilter.State.SILENCE:
                        # Buffer the last silence frame in case it contains speech beginning
                        last_frame = data
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    await ws.send_bytes(frame.data.tobytes())

                if has_ended:
                    has_ended = False

            closing_ws = True
            await ws.send_str("finalize")

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(message="Cartesia STT connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Cartesia STT message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process Cartesia STT message")

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
                    await tasks_group
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to the Cartesia STT WebSocket."""
        params = {
            "model": self._opts.model,
            "sample_rate": str(self._opts.sample_rate),
            "encoding": self._opts.encoding,
            "cartesia_version": API_VERSION,
            "api_key": self._opts.api_key,
        }

        if self._opts.language:
            params["language"] = self._opts.language

        # Build URL
        url = self._opts.get_ws_url("/stt/websocket")
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        ws_url = f"{url}?{query_string}"

        ws = await asyncio.wait_for(
            self._session.ws_connect(ws_url),
            self._conn_options.timeout,
        )
        return ws

    def _process_stream_event(self, data: dict) -> None:
        """Process incoming WebSocket messages."""
        message_type = data.get("type")

        if message_type == "transcript":
            request_id = data.get("request_id", self._request_id)
            text = data.get("text", "")
            is_final = data.get("is_final", False)
            language = data.get("language", self._opts.language or "en")

            if not text and not is_final:
                return

            speech_data = stt.SpeechData(
                language=language,
                start_time=0,  # Cartesia doesn't provide word-level timestamps in this version
                end_time=data.get("duration", 0),
                confidence=data.get("probability", 1.0),
                text=text,
            )

            if is_final:
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(event)

                if self._speaking:
                    self._speaking = False
                    end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    self._event_ch.send_nowait(end_event)
            else:
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(event)

        elif message_type == "flush_done":
            logger.debug("Received flush_done acknowledgment from Cartesia STT")

        elif message_type == "done":
            logger.debug("Received done acknowledgment from Cartesia STT - session closing")

        elif message_type == "error":
            error_msg = data.get("message", "Unknown error")
            logger.error("Cartesia STT error: %s", error_msg)
            # We could emit an error event here if needed
        else:
            logger.warning("received unexpected message from Cartesia STT: %s", data)
