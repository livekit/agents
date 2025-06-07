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
import dataclasses
import json
import os
import ssl
import weakref
from dataclasses import dataclass
from typing import Literal

import aiohttp
import numpy as np

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEvent
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

ENGLISH = "en"
DEFAULT_ENCODING = "pcm_s16le"

# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
    "pcm_mulaw": 1,
}

ssl_context = ssl._create_unverified_context()

@dataclass
class STTOptions:
    sample_rate: int
    buffer_size_seconds: float = 0.032
    encoding: NotGivenOr[Literal["pcm_s16le", "pcm_mulaw"]] = NOT_GIVEN

    # Optional metadata fields specific to Baseten
    vad_threshold: float = 0.5
    vad_min_silence_duration_ms: int = 300
    vad_speech_pad_ms: int = 30
    whisper_audio_language: str = "en"

    def __post_init__(self):
        if self.encoding not in (NOT_GIVEN, "pcm_s16le", "pcm_mulaw"):
            raise ValueError(f"Invalid encoding: {self.encoding}")


class STT(stt.STT):
    def __init__(
    self,
    *,
    api_key: NotGivenOr[str] = NOT_GIVEN,
    model_endpoint: NotGivenOr[str] = NOT_GIVEN,
    sample_rate: int = 16000,
    encoding: NotGivenOr[Literal["pcm_s16le", "pcm_mulaw"]] = NOT_GIVEN,
    buffer_size_seconds: float = 0.032,
    vad_threshold: float = 0.5,
    vad_min_silence_duration_ms: int = 300,
    vad_speech_pad_ms: int = 30,
    whisper_audio_language: str = "en",
    http_session: aiohttp.ClientSession | None = None,
):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,  # only final transcripts
            ),
        )
        self._api_key = api_key if is_given(api_key) else os.environ.get("BASETEN_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Baseten API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `BASETEN_API_KEY` environment variable"
            )

        if not model_endpoint:
            raise ValueError(
                "The model endpoint is required, you can find it in the Baseten dashboard"
            )

        self._model_endpoint = model_endpoint

        self._opts = STTOptions(
            sample_rate=sample_rate,
            encoding=encoding,
            buffer_size_seconds=buffer_size_seconds,
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            whisper_audio_language=whisper_audio_language,
        )
        self._session = http_session
        self._streams = weakref.WeakSet()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session


    async def _recognize_impl(
    self,
    buffer: AudioBuffer,
    *,
    language: NotGivenOr[str] = NOT_GIVEN,
    conn_options: APIConnectOptions,
) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            model_endpoint=self._model_endpoint,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    def update_options(
    self,
    *,
    vad_threshold: NotGivenOr[float] = NOT_GIVEN,
    vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
    vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
    whisper_audio_language: NotGivenOr[str] = NOT_GIVEN,
    buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
):
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(whisper_audio_language):
            self._opts.whisper_audio_language = whisper_audio_language
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        for stream in self._streams:
            stream.update_options(
                vad_threshold=vad_threshold,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                whisper_audio_language=whisper_audio_language,
                buffer_size_seconds=buffer_size_seconds,
            )


class SpeechStream(stt.SpeechStream):
    # Used to close websocket
    _CLOSE_MSG: str = json.dumps({"terminate_session": True})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        model_endpoint: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._session = http_session
        self._speech_duration: float = 0

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: list[SpeechEvent] = []
        self._reconnect_event = asyncio.Event()

    def update_options(
    self,
    *,
    vad_threshold: NotGivenOr[float] = NOT_GIVEN,
    vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
    vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
    whisper_audio_language: NotGivenOr[str] = NOT_GIVEN,
    buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
):
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(whisper_audio_language):
            self._opts.whisper_audio_language = whisper_audio_language
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        self._reconnect_event.set()


    async def _run(self) -> None:
        """
        Run a single websocket connection to Baseten and make sure to reconnect
        when something went wrong.
        """

        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            samples_per_buffer = 512

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    if len(frame.data) % 2 != 0:
                        logger.warning("Frame data size not aligned to float32 (multiple of 4)")

                    int16_array = np.frombuffer(frame.data, dtype=np.int16)
                    await ws.send_bytes(int16_array.tobytes())


        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    if closing_ws:
                        break
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return
                    raise APIStatusError("Baseten connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("Unexpected Baseten message type: %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("message_type")

                    if msg_type == "partial_transcript":
                        text = data.get("transcript", "")
                        confidence = data.get("confidence", 0.0)
                        segments = data.get("segments", [])

                        if text:
                            start_time = segments[0].get("start", 0.0) if segments else 0.0
                            end_time = segments[-1].get("end", 0.0) if segments else 0.0

                            event = stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        language="",
                                        text=text,
                                        confidence=confidence,
                                        start_time=start_time,
                                        end_time=end_time,
                                    )
                                ]
                            )
                            self._event_ch.send_nowait(event)

                    elif msg_type == "final_transcript":
                        text = data.get("transcript", "")
                        confidence = data.get("confidence", 0.0)
                        segments = data.get("segments", [])
                        language = data.get("language", "")

                        if text:
                            start_time=segments[0].get("start", 0.0) if segments else 0.0
                            end_time=segments[-1].get("end", 0.0) if segments else 0.0

                            event = stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        language=language,
                                        text=text,
                                        confidence=confidence,
                                        start_time=start_time,
                                        end_time=end_time,
                                    )
                                ]
                            )
                            self._final_events.append(event)
                            self._event_ch.send_nowait(event)

                    else:
                        logger.warning("Unknown message type from Baseten: %s", msg_type)

                except Exception:
                    logger.exception("Failed to process message from Baseten")


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
                    done, _ = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )  # type: ignore
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        headers = {
            "Authorization": f"Api-Key {self._api_key}",
        }

        ws = await self._session.ws_connect(self._model_endpoint, headers=headers, ssl=ssl_context)

        # Build and send the metadata payload as the first message
        metadata = {
            "vad_params": {
                "threshold": self._opts.vad_threshold,
                "min_silence_duration_ms": self._opts.vad_min_silence_duration_ms,
                "speech_pad_ms": self._opts.vad_speech_pad_ms,
            },
            "streaming_whisper_params": {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "enable_partial_transcripts": False
            }
        }

        await ws.send_str(json.dumps(metadata))
        return ws
