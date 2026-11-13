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
import os
import weakref
from dataclasses import dataclass
from typing import Any, Literal

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    http,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.rtc import AudioFrame

from .log import logger

STTEncoding = Literal["pcm_s16le"]

# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
}


SUPPORTED_SAMPLE_RATE = 24000


@dataclass
class STTOptions:
    sample_rate: int = SUPPORTED_SAMPLE_RATE
    buffer_size_seconds: float = 0.08
    encoding: str = "pcm_s16le"
    temperature: float | None = None
    language: LanguageCode = LanguageCode("en")
    vad_threshold: float = 0.6
    vad_bucket: int | None = 2
    # When set, we flush the stt state on the first time the VAD triggers
    # in order to recover the text currently being processed as soon as possible.
    vad_flush: bool = True


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_name: str = "default",
        sample_rate: int = SUPPORTED_SAMPLE_RATE,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        buffer_size_seconds: float = 0.08,
        http_session: http.HttpSession | None = None,
        vad_threshold: float = 0.9,
        vad_bucket: int | None = 2,
        vad_flush: bool = True,
        temperature: float | None = None,
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,  # only final transcripts
                aligned_transcript=False,  # only chunk start times are available
                offline_recognize=False,
            ),
        )

        api_key = api_key or os.environ.get("GRADIUM_API_KEY")

        if sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ValueError(f"Only {SUPPORTED_SAMPLE_RATE}Hz sample rate is supported")

        if not api_key:
            raise ValueError(
                "Gradium API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `GRADIUM_API_KEY` environment variable"
            )

        self._api_key = api_key

        model_endpoint = (
            model_endpoint
            or os.environ.get("GRADIUM_MODEL_ENDPOINT")
            or "wss://api.gradium.ai/api/speech/asr"
        )

        self._model_endpoint = model_endpoint
        self._model_name = model_name

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            vad_threshold=vad_threshold,
            vad_bucket=vad_bucket,
            vad_flush=vad_flush,
            temperature=temperature,
            language=LanguageCode(language),
        )

        if is_given(encoding):
            self._opts.encoding = encoding

        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Gradium"

    @property
    def session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
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
        if is_given(language):
            config.language = LanguageCode(language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            model_endpoint=self._model_endpoint,
            model_name=self._model_name,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        for stream in self._streams:
            stream.update_options(
                buffer_size_seconds=buffer_size_seconds,
            )


class SpeechStream(stt.RecognizeStream):
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
        model_name: str,
        http_session: http.HttpSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name
        self._session = http_session

        self._reconnect_event = asyncio.Event()
        self._ready_msg: dict[str, Any] | None = None

    @property
    def delay_in_tokens(self) -> int:
        if self._ready_msg is not None:
            return int(self._ready_msg.get("delay_in_tokens", 6))
        return 6

    @property
    def frame_size(self) -> int:
        if self._ready_msg is not None:
            return int(self._ready_msg.get("frame_size", 1920))
        return 1920

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        self._reconnect_event.set()

    async def _run(self) -> None:
        """
        Run a single websocket connection to Gradium and make sure to reconnect
        when something went wrong.
        """

        closing_ws = False

        async def send_task(ws: http.WebSocket) -> None:
            nonlocal closing_ws
            samples_per_buffer = 1920

            audio_bstream = utils.media.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            async for data in self.input:
                if isinstance(data, self.Flush):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.push(data.data.tobytes())

                for frame in frames:
                    if len(frame.data) % 2 != 0:
                        logger.warning("Frame data size not aligned to int16 (multiple of 2)")

                    audio_data = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                    audio_msg = {
                        "type": "audio",
                        "audio": audio_data,
                    }
                    await ws.send_str(json.dumps(audio_msg))

            closing_ws = True
            await ws.send_str(json.dumps({"type": "end_of_stream"}))

        async def recv_task(ws: http.WebSocket) -> None:
            nonlocal closing_ws
            buffered_text: list[str] = []
            speaking = False
            remaining_vad_steps: int | None = None
            while True:
                try:
                    msg = await ws.receive(timeout=5)
                except APITimeoutError:
                    if closing_ws:
                        break
                    continue

                if msg.type is http.WebSocketMessageType.CLOSE:
                    if closing_ws:
                        return
                    raise ws.close_error("Gradium connection closed unexpectedly")

                if msg.type is not http.WebSocketMessageType.TEXT:
                    logger.error("Unexpected Gradium message type: %s", msg.type)
                    continue

                data = json.loads(msg.data)

                type_ = data.get("type", "")

                if type_ == "text":
                    if speaking is False:
                        speaking = True
                        start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                        self.send(start_event)
                    buffered_text.append(data["text"])
                    event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                text=data["text"],
                                language=self._opts.language,
                                start_time=data["start_s"] + self.start_time_offset,
                            )
                        ],
                    )
                    self.send(event)

                elif type_ == "step":
                    if not speaking:
                        continue
                    if vad_bucket := self._opts.vad_bucket:
                        positive_vad = (
                            data["vad"][vad_bucket]["inactivity_prob"] > self._opts.vad_threshold
                        )
                        if positive_vad:
                            if remaining_vad_steps is None:
                                remaining_vad_steps = self.delay_in_tokens
                                if self._opts.vad_flush:
                                    samples_per_channel = self.frame_size * self.delay_in_tokens
                                    zeros = AudioFrame.create(
                                        sample_rate=self._opts.sample_rate,
                                        num_channels=1,
                                        samples_per_channel=samples_per_channel,
                                    )
                                    await self.input.send(zeros)
                            else:
                                remaining_vad_steps -= 1
                                if remaining_vad_steps <= 0:
                                    speaking = False
                                    remaining_vad_steps = None
                                    event = stt.SpeechEvent(
                                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                        alternatives=[
                                            stt.SpeechData(
                                                text=" ".join(buffered_text),
                                                language=self._opts.language,
                                            )
                                        ],
                                    )
                                    self.send(event)

                                    buffered_text = []
                                    self.send(
                                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                                    )
                        else:
                            remaining_vad_steps = None

                elif type_ == "ready":
                    self._ready_msg = data
                elif type_ == "end_text":
                    # This message provides the end timestamp of the previous word in the stop_s field.
                    pass
                elif type_ == "end_of_stream":
                    return
                elif type_ == "error":
                    # the code is an HTTP status: 429 is retried, a rejected request is not
                    raise APIStatusError(
                        "Gradium error",
                        status_code=data.get("code", -1),
                        body=data,
                        detail=data.get("message"),
                    )
                else:
                    logger.warning(f"Unknown message type from Gradium {type_}")

        ws: http.WebSocket | None = None

        while True:
            try:
                ws = await self._connect_ws()
                async with utils.aio.TaskSet() as tasks:
                    io = asyncio.gather(tasks.spawn(send_task(ws)), tasks.spawn(recv_task(ws)))
                    reconnect = tasks.spawn(self._reconnect_event.wait())
                    if await tasks.first_completed(io, reconnect) is not reconnect:
                        break

                    self._reconnect_event.clear()
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> http.WebSocket:
        headers = {"x-api-key": self._api_key, "x-api-source": "livekit"}

        ws = await self._session.ws_connect(
            self._model_endpoint, headers=headers, conn_options=self._conn_options
        )

        # Build and send the setup payload as the first message
        setup_msg: dict[str, Any] = {
            "type": "setup",
            "model_name": self._model_name,
            "input_format": "pcm",
        }
        json_config: dict[str, Any] = {"language": self._opts.language.language}
        if self._opts.temperature is not None:
            json_config["temp"] = self._opts.temperature
        setup_msg["json_config"] = json_config

        await ws.send_str(json.dumps(setup_msg))
        return ws
