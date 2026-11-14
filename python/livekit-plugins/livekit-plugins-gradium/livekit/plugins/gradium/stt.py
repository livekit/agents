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

import base64
import dataclasses
import os
import weakref
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import Field

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    http,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

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


class _Ready(TypedDict):
    type: Literal["ready"]
    frame_size: NotRequired[int]
    delay_in_tokens: NotRequired[int]


class _Text(TypedDict):
    type: Literal["text"]
    text: str
    start_s: float


class _VadBucket(TypedDict):
    inactivity_prob: float


class _Step(TypedDict):
    type: Literal["step"]
    vad: list[_VadBucket]


class _EndOfStream(TypedDict):
    type: Literal["end_of_stream"]


class _Error(TypedDict):
    type: Literal["error"]
    code: NotRequired[int]
    message: NotRequired[str]


_Message = Annotated[_Ready | _Text | _Step | _EndOfStream | _Error, Field(discriminator="type")]


class _Setup(TypedDict):
    type: Literal["setup"]
    model_name: str
    input_format: Literal["pcm"]
    json_config: dict[str, Any]


class _Audio(TypedDict):
    type: Literal["audio"]
    audio: str


class _Command(TypedDict):
    type: Literal["end_of_stream"]


class SpeechStream(stt.SocketStream[_Message, _Setup | _Audio | _Command]):
    goodbye = _Command(type="end_of_stream")
    frame_ms = 80
    # the plugin ends the speech from the inactivity the provider's steps report
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        model_endpoint: str,
        model_name: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name
        self._ready_msg: _Ready | None = None
        self._buffered_text: list[str] = []
        self._remaining_vad_steps: int | None = None

    @property
    def delay_in_tokens(self) -> int:
        if self._ready_msg is not None:
            return self._ready_msg.get("delay_in_tokens", 6)
        return 6

    @property
    def frame_size(self) -> int:
        if self._ready_msg is not None:
            return self._ready_msg.get("frame_size", 1920)
        return 1920

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        self.reconnect("options changed")

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        self._buffered_text = []
        self._remaining_vad_steps = None
        return http.SocketTarget(
            self._model_endpoint, headers={"x-api-key": self._api_key, "x-api-source": "livekit"}
        )

    def setup(self) -> http.Encoded[_Setup | _Audio | _Command]:
        json_config: dict[str, Any] = {"language": self._opts.language.language}
        if self._opts.temperature is not None:
            json_config["temp"] = self._opts.temperature
        return _Setup(
            type="setup", model_name=self._model_name, input_format="pcm", json_config=json_config
        )

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[_Setup | _Audio | _Command]:
        return self._audio(bytes(item.data)) if isinstance(item, rtc.AudioFrame) else None

    @staticmethod
    def _audio(pcm: bytes) -> _Audio:
        return _Audio(type="audio", audio=base64.b64encode(pcm).decode("utf-8"))

    def on_message(self, message: _Message) -> stt.Report | list[stt.Report] | http.Done | None:
        if message["type"] == "ready":
            self._ready_msg = message
            return None
        if message["type"] == "end_of_stream":
            return http.DONE
        if message["type"] == "error":
            # the code is an HTTP status: 429 is retried, a rejected request is not
            raise APIStatusError(
                "Gradium error",
                status_code=message.get("code", -1),
                body=dict(message),
                detail=message.get("message"),
            )
        if message["type"] == "text":
            self._buffered_text.append(message["text"])
            return stt.Transcript(
                message["text"],
                self._opts.language,
                start_time=message["start_s"] + self.start_time_offset,
            )
        if not self.speaking or not (vad_bucket := self._opts.vad_bucket):
            return None
        if message["vad"][vad_bucket]["inactivity_prob"] <= self._opts.vad_threshold:
            self._remaining_vad_steps = None
            return None
        if self._remaining_vad_steps is None:
            self._remaining_vad_steps = self.delay_in_tokens
            if self._opts.vad_flush:
                # silence pushes out the text the model still holds
                silence = bytes(2 * self.frame_size * self.delay_in_tokens)
                self.send_command(self._audio(silence))
            return None
        self._remaining_vad_steps -= 1
        if self._remaining_vad_steps > 0:
            return None
        self._remaining_vad_steps = None
        text, self._buffered_text = " ".join(self._buffered_text), []
        return [stt.Transcript(text, self._opts.language, final=True), stt.SpeechEnded()]
