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
import json
import os
from collections import deque
from dataclasses import dataclass, replace
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never

from pydantic import Field, JsonValue

from livekit.agents import APIConnectOptions, APIStatusError, http, tokenize, tts
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

SUPPORTED_SAMPLE_RATE = 48000


@dataclass
class _TTSOptions:
    voice: str | None
    voice_id: str | None
    pronunciation_id: str | None
    json_config: dict[str, JsonValue] | None = None


# Gradium's websocket: each segment is one request on the stream's socket, named by its
# `client_req_id`. Each message declares only the fields read.


class _Audio(TypedDict):
    type: Literal["audio"]
    audio: str
    client_req_id: NotRequired[str]


class _EndOfStream(TypedDict):
    type: Literal["end_of_stream"]
    client_req_id: NotRequired[str]


class _Error(TypedDict):
    type: Literal["error"]
    message: NotRequired[str]
    code: NotRequired[int]


_Message = Annotated[_Audio | _EndOfStream | _Error, Field(discriminator="type")]


class _Setup(TypedDict):
    type: Literal["setup"]
    model_name: str
    output_format: Literal["pcm"]
    close_ws_on_eos: bool
    client_req_id: str
    voice: NotRequired[str]
    voice_id: NotRequired[str]
    pronunciation_id: NotRequired[str]
    json_config: NotRequired[str]


class _Text(TypedDict):
    type: Literal["text"]
    text: str
    client_req_id: str


class _End(TypedDict):
    type: Literal["end_of_stream"]
    client_req_id: str


_Command = _Setup | _Text | _End


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_name: str = "default",
        voice: str | None = None,
        voice_id: str | None = "4SZHfMpw-p46Ywgs",
        pronunciation_id: str | None = None,
        json_config: dict[str, JsonValue] | None = None,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Initialize the Gradium TTS.

        Args:
            api_key (str): Gradium API key, or `GRADIUM_API_KEY` env var.
            model_endpoint (str): Gradium model endpoint, or `GRADIUM_MODEL_ENDPOINT` env var.
            model_name (str): Model name.
            voice (str): Speaker voice.
            voice_id (str): Speaker voice ID.
            pronunciation_id (str): Optional pronunciation ID for controlling TTS pronunciation.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, text_unit="words"),
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("GRADIUM_API_KEY")

        if not api_key:
            raise ValueError(
                "Gradium API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `GRADIUM_API_KEY` environment variable"
            )

        model_endpoint = (
            model_endpoint
            or os.environ.get("GRADIUM_MODEL_ENDPOINT")
            or "wss://api.gradium.ai/api/speech/tts"
        )

        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name

        self._opts = _TTSOptions(
            voice=voice,
            voice_id=voice_id,
            pronunciation_id=pronunciation_id,
            json_config=json_config,
        )

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Gradium"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        json_config: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(voice):
                self._opts.voice = voice
            if is_given(voice_id):
                self._opts.voice_id = voice_id
            if is_given(json_config):
                self._opts.json_config = json_config

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        return SynthesizeStream(tts=self, opts=replace(self._opts), conn_options=conn_options)

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    """Stream-based text-to-speech synthesis using Gradium WebSocket API."""

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._gradium = tts
        self._opts = opts
        self._open = deque[str]()

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        self._open.clear()
        return tts.WebSocketTarget(
            self._gradium._model_endpoint.replace("http", "ws", 1),
            headers={"x-api-key": self._gradium._api_key, "x-api-source": "livekit"},
        )

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._open.append(item.segment_id)
                return self._setup(item.segment_id)
            case tts.SegmentText():
                return _Text(type="text", text=f"{item.chunk.text} ", client_req_id=item.segment_id)
            case tts.SegmentEnd():
                return _End(type="end_of_stream", client_req_id=item.segment_id)
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Audio | _EndOfStream | _Error) -> None:
        if message["type"] == "error":
            raise APIStatusError(
                "Gradium TTS returned an error",
                status_code=message.get("code") or 500,
                body=message,
                detail=message.get("message"),
            )
        # a request on its own may go unnamed: it is the oldest one open
        segment_id = message.get("client_req_id") or (self._open[0] if self._open else "")
        if message["type"] == "audio":
            self.report(tts.SegmentAudio(segment_id, base64.b64decode(message["audio"])))
        elif message["type"] == "end_of_stream":
            if segment_id in self._open:
                self._open.remove(segment_id)
            self.report(tts.SegmentDone(segment_id))
        else:
            assert_never(message)

    def _setup(self, segment_id: str) -> _Setup:
        setup = _Setup(
            type="setup",
            model_name=self._gradium._model_name,
            output_format="pcm",
            close_ws_on_eos=False,
            client_req_id=segment_id,
        )
        # an explicit voice replaces the default voice id
        if self._opts.voice is not None:
            setup["voice"] = self._opts.voice
        elif self._opts.voice_id is not None:
            setup["voice_id"] = self._opts.voice_id
        if self._opts.pronunciation_id is not None:
            setup["pronunciation_id"] = self._opts.pronunciation_id
        if self._opts.json_config is not None:
            setup["json_config"] = json.dumps(self._opts.json_config)
        return setup
