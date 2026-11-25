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
import binascii
import os
import weakref
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

from livekit.agents import APIConnectOptions, APIError, http, tokenize, tts
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .constants import API_AUTH_HEADER, API_VERSION, API_VERSION_HEADER
from .models import TTSDefaultVoiceId, TTSEncoding, TTSModels


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice: str
    api_key: str
    language: str | None
    base_url: str


# Async's websocket: each segment is one context, ended by closeContext. No message carries a
# tag, so one declares every field read: audio and finals name their context, errors none.


class _Message(TypedDict, total=False):
    context_id: str
    audio: str
    final: bool
    error_code: str
    message: str


class _Voice(TypedDict):
    mode: Literal["id"]
    id: str


class _OutputFormat(TypedDict):
    container: Literal["raw"]
    encoding: str
    sample_rate: int


class _Init(TypedDict):
    model_id: str
    voice: _Voice
    output_format: _OutputFormat
    language: NotRequired[str]


class _Text(TypedDict):
    context_id: str
    transcript: str
    force: bool


class _CloseContext(TypedDict):
    context_id: str
    transcript: str
    close_context: bool


_Command = _Init | _Text | _CloseContext


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "async_flash_v1.0",
        language: str | None = None,
        encoding: TTSEncoding = "pcm_s16le",
        voice: str = TTSDefaultVoiceId,
        sample_rate: int = 32000,
        http_session: http.HttpSession | None = None,
        base_url: str = "https://api.async.com",
    ) -> None:
        """
        Create a new instance of Async TTS.

        See https://docs.async.com/text-to-speech-websocket-3477526w0 for more details
            on the the Async API.

        Args:
            model (TTSModels, optional): The Async TTS model to use. Defaults to "async_flash_v1.0".
            language (str, optional): The language code for synthesis.
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (str, optional): The voice ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 32000.
            api_key (str, optional): The Async API key. If not provided, it will be
                read from the ASYNCAI_API_KEY environment variable.
            http_session (http.HttpSession | None, optional): An existing HttpSession to
                use. If not provided, the job's shared session is used.
            base_url (str, optional): The base URL for the Async API. Defaults to "https://api.async.com".
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        async_api_key = api_key or os.environ.get("ASYNCAI_API_KEY")
        if not async_api_key:
            raise ValueError(
                "AsyncAI API key is required, either as argument or set"
                " ASYNCAI_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            api_key=async_api_key,
            base_url=base_url,
        )
        self._mux = _Mux(self._opts)
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "asyncai"

    def prewarm(self) -> None:
        self._mux.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language and voice.
        If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels, optional): The Async TTS model to use. Defaults to "async_flash_v1.0".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str, optional): The voice ID.
            sentence_tokenizer (tokenize.SentenceTokenizer, optional): What splits the text into the
                chunks the TTS is sent.
            pacing (tts.SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or
                ``None`` to send each chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            connection_params_changed = False
            if is_given(model):
                self._opts.model = model
                connection_params_changed = True
            if is_given(language):
                self._opts.language = language
                connection_params_changed = True
            if is_given(voice):
                self._opts.voice = voice
                connection_params_changed = True

        if connection_params_changed:
            # model, voice and language are sent once, in each socket's init payload
            self._mux.invalidate()

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._mux.aclose()


class _Mux(tts.WebSocketMultiplexer[_Message, _Command]):
    """One Async socket carries every stream's segments, each its own context; its init payload
    fixes the model, voice and format."""

    max_session_duration = 300

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        query = urlencode({API_AUTH_HEADER: self._opts.api_key, API_VERSION_HEADER: API_VERSION})
        base_url = self._opts.base_url.replace("http", "ws", 1)
        return tts.WebSocketTarget(f"{base_url}/text_to_speech/websocket/ws?{query}")

    def setup(self) -> _Init:
        init = _Init(
            model_id=self._opts.model,
            voice=_Voice(mode="id", id=self._opts.voice),
            output_format=_OutputFormat(
                container="raw", encoding=self._opts.encoding, sample_rate=self._opts.sample_rate
            ),
        )
        if self._opts.language is not None:
            init["language"] = self._opts.language
        return init

    def route(self, message: _Message) -> str | None:
        if "error_code" in message:
            # an error names no context: it fails every stream on the socket
            raise _error(message)
        return message.get("context_id")

    def close_context(self, context_id: str) -> _CloseContext:
        return _CloseContext(context_id=context_id, transcript="", close_context=True)


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        # segments whose context was closed: a final before that ends a text, not the segment
        self._closed: set[str] = set()

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case tts.SegmentText():
                return _Text(
                    context_id=item.segment_id, transcript=f"{item.chunk.text} ", force=True
                )
            case tts.SegmentEnd():
                self._closed.add(item.segment_id)
                return _CloseContext(context_id=item.segment_id, transcript="", close_context=True)
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if "error_code" in message:
            raise _error(message)
        if (context_id := message.get("context_id")) is None:
            return
        if audio := message.get("audio"):
            try:
                data = base64.b64decode(audio, validate=True)
            except binascii.Error as e:
                raise tts.protocol_error(self, f"malformed audio: {audio}") from e
            self.report(tts.SegmentAudio(context_id, data))
        if message.get("final") and context_id in self._closed:
            self._closed.discard(context_id)
            self.report(tts.SegmentDone(context_id))


def _error(message: _Message) -> APIError:
    return APIError(
        f"Async returned an error ({message.get('error_code')})",
        kind="invalid_request",
        detail=message.get("message"),
    )
