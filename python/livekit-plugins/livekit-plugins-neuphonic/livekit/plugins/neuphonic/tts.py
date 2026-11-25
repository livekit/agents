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

from __future__ import annotations  # noqa: I001

import base64
import binascii
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import assert_never

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    tokenize,
    tts,
    utils,
)
from livekit.agents import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents import http
from livekit.agents.utils import is_given

from .models import TTSLangCodes  # noqa: I001

API_AUTH_HEADER = "x-api-key"


def _encoding_to_mime_type(encoding: str) -> str:
    if encoding.startswith("pcm"):
        return "audio/pcm"
    elif encoding.startswith("mp3"):
        return "audio/mp3"
    return "audio/pcm"


@dataclass
class _TTSOptions:
    lang_code: LanguageCode
    encoding: str
    sample_rate: int
    voice_id: str
    speed: float | None
    api_key: str | None
    jwt_token: str | None
    base_url: str

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        jwt_token: str | None = None,
        lang_code: TTSLangCodes | str = "en",
        encoding: str = "pcm_linear",
        voice_id: str = "8e9c4bc8-3979-48ab-8626-df53befc2090",
        speed: float | None = 1.0,
        sample_rate: int = 22050,
        http_session: http.HttpSession | None = None,
        base_url: str = "https://api.neuphonic.com",
    ) -> None:
        """
        Create a new instance of NeuPhonic TTS.

        See https://docs.neuphonic.com for more details on the NeuPhonic API.

        Args:
            lang_code (TTSLangCodes | str, optional): The language code for synthesis. Defaults to "en".
            encoding (str, optional): The audio encoding format. Defaults to "pcm_linear".
            voice_id (str, optional): The voice ID for the desired voice.
            speed (float, optional): The audio playback speed. Defaults to 1.0.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 22050.
            api_key (str, optional): The NeuPhonic API key. If not provided, it will be read from the NEUPHONIC_API_KEY environment variable.
            jwt_token (str, optional): The NeuPhonic JWT token.
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, the job's shared session is used.
            base_url (str, optional): The base URL for the NeuPhonic API. Defaults to "https://api.neuphonic.com".
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        neuphonic_api_key = api_key or os.environ.get("NEUPHONIC_API_KEY")
        if not neuphonic_api_key and not jwt_token:
            raise ValueError(
                "Neuphonic API key or JWT token is required, either as argument or set"
                " NEUPHONIC_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            lang_code=LanguageCode(lang_code),
            encoding=encoding,
            sample_rate=sample_rate,
            voice_id=voice_id,
            speed=speed,
            api_key=neuphonic_api_key,
            jwt_token=jwt_token,
            base_url=base_url,
        )
        self._session = http_session
        self._mux = _Mux(self._opts)
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return "Octave"

    @property
    def provider(self) -> str:
        return "neuphonic"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def prewarm(self) -> None:
        self._mux.prewarm()

    def update_options(
        self,
        *,
        lang_code: NotGivenOr[TTSLangCodes | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This allows updating the TTS settings, including lang_code, voice_id, and speed.
        If any parameter is not provided, the existing value will be retained.

        Args:
            lang_code (TTSLangCodes | str, optional): The language code for synthesis.
            voice_id (str, optional): The voice ID for the desired voice.
            speed (float, optional): The audio playback speed.
            sentence_tokenizer (SentenceTokenizer, optional): What splits the text into the chunks
                the TTS is sent.
            pacing (SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or ``None`` to
                send each chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            connection_params_changed = False
            if is_given(lang_code):
                self._opts.lang_code = LanguageCode(lang_code)
                connection_params_changed = True
            if is_given(voice_id):
                self._opts.voice_id = voice_id
                connection_params_changed = True
            if is_given(speed):
                self._opts.speed = speed
                connection_params_changed = True

        if connection_params_changed:
            # lang_code, voice_id and speed are part of the socket URL
            self._mux.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        if not self._opts.api_key:
            raise ValueError(
                "Neuphonic API key is required, either as argument or set"
                " NEUPHONIC_API_KEY environment variable"
            )

        return ChunkedStream(
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
            opts=replace(self._opts),
            ensure_session=self._ensure_session,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options, opts=replace(self._opts))
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._mux.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the SSE endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        ensure_session: Callable[[], http.HttpSession],
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._ensure_session = ensure_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            if self._opts.api_key:
                headers = {API_AUTH_HEADER: self._opts.api_key}
            else:
                headers = None

            async with self._ensure_session().post(
                f"{self._opts.base_url}/sse/speak/{self._opts.lang_code.language}",
                headers=headers,
                json={
                    "text": self._input_text,
                    "voice_id": self._opts.voice_id,
                    "lang_code": self._opts.lang_code.language,
                    "encoding": self._opts.encoding,
                    "sampling_rate": self._opts.sample_rate,
                    "speed": self._opts.speed,
                },
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=_encoding_to_mime_type(self._opts.encoding),
                )

                async for line in resp.iter_lines():
                    message = line.decode("utf-8")
                    if not message:
                        continue

                    parsed_message = _parse_sse_message(message)

                    if parsed_message is not None:
                        audio = (parsed_message.get("data") or {}).get("audio")
                        if audio is not None:
                            output_emitter.push(_decode_audio(audio))

                output_emitter.flush()
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class _AudioData(TypedDict, total=False):
    audio: str | None
    context_id: str | None
    stop: bool | None


class _Speak(TypedDict):
    text: str
    context_id: str


class _Response(TypedDict, total=False):
    data: _AudioData | None
    type: str
    status_code: int | None
    errors: object


_RESPONSE = TypeAdapter(_Response)


def _decode_audio(audio: str) -> bytes:
    try:
        return base64.b64decode(audio, validate=True)
    except binascii.Error as e:
        raise APIError("NeuPhonic sent malformed audio", kind="protocol") from e


def _error(resp: _Response) -> APIError:
    # the status code decides retrying: auth, quota and invalid requests are 4xx
    detail = str(resp.get("errors"))
    if status_code := resp.get("status_code"):
        return APIStatusError(
            "NeuPhonic returned an error", status_code=status_code, body=resp, detail=detail
        )
    return APIError("NeuPhonic returned an error", kind="invalid_request", body=resp, detail=detail)


def _parse_sse_message(message: str) -> _Response | None:
    """
    Parse each response from the SSE endpoint.

    The message will either be a string reading:
    - `event: error`
    - `event: message`
    - `data: { "status_code": 200, "data": {"audio": ... } }`
    """
    message = message.strip()
    # only a data line carries a response: an event name (one Neuphonic added since), an id or
    # a comment is not read
    if not message.startswith("data:"):
        return None

    value = message[len("data:") :].strip()
    try:
        message_dict = _RESPONSE.validate_json(value)
    except ValidationError as error:
        raise APIError("Malformed Neuphonic response", kind="protocol", detail=value) from error

    if message_dict.get("errors") is not None:
        raise _error(message_dict)

    return message_dict


class _Mux(tts.WebSocketMultiplexer[_Response, _Speak]):
    """One NeuPhonic socket carries every stream's segments, each its own context."""

    max_session_duration = 300

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        opts = self._opts
        lang = opts.lang_code.language
        params = {
            "speed": str(opts.speed),
            "lang_code": lang,
            "encoding": opts.encoding,
            "sampling_rate": str(opts.sample_rate),
            "voice_id": opts.voice_id,
        }
        if opts.jwt_token:
            params["jwt_token"] = opts.jwt_token
        headers = {API_AUTH_HEADER: opts.api_key} if opts.api_key else {}
        return tts.WebSocketTarget(
            opts.get_ws_url(f"/speak/{lang}"), headers=headers, params=params
        )

    def route(self, message: _Response) -> str | None:
        if message.get("type") == "error":
            raise _error(message)
        return (message.get("data") or {}).get("context_id")


class SynthesizeStream(tts.WebSocketStream[_Response, _Speak]):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions, opts: _TTSOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self.mime_type = _encoding_to_mime_type(opts.encoding)
        # each text is sent ending in <STOP> and answered with a stop: the stops each segment
        # still owes, and the segments whose text ended
        self._owed: dict[str, int] = {}
        self._ended: set[str] = set()

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Speak | None:
        match item:
            case tts.SegmentText():
                self._owed[item.segment_id] = self._owed.get(item.segment_id, 0) + 1
                return {"text": f"{item.chunk.text}<STOP>", "context_id": item.segment_id}
            case tts.SegmentEnd():
                self._ended.add(item.segment_id)
                if not self._owed.get(item.segment_id):
                    # every stop already came: nothing more will
                    self._forget(item.segment_id)
                    self.report(tts.SegmentDone(item.segment_id))
                return None
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Response) -> None:
        data = message.get("data") or {}
        context_id = data.get("context_id")
        if context_id is None:
            return
        if audio := data.get("audio"):
            self.report(tts.SegmentAudio(context_id, _decode_audio(audio)))
        if data.get("stop"):
            self._owed[context_id] = self._owed.get(context_id, 0) - 1
            if self._owed[context_id] <= 0 and context_id in self._ended:
                self._forget(context_id)
                self.report(tts.SegmentDone(context_id))

    def _forget(self, context_id: str) -> None:
        self._owed.pop(context_id, None)
        self._ended.discard(context_id)
