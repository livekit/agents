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
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, NotRequired, TypedDict, assert_never

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import (
    TTSDefaultVoiceId,
    TTSDefaultVoiceStyle,
    TTSEncoding,
    TTSLocales,
    TTSModels,
    TTSStyles,
)

API_AUTH_HEADER = "api-key"


@dataclass
class _TTSOptions:
    api_key: str
    locale: TTSLocales | str | None
    model: TTSModels | str
    voice: str
    style: str | None
    speed: int | None
    pitch: int | None
    sample_rate: int
    encoding: TTSEncoding
    base_url: str
    min_buffer_size: int
    max_buffer_delay_in_ms: int

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "FALCON",
        locale: TTSLocales | str | None = None,
        voice: str = TTSDefaultVoiceId,
        style: TTSStyles | str | None = None,
        speed: int | None = None,
        pitch: int | None = None,
        sample_rate: int = 24000,
        encoding: TTSEncoding = "pcm",
        base_url: str = "https://global.api.murf.ai",
        http_session: http.HttpSession | None = None,
        min_buffer_size: int = 3,
        max_buffer_delay_in_ms: int = 0,
        streaming: bool = True,
    ) -> None:
        """
        Create a new instance of Murf AI TTS.

        See https://murf.ai/api/docs/api-reference/text-to-speech/stream-input for more details on the the Murf AI API.

        Args:
            api_key (str | None, optional): The Murf AI API key. If not provided, it will be read from the MURF_API_KEY environment variable.
            model (TTSModels | str, optional): The Murf AI TTS model to use. Defaults to "FALCON".
            locale (str | None, optional): The locale for synthesis (e.g., "en-US", "en-UK"). If not provided, will be inferred from voice.
            voice (str, optional): The voice ID from Murf AI's voice library (e.g., "en-US-matthew"). Defaults to TTSDefaultVoiceId.
            style (TTSStyles | str | None, optional): The voice style to apply (e.g., "Conversation"). Can be None for default style.
            speed (int | None, optional): The speech speed control. Higher values = faster speech. None for default speed.
            pitch (int | None, optional): The speech pitch control. Higher values = higher pitch. None for default pitch.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            encoding (str, optional): The audio encoding format. Defaults to "pcm".
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, the job's shared session is used.
            base_url (str, optional): The base URL for the Murf AI API. Defaults to "https://global.api.murf.ai".
            min_buffer_size (int, optional):Minimum characters to buffer before sending text to audio when no sentence boundary is detected. Higher values improve quality; lower values reduce TTFB. Defaults to 3.
            max_buffer_delay_in_ms (int, optional): Maximum wait time before sending buffered text if min_buffer_size isn’t reached. Defaults to 0.
            streaming (bool, optional): If True, uses WebSocket streaming for real-time audio. If False, uses HTTP requests. Defaults to True.
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=sample_rate,
            num_channels=1,
        )

        murf_api_key = api_key or os.environ.get("MURF_API_KEY")
        if not murf_api_key:
            raise ValueError("MURF_API_KEY must be set")

        self._opts = _TTSOptions(
            api_key=murf_api_key,
            model=model,
            locale=locale,
            voice=voice,
            style=style or TTSDefaultVoiceStyle,
            speed=speed,
            pitch=pitch,
            sample_rate=sample_rate,
            encoding=encoding,
            base_url=base_url,
            min_buffer_size=min_buffer_size,
            max_buffer_delay_in_ms=max_buffer_delay_in_ms,
        )
        self._session = http_session
        self._mux = _Mux(self._opts)
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Murf"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def prewarm(self) -> None:
        self._mux.prewarm()

    def update_options(
        self,
        *,
        locale: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        style: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[int | None] = NOT_GIVEN,
        pitch: NotGivenOr[int | None] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model, locale, voice, style,
        speed and pitch. If any parameter is not provided, the existing value will be retained.

        Args:
            locale (str, optional): The locale for synthesis (e.g., "en-US", "en-UK").
            voice (str, optional): The voice ID from Murf AI's voice library. (e.g. "en-US-matthew")
            style (str | None, optional): The voice style to apply (e.g., "Conversation", "Promo").
            speed (int | None, optional): Controls the speech speed. Positive values increase speed, negative values decrease it. Valid range: -50 to 50.
            pitch (int | None, optional): Controls the speech pitch. Positive values raise pitch, negative values lower it. Valid range: -50 to 50.
            sentence_tokenizer (SentenceTokenizer, optional): What splits the text into the chunks
                the TTS is sent.
            pacing (SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or ``None`` to
                send each chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(locale):
                self._opts.locale = locale
            if is_given(voice):
                self._opts.voice = voice
            if is_given(style):
                self._opts.style = style
            if is_given(speed):
                self._opts.speed = speed
            if is_given(pitch):
                self._opts.pitch = pitch

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            get_session=self._ensure_session,
            opts=replace(self._opts),
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(
            opts=replace(self._opts),
            tts=self,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._mux.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the http streaming output endpoint"""

    def __init__(
        self,
        *,
        get_session: Callable[[], http.HttpSession],
        opts: _TTSOptions,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._get_session = get_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()

        try:
            async with self._get_session().post(
                self._opts.get_http_url("/v1/speech/stream"),
                headers={API_AUTH_HEADER: self._opts.api_key},
                json={
                    "text": self._input_text,
                    "model": self._opts.model,
                    "multiNativeLocale": self._opts.locale,
                    "voice_id": self._opts.voice,
                    "style": self._opts.style,
                    "rate": self._opts.speed,
                    "pitch": self._opts.pitch,
                    "format": self._opts.encoding,
                    "sample_rate": self._opts.sample_rate,
                },
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


# Murf's stream-input websocket: each segment is one context, ended by `end`. Each message
# declares only the fields read.


class _Message(TypedDict):
    context_id: NotRequired[str | None]
    audio: NotRequired[str]
    final: NotRequired[bool]
    error: NotRequired[str]
    error_code: NotRequired[str | int]


class _Config(TypedDict):
    voice_config: dict[str, Any]
    min_buffer_size: int
    max_buffer_delay_in_ms: int


class _Speak(_Config):
    context_id: str
    text: str
    end: bool


class _Clear(TypedDict):
    clear: bool
    context_id: str


class _Mux(tts.WebSocketMultiplexer[_Message, _Speak | _Clear]):
    """One Murf socket carries every stream's segments, each its own context."""

    max_session_duration = 300

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._opts.get_ws_url("/v1/speech/stream-input"),
            headers={API_AUTH_HEADER: self._opts.api_key},
            params={
                "sample_rate": str(self._opts.sample_rate),
                "format": self._opts.encoding,
                "model": self._opts.model,
            },
        )

    def route(self, message: _Message) -> str | None:
        if context_id := message.get("context_id"):
            return context_id
        if "error" in message:
            # an error naming no context fails every stream on the socket
            raise _error(message)
        return None

    def close_context(self, context_id: str) -> _Clear:
        return _Clear(clear=True, context_id=context_id)


class SynthesizeStream(tts.WebSocketStream[_Message, _Speak | _Clear]):
    def __init__(self, *, opts: _TTSOptions, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self._opts = opts
        # contexts whose text ended: a final before the end is not the context's last answer
        self._ended: set[str] = set()

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Speak | None:
        match item:
            case tts.SegmentText():
                return self._speak(item.segment_id, item.chunk.text + " ", end=False)
            case tts.SegmentEnd():
                self._ended.add(item.segment_id)
                return self._speak(item.segment_id, "", end=True)
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        context_id = message.get("context_id") or ""
        if "error" in message:
            raise _error(message)
        if audio := message.get("audio"):
            self.report(tts.SegmentAudio(context_id, base64.b64decode(audio)))
        elif message.get("final") and context_id in self._ended:
            self._ended.discard(context_id)
            self.report(tts.SegmentDone(context_id))

    def _speak(self, context_id: str, text: str, *, end: bool) -> _Speak:
        config = _to_murf_websocket_pkt(self._opts)
        return {**config, "context_id": context_id, "text": text, "end": end}


def _error(message: _Message) -> APIError:
    return APIError(
        f"Murf returned an error ({message.get('error_code')})",
        kind="invalid_request",
        body=message,
        detail=message.get("error"),
    )


def _to_murf_websocket_pkt(opts: _TTSOptions) -> _Config:
    voice_config: dict[str, Any] = {}

    if opts.voice:
        voice_config["voice_id"] = opts.voice

    if opts.style:
        voice_config["style"] = opts.style

    if opts.speed:
        voice_config["rate"] = opts.speed

    if opts.pitch:
        voice_config["pitch"] = opts.pitch

    if opts.locale:
        voice_config["multi_native_locale"] = opts.locale

    return {
        "voice_config": voice_config,
        "min_buffer_size": opts.min_buffer_size,
        "max_buffer_delay_in_ms": opts.max_buffer_delay_in_ms,
    }
