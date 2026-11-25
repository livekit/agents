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
from dataclasses import dataclass, field, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from pydantic import BaseModel, TypeAdapter

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    LanguageCode,
    create_api_error_from_http,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from ._utils import TRACE_ID_HEADER, trace_id_from_headers
from .log import logger
from .models import TTSEncoding, TTSEndpoint, TTSModels, tts_endpoint

# by default, use 22.05kHz sample rate at 32kbps
# in our testing,  reduce TTFB by about ~110ms
_DefaultEncoding: TTSEncoding = "mp3_22050_32"


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_44100
    return int(split[1])


def _encoding_to_mimetype(encoding: TTSEncoding) -> str:
    if encoding.startswith("mp3"):
        return "audio/mp3"
    elif encoding.startswith("opus"):
        return "audio/opus"
    elif encoding.startswith("pcm"):
        return "audio/pcm"
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: NotGivenOr[float] = NOT_GIVEN  # [0.0 - 1.0]
    speed: NotGivenOr[float] = NOT_GIVEN  # [0.8 - 1.2]
    use_speaker_boost: NotGivenOr[bool] = NOT_GIVEN


@dataclass
class Voice:
    id: str
    name: str
    category: str


@dataclass
class PronunciationDictionaryLocator:
    pronunciation_dictionary_id: str
    version_id: str


DEFAULT_VOICE_ID = "hpp4J3VqNfWAUOO0d1Us"
API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
WS_INACTIVITY_TIMEOUT = 180


# ElevenLabs' websockets. Text-to-speech spells its keys in camelCase, text-to-dialogue in
# snake_case; no message carries a tag, so one frame declares every field read.


class _Alignment(TypedDict, total=False):
    chars: list[str]
    charStartTimesMs: list[float]
    charDurationsMs: list[float]
    char_start_times_ms: list[float]
    char_durations_ms: list[float]


class _Frame(TypedDict, total=False):
    contextId: str
    context_id: str
    audio: str | None
    isFinal: bool | None
    is_final: bool | None
    alignment: _Alignment | None
    normalizedAlignment: _Alignment | None
    error: str
    message: str


class _VoiceSettings(TypedDict, total=False):
    stability: float
    similarity_boost: float
    style: float
    speed: float
    use_speaker_boost: bool


class _Locator(TypedDict):
    pronunciation_dictionary_id: str
    version_id: str


class _GenerationConfig(TypedDict):
    chunk_length_schedule: list[int]


class _Init(TypedDict):
    text: str
    voice_settings: _VoiceSettings
    context_id: str
    generation_config: NotRequired[_GenerationConfig]
    pronunciation_dictionary_locators: NotRequired[list[_Locator]]


class _Text(TypedDict):
    text: str
    context_id: str
    flush: NotRequired[bool]


class _Input(TypedDict):
    text: str
    voice_id: str


class _DialogueInit(TypedDict):
    voices: list[str]
    context_id: NotRequired[str]
    voice_settings: NotRequired[_VoiceSettings]
    pronunciation_dictionary_locators: NotRequired[list[_Locator]]


class _DialogueText(TypedDict):
    inputs: list[_Input]
    context_id: NotRequired[str]
    flush: NotRequired[bool]


class _CloseContext(TypedDict):
    context_id: str
    close_context: bool


class _CloseSocket(TypedDict):
    close_socket: bool


_Command = _Init | _Text | _DialogueInit | _DialogueText | _CloseContext | _CloseSocket


class _SynthesizeBody(TypedDict):
    text: str
    model_id: str
    voice_settings: _VoiceSettings | None
    apply_text_normalization: str
    language_code: NotRequired[str]
    apply_language_text_normalization: NotRequired[bool]


class _DialogueBody(TypedDict):
    inputs: list[_Input]
    model_id: str
    apply_text_normalization: str
    settings: NotRequired[_VoiceSettings]
    language_code: NotRequired[str]
    pronunciation_dictionary_locators: NotRequired[list[_Locator]]


class _VoiceRecord(BaseModel):
    voice_id: str
    name: str
    category: str


class _Voices(BaseModel):
    voices: list[_VoiceRecord]


_VOICES = TypeAdapter(_Voices)


@dataclass
class _TTSOptions:
    api_key: str
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    model: TTSModels | str
    language: NotGivenOr[LanguageCode]
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    chunk_length_schedule: NotGivenOr[list[int]]
    enable_ssml_parsing: bool
    enable_logging: bool
    inactivity_timeout: int
    sync_alignment: bool
    apply_text_normalization: Literal["auto", "on", "off"]
    apply_language_text_normalization: NotGivenOr[bool]
    preferred_alignment: NotGivenOr[Literal["normalized", "original"]]
    auto_mode: NotGivenOr[bool]
    pronunciation_dictionary_locators: NotGivenOr[list[PronunciationDictionaryLocator]]

    @property
    def endpoint(self) -> TTSEndpoint:
        return tts_endpoint(self.model)

    @property
    def ws_base_url(self) -> str:
        return self.base_url.replace("https://", "wss://").replace("http://", "ws://")


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: TTSModels | str = "eleven_turbo_v2_5",
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
        auto_mode: NotGivenOr[bool] = NOT_GIVEN,
        apply_text_normalization: Literal["auto", "off", "on"] = "auto",
        apply_language_text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        enable_ssml_parsing: bool = False,
        enable_logging: bool = True,
        chunk_length_schedule: NotGivenOr[list[int]] = NOT_GIVEN,  # range is [50, 500]
        http_session: http.HttpSession | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
        sync_alignment: bool = True,
        preferred_alignment: NotGivenOr[Literal["normalized", "original"]] = NOT_GIVEN,
        pronunciation_dictionary_locators: NotGivenOr[
            list[PronunciationDictionaryLocator]
        ] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of ElevenLabs TTS.

        Args:
            voice_id (str): Voice ID. Defaults to `DEFAULT_VOICE_ID`.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
                "eleven_v3", "eleven_v3_conversational", "eleven_v4" and "eleven_v4_turbo"
                go through ElevenLabs' text-to-dialogue API instead (single voice per
                instance, same as other models); eleven_v4 streams one context a socket.
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            inactivity_timeout (int): Inactivity timeout in seconds for the websocket connection. Defaults to 300.
            auto_mode (bool): Reduces latency by disabling chunk schedule and buffers.
                Text is synthesized one sentence at a time.
                Defaults to True unless ``chunk_length_schedule`` is provided.
            apply_text_normalization (Literal["auto", "off", "on"]): This parameter controls text normalization with three modes: ‘auto’, ‘on’, and ‘off’. When set to ‘auto’, the system will automatically decide whether to apply text normalization (e.g., spelling out numbers). With ‘on’, text normalization will always be applied, while with ‘off’, it will be skipped.
            apply_language_text_normalization (bool): This parameter controls language text normalization. This helps with proper pronunciation of text in some supported languages.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            enable_logging (bool): Enable logging of the request. When set to false, zero retention mode will be used. Defaults to True.
            chunk_length_schedule (NotGivenOr[list[int]]): Schedule for chunk lengths, ranging from 50 to 500. Defaults are [120, 160, 250, 290].
            http_session (http.HttpSession | None): Custom HTTP session for API requests. Optional.
            language (NotGivenOr[str]): Language code used to enforce a language for the model and text normalization. If the model does not support language overrides, it will be ignored.
            sync_alignment (bool): Enable sync alignment for the TTS model. Defaults to True.
            preferred_alignment (Literal["normalized", "original"]): Use normalized or original alignment. Defaults to "normalized", or "original" for CJK (ja, ko, zh) languages.
            pronunciation_dictionary_locators (NotGivenOr[list[PronunciationDictionaryLocator]]): List of pronunciation dictionary locators to use for pronunciation control.
        """  # noqa: E501
        if not is_given(encoding):
            encoding = _DefaultEncoding
        if not is_given(auto_mode):
            auto_mode = not is_given(chunk_length_schedule)

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=sync_alignment,
                text_unit="sentences" if auto_mode else "words",
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set ELEVEN_API_KEY environmental variable"  # noqa: E501
            )

        self._opts = _TTSOptions(
            voice_id=voice_id,
            voice_settings=voice_settings,
            model=model,
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            enable_logging=enable_logging,
            language=LanguageCode(language) if is_given(language) else NOT_GIVEN,
            inactivity_timeout=inactivity_timeout,
            sync_alignment=sync_alignment,
            auto_mode=auto_mode,
            apply_text_normalization=apply_text_normalization,
            apply_language_text_normalization=apply_language_text_normalization,
            preferred_alignment=preferred_alignment,
            pronunciation_dictionary_locators=pronunciation_dictionary_locators,
        )
        self._session = http_session.with_request_id(TRACE_ID_HEADER) if http_session else None
        self._streams = weakref.WeakSet[tts.SynthesizeStream]()
        self._mux = _Mux(self._opts)
        # muxes of an earlier endpoint: their streams finish on them, closed with the model
        self._retired: list[_Mux] = []
        self._warn_if_dialogue_model_ignores_options()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "elevenlabs"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session().with_request_id(TRACE_ID_HEADER)
        return self._session

    def _warn_if_dialogue_model_ignores_options(self) -> None:
        """Warn if options unsupported by text-to-dialogue are set for a dialogue model."""
        if self._opts.endpoint == "speech":
            return

        ignored = [
            name
            for name, is_set in (
                ("chunk_length_schedule", is_given(self._opts.chunk_length_schedule)),
                ("enable_ssml_parsing", self._opts.enable_ssml_parsing),
                (
                    "apply_language_text_normalization",
                    is_given(self._opts.apply_language_text_normalization),
                ),
            )
            if is_set
        ]
        if is_given(self._opts.voice_settings):
            settings = _voice_settings(self._opts.voice_settings, dialogue=False)
            ignored.extend(f"voice_settings.{name}" for name in settings if name != "stability")
        if ignored:
            logger.warning(
                "model '%s' is synthesized via ElevenLabs' text-to-dialogue API, which "
                "does not support these options; they will be ignored: %s",
                self._opts.model,
                ", ".join(ignored),
            )

    async def list_voices(self) -> list[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            voices = _VOICES.validate_python(await resp.json()).voices
        return [Voice(id=v.voice_id, name=v.name, category=v.category) for v in voices]

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        pronunciation_dictionary_locators: NotGivenOr[
            list[PronunciationDictionaryLocator]
        ] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice_id (NotGivenOr[str]): Voice ID.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[str]): Language code for the TTS model.
            pronunciation_dictionary_locators (NotGivenOr[list[PronunciationDictionaryLocator]]): List of pronunciation dictionary locators.
            sentence_tokenizer (NotGivenOr[tokenize.SentenceTokenizer]): What splits the text into
                the chunks the TTS is sent.
            pacing (NotGivenOr[tts.SentenceStreamPacer | None]): A ``SentenceStreamPacer``, or
                ``None`` to send each chunk as soon as it is split.
        """  # noqa: E501
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(language):
                language = LanguageCode(language)
            endpoint = self._opts.endpoint
            changed = False

            if is_given(model) and model != self._opts.model:
                self._opts.model = model
                changed = True
                self._warn_if_dialogue_model_ignores_options()

            if is_given(voice_id) and voice_id != self._opts.voice_id:
                self._opts.voice_id = voice_id
                changed = True

            if is_given(voice_settings):
                self._opts.voice_settings = voice_settings
                changed = True

            if is_given(language) and language != self._opts.language:
                self._opts.language = language
                changed = True

            if is_given(pronunciation_dictionary_locators):
                self._opts.pronunciation_dictionary_locators = pronunciation_dictionary_locators
                changed = True

        if self._opts.endpoint != endpoint:
            # another endpoint shares its sockets differently: a mux of its own
            self._mux.invalidate()
            self._retired.append(self._mux)
            self._mux = _Mux(self._opts)
        elif changed:
            self._mux.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self, opts=replace(self._opts), input_text=chunk.text, conn_options=conn_options
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> tts.SynthesizeStream:
        opts = replace(self._opts)
        stream: tts.SynthesizeStream
        if opts.endpoint == "dialogue_single":
            stream = _SingleContextStream(tts=self, opts=opts, conn_options=conn_options)
        else:
            stream = SynthesizeStream(tts=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Open the websocket connection before the first synthesis request."""
        if self._opts.endpoint != "dialogue_single":
            self._mux.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        for mux in [*self._retired, self._mux]:
            await mux.aclose()
        self._retired.clear()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self, *, opts: _TTSOptions, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._opts
        query = f"output_format={opts.encoding}&enable_logging={str(opts.enable_logging).lower()}"
        body: _SynthesizeBody | _DialogueBody
        if opts.endpoint == "speech":
            url = f"{opts.base_url}/text-to-speech/{opts.voice_id}/stream?{query}"
            body = _synthesize_body(opts, self._input_text)
        else:
            url = f"{opts.base_url}/text-to-dialogue/stream?{query}"
            body = _dialogue_body(opts, self._input_text)

        try:
            async with self._tts._ensure_session().post(
                url,
                headers={AUTHORIZATION_HEADER: opts.api_key},
                json=body,
                conn_options=self._conn_options,
            ) as resp:
                if not resp.ok:
                    raise create_api_error_from_http(
                        resp.reason or "",
                        status=resp.status,
                        request_id=trace_id_from_headers(resp.headers),
                        body=await resp.text(),
                    )

                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    raise APIError(
                        "11labs returned non-audio data", kind="protocol", detail=content
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=opts.sample_rate,
                    num_channels=1,
                    mime_type=_encoding_to_mimetype(opts.encoding),
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                if (billed := resp.headers.get("character-cost")) is not None:
                    output_emitter.report_usage(characters_count=int(billed))
                output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class _Mux(tts.WebSocketMultiplexer[_Frame, _Command]):
    """One multi-stream-input socket carries every stream's segments, each its own context: the
    text-to-speech endpoint, or text-to-dialogue's for a dialogue model, whose contexts the
    server finalizes once idle for 20 s and whose idle sockets it closes."""

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts
        # contexts whose close went: a stream leaving one owes no second close
        self.closing: set[str] = set()
        self.keep_warm = opts.endpoint == "dialogue"

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        opts = self._opts
        params = {
            "model_id": opts.model,
            "output_format": opts.encoding,
            "enable_logging": _flag(opts.enable_logging),
            "apply_text_normalization": opts.apply_text_normalization,
        }
        if is_given(opts.language):
            params["language_code"] = opts.language.language
        if opts.sync_alignment:
            params["sync_alignment"] = "true"
        if opts.endpoint != "speech":
            # no voice in the URL (each context registers it), and a narrower set of params
            url = f"{opts.ws_base_url}/text-to-dialogue/multi-stream-input"
        else:
            url = f"{opts.ws_base_url}/text-to-speech/{opts.voice_id}/multi-stream-input"
            params["enable_ssml_parsing"] = _flag(opts.enable_ssml_parsing)
            params["inactivity_timeout"] = str(opts.inactivity_timeout)
            if is_given(opts.apply_language_text_normalization):
                params["apply_language_text_normalization"] = _flag(
                    opts.apply_language_text_normalization
                )
            if is_given(opts.auto_mode):
                params["auto_mode"] = _flag(opts.auto_mode)
        return tts.WebSocketTarget(
            url,
            headers={AUTHORIZATION_HEADER: opts.api_key},
            params=params,
            request_id_header=TRACE_ID_HEADER,
        )

    def route(self, message: _Frame) -> str | None:
        if (context_id := _context(message)) is None and "error" in message:
            # an error naming no context fails every stream on the socket
            raise _error(message)
        return context_id

    def close_context(self, context_id: str) -> _CloseContext | None:
        if context_id in self.closing:
            self.closing.discard(context_id)
            return None
        return _CloseContext(context_id=context_id, close_context=True)


class SynthesizeStream(tts.WebSocketStream[_Frame, _Command]):
    """Each segment is a context on the shared multi-stream-input socket."""

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = self._shared = tts._mux
        self.mime_type = _encoding_to_mimetype(opts.encoding)
        self._opts = opts
        self._words = _Words(opts)

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        opts = self._opts
        dialogue = opts.endpoint != "speech"
        # in auto_mode each chunk is a sentence, flushed on its own
        flush = is_given(opts.auto_mode) and opts.auto_mode
        match item:
            case tts.SegmentStart(segment_id=context_id):
                self._words.start(context_id)
                return _dialogue_init(opts, context_id) if dialogue else _init(opts, context_id)
            case tts.SegmentText(segment_id=context_id):
                # must always end with a space
                text = f"{item.chunk.text} "
                if dialogue:
                    return _DialogueText(
                        context_id=context_id,
                        inputs=[_Input(text=text, voice_id=opts.voice_id)],
                        flush=flush,
                    )
                return _Text(text=text, context_id=context_id, flush=flush)
            case tts.SegmentEnd(segment_id=context_id):
                if not dialogue:
                    # closing a text-to-speech context does not speak its buffered text
                    self.send_now(_Text(text="", context_id=context_id, flush=True))
                self._shared.closing.add(context_id)
                return _CloseContext(context_id=context_id, close_context=True)
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Frame) -> None:
        if (context_id := _context(message)) is not None:
            if self._words.report(self, context_id, message):
                self._shared.closing.discard(context_id)


class _SingleContextStream(tts.WebSocketStream[_Frame, _Command]):
    """Text-to-dialogue stream-input, the endpoint an eleven_v4 model streams on: one context a
    socket, so each segment is a session of its own, ended by ``close_socket``."""

    connection_per_segment = True

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.mime_type = _encoding_to_mimetype(opts.encoding)
        self._opts = opts
        self._words = _Words(opts)
        self._segment_id = ""

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        opts = self._opts
        params = {
            "model_id": opts.model,
            "output_format": opts.encoding,
            "enable_logging": _flag(opts.enable_logging),
            "apply_text_normalization": opts.apply_text_normalization,
        }
        if is_given(opts.language):
            params["language_code"] = opts.language.language
        if opts.sync_alignment:
            params["sync_alignment"] = "true"
        return tts.WebSocketTarget(
            f"{opts.ws_base_url}/text-to-dialogue/stream-input",
            headers={AUTHORIZATION_HEADER: opts.api_key},
            params=params,
            request_id_header=TRACE_ID_HEADER,
        )

    def setup(self) -> _DialogueInit:
        return _dialogue_init(self._opts, None)

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        opts = self._opts
        match item:
            case tts.SegmentStart(segment_id=segment_id):
                self._segment_id = segment_id
                self._words.start(segment_id)
                return None
            case tts.SegmentText():
                return _DialogueText(
                    inputs=[_Input(text=f"{item.chunk.text} ", voice_id=opts.voice_id)],
                    flush=is_given(opts.auto_mode) and opts.auto_mode,
                )
            case tts.SegmentEnd():
                return _CloseSocket(close_socket=True)
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Frame) -> None:
        self._words.report(self, self._segment_id, message)


@dataclass
class _AlignmentBuffer:
    text: str = ""
    start_times_ms: list[float] = field(default_factory=list[float])
    durations_ms: list[float] = field(default_factory=list[float])


class _Words:
    """Each segment's audio, and its character alignment as timed words, reported."""

    def __init__(self, opts: _TTSOptions) -> None:
        if opts.endpoint != "speech":
            normalized = False  # text-to-dialogue sends one alignment
        elif is_given(opts.preferred_alignment):
            normalized = opts.preferred_alignment == "normalized"
        else:
            cjk = is_given(opts.language) and opts.language.language in {"ja", "ko", "zh"}
            normalized = not cjk
        self._key: Literal["normalizedAlignment", "alignment"] = (
            "normalizedAlignment" if normalized else "alignment"
        )
        self._buffers: dict[str, _AlignmentBuffer] = {}

    def start(self, segment_id: str) -> None:
        self._buffers[segment_id] = _AlignmentBuffer()

    def report(
        self, stream: tts.WebSocketStream[_Frame, _Command], segment_id: str, frame: _Frame
    ) -> bool:
        """Whether the frame finished the segment."""
        if "error" in frame:
            raise _error(frame)
        buffer = self._buffers.setdefault(segment_id, _AlignmentBuffer())
        if alignment := frame.get(self._key):
            words = _accumulate_alignment(buffer, alignment)
            if words:
                stream.report(tts.SegmentWords(segment_id, words))
        if audio := frame.get("audio"):
            stream.report(tts.SegmentAudio(segment_id, base64.b64decode(audio)))
        if frame.get("isFinal") or frame.get("is_final"):
            del self._buffers[segment_id]
            words, _ = _to_timed_words(
                buffer.text, buffer.start_times_ms, buffer.durations_ms, flush=True
            )
            if words:
                stream.report(tts.SegmentWords(segment_id, words))
            stream.report(tts.SegmentDone(segment_id))
            return True
        return False


def _context(frame: _Frame) -> str | None:
    # ElevenLabs sends snake_case context ids on the websocket API, older responses camelCase
    return frame.get("context_id") or frame.get("contextId")


def _error(frame: _Frame) -> APIError:
    return APIError(
        "ElevenLabs returned an error",
        kind="invalid_request",
        detail=frame.get("message") or frame.get("error"),
    )


def _flag(value: bool) -> str:
    return str(value).lower()


def _voice_settings(settings: VoiceSettings, *, dialogue: bool) -> _VoiceSettings:
    """The settings set, without unset fields; text-to-dialogue takes only ``stability``."""
    payload = _VoiceSettings(stability=settings.stability)
    if dialogue:
        return payload
    payload["similarity_boost"] = settings.similarity_boost
    if is_given(settings.style):
        payload["style"] = settings.style
    if is_given(settings.speed):
        payload["speed"] = settings.speed
    if is_given(settings.use_speaker_boost):
        payload["use_speaker_boost"] = settings.use_speaker_boost
    return payload


def _locators(opts: _TTSOptions) -> list[_Locator] | None:
    if not is_given(opts.pronunciation_dictionary_locators):
        return None
    return [
        _Locator(
            pronunciation_dictionary_id=locator.pronunciation_dictionary_id,
            version_id=locator.version_id,
        )
        for locator in opts.pronunciation_dictionary_locators
    ]


def _init(opts: _TTSOptions, context_id: str) -> _Init:
    init = _Init(
        text=" ",
        voice_settings=(
            _voice_settings(opts.voice_settings, dialogue=False)
            if is_given(opts.voice_settings)
            else _VoiceSettings()
        ),
        context_id=context_id,
    )
    if is_given(opts.chunk_length_schedule):
        init["generation_config"] = _GenerationConfig(
            chunk_length_schedule=opts.chunk_length_schedule
        )
    if (locators := _locators(opts)) is not None:
        init["pronunciation_dictionary_locators"] = locators
    return init


def _dialogue_init(opts: _TTSOptions, context_id: str | None) -> _DialogueInit:
    # voices, voice_settings and pronunciation_dictionary_locators are only valid on a
    # context's first message; text is sent separately, after this
    init = _DialogueInit(voices=[opts.voice_id])
    if context_id is not None:
        init["context_id"] = context_id
    if is_given(opts.voice_settings):
        init["voice_settings"] = _voice_settings(opts.voice_settings, dialogue=True)
    if (locators := _locators(opts)) is not None:
        init["pronunciation_dictionary_locators"] = locators
    return init


def _synthesize_body(opts: _TTSOptions, text: str) -> _SynthesizeBody:
    body = _SynthesizeBody(
        text=text,
        model_id=opts.model,
        voice_settings=(
            _voice_settings(opts.voice_settings, dialogue=False)
            if is_given(opts.voice_settings)
            else None
        ),
        apply_text_normalization=opts.apply_text_normalization,
    )
    if is_given(opts.language):
        body["language_code"] = opts.language.language
    if is_given(opts.apply_language_text_normalization):
        body["apply_language_text_normalization"] = opts.apply_language_text_normalization
    return body


def _dialogue_body(opts: _TTSOptions, text: str) -> _DialogueBody:
    """Single-turn body for /text-to-dialogue/stream: `settings`, not `voice_settings`,
    and voice_id is per-input rather than part of the URL."""
    body = _DialogueBody(
        inputs=[_Input(text=text, voice_id=opts.voice_id)],
        model_id=opts.model,
        apply_text_normalization=opts.apply_text_normalization,
    )
    if is_given(opts.voice_settings):
        body["settings"] = _voice_settings(opts.voice_settings, dialogue=True)
    if is_given(opts.language):
        body["language_code"] = opts.language.language
    if (locators := _locators(opts)) is not None:
        body["pronunciation_dictionary_locators"] = locators
    return body


def _accumulate_alignment(buffer: _AlignmentBuffer, alignment: _Alignment) -> list[TranscriptChunk]:
    chars = alignment.get("chars")
    starts = alignment.get("charStartTimesMs") or alignment.get("char_start_times_ms")
    durs = alignment.get("charDurationsMs") or alignment.get("char_durations_ms")
    if not (chars and starts and durs and len(chars) == len(starts) == len(durs)):
        return []

    buffer.text += "".join(chars)
    # in case item in chars has multiple characters
    for char, start, dur in zip(chars, starts, durs, strict=False):
        if len(char) > 1:
            buffer.start_times_ms += [start] * (len(char) - 1)
            buffer.durations_ms += [0] * (len(char) - 1)
        buffer.start_times_ms.append(start)
        buffer.durations_ms.append(dur)

    timed_words, buffer.text = _to_timed_words(
        buffer.text, buffer.start_times_ms, buffer.durations_ms
    )
    buffer.start_times_ms = buffer.start_times_ms[-len(buffer.text) :]
    buffer.durations_ms = buffer.durations_ms[-len(buffer.text) :]
    return timed_words


def _to_timed_words(
    text: str, start_times_ms: list[float], durations_ms: list[float], flush: bool = False
) -> tuple[list[TranscriptChunk], str]:
    """Return timed words and the remaining text"""
    if not text:
        return [], ""

    timestamps = start_times_ms + [start_times_ms[-1] + durations_ms[-1]]  # N+1

    words = tokenize.split_words(text, ignore_punctuation=False)
    if not words:
        return [], text

    timed_words: list[TranscriptChunk] = []
    start_indices = [word.start for word in words]
    end = 0
    # we don't know if the last word is complete, always leave it as remaining
    for start, end in zip(start_indices[:-1], start_indices[1:], strict=False):
        start_t = timestamps[start] / 1000
        end_t = timestamps[end] / 1000
        timed_words.append(
            TranscriptChunk(text=text[start:end], start_time=start_t, end_time=end_t),
        )

    if flush:
        start_t = timestamps[end] / 1000
        end_t = timestamps[-1] / 1000
        timed_words.append(TranscriptChunk(text=text[end:], start_time=start_t, end_time=end_t))
        end = len(text)

    return timed_words, text[end:]
