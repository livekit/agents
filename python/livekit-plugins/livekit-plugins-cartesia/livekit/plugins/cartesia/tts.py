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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, NotRequired, TypedDict, assert_never, cast

from pydantic import Field

from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from .constants import (
    API_AUTH_HEADER,
    API_VERSION,
    API_VERSION_HEADER,
    API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS,
    AUDIO_ENCODING,
    MODEL_ID_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS,
    REQUEST_ID_HEADER,
    USER_AGENT,
)
from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSModels,
    TTSVoiceEmotion,
    TTSVoiceSpeed,
)

# Cartesia's websocket: each segment is one context, ended by `continue: false`. Each message
# declares only the fields read.


class _WordTimestamps(TypedDict):
    words: list[str]
    start: list[float]
    end: list[float]


class _Chunk(TypedDict):
    type: Literal["chunk"]
    context_id: str
    data: str


class _Timestamps(TypedDict):
    type: Literal["timestamps"]
    context_id: str
    word_timestamps: NotRequired[_WordTimestamps]


class _Done(TypedDict):
    type: Literal["done"]
    context_id: str


class _Error(TypedDict):
    type: Literal["error"]
    context_id: NotRequired[str]
    message: NotRequired[str]
    error: NotRequired[str]
    status_code: NotRequired[int]


_Message = Annotated[_Chunk | _Timestamps | _Done | _Error, Field(discriminator="type")]


class _Options(TypedDict):
    model_id: str
    voice: dict[str, Any]
    output_format: dict[str, Any]
    language: str | None
    pronunciation_dict_id: NotRequired[str]
    generation_config: NotRequired[dict[str, Any]]
    add_timestamps: NotRequired[bool]


_Continue = TypedDict("_Continue", {"continue": bool})  # a keyword: the functional form


class _Speak(_Options, _Continue):
    max_buffer_delay_ms: int
    context_id: str
    transcript: str


class _Cancel(TypedDict):
    context_id: str
    cancel: bool


def _is_sonic_3(model: str) -> bool:
    return model.startswith("sonic-3")


@dataclass
class _TTSOptions:
    model: TTSModels | str
    sample_rate: int
    voice: str | list[float]
    speed: TTSVoiceSpeed | float | None
    emotion: list[TTSVoiceEmotion | str] | None
    volume: float | None
    word_timestamps: bool
    api_key: str
    language: LanguageCode | None
    base_url: str
    api_version: str
    pronunciation_dict_id: str | None

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "sonic-3",
        language: str | None = "en",
        voice: str | list[float] = TTSDefaultVoiceId,
        speed: TTSVoiceSpeed | float | None = None,
        emotion: TTSVoiceEmotion | str | list[TTSVoiceEmotion | str] | None = None,
        volume: float | None = None,
        sample_rate: int = 24000,
        word_timestamps: bool = True,
        pronunciation_dict_id: str | None = None,
        http_session: http.HttpSession | None = None,
        base_url: str = "https://api.cartesia.ai",
        api_version: str = API_VERSION,
    ) -> None:
        """
        Create a new instance of Cartesia TTS.

        See https://docs.cartesia.ai/reference/web-socket/stream-speech/stream-speech for more details on the Cartesia API.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-3".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Speed of speech, with sonic-3, the value is valid between 0.6 and 2.0 (https://docs.cartesia.ai/api-reference/tts/bytes#body-generation-config-speed)
            emotion (list[TTSVoiceEmotion], optional): Emotion of the speech (https://docs.cartesia.ai/api-reference/tts/bytes#body-generation-config-emotion)
            volume (float, optional): Volume of the speech, with sonic-3, the value is valid between 0.5 and 2.0
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            word_timestamps (bool, optional): Whether to add word timestamps to the output. Defaults to True.
            pronunciation_dict_id (str, optional): The pronunciation dictionary ID to use for custom pronunciations. Defaults to None.
            api_key (str, optional): The Cartesia API key. If not provided, it will be read from the CARTESIA_API_KEY environment variable.
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, the job's shared session is used.
            base_url (str, optional): The base URL for the Cartesia API. Defaults to "https://api.cartesia.ai".
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        cartesia_api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise ValueError(
                "Cartesia API key is required, either as argument or set"
                " CARTESIA_API_KEY environment variable"
            )

        if isinstance(emotion, str):
            emotion = [emotion]

        self._opts = _TTSOptions(
            model=model,
            language=LanguageCode(language) if language else None,
            sample_rate=sample_rate,
            voice=voice,
            speed=speed,
            emotion=emotion,
            volume=volume,
            api_key=cartesia_api_key,
            base_url=base_url,
            word_timestamps=word_timestamps,
            api_version=api_version,
            pronunciation_dict_id=pronunciation_dict_id,
        )

        if speed or emotion or volume or pronunciation_dict_id:
            _check_generation_config(self._opts)

        self._session = http_session
        self._mux = _Mux(self)
        self._streams = weakref.WeakSet[SynthesizeStream]()

        if word_timestamps:
            if "preview" not in self._opts.model and (
                self._opts.language is not None
                and self._opts.language.language
                not in {
                    "en",
                    "de",
                    "es",
                    "fr",
                }
            ):
                # https://docs.cartesia.ai/api-reference/tts/compare-tts-endpoints
                logger.warning(
                    "word_timestamps is only supported for languages en, de, es, and fr with `sonic` models"
                    " or all languages with `preview` models"
                )

    class Markup(tts.TTS.Markup):
        # markup delegation lives in the base class, keyed on _provider_key()
        def _provider_key(self) -> str:
            return "cartesia"

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Cartesia"

    def prewarm(self) -> None:
        self._mux.prewarm()

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        voice: NotGivenOr[str | list[float]] = NOT_GIVEN,
        speed: NotGivenOr[TTSVoiceSpeed | float] = NOT_GIVEN,
        emotion: NotGivenOr[TTSVoiceEmotion | str | list[TTSVoiceEmotion | str]] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        pronunciation_dict_id: NotGivenOr[str] = NOT_GIVEN,
        api_version: NotGivenOr[str] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language, voice, speed,
        and emotion. If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-3".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Voice Control - Speed (https://docs.cartesia.ai/user-guides/voice-control)
            emotion (list[TTSVoiceEmotion], optional): Voice Control - Emotion (https://docs.cartesia.ai/user-guides/voice-control)
            pronunciation_dict_id (str, optional): The pronunciation dictionary ID to use for custom pronunciations.
            sentence_tokenizer (tokenize.SentenceTokenizer, optional): What splits the text into the
                chunks the TTS is sent.
            pacing (tts.SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or
                ``None`` to send each chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            opts = replace(self._opts)
            if is_given(model):
                opts.model = model
            if is_given(language):
                opts.language = LanguageCode(language) if language else None
            if is_given(voice):
                opts.voice = voice
            if is_given(speed):
                opts.speed = cast(TTSVoiceSpeed | float, speed)
            if is_given(emotion):
                emotion = [emotion] if isinstance(emotion, str) else emotion
                opts.emotion = emotion
            if is_given(volume):
                opts.volume = volume
            if is_given(pronunciation_dict_id):
                opts.pronunciation_dict_id = pronunciation_dict_id
            if is_given(api_version):
                opts.api_version = api_version

            if speed or emotion or volume or pronunciation_dict_id:
                _check_generation_config(opts)
        # cartesia_version is a query parameter on the websocket URL, so a shared socket keeps
        # speaking the version it was opened with while _to_cartesia_options starts shaping the
        # body for the new one.
        version_changed = opts.api_version != self._opts.api_version
        self._opts = opts
        if version_changed:
            self._mux.invalidate()

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
        stream = SynthesizeStream(opts=replace(self._opts), tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._mux.aclose()


def _check_generation_config(opts: _TTSOptions) -> None:
    if _is_sonic_3(opts.model):
        if opts.speed:
            if not isinstance(opts.speed, float):
                raise ValueError("speed must be a float for sonic-3")
            if not 0.6 <= opts.speed <= 2.0:
                logger.warning("speed must be between 0.6 and 2.0 for sonic-3")
        if opts.volume is not None and not 0.5 <= opts.volume <= 2.0:
            logger.warning("volume must be between 0.5 and 2.0 for sonic-3")
    elif (
        opts.api_version != API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS
        or opts.model != MODEL_ID_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS
    ):
        logger.warning(
            f"speed and emotion controls are only supported for model '{MODEL_ID_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS}', and API version '{API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS}', "
            "see https://docs.cartesia.ai/developer-tools/changelog for details",
            extra={"model": opts.model, "speed": opts.speed, "emotion": opts.emotion},
        )

    if opts.pronunciation_dict_id and not _is_sonic_3(opts.model):
        logger.warning(
            "pronunciation_dict_id is only supported for sonic-3 models",
            extra={"model": opts.model, "pronunciation_dict_id": opts.pronunciation_dict_id},
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the bytes endpoint"""

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
        body = {**_to_cartesia_options(self._opts, streaming=False), "transcript": self._input_text}

        async with self._get_session().post(
            self._opts.get_http_url("/tts/bytes"),
            headers={
                API_AUTH_HEADER: self._opts.api_key,
                API_VERSION_HEADER: self._opts.api_version,
                "User-Agent": USER_AGENT,
            },
            json=body,
            conn_options=self._conn_options,
        ) as resp:
            await resp.raise_for_status()

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )

            async for data in resp.iter_bytes():
                output_emitter.push(data)

            output_emitter.flush()


class _Mux(tts.WebSocketMultiplexer[_Message, _Speak | _Cancel]):
    """One Cartesia socket carries every stream's segments, each its own context."""

    max_session_duration = 300

    def __init__(self, owner: TTS) -> None:
        self._owner = owner

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        # the TTS's options as they are now: update_options replaces them
        opts = self._owner._opts
        return tts.WebSocketTarget(
            opts.get_ws_url("/tts/websocket"),
            headers={"User-Agent": USER_AGENT, API_AUTH_HEADER: opts.api_key},
            params={"cartesia_version": opts.api_version},
            request_id_header=REQUEST_ID_HEADER,
        )

    def route(self, message: _Chunk | _Timestamps | _Done | _Error) -> str:
        if "context_id" not in message:
            # an error naming no context fails every stream on the socket
            raise _error(message)
        return message["context_id"]

    def close_context(self, context_id: str) -> _Cancel:
        return _Cancel(context_id=context_id, cancel=True)


class SynthesizeStream(tts.WebSocketStream[_Message, _Speak | _Cancel]):
    def __init__(self, *, opts: _TTSOptions, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self._options = _to_cartesia_options(opts, streaming=True)
        self._aligned: dict[str, _Alignment] = {}

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Speak | None:
        match item:
            case tts.SegmentStart():
                self._aligned[item.segment_id] = _Alignment()
                return None
            case tts.SegmentText():
                self._aligned[item.segment_id].sent.append(item.chunk.said + " ")
                return self._speak(item.segment_id, item.chunk.text + " ", more=True)
            case tts.SegmentEnd():
                self._aligned[item.segment_id].sent.append(" ")
                return self._speak(item.segment_id, " ", more=False)
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Chunk | _Timestamps | _Done | _Error) -> None:
        if message["type"] == "chunk":
            audio = base64.b64decode(message["data"])
            self.report(tts.SegmentAudio(message["context_id"], audio))
        elif message["type"] == "timestamps":
            context_id, words = message["context_id"], message.get("word_timestamps")
            aligned = self._aligned.setdefault(context_id, _Alignment())
            self.report(tts.SegmentWords(context_id, aligned.words(words) if words else []))
        elif message["type"] == "done":
            self._aligned.pop(message["context_id"], None)
            self.report(tts.SegmentDone(message["context_id"]))
        elif message["type"] == "error":
            raise _error(message)
        else:
            self._skip_unknown_message(message)

    def _speak(self, context_id: str, transcript: str, *, more: bool) -> _Speak:
        return {
            **self._options,
            "max_buffer_delay_ms": 0,
            "context_id": context_id,
            "transcript": transcript,
            "continue": more,
        }


class _Alignment:
    """Cartesia's words matched to the text sent, so each keeps the text's punctuation and
    spacing; once a word does not match, the rest are taken as Cartesia spells them."""

    def __init__(self) -> None:
        self.sent = deque[str]()
        self._lost = False

    def words(self, timestamps: _WordTimestamps) -> list[TranscriptChunk]:
        chunks = []
        for word, start, end in zip(
            timestamps["words"], timestamps["start"], timestamps["end"], strict=False
        ):
            if not self.sent or self._lost:
                word = f"{word} "
                self._lost = True
            else:
                sent = self.sent.popleft()
                if (idx := sent.find(word)) != -1:
                    word, sent = sent[: idx + len(word)], sent[idx + len(word) :]
                    if sent.strip():
                        self.sent.appendleft(sent)
                    elif sent and self.sent:
                        # the remaining whitespace goes to the next sentence
                        self.sent[0] = sent + self.sent[0]
                else:
                    word = f"{word} "
                    self._lost = True
            chunks.append(TranscriptChunk(text=word, start_time=start, end_time=end))
        return chunks


def _error(message: _Error) -> APIStatusError:
    # an HTTP-style status: a 5xx is retried, a rejected request is not
    return APIStatusError(
        f"Cartesia returned error (cartesia_context_id={message.get('context_id')})",
        status_code=message.get("status_code") or 500,
        body=message,
        detail=message.get("message") or message.get("error"),
    )


def _to_cartesia_options(opts: _TTSOptions, *, streaming: bool) -> _Options:
    voice: dict[str, Any] = {}
    if isinstance(opts.voice, str):
        voice["mode"] = "id"
        voice["id"] = opts.voice
    else:
        voice["mode"] = "embedding"
        voice["embedding"] = opts.voice

    if opts.api_version == API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS:
        voice_controls: dict[str, float | str | list[str]] = {}
        if opts.speed:
            voice_controls["speed"] = opts.speed

        if opts.emotion:
            voice_controls["emotion"] = opts.emotion

        if voice_controls:
            voice["__experimental_controls"] = voice_controls

    options: _Options = {
        "model_id": opts.model,
        "voice": voice,
        "output_format": {
            "container": "raw",
            "encoding": AUDIO_ENCODING,
            "sample_rate": opts.sample_rate,
        },
        "language": opts.language.language if opts.language else None,
    }

    if opts.pronunciation_dict_id:
        options["pronunciation_dict_id"] = opts.pronunciation_dict_id

    if opts.api_version > API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS and _is_sonic_3(
        opts.model
    ):
        generation_config: dict[str, Any] = {}
        if opts.speed:
            generation_config["speed"] = opts.speed
        if opts.emotion:
            generation_config["emotion"] = opts.emotion[0]
        if opts.volume:
            generation_config["volume"] = opts.volume
        if generation_config:
            options["generation_config"] = generation_config

    if streaming:
        options["add_timestamps"] = opts.word_timestamps

    return options
