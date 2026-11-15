# Copyright 2025 LiveKit, Inc.
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
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, NotRequired, TypedDict, assert_never, cast, get_args
from urllib.parse import urljoin

from pydantic import TypeAdapter

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    APIConnectOptions,
    ErrorKind,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from .log import logger
from .version import __version__

USER_AGENT = f"livekit-agents-py/{__version__}"

DEFAULT_BIT_RATE = 64000
DEFAULT_ENCODING: Encoding = "PCM"
DEFAULT_MODEL = "inworld-tts-1.5-max"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_URL = "https://api.inworld.ai/"
DEFAULT_WS_URL = "wss://api.inworld.ai/"
DEFAULT_VOICE = "Ashley"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SPEAKING_RATE = 1.0
DEFAULT_BUFFER_CHAR_THRESHOLD = 120
DEFAULT_MAX_BUFFER_DELAY_MS = 3000
NUM_CHANNELS = 1

# Inworld reports errors as gRPC status codes; an unlisted code refused the request
_GRPC_ERROR_KINDS: dict[int, ErrorKind] = {
    4: "timeout",  # DEADLINE_EXCEEDED
    7: "auth",  # PERMISSION_DENIED
    8: "rate_limit",  # RESOURCE_EXHAUSTED
    13: "server",  # INTERNAL
    14: "server",  # UNAVAILABLE
    16: "auth",  # UNAUTHENTICATED
}


def _grpc_error_kind(code: object) -> ErrorKind:
    if not isinstance(code, int):
        return "invalid_request"
    return _GRPC_ERROR_KINDS.get(code, "invalid_request")


TTSModels = Literal["inworld-tts-2", "inworld-tts-1.5-max"]
Encoding = Literal["LINEAR16", "PCM", "MP3", "OGG_OPUS", "FLAC"]
TimestampType = Literal["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"]
_TextNormalizationStr = Literal["APPLY_TEXT_NORMALIZATION_UNSPECIFIED", "ON", "OFF"]
TextNormalization = _TextNormalizationStr | bool
TimestampTransportStrategy = Literal["TIMESTAMP_TRANSPORT_STRATEGY_UNSPECIFIED", "SYNC", "ASYNC"]
DeliveryMode = Literal["DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", "CREATIVE"]

DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY: TimestampTransportStrategy = "ASYNC"


def _validate_str_param(value: object, name: str, literal_type: object) -> None:
    valid = get_args(literal_type)
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"Invalid {name}: {value!r}. Must be one of {sorted(valid)}")


def _resolve_text_normalization(value: object) -> _TextNormalizationStr:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    _validate_str_param(value, "text_normalization", _TextNormalizationStr)
    return cast(_TextNormalizationStr, value)


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: Encoding
    voice: str
    sample_rate: int
    bit_rate: int
    speaking_rate: float
    temperature: float
    language: NotGivenOr[str] = NOT_GIVEN
    timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN
    text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN
    delivery_mode: NotGivenOr[DeliveryMode] = NOT_GIVEN
    timestamp_transport_strategy: TimestampTransportStrategy = DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY
    buffer_char_threshold: int = DEFAULT_BUFFER_CHAR_THRESHOLD
    max_buffer_delay_ms: int = DEFAULT_MAX_BUFFER_DELAY_MS

    @property
    def mime_type(self) -> str:
        if self.encoding == "PCM":
            return "audio/pcm"
        elif self.encoding == "MP3":
            return "audio/mpeg"
        elif self.encoding == "OGG_OPUS":
            return "audio/ogg"
        elif self.encoding == "FLAC":
            return "audio/flac"
        else:
            return "audio/wav"


# Inworld's bidirectional websocket: each segment is one context, created with the stream's
# options, closed once its text ended. Each message declares only the fields read.


class _Status(TypedDict):
    code: NotRequired[int]
    message: NotRequired[str]


class _WordAlignment(TypedDict):
    words: NotRequired[list[str]]
    wordStartTimeSeconds: NotRequired[list[float]]
    wordEndTimeSeconds: NotRequired[list[float]]


class _CharacterAlignment(TypedDict):
    characters: NotRequired[list[str]]
    characterStartTimeSeconds: NotRequired[list[float]]
    characterEndTimeSeconds: NotRequired[list[float]]


class _TimestampInfo(TypedDict):
    wordAlignment: NotRequired[_WordAlignment]
    characterAlignment: NotRequired[_CharacterAlignment]


class _Usage(TypedDict):
    processedCharactersCount: NotRequired[int]


class _AudioChunk(TypedDict):
    audioContent: NotRequired[str]
    timestampInfo: NotRequired[_TimestampInfo]
    usage: NotRequired[_Usage]


class _Result(TypedDict):
    contextId: NotRequired[str]
    status: NotRequired[_Status]
    audioChunk: NotRequired[_AudioChunk]
    flushCompleted: NotRequired[dict[str, Any]]
    contextClosed: NotRequired[dict[str, Any]]


class _Message(TypedDict):
    result: NotRequired[_Result]
    error: NotRequired[_Status]


class _Text(TypedDict):
    text: str


class _Create(TypedDict):
    create: dict[str, Any]
    contextId: str


class _SendText(TypedDict):
    send_text: _Text
    contextId: str


class _Flush(TypedDict):
    flush_context: dict[str, Any]
    contextId: str


class _Close(TypedDict):
    close_context: dict[str, Any]
    contextId: str


class _Cancel(TypedDict):
    cancel_context: dict[str, Any]
    contextId: str


_Command = _Create | _SendText | _Flush | _Close | _Cancel


class _Mux(tts.WebSocketMultiplexer[_Message, _Command]):
    """One Inworld socket carries up to five contexts."""

    max_contexts = 5

    def __init__(self, ws_url: str, authorization: str) -> None:
        self._ws_url = ws_url
        self._authorization = authorization

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            urljoin(self._ws_url, "/tts/v1/voice:streamBidirectional").replace("http", "ws", 1),
            headers={
                "Authorization": self._authorization,
                "X-User-Agent": USER_AGENT,
                "X-Request-Id": str(uuid.uuid4()),
            },
        )

    def route(self, message: _Message) -> str | None:
        if "result" not in message:
            # an error naming no context is the socket's, and fails none of its streams
            logger.warning(
                "Received error from Inworld",
                extra={
                    "error_code": message.get("error", {}).get("code", 0),
                    "error_message": message.get("error", {}).get("message", "Unknown error"),
                },
            )
            return None
        return message["result"].get("contextId")

    def close_context(self, context_id: str) -> _Cancel:
        return _Cancel(cancel_context={}, contextId=context_id)


@dataclass
class _Segment:
    """One context's timeline: the server restarts its timestamps after each generation (a
    flush in auto mode), so each generation is offset by where the audio before it ended."""

    pcm_byte_rate: int
    cumulative_time: float = 0.0
    generation_end_time: float = 0.0
    audio_time: float = 0.0
    # on the websocket, `usage.processedCharactersCount` is the characters processed so far
    billed_characters: int | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        delivery_mode: NotGivenOr[DeliveryMode] = NOT_GIVEN,
        timestamp_transport_strategy: NotGivenOr[TimestampTransportStrategy] = NOT_GIVEN,
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        base_url: str = DEFAULT_URL,
        ws_url: str = DEFAULT_WS_URL,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Create a new instance of Inworld TTS.

        Args:
            api_key (str, optional): The Inworld API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            voice (str, optional): The voice to use. Defaults to "Ashley".
            model (str, optional): The Inworld model to use. Defaults to "inworld-tts-1.5-max".
            encoding (str, optional): The encoding to use. Defaults to "PCM".
            bit_rate (int, optional): Bits per second of the audio. Defaults to 64000.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            speaking_rate (float, optional): The speed of the voice, in the range [0.5, 1.5].
                Defaults to 1.0.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response. Range (0, 2]. Defaults to 1.0.
            language (str, optional): BCP-47 language tag (e.g., "en-US", "fr-FR", "ja-JP")
                specifying the language that the given voice should speak the text in.
                If not set, the model default applies.
            timestamp_type (str, optional): Controls timestamp metadata returned with the audio.
                Use "WORD" for word-level timestamps or "CHARACTER" for character-level.
                Useful for karaoke-style captions, word highlighting, and lipsync.
            text_normalization (str, optional): Controls text normalization. When "ON", numbers,
                dates, and abbreviations are expanded (e.g., "Dr." -> "Doctor"). When "OFF",
                text is read exactly as written. Defaults to automatic.
            delivery_mode (str, optional): Controls output variation on ``inworld-tts-2`` only.
                One of "DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", or "CREATIVE".
                The Inworld API ignores ``temperature`` on ``inworld-tts-2`` — use
                ``delivery_mode`` to steer output variation on that model instead.
                Defaults to the server-side default ("BALANCED").
            timestamp_transport_strategy (str, optional): Controls how timestamp info is
                transported relative to audio data. "SYNC" returns timestamps in the same
                message as audio data. "ASYNC" allows timestamps to return in trailing
                messages after the audio data. Defaults to "ASYNC".
            buffer_char_threshold (int, optional): For streaming, the minimum number of characters
                in the buffer that automatically triggers audio generation. Defaults to 1000.
            max_buffer_delay_ms (int, optional): For streaming, the maximum time in ms to buffer
                before starting generation. Defaults to 3000.
            base_url (str, optional): The base URL for the Inworld TTS API.
                Defaults to "https://api.inworld.ai/".
            ws_url (str, optional): The WebSocket URL for streaming TTS.
                Defaults to "wss://api.inworld.ai/".
            http_session (http.HttpSession, optional): The HTTP session to use.
        """
        if not is_given(sample_rate):
            sample_rate = DEFAULT_SAMPLE_RATE
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=is_given(timestamp_type)
                and timestamp_type != "TIMESTAMP_TYPE_UNSPECIFIED",
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        key = api_key if is_given(api_key) else os.getenv("INWORLD_API_KEY")
        if not key:
            raise ValueError(
                "Inworld API key is required, either as argument or set"
                " INWORLD_API_KEY environment variable"
            )

        self._authorization = f"Basic {key}"
        self._base_url = base_url
        self._ws_url = ws_url
        self._session = http_session

        if is_given(encoding):
            _validate_str_param(encoding, "encoding", Encoding)
        if is_given(timestamp_type):
            _validate_str_param(timestamp_type, "timestamp_type", TimestampType)
        if is_given(text_normalization):
            text_normalization = _resolve_text_normalization(text_normalization)
        if is_given(delivery_mode):
            _validate_str_param(delivery_mode, "delivery_mode", DeliveryMode)
        if is_given(timestamp_transport_strategy):
            _validate_str_param(
                timestamp_transport_strategy,
                "timestamp_transport_strategy",
                TimestampTransportStrategy,
            )

        self._opts = _TTSOptions(
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            model=model if is_given(model) else DEFAULT_MODEL,
            encoding=encoding if is_given(encoding) else DEFAULT_ENCODING,
            bit_rate=bit_rate if is_given(bit_rate) else DEFAULT_BIT_RATE,
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            speaking_rate=speaking_rate if is_given(speaking_rate) else DEFAULT_SPEAKING_RATE,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            language=language,
            timestamp_type=timestamp_type,
            text_normalization=text_normalization,
            delivery_mode=delivery_mode,
            timestamp_transport_strategy=timestamp_transport_strategy
            if is_given(timestamp_transport_strategy)
            else DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY,
            buffer_char_threshold=buffer_char_threshold
            if is_given(buffer_char_threshold)
            else DEFAULT_BUFFER_CHAR_THRESHOLD,
            max_buffer_delay_ms=max_buffer_delay_ms
            if is_given(max_buffer_delay_ms)
            else DEFAULT_MAX_BUFFER_DELAY_MS,
        )

        self._mux = _Mux(ws_url, self._authorization)
        self._streams = weakref.WeakSet[SynthesizeStream]()

    class Markup(tts.TTS.Markup):
        # markup delegation lives in the base class, keyed on _provider_key()
        def _provider_key(self) -> str:
            # only inworld-tts-2 understands the markup tags; older models get no
            # markup so the tags aren't injected, converted, or stripped (matches
            # the inference gateway's behavior)
            return "inworld" if "tts-2" in self._tts.model else ""

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Inworld"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        delivery_mode: NotGivenOr[DeliveryMode] = NOT_GIVEN,
        timestamp_transport_strategy: NotGivenOr[TimestampTransportStrategy] = NOT_GIVEN,
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS configuration options.

        Args:
            voice (str, optional): The voice to use.
            model (str, optional): The Inworld model to use.
            encoding (str, optional): The encoding to use.
            bit_rate (int, optional): Bits per second of the audio.
            sample_rate (int, optional): The audio sample rate in Hz.
            speaking_rate (float, optional): The speed of the voice.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response.
            language (str, optional): BCP-47 language tag (e.g., "en-US", "fr-FR").
            timestamp_type (str, optional): Controls timestamp metadata ("WORD" or "CHARACTER").
            text_normalization (str, optional): Controls text normalization ("ON" or "OFF").
            delivery_mode (str, optional): Controls output variation on ``inworld-tts-2`` only
                ("DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", or "CREATIVE"). The Inworld
                API ignores ``temperature`` on TTS-2; use this instead.
            timestamp_transport_strategy (str, optional): Controls timestamp transport strategy
                ("SYNC" or "ASYNC").
            buffer_char_threshold (int, optional): For streaming, min characters before triggering.
            max_buffer_delay_ms (int, optional): For streaming, max time to buffer.
            sentence_tokenizer (tokenize.SentenceTokenizer, optional): What splits the text into the
                chunks the TTS is sent.
            pacing (tts.SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or
                ``None`` to send each chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(encoding):
                _validate_str_param(encoding, "encoding", Encoding)
            if is_given(timestamp_type):
                _validate_str_param(timestamp_type, "timestamp_type", TimestampType)
            normalization = (
                _resolve_text_normalization(text_normalization)
                if is_given(text_normalization)
                else None
            )
            if is_given(delivery_mode):
                _validate_str_param(delivery_mode, "delivery_mode", DeliveryMode)
            if is_given(timestamp_transport_strategy):
                _validate_str_param(
                    timestamp_transport_strategy,
                    "timestamp_transport_strategy",
                    TimestampTransportStrategy,
                )
        if is_given(voice):
            self._opts.voice = voice
        if is_given(model):
            self._opts.model = model
        if is_given(encoding):
            self._opts.encoding = encoding
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            # keep the base class property in sync; the framework reads tts.sample_rate
            self._sample_rate = sample_rate
        if is_given(speaking_rate):
            self._opts.speaking_rate = speaking_rate
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(language):
            self._opts.language = language
        if is_given(timestamp_type):
            self._opts.timestamp_type = timestamp_type
        if normalization is not None:
            self._opts.text_normalization = normalization
        if is_given(delivery_mode):
            self._opts.delivery_mode = delivery_mode
        if is_given(timestamp_transport_strategy):
            self._opts.timestamp_transport_strategy = timestamp_transport_strategy
        if is_given(buffer_char_threshold):
            self._opts.buffer_char_threshold = buffer_char_threshold
        if is_given(max_buffer_delay_ms):
            self._opts.max_buffer_delay_ms = max_buffer_delay_ms

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def prewarm(self) -> None:
        self._mux.prewarm()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return ChunkedStream(
            tts=self,
            opts=replace(self._opts),
            get_session=self._ensure_session,
            base_url=self._base_url,
            authorization=self._authorization,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            opts=replace(self._opts),
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._mux.aclose()

    async def list_voices(self, language: str | None = None) -> list[dict[str, object]]:
        """
        List all available voices in the workspace associated with the API key.

        Args:
            language (str, optional): ISO 639-1 language code to filter voices (e.g., 'en', 'es', 'fr').
        """
        url = urljoin(self._base_url, "tts/v1/voices")
        params: dict[str, str] = {}
        if language:
            params["filter"] = f"language={language}"

        async with self._ensure_session().get(
            url,
            headers={
                "Authorization": self._authorization,
                "X-User-Agent": USER_AGENT,
                "X-Request-Id": str(uuid.uuid4()),
            },
            params=params,
        ) as resp:
            if not resp.ok:
                error_body = cast(dict[str, Any], await resp.json())
                raise APIStatusError(
                    "Inworld list voices failed",
                    status_code=resp.status,
                    request_id=None,
                    body=None,
                    detail=error_body.get("message"),
                )

            data = cast(dict[str, Any], await resp.json())
            return TypeAdapter(list[dict[str, object]]).validate_python(data.get("voices", []))


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        get_session: Callable[[], http.HttpSession],
        base_url: str,
        authorization: str,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._get_session = get_session
        self._base_url = base_url
        self._authorization = authorization

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        x_request_id = str(uuid.uuid4())
        try:
            audio_config: dict[str, Any] = {
                "audioEncoding": self._opts.encoding,
                "bitrate": self._opts.bit_rate,
                "sampleRateHertz": self._opts.sample_rate,
                "temperature": self._opts.temperature,
                "speakingRate": self._opts.speaking_rate,
            }

            body_params: dict[str, Any] = {
                "text": self._input_text,
                "voiceId": self._opts.voice,
                "modelId": self._opts.model,
                "audioConfig": audio_config,
            }
            if utils.is_given(self._opts.language):
                body_params["language"] = self._opts.language
            if utils.is_given(self._opts.timestamp_type):
                body_params["timestampType"] = self._opts.timestamp_type
            if utils.is_given(self._opts.text_normalization):
                body_params["applyTextNormalization"] = self._opts.text_normalization
            if utils.is_given(self._opts.delivery_mode):
                body_params["deliveryMode"] = self._opts.delivery_mode
            body_params["timestampTransportStrategy"] = self._opts.timestamp_transport_strategy

            async with self._get_session().post(
                urljoin(self._base_url, "/tts/v1/voice:stream"),
                headers={
                    "Authorization": self._authorization,
                    "X-User-Agent": USER_AGENT,
                    "X-Request-Id": x_request_id,
                },
                json=body_params,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                request_id = utils.shortuuid()
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=self._opts.mime_type,
                )

                async for raw_line in resp.iter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        data = None
                    if not isinstance(data, dict):
                        raise APIError(
                            "Inworld TTS sent a malformed response line",
                            kind="protocol",
                            detail=str(line),
                        )

                    if result := data.get("result"):
                        # Handle timestamp info if present
                        if timestamp_info := result.get("timestampInfo"):
                            timed_strings = _parse_timestamp_info(timestamp_info)
                            if timed_strings:
                                output_emitter.push_timed_transcript(timed_strings)

                        # HTTP streaming reports per message: the first carries the
                        # request's count, the rest zero, so the reports add up.
                        if usage := result.get("usage"):
                            output_emitter.report_usage(
                                characters_count=usage.get("processedCharactersCount", 0)
                            )
                        if audio_content := result.get("audioContent"):
                            output_emitter.push(base64.b64decode(audio_content))
                            if self._opts.encoding == "PCM":
                                output_emitter.flush()
                    elif error := data.get("error"):
                        raise APIError(
                            "Inworld TTS error",
                            kind=_grpc_error_kind(error.get("code")),
                            detail=error.get("message"),
                        )
        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self.mime_type = opts.mime_type
        self._opts = opts
        self._contexts: dict[str, _Segment] = {}

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                rate = self._opts.sample_rate * 2 * NUM_CHANNELS
                pcm = self._opts.encoding in ("PCM", "LINEAR16")
                self._contexts[item.segment_id] = _Segment(pcm_byte_rate=rate if pcm else 0)
                return _Create(create=_create_config(self._opts), contextId=item.segment_id)
            case tts.SegmentText():
                text = _Text(text=item.chunk.text + " ")
                return _SendText(send_text=text, contextId=item.segment_id)
            case tts.SegmentEnd():
                self.send_now(_Flush(flush_context={}, contextId=item.segment_id))
                return _Close(close_context={}, contextId=item.segment_id)
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        result = message.get("result", {})
        segment_id = result.get("contextId", "")
        status = result.get("status", {})
        if status.get("code", 0) != 0:
            raise APIError(
                "Inworld TTS context error",
                kind=_grpc_error_kind(status.get("code")),
                detail=status.get("message"),
            )
        segment = self._contexts.get(segment_id)
        if segment is None:
            return
        if (chunk := result.get("audioChunk")) is not None:
            if timestamp_info := chunk.get("timestampInfo"):
                words = _parse_timestamp_info(
                    cast(dict[str, Any], timestamp_info), cumulative_time=segment.cumulative_time
                )
                if words and words[-1].end_time is not None:
                    segment.generation_end_time = words[-1].end_time
                self.report(tts.SegmentWords(segment_id, words))
            if usage := chunk.get("usage"):
                segment.billed_characters = max(
                    segment.billed_characters or 0, usage.get("processedCharactersCount", 0)
                )
            if audio_content := chunk.get("audioContent"):
                audio = base64.b64decode(audio_content)
                if segment.pcm_byte_rate:
                    pcm_len = len(audio) - (44 if audio[:4] == b"RIFF" else 0)
                    segment.audio_time += pcm_len / segment.pcm_byte_rate
                self.report(tts.SegmentAudio(segment_id, audio))
        if "flushCompleted" in result:
            # the next generation's timestamps restart at 0
            segment.cumulative_time = (
                segment.audio_time if segment.pcm_byte_rate else segment.generation_end_time
            )
        if "contextClosed" in result:
            del self._contexts[segment_id]
            if segment.billed_characters is not None:
                billed = tts.SegmentUsage(segment_id, characters_count=segment.billed_characters)
                self.report(billed)
            self.report(tts.SegmentDone(segment_id))


def _create_config(opts: _TTSOptions) -> dict[str, Any]:
    config: dict[str, Any] = {
        "voiceId": opts.voice,
        "modelId": opts.model,
        "audioConfig": {
            "audioEncoding": opts.encoding,
            "sampleRateHertz": opts.sample_rate,
            "bitrate": opts.bit_rate,
            "speakingRate": opts.speaking_rate,
        },
        "temperature": opts.temperature,
        "bufferCharThreshold": opts.buffer_char_threshold,
        "maxBufferDelayMs": opts.max_buffer_delay_ms,
        "timestampTransportStrategy": opts.timestamp_transport_strategy,
        # each chunk is a sentence of the framework's tokenizer
        "autoMode": True,
    }
    if is_given(opts.language):
        config["language"] = opts.language
    if is_given(opts.timestamp_type):
        config["timestampType"] = opts.timestamp_type
    if is_given(opts.text_normalization):
        config["applyTextNormalization"] = opts.text_normalization
    if is_given(opts.delivery_mode):
        config["deliveryMode"] = opts.delivery_mode
    return config


def _parse_timestamp_info(
    timestamp_info: dict[str, Any], cumulative_time: float = 0.0
) -> list[TranscriptChunk]:
    """Parse timestamp info from API response into TranscriptChunk objects.

    Args:
        timestamp_info: The timestamp info from the API response.
        cumulative_time: Offset to add to all timestamps for monotonic ordering
            across multiple generations within a single context.
    """
    timed_strings: list[TranscriptChunk] = []

    # Handle word-level alignment
    if word_align := timestamp_info.get("wordAlignment"):
        words = word_align.get("words", [])
        starts = word_align.get("wordStartTimeSeconds", [])
        ends = word_align.get("wordEndTimeSeconds", [])

        for word, start, end in zip(words, starts, ends, strict=False):
            text = f"{word}"
            timed_strings.append(
                TranscriptChunk(
                    text,
                    start_time=cumulative_time + start,
                    end_time=cumulative_time + end,
                )
            )

    # Character-level alignment, grouped into words (each word keeps its trailing space)
    elif char_align := timestamp_info.get("characterAlignment"):
        chars = char_align.get("characters", [])
        starts = char_align.get("characterStartTimeSeconds", [])
        ends = char_align.get("characterEndTimeSeconds", [])

        word, word_start, word_end = "", 0.0, 0.0
        for char, start, end in zip(chars, starts, ends, strict=False):
            if not word:
                word_start = start
            word += char
            word_end = end
            if char.isspace():
                timed_strings.append(
                    TranscriptChunk(
                        word,
                        start_time=cumulative_time + word_start,
                        end_time=cumulative_time + word_end,
                    )
                )
                word = ""
        if word:
            timed_strings.append(
                TranscriptChunk(
                    word,
                    start_time=cumulative_time + word_start,
                    end_time=cumulative_time + word_end,
                )
            )

    return timed_strings
