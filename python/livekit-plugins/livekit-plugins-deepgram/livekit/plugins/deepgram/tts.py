from __future__ import annotations

import os
import weakref
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

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

from ._utils import to_deepgram_url
from .log import logger
from .models import FluxTTSModels, TTSModels

BASE_URL = "https://api.deepgram.com/v1/speak"
# Flux models are served on /v2/speak, over both the websocket and REST
FLUX_BASE_URL = "https://api.deepgram.com/v2/speak"
NUM_CHANNELS = 1
# Speaking rate bounds accepted by the Deepgram TTS API; values outside are rejected
# with a 400, which on the WebSocket path surfaces only as a failed handshake.
MIN_SPEED = 0.7
MAX_SPEED = 1.5

_ENCODING_TO_MIMETYPE: dict[str, str] = {
    "linear16": "audio/pcm",
    "mulaw": "audio/pcmu",
    "alaw": "audio/pcma",
    "mp3": "audio/mp3",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
}
# the encodings Deepgram streams over the websocket
_STREAM_ENCODINGS = ("linear16", "mulaw", "alaw")


def _is_flux(model: str) -> bool:
    return model.startswith("flux-")


# Deepgram's speak websocket: Speak and Flush in, binary audio out. Aura ends a flushed turn with
# Flushed, Flux with SpeechMetadata (and a trailing Flushed it ignores). Each message declares
# only the fields read.


class _Flushed(TypedDict):
    type: Literal["Flushed"]
    billable_character_count: NotRequired[int]


class _SpeechMetadata(TypedDict):
    type: Literal["SpeechMetadata"]
    billable_character_count: NotRequired[int]


class _Warning(TypedDict):
    type: Literal["Warning"]
    description: NotRequired[str]
    warn_msg: NotRequired[str]


class _Error(TypedDict):
    type: Literal["Error", "error"]
    description: NotRequired[str]


_Message = bytes | _Flushed | _SpeechMetadata | _Warning | _Error


class _Speak(TypedDict):
    type: Literal["Speak"]
    text: str


class _Flush(TypedDict):
    type: Literal["Flush"]


class _Close(TypedDict):
    type: Literal["Close"]


_Command = _Speak | _Flush | _Close


@dataclass
class _TTSOptions:
    model: TTSModels | FluxTTSModels | str
    encoding: str
    sample_rate: int
    base_url: str | None
    api_key: str
    mip_opt_out: bool = False
    bit_rate: int | None = None
    speed: float | None = None

    @property
    def url(self) -> str:
        if self.base_url is not None:
            return self.base_url
        return FLUX_BASE_URL if _is_flux(self.model) else BASE_URL

    @property
    def mime_type(self) -> str:
        try:
            return _ENCODING_TO_MIMETYPE[self.encoding]
        except KeyError:
            raise ValueError(
                f"unsupported Deepgram TTS encoding {self.encoding!r}; "
                f"supported encodings are: {', '.join(_ENCODING_TO_MIMETYPE)}"
            ) from None


def _check_speed(model: str, speed: float | None) -> None:
    if speed is None:
        return
    if _is_flux(model):
        raise ValueError("speed is not supported by Flux models")
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, but got {speed}")


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | FluxTTSModels | str = "aura-2-andromeda-en",
        encoding: str = "linear16",
        sample_rate: int = 24000,
        bit_rate: int | None = None,
        speed: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: http.HttpSession | None = None,
        mip_opt_out: bool = False,
    ) -> None:
        """
        Create a new instance of Deepgram TTS.

        Args:
            model (TTSModels | FluxTTSModels | str): TTS model to use. Defaults to
                "aura-2-andromeda-en". ``flux-*`` models speak on the /v2/speak endpoint.
                See https://developers.deepgram.com/docs/tts-models for available models.
            encoding (str): Audio encoding to use. Defaults to "linear16". stream() supports
                "linear16", "mulaw" and "alaw"; compressed encodings are batch only.
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            bit_rate (int | None): Bit rate for compressed encodings (e.g. mp3). Defaults to None.
                See https://developers.deepgram.com/reference/text-to-speech-api#query-bit_rate
            speed (float | None): Speaking rate, from 0.7 to 1.5; 1.0 is the normal rate. Defaults to None,
                which leaves it out of the request. Deepgram supports it on Aura-2 English and Spanish
                voices; Flux models do not take it.
                See https://developers.deepgram.com/docs/tts-voice-controls
            api_key (str): Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY in environment.
            base_url (str | None): Endpoint of the Deepgram TTS API. Defaults to
                "https://api.deepgram.com/v1/speak", or "https://api.deepgram.com/v2/speak" for Flux.
            http_session (http.HttpSession): Optional HTTP session to use for requests.
            mip_opt_out (bool): Opt out of the Deepgram Model Improvement Program.
        """  # noqa: E501
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY or provide api_key.")
        _check_speed(model, speed)

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, text_unit="words", max_text_len=2000),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
            speed=speed,
            base_url=base_url,
            api_key=api_key,
            mip_opt_out=mip_opt_out,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._mux = _Mux(self._opts)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepgram"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | FluxTTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        bit_rate: NotGivenOr[int | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            model (TTSModels | FluxTTSModels | str): TTS model to use; ``flux-*`` speaks on
                /v2/speak.
            encoding (str): Audio encoding to use.
            sample_rate (int): Sample rate of audio in Hz.
            bit_rate (int | None): Bit rate for compressed encodings (e.g. mp3).
                See https://developers.deepgram.com/reference/text-to-speech-api#query-bit_rate
            speed (float | None): Speaking rate, from 0.7 to 1.5; None leaves it out of the request.
                See https://developers.deepgram.com/docs/tts-voice-controls
            sentence_tokenizer (tokenize.SentenceTokenizer): What splits the text into the chunks
                the TTS is sent.
            pacing (tts.SentenceStreamPacer | None): A ``SentenceStreamPacer``, or ``None`` to send
                each chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            _check_speed(
                model if is_given(model) else self._opts.model,
                speed if is_given(speed) else self._opts.speed,
            )
        changed = False
        if is_given(model):
            self._opts.model = model
            changed = True
        if is_given(encoding):
            self._opts.encoding = encoding
            changed = True
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            self._sample_rate = sample_rate  # keep base class property in sync
            changed = True
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
            changed = True
        if is_given(speed):
            self._opts.speed = speed
            changed = True
        if changed:
            # the options are baked into the websocket URL at connection time
            self._mux.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            opts=replace(self._opts), tts=self, input_text=chunk.text, conn_options=conn_options
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        if self._opts.encoding not in _STREAM_ENCODINGS:
            raise ValueError(
                f"Deepgram TTS streams {', '.join(_STREAM_ENCODINGS)}, got "
                f"{self._opts.encoding!r}; use synthesize() for compressed output."
            )
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._mux.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._mux.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, opts: _TTSOptions, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            http_params: dict[str, str | float | int | bool] = {
                "encoding": self._opts.encoding,
                "container": "none",
                "model": self._opts.model,
                "sample_rate": self._opts.sample_rate,
                "mip_opt_out": self._opts.mip_opt_out,
            }
            if self._opts.bit_rate is not None:
                http_params["bit_rate"] = self._opts.bit_rate
            if self._opts.speed is not None:
                http_params["speed"] = self._opts.speed
            async with self._tts._ensure_session().post(
                to_deepgram_url(http_params, self._opts.url, websocket=False),
                headers={
                    "Authorization": f"Token {self._opts.api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": self._input_text},
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=self._opts.mime_type,
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                if (billed := resp.headers.get("dg-char-count")) is not None:
                    output_emitter.report_usage(characters_count=int(billed))
                output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class _Mux(tts.WebSocketMultiplexer[_Message, _Command]):
    """A pool of Deepgram sockets, each carrying one stream at a time: the URL holds the options,
    so a socket is reused by the next stream."""

    routed = False
    max_session_duration = 3600

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    def goodbye(self) -> _Close:
        # ends the session on Deepgram's side, so none lingers against the concurrency limit
        return _Close(type="Close")

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        config: dict[str, str | float | int | bool] = {
            "encoding": self._opts.encoding,
            "model": self._opts.model,
            "sample_rate": self._opts.sample_rate,
            "mip_opt_out": self._opts.mip_opt_out,
        }
        if self._opts.speed is not None:
            config["speed"] = self._opts.speed
        return tts.WebSocketTarget(
            to_deepgram_url(config, self._opts.url, websocket=True),
            headers={"Authorization": f"Token {self._opts.api_key}"},
            request_id_header="dg-request-id",
        )


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self._turn_end = "SpeechMetadata" if _is_flux(tts._opts.model) else "Flushed"
        # Deepgram answers the flushed segments in order: audio is the oldest one's
        self._open = deque[str]()

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._open.append(item.segment_id)
                return None
            case tts.SegmentText():
                return _Speak(type="Speak", text=f"{item.chunk.text} ")
            case tts.SegmentEnd():
                return _Flush(type="Flush")
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if isinstance(message, bytes):
            if self._open:
                self.report(tts.SegmentAudio(self._open[0], message))
        elif message["type"] == "Flushed" or message["type"] == "SpeechMetadata":
            if message["type"] != self._turn_end or not self._open:
                return  # Flux's trailing Flushed
            segment_id = self._open.popleft()
            if (billed := message.get("billable_character_count")) is not None:
                self.report(tts.SegmentUsage(segment_id, characters_count=billed))
            self.report(tts.SegmentDone(segment_id))
        elif message["type"] == "Warning":
            logger.warning(
                "Deepgram warning: %s", message.get("description") or message.get("warn_msg")
            )
        else:  # Error
            raise APIError(
                "Deepgram TTS returned error",
                kind="invalid_request",
                body=message,
                detail=message.get("description"),
            )
