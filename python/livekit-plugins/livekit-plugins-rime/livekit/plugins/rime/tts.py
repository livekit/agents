# Copyright 202 LiveKit, Inc.
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

import os
import weakref
from dataclasses import dataclass, replace
from typing import NotRequired, TypedDict, assert_never

from rime_api import text_to_speech_pb2 as proto

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    http,
    redacted_cause,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import _websocket_v1 as v1
from .langs import TTSLangs
from .models import (
    MODEL_CODA,
    MODEL_MIST_V2,
    MODEL_MIST_V3,
    WEBSOCKET_URLS,
    DefaultCodaVoice,
    DefaultMistVoice,
    TTSModels,
    is_mist_model,
    supports_time_scale_factor,
)

RIME_BASE_URL = "https://users.rime.ai/v1/rime-tts"
NUM_CHANNELS = 1
DEFAULT_AUDIO_FORMAT: v1.RimeAudioFormat = "audio/pcm"


@dataclass
class _TTSOptions:
    model: str
    speaker: str
    language: str
    audio_format: v1.RimeAudioFormat
    sample_rate: int | None
    time_scale_factor: float | None
    pause_between_brackets: bool | None
    phonemize_between_brackets: bool | None
    # HTTP synthesis only
    repetition_penalty: float | None
    temperature: float | None
    top_p: float | None
    max_tokens: int | None


class _HttpRequest(TypedDict):
    speaker: str
    text: str
    modelId: str
    lang: str
    samplingRate: int
    timeScaleFactor: NotRequired[float]
    repetition_penalty: NotRequired[float]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    max_tokens: NotRequired[int]
    pauseBetweenBrackets: NotRequired[bool]
    phonemizeBetweenBrackets: NotRequired[bool]


def _default_sample_rate(model: str) -> int:
    return 22050 if model == MODEL_MIST_V2 else 24000


def _endpoint(
    model: str | None, websocket_url: str | None, *, default_model: str, allow_custom: bool
) -> tuple[str, str | None]:
    """The model and its WebSocket v1 endpoint (``None``: the model does not stream). A
    ``/{model}/ws`` URL names its model; a ``/ws`` endpoint takes ``model``."""
    if websocket_url is not None:
        named = v1.model_from_websocket_url(websocket_url, allow_custom_endpoint=allow_custom)
        if named is None:
            return model or default_model, websocket_url
        named = MODEL_MIST_V3 if named == "mist" else named
        if model is not None and model != named:
            raise ValueError(f"websocket_url serves {named!r}, not model {model!r}")
        return named, websocket_url
    model = model or default_model
    return model, WEBSOCKET_URLS.get(model)


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        audio_format: v1.RimeAudioFormat = DEFAULT_AUDIO_FORMAT,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        websocket_url: NotGivenOr[str] = NOT_GIVEN,
        websocket_protocol: v1.WebSocketProtocol = "binary",
        base_url: str = RIME_BASE_URL,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: http.HttpSession | None = None,
        allow_custom_endpoint: bool = False,
    ) -> None:
        """Rime TTS: ``stream()`` speaks over the WebSocket v1 API, ``synthesize()`` over HTTP.

        Args:
            model: ``coda`` (the default) or ``mistv3`` stream; ``mistv2`` synthesizes over
                HTTP only.
            speaker: The voice; ``lyra`` for Coda, ``cove`` for Mist by default.
            lang: The language code.
            audio_format: The streamed audio's format (HTTP synthesis returns PCM).
            sample_rate: The audio's rate; the model's default if not given.
            time_scale_factor: The speed; not supported by ``mistv2``.
            pause_between_brackets, phonemize_between_brackets: Mist bracket options.
            repetition_penalty, temperature, top_p, max_tokens: Coda generation controls, for
                HTTP synthesis only.
            websocket_url: The WebSocket v1 endpoint; the model's public endpoint by default. A
                ``/{model}/ws`` path names its model; a dedicated ``/ws`` endpoint takes
                ``model``.
            websocket_protocol: ``binary`` (protobuf) or ``json`` frames.
            base_url: The HTTP endpoint.
            api_key: The Rime API key, or ``RIME_API_KEY``.
            http_session: The HTTP session; the job's shared session by default.
            allow_custom_endpoint: Send the API key to a host outside ``rime.ai``.
        """
        resolved_model, url = _endpoint(
            model if is_given(model) else None,
            websocket_url if is_given(websocket_url) else None,
            default_model=MODEL_CODA,
            allow_custom=allow_custom_endpoint,
        )
        v1.validate_endpoint_host(base_url, allow_custom_endpoint=allow_custom_endpoint)
        if not v1.is_audio_format(audio_format):
            raise ValueError(f"unsupported Rime audio_format {audio_format!r}")
        if url is None and audio_format != DEFAULT_AUDIO_FORMAT:
            raise ValueError(f"{resolved_model} does not stream: audio_format does not apply")
        _check_model_options(
            resolved_model,
            time_scale_factor=is_given(time_scale_factor),
            brackets=is_given(pause_between_brackets) or is_given(phonemize_between_brackets),
        )
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=url is not None),
            sample_rate=sample_rate
            if is_given(sample_rate)
            else _default_sample_rate(resolved_model),
            num_channels=NUM_CHANNELS,
        )
        resolved_api_key = api_key if is_given(api_key) else os.environ.get("RIME_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Rime API key is required, either as argument or set RIME_API_KEY environmental variable"  # noqa: E501
            )
        self._api_key = resolved_api_key
        self._allow_custom_endpoint = allow_custom_endpoint
        self._base_url = base_url
        self._session = http_session
        self._custom_websocket_url = websocket_url if is_given(websocket_url) else None
        self._opts = _TTSOptions(
            model=resolved_model,
            speaker=speaker if is_given(speaker) else _default_speaker(resolved_model),
            language=lang,
            audio_format=audio_format,
            sample_rate=sample_rate if is_given(sample_rate) else None,
            time_scale_factor=time_scale_factor if is_given(time_scale_factor) else None,
            pause_between_brackets=(
                pause_between_brackets if is_given(pause_between_brackets) else None
            ),
            phonemize_between_brackets=(
                phonemize_between_brackets if is_given(phonemize_between_brackets) else None
            ),
            repetition_penalty=repetition_penalty if is_given(repetition_penalty) else None,
            temperature=temperature if is_given(temperature) else None,
            top_p=top_p if is_given(top_p) else None,
            max_tokens=max_tokens if is_given(max_tokens) else None,
        )
        self._mux = (
            _Mux(url, protocol=websocket_protocol, api_key=self._api_key)
            if url is not None
            else None
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Rime"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def prewarm(self) -> None:
        if self._mux is not None:
            self._mux.prewarm()

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        if self._mux is None:
            raise NotImplementedError(f"Rime {self._opts.model} does not stream")
        stream = SynthesizeStream(
            tts=self, mux=self._mux, opts=replace(self._opts), conn_options=conn_options
        )
        self._streams.add(stream)
        return stream

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self, input_text=chunk.text, opts=replace(self._opts), conn_options=conn_options
        )

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._mux is not None:
            await self._mux.aclose()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        audio_format: NotGivenOr[v1.RimeAudioFormat] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        websocket_url: NotGivenOr[str] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Change the options of the next synthesis. A model that would change whether the TTS
        streams is refused, and a refused update changes nothing."""
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            custom_url = websocket_url if is_given(websocket_url) else self._custom_websocket_url
            new_model, url = _endpoint(
                model if is_given(model) else None,
                custom_url,
                default_model=self._opts.model,
                allow_custom=self._allow_custom_endpoint,
            )
            if (url is not None) != self.capabilities.streaming:
                raise ValueError(f"Rime {new_model} would change whether this TTS streams")
            if is_given(audio_format) and not v1.is_audio_format(audio_format):
                raise ValueError(f"unsupported Rime audio_format {audio_format!r}")
            brackets = is_given(pause_between_brackets) or is_given(phonemize_between_brackets)
            _check_model_options(
                new_model,
                time_scale_factor=is_given(time_scale_factor)
                or self._opts.time_scale_factor is not None,
                brackets=brackets,
            )
            if is_given(base_url):
                v1.validate_endpoint_host(
                    base_url, allow_custom_endpoint=self._allow_custom_endpoint
                )

        if is_given(base_url):
            self._base_url = base_url

        opts = self._opts
        if new_model != opts.model and not is_given(speaker):
            opts.speaker = _default_speaker(new_model)
        opts.model = new_model
        self._custom_websocket_url = custom_url
        if self._mux is not None and url is not None:
            self._mux.update_endpoint(url)
        if is_given(speaker):
            opts.speaker = speaker
        if is_given(lang):
            opts.language = lang
        if is_given(sample_rate):
            opts.sample_rate = sample_rate
        if is_given(audio_format):
            opts.audio_format = audio_format
        if is_given(time_scale_factor):
            opts.time_scale_factor = time_scale_factor
        if is_given(pause_between_brackets):
            opts.pause_between_brackets = pause_between_brackets
        if is_given(phonemize_between_brackets):
            opts.phonemize_between_brackets = phonemize_between_brackets
        if is_given(repetition_penalty):
            opts.repetition_penalty = repetition_penalty
        if is_given(temperature):
            opts.temperature = temperature
        if is_given(top_p):
            opts.top_p = top_p
        if is_given(max_tokens):
            opts.max_tokens = max_tokens
        self._sample_rate = opts.sample_rate or _default_sample_rate(opts.model)


def _default_speaker(model: str) -> str:
    return DefaultMistVoice if is_mist_model(model) else DefaultCodaVoice


def _check_model_options(model: str, *, time_scale_factor: bool, brackets: bool) -> None:
    if time_scale_factor and not supports_time_scale_factor(model):
        raise ValueError("time_scale_factor is not supported by the mistv2 model")
    if brackets and not is_mist_model(model):
        raise ValueError("pause_between_brackets and phonemize_between_brackets need a Mist model")


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a whole text over Rime's HTTP API."""

    def __init__(
        self, *, tts: TTS, input_text: str, opts: _TTSOptions, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts, sample_rate = self._opts, self._tts.sample_rate
        payload = _HttpRequest(
            speaker=opts.speaker,
            text=self._input_text,
            modelId=opts.model,
            lang=opts.language,
            samplingRate=sample_rate,
        )
        if opts.time_scale_factor is not None:
            payload["timeScaleFactor"] = opts.time_scale_factor
        if opts.model == MODEL_CODA:
            if opts.repetition_penalty is not None:
                payload["repetition_penalty"] = opts.repetition_penalty
            if opts.temperature is not None:
                payload["temperature"] = opts.temperature
            if opts.top_p is not None:
                payload["top_p"] = opts.top_p
            if opts.max_tokens is not None:
                payload["max_tokens"] = opts.max_tokens
        if opts.pause_between_brackets is not None:
            payload["pauseBetweenBrackets"] = opts.pause_between_brackets
        if opts.phonemize_between_brackets is not None:
            payload["phonemizeBetweenBrackets"] = opts.phonemize_between_brackets

        try:
            async with self._tts._ensure_session().post(
                self._tts._base_url,
                headers={
                    "accept": "audio/pcm",
                    "Authorization": f"Bearer {self._tts._api_key}",
                    "content-type": "application/json",
                },
                json=payload,
                conn_options=self._conn_options,
            ) as resp:
                try:
                    await resp.raise_for_status()
                except APIStatusError as e:
                    # the provider's reason and body stay out of the error
                    raise APIStatusError(
                        "Rime HTTP request failed", status_code=e.status_code
                    ) from redacted_cause(e)

                if not resp.content_type.startswith("audio"):
                    raise APIError(
                        "Rime returned non-audio data", kind="protocol", detail=await resp.text()
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )
                async for data in resp.iter_bytes():
                    output_emitter.push(data)
        except APIError:
            raise
        except Exception as cause:
            raise APIConnectionError("Rime HTTP request failed") from redacted_cause(cause)


_Frame = bytes | str


class _Mux(tts.WebSocketMultiplexer[proto.WebSocketResponse, _Frame]):
    """Rime WebSocket v1 sockets, one stream at a time each, reused by the next stream."""

    max_session_duration = 300
    max_contexts = 1

    def __init__(self, url: str, *, protocol: v1.WebSocketProtocol, api_key: str) -> None:
        self._url = url
        self.protocol: v1.WebSocketProtocol = protocol
        self._api_key = api_key

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            protocols=[v1.SUBPROTOCOLS[self.protocol]],
        )

    def decode(self, frame: bytes | str) -> proto.WebSocketResponse | None:
        return v1.decode(self.protocol, frame)

    def route(self, message: proto.WebSocketResponse) -> str | None:
        if message.context_id:
            return message.context_id
        payload = message.WhichOneof("payload")
        if payload == "error":
            raise v1.rime_error(message.error, fallback_request_id=None)
        if payload != "ready":
            raise v1.protocol_error(f"Rime v1 sent a {payload} event naming no context")
        if message.ready.protocol != v1.PROTOCOL_VERSION:
            raise v1.protocol_error("Rime v1 reported an unsupported protocol version")
        return None

    def close_context(self, context_id: str) -> _Frame:
        request = proto.WebSocketRequest(context_id=context_id)
        request.cancel.SetInParent()
        return v1.encode(self.protocol, request)

    def update_endpoint(self, url: str) -> None:
        if url != self._url:
            self._url = url
            self.invalidate()


class SynthesizeStream(tts.WebSocketStream[proto.WebSocketResponse, _Frame]):
    """A stream on a Rime WebSocket v1 socket, each segment its own Rime context."""

    def __init__(
        self, *, tts: TTS, mux: _Mux, opts: _TTSOptions, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = mux
        self.mime_type = opts.audio_format.partition(";")[0]
        self._protocol: v1.WebSocketProtocol = mux.protocol
        self._start = _start(opts, sample_rate=tts.sample_rate)

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Frame | None:
        match item:
            case tts.SegmentStart():
                request = proto.WebSocketRequest(context_id=item.segment_id)
                request.start.CopyFrom(self._start)
            case tts.SegmentText():
                request = proto.WebSocketRequest(
                    context_id=item.segment_id, text=item.chunk.text + " "
                )
            case tts.SegmentEnd():
                request = proto.WebSocketRequest(context_id=item.segment_id)
                request.end.SetInParent()
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)
        return v1.encode(self._protocol, request)

    def on_message(self, message: proto.WebSocketResponse) -> None:
        context_id = message.context_id
        payload = message.WhichOneof("payload")
        if payload == "audio":
            self.report(tts.SegmentAudio(context_id, message.audio))
        elif payload == "done":
            self.report(tts.SegmentDone(context_id))
        elif payload == "error":
            raise v1.rime_error(message.error, fallback_request_id=self.request_id)
        elif payload == "cancelled":
            raise APIConnectionError("Rime v1 cancelled an active context unexpectedly")
        elif payload == "started":
            if not message.started.request_id:
                raise v1.protocol_error("Rime v1 sent a started event naming no request")
            # Rime's id for the context: on the audio, the metrics and a later error
            self.report(tts.ProviderRequestId(message.started.request_id))


def _start(opts: _TTSOptions, *, sample_rate: int) -> proto.SynthesisRequest:
    audio = proto.AudioParameters(audio_format=opts.audio_format, sampling_rate=sample_rate)
    if opts.time_scale_factor is not None:
        audio.time_scale_factor = opts.time_scale_factor
    request = proto.SynthesisRequest(
        speaker=opts.speaker, language=opts.language, text="", audio_parameters=audio
    )
    if opts.pause_between_brackets is not None:
        request.mist_parameters.pause_between_brackets = opts.pause_between_brackets
    if opts.phonemize_between_brackets is not None:
        request.mist_parameters.phonemize_between_brackets = opts.phonemize_between_brackets
    return request
