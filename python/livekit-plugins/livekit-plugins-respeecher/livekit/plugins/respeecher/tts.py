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
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, NotRequired, TypedDict, assert_never, cast
from urllib.parse import urlsplit

from pydantic import Field

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSEncoding, TTSModels, Voice, VoiceSettings
from .version import __version__

API_VERSION = __version__
API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "LiveKit-Plugin-Respeecher-Version"
API_BASE_URL = "https://api.respeecher.com/v1"

DEFAULT_VOICES: dict[str, str] = {
    "/public/tts/en-rt": "samantha",
    "/public/tts/ua-rt": "olesia-conversation",
}


_LOOPBACK = ("localhost", "127.0.0.1", "::1")

# Respeecher's websocket: each segment is one context, ended by `continue: false`. Each message
# declares only the fields read.


class _Chunk(TypedDict):
    type: Literal["chunk"]
    context_id: str
    data: str


class _Done(TypedDict):
    type: Literal["done"]
    context_id: str


class _Error(TypedDict):
    type: Literal["error"]
    context_id: NotRequired[str]
    error: NotRequired[object]


_Message = Annotated[_Chunk | _Done | _Error, Field(discriminator="type")]

_Continue = TypedDict("_Continue", {"continue": bool})  # a keyword: the functional form


class _Speak(_Continue):
    context_id: str
    transcript: str
    voice: dict[str, object]
    output_format: dict[str, object]


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    api_key: str
    base_url: str


async def list_voices(
    *,
    model: TTSModels | str = "/public/tts/en-rt",
    api_key: NotGivenOr[str] = NOT_GIVEN,
    base_url: str = API_BASE_URL,
    http_session: http.HttpSession | None = None,
) -> list[Voice]:
    """List available voices for the given Respeecher model.

    Args:
        model: The Respeecher TTS model whose voices should be listed.
        api_key: Respeecher API key. If not provided, uses RESPEECHER_API_KEY env variable.
        base_url: The base URL for the Respeecher API.
        http_session: Optional HttpSession to use for the request.
    """
    resolved_api_key = api_key if is_given(api_key) else os.environ.get("RESPEECHER_API_KEY")
    if not resolved_api_key:
        raise ValueError("RESPEECHER_API_KEY must be set")

    session = http_session or http.http_session()
    async with session.get(
        f"{base_url}{model}/voices",
        headers={
            API_AUTH_HEADER: resolved_api_key,
            API_VERSION_HEADER: API_VERSION,
        },
    ) as resp:
        await resp.raise_for_status()
        data = cast(list[dict[str, Any]], await resp.json())
        voices = [Voice(voice_data) for voice_data in data]
        if not voices:
            raise APIError("No voices are available", kind="invalid_request")
        return voices


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = "/public/tts/en-rt",
        encoding: TTSEncoding = "pcm_s16le",
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        sample_rate: int = 24000,
        http_session: http.HttpSession | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        """
        Create a new instance of Respeecher TTS.

        Args:
            voice_id: ID of the voice to use. If not provided, a model-specific default is used (see `DEFAULT_VOICES`). Each model exposes a different set of voices; call the module-level `list_voices()` helper to discover the IDs available for the chosen model.
            api_key: Respeecher API key. If not provided, uses RESPEECHER_API_KEY env variable.
            model: The Respeecher TTS model to use.
            encoding: Audio encoding format.
            voice_settings: Optional voice settings including sampling parameters.
            sample_rate: Audio sample rate in Hz.
            http_session: Optional HttpSession to use for requests.
            base_url: The base URL for the Respeecher API.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        respeecher_api_key = api_key if is_given(api_key) else os.environ.get("RESPEECHER_API_KEY")
        if not respeecher_api_key:
            raise ValueError("RESPEECHER_API_KEY must be set")

        resolved_voice_id = voice_id if is_given(voice_id) else DEFAULT_VOICES.get(model)
        if not resolved_voice_id:
            raise ValueError(
                f"voice_id is required for model {model!r} (no default voice is configured); "
                "pass voice_id explicitly or use one of the supported models."
            )

        self._opts = _TTSOptions(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            voice_id=resolved_voice_id,
            voice_settings=voice_settings,
            api_key=respeecher_api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._mux = _Mux(self._opts)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Respeecher"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update TTS options"""
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            model_changed = is_given(model) and model != self._opts.model
            if is_given(model):
                self._opts.model = model
            if is_given(voice_id):
                self._opts.voice_id = voice_id
            if is_given(voice_settings):
                self._opts.voice_settings = voice_settings

        if model_changed:
            # the model is part of the socket URL: idle sockets close now, busy ones once
            # their streams ended
            self._mux.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
            opts=replace(self._opts),
            ensure_session=self._ensure_session,
        )

    def prewarm(self) -> None:
        self._mux.prewarm()

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
    """Synthesize text using Respeecher HTTPS endpoint"""

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
        """Run the TTS synthesis"""
        voice: dict[str, object] = {"id": self._opts.voice_id}
        json_data = {
            "transcript": self._input_text,
            "voice": voice,
            "output_format": {
                "sample_rate": self._opts.sample_rate,
                "encoding": self._opts.encoding,
            },
        }

        if is_given(self._opts.voice_settings) and self._opts.voice_settings.sampling_params:
            voice["sampling_params"] = self._opts.voice_settings.sampling_params

        http_url = f"{self._opts.base_url}{self._opts.model}/tts/bytes"

        try:
            async with self._ensure_session().post(
                http_url,
                headers={
                    API_AUTH_HEADER: self._opts.api_key,
                    API_VERSION_HEADER: API_VERSION,
                    "Content-Type": "application/json",
                },
                json=json_data,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    # /tts/bytes returns WAV-wrapped PCM; the WebSocket stream returns raw PCM.
                    mime_type="audio/wav",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()
        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class _Mux(tts.WebSocketMultiplexer[_Message, _Speak]):
    """One Respeecher socket carries every stream's segments, each its own context."""

    def __init__(self, opts: _TTSOptions) -> None:
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        url = self._opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
        parts = urlsplit(url)
        # the API key travels in the query: plain ws only to this machine
        if parts.scheme != "wss" and parts.hostname not in _LOOPBACK:
            raise APIError("Secure WebSocket connection (wss://) required", kind="invalid_request")
        return tts.WebSocketTarget(
            f"{url}{self._opts.model}/tts/websocket",
            params={
                "api_key": self._opts.api_key,
                "source": API_VERSION_HEADER,
                "version": API_VERSION,
            },
        )

    def route(self, message: _Chunk | _Done | _Error) -> str:
        if "context_id" not in message:
            # an error naming no context fails every stream on the socket
            raise _error(message)
        return message["context_id"]


class SynthesizeStream(tts.WebSocketStream[_Message, _Speak]):
    """Streamed API using WebSocket for real-time synthesis"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions, opts: _TTSOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        voice: dict[str, object] = {"id": opts.voice_id}
        if is_given(opts.voice_settings) and opts.voice_settings.sampling_params:
            voice["sampling_params"] = opts.voice_settings.sampling_params
        self._voice = voice
        self._output_format = {"encoding": opts.encoding, "sample_rate": opts.sample_rate}

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Speak | None:
        match item:
            case tts.SegmentText():
                return self._speak(item.segment_id, item.chunk.text + " ", more=True)
            case tts.SegmentEnd():
                return self._speak(item.segment_id, " ", more=False)
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Chunk | _Done | _Error) -> None:
        if message["type"] == "chunk":
            audio = base64.b64decode(message["data"])
            self.report(tts.SegmentAudio(message["context_id"], audio))
        elif message["type"] == "done":
            self.report(tts.SegmentDone(message["context_id"]))
        elif message["type"] == "error":
            raise _error(message)
        else:
            assert_never(message)

    def _speak(self, context_id: str, transcript: str, *, more: bool) -> _Speak:
        return {
            "context_id": context_id,
            "transcript": transcript,
            "voice": self._voice,
            "continue": more,
            "output_format": self._output_format,
        }


def _error(message: _Error) -> APIError:
    # in-band errors reject the request (voice, format, quota): retrying repeats it
    return APIError(
        "Respeecher returned an error", kind="invalid_request", detail=str(message.get("error"))
    )
