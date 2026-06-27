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

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, TypedDict
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .version import __version__


class VoiceById(TypedDict, total=False):
    id: str
    provider: VoiceProvider | None


class VoiceByName(TypedDict, total=False):
    name: str
    provider: VoiceProvider | None


class Utterance(TypedDict, total=False):
    """Utterance for TTS synthesis."""

    text: str
    description: str | None
    speed: float | None
    voice: VoiceById | VoiceByName | None
    trailing_silence: float | None


class VoiceProvider(str, Enum):
    """Voice provider for the voice library."""

    hume = "HUME_AI"
    custom = "CUSTOM_VOICE"


class AudioFormat(str, Enum):
    """Audio format for the synthesized speech."""

    mp3 = "mp3"
    wav = "wav"
    pcm = "pcm"


ModelVersion = Literal["1", "2"]


DEFAULT_HEADERS = {
    "X-Hume-Client-Name": "livekit",
    "X-Hume-Client-Version": __version__,
}
API_AUTH_HEADER = "X-Hume-Api-Key"
STREAM_PATH = "/v0/tts/stream/json"
INPUT_STREAM_PATH = "/v0/tts/stream/input"
DEFAULT_BASE_URL = "https://api.hume.ai"
SUPPORTED_SAMPLE_RATE = 48000
DEFAULT_VOICE = VoiceByName(name="Male English Actor", provider=VoiceProvider.hume)


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    voice: VoiceById | VoiceByName | None
    model_version: ModelVersion | None
    description: str | None
    speed: float | None
    trailing_silence: float | None
    context: str | list[Utterance] | None
    instant_mode: bool | None
    audio_format: AudioFormat

    def http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def websocket_url(self, path: str, query_params: dict[str, str]) -> str:
        base_url = self.base_url
        if base_url.startswith("https://"):
            base_url = f"wss://{base_url.removeprefix('https://')}"
        elif base_url.startswith("http://"):
            base_url = f"ws://{base_url.removeprefix('http://')}"

        query = urlencode(query_params)
        return f"{base_url}{path}?{query}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: VoiceById | VoiceByName | None = DEFAULT_VOICE,
        model_version: ModelVersion | None = "1",
        description: str | None = None,
        speed: float | None = None,
        trailing_silence: float | None = None,
        context: str | list[Utterance] | None = None,
        instant_mode: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: AudioFormat = AudioFormat.mp3,
        streaming: bool = True,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ):
        """Initialize the Hume AI TTS client. Options will be used for all future synthesis
        (until updated with update_options).

        Args:
            api_key: Hume AI API key. If not provided, will look for HUME_API_KEY environment
                variable.
            voice: A voice from the voice library specified by name or id.
            model_version: Specifies which version of Octave to use. See Hume's documentation for
                details on model version differences: https://dev.hume.ai/docs/text-to-speech-tts/overview.
            description: Natural language instructions describing how the synthesized speech
                should sound (≤1000 characters).
            speed: Speed multiplier for the synthesized speech (≥0.5, ≤2.0, default: 1.0).
            trailing_silence: Duration of trailing silence (in seconds) to add to each utterance
                (≥0, ≤5.0, default: 0.35).
            context: Optional context for synthesis, either as text or list of utterances.
            instant_mode: Whether to use instant mode. Defaults to True if voice specified,
                False otherwise. Requires a voice to be specified when enabled.
            audio_format: Output audio format (mp3, wav, or pcm). Defaults to mp3.
            streaming: Whether this TTS advertises native websocket input streaming to LiveKit.
                Defaults to True. Set to False to let LiveKit's default TTS node wrap
                `synthesize()` with a sentence-based StreamAdapter.
            base_url: Base URL for Hume AI API. Defaults to https://api.hume.ai
            http_session: Optional aiohttp ClientSession to use for requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
        )
        key = api_key or os.environ.get("HUME_API_KEY")
        if not key:
            raise ValueError("Hume API key is required via api_key or HUME_API_KEY env var")

        has_voice = voice is not None

        # Default instant_mode is True if a voice is specified, otherwise False
        # (Hume API requires a voice for instant mode)
        if not is_given(instant_mode):
            resolved_instant_mode = has_voice
        elif instant_mode and not has_voice:
            raise ValueError("Hume TTS: instant_mode cannot be enabled without specifying a voice")
        else:
            resolved_instant_mode = instant_mode

        self._opts = _TTSOptions(
            api_key=key,
            voice=voice,
            model_version=model_version,
            description=description,
            speed=speed,
            trailing_silence=trailing_silence,
            context=context,
            instant_mode=resolved_instant_mode,
            audio_format=audio_format,
            base_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return "Octave"

    @property
    def provider(self) -> str:
        return "Hume"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        description: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        voice: NotGivenOr[VoiceById | VoiceByName | None] = NOT_GIVEN,
        trailing_silence: NotGivenOr[float | None] = NOT_GIVEN,
        context: NotGivenOr[str | list[Utterance] | None] = NOT_GIVEN,
        instant_mode: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat] = NOT_GIVEN,
    ) -> None:
        """Update TTS options used for all future synthesis (until updated again)

        Args:
            voice: A voice from the voice library specified by name or id.
            description: Natural language instructions describing how the synthesized speech
                should sound (≤1000 characters).
            speed: Speed multiplier for the synthesized speech (≥0.5, ≤2.0, default: 1.0).
            trailing_silence: Duration of trailing silence (in seconds) to add to each utterance.
            context: Optional context for synthesis, either as text or list of utterances.
            instant_mode: Whether to use instant mode.
            audio_format: Output audio format (mp3, wav, or pcm).
        """
        if is_given(description):
            self._opts.description = description
        if is_given(speed):
            self._opts.speed = speed
        if is_given(voice):
            self._opts.voice = voice
        if is_given(trailing_silence):
            self._opts.trailing_silence = trailing_silence
        if is_given(context):
            self._opts.context = context
        if is_given(instant_mode):
            self._opts.instant_mode = instant_mode
        if is_given(audio_format):
            self._opts.audio_format = audio_format

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if isinstance(self._opts.context, list):
            raise ValueError(
                "Hume TTS input streaming only supports context as a generation ID string. "
                "Use synthesize() for utterance-list context."
            )

        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        utterance: Utterance = {
            "text": self._input_text,
        }

        if self._opts.voice:
            utterance["voice"] = self._opts.voice
        if self._opts.description:
            utterance["description"] = self._opts.description
        if self._opts.speed:
            utterance["speed"] = self._opts.speed
        if self._opts.trailing_silence:
            utterance["trailing_silence"] = self._opts.trailing_silence

        payload: dict[str, Any] = {
            "utterances": [utterance],
            "version": self._opts.model_version,
            "strip_headers": True,
            "instant_mode": self._opts.instant_mode,
            "format": {"type": self._opts.audio_format.value},
        }
        if isinstance(self._opts.context, str):
            payload["context"] = {"generation_id": self._opts.context}
        elif isinstance(self._opts.context, list):
            payload["context"] = {"utterances": self._opts.context}

        try:
            async with self._tts._ensure_session().post(
                self._opts.http_url(STREAM_PATH),
                headers={**DEFAULT_HEADERS, API_AUTH_HEADER: self._opts.api_key},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=self._conn_options.timeout),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SUPPORTED_SAMPLE_RATE,
                    num_channels=self._tts.num_channels,
                    mime_type=f"audio/{self._opts.audio_format.value}",
                )

                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue

                    data = json.loads(line.decode())
                    if data.get("type") == "error":
                        raise APIError(message=str(data))

                    audio_b64 = data.get("audio")
                    if audio_b64:
                        output_emitter.push(base64.b64decode(audio_b64))

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        output_started = False
        input_sent_event = asyncio.Event()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=self._tts.num_channels,
            mime_type=f"audio/{self._opts.audio_format.value}",
            stream=True,
        )

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    if not self._input_ended:
                        await ws.send_str(json.dumps({"flush": True}))
                        input_sent_event.set()
                    continue

                self._mark_started()
                await ws.send_str(json.dumps(_input_message(text=data, opts=self._opts)))
                input_sent_event.set()

            await ws.send_str(json.dumps({"close": True}))
            input_sent_event.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal output_started

            await input_sent_event.wait()
            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break

                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(str(ws.exception()))

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not output_started:
                        output_emitter.start_segment(segment_id=segment_id)
                        output_started = True
                    output_emitter.push(msg.data)
                    continue

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError as e:
                    raise APIError(f"received invalid JSON from Hume TTS: {msg.data}") from e

                if data.get("type") == "error" or data.get("error"):
                    raise APIError(message=f"Hume TTS returned error: {data}", body=data)

                provider_request_id = (
                    data.get("request_id") or data.get("snippet_id") or data.get("generation_id")
                )
                if provider_request_id:
                    output_emitter._note_provider_request_id(provider_request_id)

                if data.get("type") == "timestamp":
                    continue

                audio_b64 = data.get("audio")
                if not audio_b64:
                    continue

                if not output_started:
                    output_emitter.start_segment(segment_id=segment_id)
                    output_started = True

                output_emitter.push(base64.b64decode(audio_b64))
                if data.get("is_last_chunk") and self._input_ended:
                    break

            if output_started:
                output_emitter.end_segment()

        try:
            ws = await asyncio.wait_for(
                self._tts._ensure_session().ws_connect(
                    self._opts.websocket_url(
                        INPUT_STREAM_PATH, _stream_input_query_params(self._opts)
                    ),
                    headers=DEFAULT_HEADERS,
                ),
                timeout=self._conn_options.timeout,
            )
            try:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await utils.aio.gracefully_cancel(*tasks)
            finally:
                await ws.close()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


def _stream_input_query_params(opts: _TTSOptions) -> dict[str, str]:
    params = {
        "api_key": opts.api_key,
        "format_type": opts.audio_format.value,
        "no_binary": "true",
        "strip_headers": "true",
    }

    if opts.instant_mode is not None:
        params["instant_mode"] = _bool_str(opts.instant_mode)
    if opts.model_version is not None:
        params["version"] = opts.model_version
    if isinstance(opts.context, str):
        params["context_generation_id"] = opts.context

    return params


def _input_message(*, text: str, opts: _TTSOptions) -> dict[str, Any]:
    message: dict[str, Any] = {"text": text}

    if opts.voice:
        message["voice"] = opts.voice
    if opts.description:
        message["description"] = opts.description
    if opts.speed:
        message["speed"] = opts.speed
    if opts.trailing_silence:
        message["trailing_silence"] = opts.trailing_silence

    return message


def _bool_str(value: bool) -> str:
    return "true" if value else "false"
