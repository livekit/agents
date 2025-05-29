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
import dataclasses
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncoding, TTSModels

# by default, use 22.05kHz sample rate at 32kbps
# in our testing,  reduce TTFB by about ~110ms
_DefaultEncoding: TTSEncoding = "mp3_22050_32"


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_44100
    return int(split[1])


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


DEFAULT_VOICE_ID = "bIHbv24MWmeRgasZH58o"
API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
WS_INACTIVITY_TIMEOUT = 300


@dataclass
class _TTSOptions:
    api_key: str
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    model: TTSModels | str
    language: NotGivenOr[str]
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    streaming_latency: NotGivenOr[int]
    word_tokenizer: tokenize.WordTokenizer
    chunk_length_schedule: NotGivenOr[list[int]]
    enable_ssml_parsing: bool
    inactivity_timeout: int


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
        streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        enable_ssml_parsing: bool = False,
        chunk_length_schedule: NotGivenOr[list[int]] = NOT_GIVEN,  # range is [50, 500]
        http_session: aiohttp.ClientSession | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of ElevenLabs TTS.

        Args:
            voice_id (str): Voice ID. Defaults to `DEFAULT_VOICE_ID`.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            streaming_latency (NotGivenOr[int]): Optimize for streaming latency, defaults to 0 - disabled. 4 for max latency optimizations. deprecated
            inactivity_timeout (int): Inactivity timeout in seconds for the websocket connection. Defaults to 300.
            word_tokenizer (NotGivenOr[tokenize.WordTokenizer]): Tokenizer for processing text. Defaults to basic WordTokenizer.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            chunk_length_schedule (NotGivenOr[list[int]]): Schedule for chunk lengths, ranging from 50 to 500. Defaults are [120, 160, 250, 290].
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (NotGivenOr[str]): Language code for the TTS model, as of 10/24/24 only valid for "eleven_turbo_v2_5".
        """  # noqa: E501

        if not is_given(encoding):
            encoding = _DefaultEncoding

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set ELEVEN_API_KEY environmental variable"  # noqa: E501
            )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(
                ignore_punctuation=False  # punctuation can help for intonation
            )

        self._opts = _TTSOptions(
            voice_id=voice_id,
            voice_settings=voice_settings,
            model=model,
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            language=language,
            inactivity_timeout=inactivity_timeout,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def list_voices(self) -> list[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice_id (NotGivenOr[str]): Voice ID.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[str]): Language code for the TTS model.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(voice_settings):
            self._opts.voice_settings = voice_settings
        if is_given(language):
            self._opts.language = language

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice_settings))
            if is_given(self._opts.voice_settings)
            else None
        )
        try:
            async with self._tts._ensure_session().post(
                _synthesize_url(self._opts),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
                json={
                    "text": self._input_text,
                    "model_id": self._opts.model,
                    "voice_settings": voice_settings,
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    raise APIError(message="11labs returned non-audio data", body=content)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/mp3",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/mp3",
        )

        async def _tokenize_input() -> None:
            """tokenize text from the input_ch to words"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)

                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream is not None:
                        word_stream.end_input()

                    word_stream = None

            if word_stream is not None:
                word_stream.end_input()

            self._segments_ch.close()

        async def _process_segments() -> None:
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, word_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        ws_conn = await asyncio.wait_for(
            self._tts._ensure_session().ws_connect(
                _stream_url(self._opts), headers={AUTHORIZATION_HEADER: self._opts.api_key}
            ),
            timeout=self._conn_options.timeout,
        )

        # 11labs protocol expects the first message to be an "init msg"
        init_pkt: dict = {
            "text": " ",
        }
        if is_given(self._opts.chunk_length_schedule):
            init_pkt["generation_config"] = {
                "chunk_length_schedule": self._opts.chunk_length_schedule
            }
        if is_given(self._opts.voice_settings):
            init_pkt["voice_settings"] = _strip_nones(dataclasses.asdict(self._opts.voice_settings))
        await ws_conn.send_str(json.dumps(init_pkt))
        eos_sent = False

        @utils.log_exceptions(logger=logger)
        async def send_task() -> None:
            nonlocal eos_sent
            xml_content: list[str] = []
            async for data in word_stream:
                text = data.token
                # send xml tags fully formed
                xml_start_tokens = ["<phoneme", "<break"]
                xml_end_tokens = ["</phoneme>", "/>"]

                if (
                    self._opts.enable_ssml_parsing
                    and any(data.token.startswith(start) for start in xml_start_tokens)
                    or xml_content
                ):
                    xml_content.append(text)

                    if any(data.token.find(end) > -1 for end in xml_end_tokens):
                        text = self._opts.word_tokenizer.format_words(xml_content)
                        xml_content = []
                    else:
                        continue

                data_pkt = {"text": f"{text} "}  # must always end with a space

                self._mark_started()
                await ws_conn.send_str(json.dumps(data_pkt))
            if xml_content:
                logger.warning("11labs stream ended with incomplete xml content")

            # no more token, mark eos
            eos_pkt = {"text": ""}
            await ws_conn.send_str(json.dumps(eos_pkt))
            eos_sent = True

        # receives from ws and decodes audio
        @utils.log_exceptions(logger=logger)
        async def recv_task() -> None:
            nonlocal eos_sent

            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not eos_sent:
                        raise APIStatusError(
                            "11labs connection closed unexpectedly, not all tokens have been consumed",  # noqa: E501
                        )
                    return

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    output_emitter.push(b64data)
                elif data.get("isFinal"):
                    output_emitter.end_input()
                    return  # 11labs only allow one segment per connection
                elif data.get("error"):
                    raise APIError(message=data["error"])
                else:
                    raise APIError("unexpected 11labs message {data}")

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await ws_conn.close()


def _dict_to_voices_list(data: dict[str, Any]) -> list[Voice]:
    voices: list[Voice] = []
    for voice in data["voices"]:
        voices.append(Voice(id=voice["voice_id"], name=voice["name"], category=voice["category"]))

    return voices


def _strip_nones(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if is_given(v) and v is not None}


def _synthesize_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice_id
    model_id = opts.model
    output_format = opts.encoding
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream?"
        f"model_id={model_id}&output_format={output_format}"
    )
    if is_given(opts.streaming_latency):
        url += f"&optimize_streaming_latency={opts.streaming_latency}"
    return url


def _stream_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice_id
    model_id = opts.model
    output_format = opts.encoding
    enable_ssml = str(opts.enable_ssml_parsing).lower()
    language = opts.language
    inactivity_timeout = opts.inactivity_timeout
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream-input?"
        f"model_id={model_id}&output_format={output_format}&"
        f"enable_ssml_parsing={enable_ssml}&inactivity_timeout={inactivity_timeout}"
    )
    if is_given(language):
        url += f"&language_code={language}"
    if is_given(opts.streaming_latency):
        url += f"&optimize_streaming_latency={opts.streaming_latency}"

    return url
