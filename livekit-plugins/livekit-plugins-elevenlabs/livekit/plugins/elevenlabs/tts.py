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
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncoding, TTSModels

_DefaultEncoding: TTSEncoding = "mp3_44100"


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
    settings: NotGivenOr[VoiceSettings] = NOT_GIVEN


DEFAULT_VOICE = Voice(
    id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    category="premade",
    settings=VoiceSettings(
        stability=0.71,
        speed=1.0,
        similarity_boost=0.5,
        style=0.0,
        use_speaker_boost=True,
    ),
)

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
WS_INACTIVITY_TIMEOUT = 300


@dataclass
class _TTSOptions:
    api_key: str
    voice: Voice
    model: TTSModels | str
    language: NotGivenOr[str]
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    streaming_latency: NotGivenOr[int]
    word_tokenizer: tokenize.WordTokenizer
    chunk_length_schedule: list[int]
    enable_ssml_parsing: bool
    inactivity_timeout: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model: TTSModels | str = "eleven_flash_v2_5",
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
            voice (Voice): Voice configuration. Defaults to `DEFAULT_VOICE`.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            api_key (str | None): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (str | None): Custom base URL for the API. Optional.
            streaming_latency (int): Optimize for streaming latency, defaults to 0 - disabled. 4 for max latency optimizations. deprecated
            inactivity_timeout (int): Inactivity timeout in seconds for the websocket connection. Defaults to 300.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            chunk_length_schedule (list[int]): Schedule for chunk lengths, ranging from 50 to 500. Defaults to [80, 120, 200, 260].
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (NotGivenOr[str]): Language code for the TTS model, as of 10/24/24 only valid for "eleven_turbo_v2_5".
        """  # noqa: E501

        if not is_given(chunk_length_schedule):
            chunk_length_schedule = [80, 120, 200, 260]
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=_sample_rate_from_format(_DefaultEncoding),
            num_channels=1,
        )

        api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not is_given(api_key):
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set ELEVEN_API_KEY environmental variable"  # noqa: E501
            )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(
                ignore_punctuation=False  # punctuation can help for intonation
            )

        self._opts = _TTSOptions(
            voice=voice,
            model=model,
            api_key=api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            encoding=_DefaultEncoding,
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
        voice: NotGivenOr[Voice] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (NotGivenOr[Voice]): Voice configuration.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[str]): Language code for the TTS model.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if is_given(self._opts.voice.settings)
            else None
        )
        data = {
            "text": self._input_text,
            "model_id": self._opts.model,
            "voice_settings": voice_settings,
        }

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )

        decode_task: asyncio.Task | None = None
        try:
            async with self._session.post(
                _synthesize_url(self._opts),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
                json=data,
            ) as resp:
                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    logger.error("11labs returned non-audio data: %s", content)
                    return

                async def _decode_loop():
                    try:
                        async for bytes_data, _ in resp.content.iter_chunks():
                            decoder.push(bytes_data)
                    finally:
                        decoder.end_input()

                decode_task = asyncio.create_task(_decode_loop())
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()
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
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""

    def __init__(
        self,
        *,
        tts: TTS,
        session: aiohttp.ClientSession,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        # new segment (after flush for e.g)
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

        @utils.log_exceptions(logger=logger)
        async def _process_segments():
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, request_id)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        word_stream: tokenize.WordStream,
        request_id: str,
    ) -> None:
        ws_conn = await self._session.ws_connect(
            _stream_url(self._opts),
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        )

        segment_id = utils.shortuuid()
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )

        # 11labs protocol expects the first message to be an "init msg"
        init_pkt = {
            "text": " ",
            "voice_settings": _strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if is_given(self._opts.voice.settings)
            else None,
            "generation_config": {"chunk_length_schedule": self._opts.chunk_length_schedule},
        }
        await ws_conn.send_str(json.dumps(init_pkt))
        eos_sent = False

        @utils.log_exceptions(logger=logger)
        async def send_task():
            nonlocal eos_sent
            xml_content = []
            async for data in word_stream:
                text = data.token
                # send the xml phoneme in one go
                if (
                    self._opts.enable_ssml_parsing
                    and data.token.startswith("<phoneme")
                    or xml_content
                ):
                    xml_content.append(text)
                    if data.token.find("</phoneme>") > -1:
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

        # consumes from decoder and generates events
        @utils.log_exceptions(logger=logger)
        async def generate_task():
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
                segment_id=segment_id,
            )
            async for frame in decoder:
                emitter.push(frame)
            emitter.flush()

        # receives from ws and decodes audio
        @utils.log_exceptions(logger=logger)
        async def recv_task():
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
                            request_id=request_id,
                        )
                    return

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    decoder.push(b64data)

                elif data.get("isFinal"):
                    decoder.end_input()
                    break
                elif data.get("error"):
                    raise APIStatusError(
                        message=data["error"],
                        status_code=500,
                        request_id=request_id,
                        body=None,
                    )
                else:
                    raise APIStatusError(
                        message=f"unexpected 11labs message {data}",
                        status_code=500,
                        request_id=request_id,
                        body=None,
                    )

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
            asyncio.create_task(generate_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await decoder.aclose()
            if ws_conn is not None:
                await ws_conn.close()


def _dict_to_voices_list(data: dict[str, Any]):
    voices: list[Voice] = []
    for voice in data["voices"]:
        voices.append(
            Voice(
                id=voice["voice_id"],
                name=voice["name"],
                category=voice["category"],
                settings=None,
            )
        )
    return voices


def _strip_nones(data: dict[str, Any]):
    return {k: v for k, v in data.items() if is_given(v)}


def _synthesize_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice.id
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
    voice_id = opts.voice.id
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
