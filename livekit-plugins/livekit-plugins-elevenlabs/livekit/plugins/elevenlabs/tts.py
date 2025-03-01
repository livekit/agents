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
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import aiohttp
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)

from .log import logger
from .models import TTSEncoding, TTSModels

_Encoding = Literal["mp3", "pcm"]


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_22050_32
    return int(split[1])


def _encoding_from_format(output_format: TTSEncoding) -> _Encoding:
    if output_format.startswith("mp3"):
        return "mp3"
    elif output_format.startswith("pcm"):
        return "pcm"

    raise ValueError(f"Unknown format: {output_format}")


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: float | None = None  # [0.0 - 1.0]
    speed: float | None = 1.0  # [0.8 - 1.2]
    use_speaker_boost: bool | None = False


@dataclass
class Voice:
    id: str
    name: str
    category: str
    settings: VoiceSettings | None = None


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


@dataclass
class _TTSOptions:
    api_key: str
    voice: Voice
    model: TTSModels | str
    language: str | None
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    streaming_latency: int
    word_tokenizer: tokenize.WordTokenizer
    chunk_length_schedule: list[int]
    enable_ssml_parsing: bool


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model: TTSModels | str = "eleven_flash_v2_5",
        api_key: str | None = None,
        base_url: str | None = None,
        encoding: TTSEncoding = "mp3_22050_32",
        streaming_latency: int = 3,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False  # punctuation can help for intonation
        ),
        enable_ssml_parsing: bool = False,
        chunk_length_schedule: list[int] = [80, 120, 200, 260],  # range is [50, 500]
        http_session: aiohttp.ClientSession | None = None,
        # deprecated
        model_id: TTSModels | str | None = None,
        language: str | None = None,
    ) -> None:
        """
        Create a new instance of ElevenLabs TTS.

        Args:
            voice (Voice): Voice configuration. Defaults to `DEFAULT_VOICE`.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            api_key (str | None): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (str | None): Custom base URL for the API. Optional.
            encoding (TTSEncoding): Audio encoding format. Defaults to "mp3_22050_32".
            streaming_latency (int): Latency in seconds for streaming. Defaults to 3.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            chunk_length_schedule (list[int]): Schedule for chunk lengths, ranging from 50 to 500. Defaults to [80, 120, 200, 260].
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (str | None): Language code for the TTS model, as of 10/24/24 only valid for "eleven_turbo_v2_5". Optional.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )

        if model_id is not None:
            logger.warning(
                "model_id is deprecated and will be removed in 1.5.0, use model instead",
            )
            model = model_id

        api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set ELEVEN_API_KEY environmental variable"
            )

        self._opts = _TTSOptions(
            voice=voice,
            model=model,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            language=language,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def list_voices(self) -> List[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def update_options(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model: TTSModels | str = "eleven_turbo_v2_5",
        language: str | None = None,
    ) -> None:
        """
        Args:
            voice (Voice): Voice configuration. Defaults to `DEFAULT_VOICE`.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            language (str | None): Language code for the TTS model. Optional.
        """
        self._opts.model = model or self._opts.model
        self._opts.voice = voice or self._opts.voice
        self._opts.language = language or self._opts.language

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    def stream(self, *, conn_options: Optional[APIConnectOptions] = None) -> "SynthesizeStream":
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: Optional[APIConnectOptions] = None,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session
        if _encoding_from_format(self._opts.encoding) == "mp3":
            self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(sample_rate=self._opts.sample_rate, num_channels=1)

        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if self._opts.voice.settings
            else None
        )
        data = {
            "text": self._input_text,
            "model_id": self._opts.model,
            "voice_settings": voice_settings,
        }

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

                encoding = _encoding_from_format(self._opts.encoding)
                if encoding == "mp3":
                    async for bytes_data, _ in resp.content.iter_chunks():
                        for frame in self._mp3_decoder.decode_chunk(bytes_data):
                            for frame in bstream.write(frame.data.tobytes()):
                                self._event_ch.send_nowait(
                                    tts.SynthesizedAudio(
                                        request_id=request_id,
                                        frame=frame,
                                    )
                                )
                else:
                    async for bytes_data, _ in resp.content.iter_chunks():
                        for frame in bstream.write(bytes_data):
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    frame=frame,
                                )
                            )

                for frame in bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(request_id=request_id, frame=frame)
                    )

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

    def __init__(
        self,
        *,
        tts: TTS,
        session: aiohttp.ClientSession,
        opts: _TTSOptions,
        conn_options: Optional[APIConnectOptions] = None,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts, self._session = opts, session
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
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

            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _run():
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        word_stream: tokenize.WordStream,
        max_retry: int = 3,
    ) -> None:
        ws_conn: aiohttp.ClientWebSocketResponse | None = None
        for try_i in range(max_retry):
            retry_delay = 5
            try:
                if try_i > 0:
                    await asyncio.sleep(retry_delay)

                ws_conn = await self._session.ws_connect(
                    _stream_url(self._opts),
                    headers={AUTHORIZATION_HEADER: self._opts.api_key},
                )
                break
            except Exception as e:
                logger.warning(
                    f"failed to connect to 11labs, retrying in {retry_delay}s",
                    exc_info=e,
                )

        if ws_conn is None:
            raise Exception(f"failed to connect to 11labs after {max_retry} retries")

        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        # 11labs protocol expects the first message to be an "init msg"
        init_pkt = dict(
            text=" ",
            try_trigger_generation=True,
            voice_settings=_strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if self._opts.voice.settings
            else None,
            generation_config=dict(chunk_length_schedule=self._opts.chunk_length_schedule),
        )
        await ws_conn.send_str(json.dumps(init_pkt))
        eos_sent = False

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

                # try_trigger_generation=True is a bad practice, we expose
                # chunk_length_schedule instead
                data_pkt = dict(
                    text=f"{text} ",  # must always end with a space
                    try_trigger_generation=False,
                )
                self._mark_started()
                await ws_conn.send_str(json.dumps(data_pkt))

            if xml_content:
                logger.warning("11labs stream ended with incomplete xml content")

            # no more token, mark eos
            eos_pkt = dict(text="")
            await ws_conn.send_str(json.dumps(eos_pkt))
            eos_sent = True

        async def recv_task():
            nonlocal eos_sent
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
            )

            last_frame: rtc.AudioFrame | None = None

            def _send_last_frame(*, segment_id: str, is_final: bool) -> None:
                nonlocal last_frame
                if last_frame is not None:
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            segment_id=segment_id,
                            frame=last_frame,
                            is_final=is_final,
                        )
                    )

                    last_frame = None

            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not eos_sent:
                        raise APIStatusError(
                            "11labs connection closed unexpectedly, not all tokens have been consumed",
                            request_id=request_id,
                        )
                    return

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                encoding = _encoding_from_format(self._opts.encoding)
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    if encoding == "mp3":
                        for frame in self._mp3_decoder.decode_chunk(b64data):
                            for frame in audio_bstream.write(frame.data.tobytes()):
                                _send_last_frame(segment_id=segment_id, is_final=False)
                                last_frame = frame

                    else:
                        for frame in audio_bstream.write(b64data):
                            _send_last_frame(segment_id=segment_id, is_final=False)
                            last_frame = frame

                elif data.get("isFinal"):
                    for frame in audio_bstream.flush():
                        _send_last_frame(segment_id=segment_id, is_final=False)
                        last_frame = frame

                    _send_last_frame(segment_id=segment_id, is_final=True)

                    pass
                elif data.get("error"):
                    logger.error("11labs reported an error: %s", data["error"])
                else:
                    logger.error("unexpected 11labs message %s", data)

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            if ws_conn is not None:
                await ws_conn.close()


def _dict_to_voices_list(data: dict[str, Any]):
    voices: List[Voice] = []
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
    return {k: v for k, v in data.items() if v is not None}


def _synthesize_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice.id
    model_id = opts.model
    output_format = opts.encoding
    latency = opts.streaming_latency
    return (
        f"{base_url}/text-to-speech/{voice_id}/stream?"
        f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}"
    )


def _stream_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice.id
    model_id = opts.model
    output_format = opts.encoding
    latency = opts.streaming_latency
    enable_ssml = str(opts.enable_ssml_parsing).lower()
    language = opts.language
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream-input?"
        f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}&"
        f"enable_ssml_parsing={enable_ssml}"
    )
    if language is not None:
        url += f"&language_code={language}"
    return url
