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
import contextlib
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import aiohttp
from livekit import rtc
from livekit.agents import tokenize, tts, utils

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
        stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
    ),
)

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"


@dataclass
class _TTSOptions:
    api_key: str
    voice: Voice
    model_id: TTSModels
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    streaming_latency: int
    word_tokenizer: tokenize.WordTokenizer
    chunk_length_schedule: list[int]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model_id: TTSModels = "eleven_turbo_v2",
        api_key: str | None = None,
        base_url: str | None = None,
        encoding: TTSEncoding = "mp3_22050_32",
        streaming_latency: int = 3,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False  # punctuation can help for intonation
        ),
        # default value of 11labs is [120, 160, 250, 290], but we want faster responses by default
        # (range is 50-500)
        chunk_length_schedule: list[int] = [80, 120, 200, 260],
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )
        api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ELEVEN_API_KEY must be set")

        self._opts = _TTSOptions(
            voice=voice,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
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

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())

    def stream(self) -> "SynthesizeStream":
        return SynthesizeStream(self._ensure_session(), self._opts)

    async def aclose(self) -> None:
        pass


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        self._opts = opts
        self._text = text
        self._session = session
        self._task: asyncio.Task[None] | None = None
        self._queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()

    def _synthesize_url(self) -> str:
        base_url = self._opts.base_url
        voice_id = self._opts.voice.id
        model_id = self._opts.model_id
        sample_rate = _sample_rate_from_format(self._opts.encoding)
        latency = self._opts.streaming_latency
        url = (
            f"{base_url}/text-to-speech/{voice_id}/stream?"
            f"model_id={model_id}&output_format=pcm_{sample_rate}&optimize_streaming_latency={latency}"
        )
        return url

    async def _main_task(self):
        try:
            await self._run()
        except Exception:
            logger.exception("11labs main task failed in chunked stream")
        finally:
            self._queue.put_nowait(None)

    async def _run(self) -> None:
        segment_id = utils.nanoid()
        async with self._session.post(
            self._synthesize_url(),
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
            json=dict(
                text=self._text,
                model_id=self._opts.model_id,
                voice_settings=(
                    dataclasses.asdict(self._opts.voice.settings)
                    if self._opts.voice.settings
                    else None
                ),
            ),
        ) as resp:
            # avoid very small frames. chunk by 10ms 16bits
            bytes_per_frame = (self._opts.sample_rate // 100) * 2
            buf = bytearray()
            async for data, _ in resp.content.iter_chunks():
                buf.extend(data)

                while len(buf) >= bytes_per_frame:
                    frame_data = buf[:bytes_per_frame]
                    buf = buf[bytes_per_frame:]

                    self._queue.put_nowait(
                        tts.SynthesizedAudio(
                            segment_id=segment_id,
                            frame=rtc.AudioFrame(
                                data=frame_data,
                                sample_rate=self._opts.sample_rate,
                                num_channels=1,
                                samples_per_channel=len(frame_data) // 2,
                            ),
                        )
                    )

            # send any remaining data
            if len(buf) > 0:
                self._queue.put_nowait(
                    tts.SynthesizedAudio(
                        segment_id=segment_id,
                        frame=rtc.AudioFrame(
                            data=buf,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(buf) // 2,
                        ),
                    )
                )

    async def __anext__(self) -> tts.SynthesizedAudio:
        if not self._task:
            self._task = asyncio.create_task(self._main_task())

        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration

        return frame

    async def aclose(self) -> None:
        if not self._task:
            return

        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""

    @dataclass
    class _SegmentConnection:
        audio_rx: utils.aio.ChanReceiver[tts.SynthesizedAudio]
        task: asyncio.Task[None]

    def __init__(
        self,
        session: aiohttp.ClientSession,
        opts: _TTSOptions,
        max_retry_per_segment: int = 3,
    ):
        self._opts = opts
        self._session = session
        self._main_task = asyncio.create_task(self._run(max_retry_per_segment))
        self._event_queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()
        self._closed = False
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()
        self._word_stream = opts.word_tokenizer.stream()

    def _stream_url(self) -> str:
        base_url = self._opts.base_url
        voice_id = self._opts.voice.id
        model_id = self._opts.model_id
        output_format = self._opts.encoding
        latency = self._opts.streaming_latency
        url = (
            f"{base_url}/text-to-speech/{voice_id}/stream-input?"
            f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}"
        )

        return url

    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if token is None:
            self._word_stream.mark_segment_end()
            return

        self._word_stream.push_text(token)

    async def aclose(self) -> None:
        self._closed = True
        await self._word_stream.aclose()

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self, max_retry_per_segment: int) -> None:
        try:
            token_tx: utils.aio.ChanSender[str] | None = None
            task: asyncio.Task[None] | None = None
            async for ev in self._word_stream:
                if ev.type == tokenize.TokenEventType.STARTED:
                    token_tx = token_rx = utils.aio.Chan[str]()
                    task = asyncio.create_task(
                        self._run_ws(max_retry_per_segment, token_rx)
                    )
                elif ev.type == tokenize.TokenEventType.TOKEN:
                    assert token_tx is not None
                    token_tx.send_nowait(ev.token)
                elif ev.type == tokenize.TokenEventType.FINISHED:
                    assert token_tx is not None
                    token_tx.close()
                    await task
        except Exception:
            logger.exception("11labs task failed")

        self._event_queue.put_nowait(None)

    async def _run_ws(
        self,
        max_retry: int,
        token_rx: utils.aio.ChanReceiver[str],
    ) -> None:
        # try to connect to 11labs
        ws_conn: aiohttp.ClientWebSocketResponse | None = None
        for try_i in range(max_retry):
            try:
                ws_conn = await self._session.ws_connect(
                    self._stream_url(),
                    headers={AUTHORIZATION_HEADER: self._opts.api_key},
                )

                voice_settings = None
                if self._opts.voice.settings is not None:
                    voice_settings = dataclasses.asdict(self._opts.voice.settings)

                init_pkt = dict(
                    text=" ",
                    try_trigger_generation=True,
                    voice_settings=voice_settings,
                    generation_config=dict(
                        chunk_length_schedule=self._opts.chunk_length_schedule
                    ),
                )
                await ws_conn.send_str(json.dumps(init_pkt))
            except Exception:
                if try_i + 1 == max_retry:
                    logger.exception(
                        f"failed to connect to 11labs after {max_retry} retries"
                    )
                    return

                retry_delay = min(try_i * 5, 5)  # max 5s
                logger.warning(
                    f"failed to connect to 11labs, retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

        assert ws_conn is not None

        all_tokens_consumed = False

        async def send_task():
            nonlocal all_tokens_consumed

            async for token in token_rx:
                if token == "":
                    continue  # empty token is closing the stream in 11labs protocol

                # try_trigger_generation=True is a bad practice, we expose
                # chunk_length_schedule instead
                data_pkt = dict(
                    text=f"{token} ",  # must always end with a space
                    try_trigger_generation=False,
                )
                await ws_conn.send_str(json.dumps(data_pkt))

            # no more token, mark eos
            eos_pkt = dict(text="")
            await ws_conn.send_str(json.dumps(eos_pkt))

            all_tokens_consumed = True

        async def recv_task():
            segment_id = utils.nanoid()
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if all_tokens_consumed:
                        return  # close is expected

                    raise Exception(
                        "11labs connection closed unexpectedly, not all tokens have been consumed"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    # audio frames are serialized in base64..
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data), segment_id)
                except Exception:
                    logger.exception("failed to process 11labs message")

        try:
            await asyncio.gather(send_task(), recv_task())
        except Exception:
            logger.exception("11labs ws connection failed")

    def _process_stream_event(self, data: dict, segment_id: str) -> None:
        encoding = _encoding_from_format(self._opts.encoding)
        audio = data.get("audio")

        if data.get("error"):
            logger.error("11labs reported an error: %s", data["error"])
            return
        elif audio:
            b64data = base64.b64decode(audio)
            if encoding == "mp3":
                for frame in self._mp3_decoder.decode_chunk(b64data):
                    self._event_queue.put_nowait(
                        tts.SynthesizedAudio(
                            segment_id=segment_id,
                            frame=frame,
                        )
                    )
            else:
                self._event_queue.put_nowait(
                    tts.SynthesizedAudio(
                        segment_id=segment_id,
                        frame=rtc.AudioFrame(
                            data=b64data,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(b64data) // 2,
                        ),
                    )
                )

            return
        elif data.get("isFinal"):
            self._event_queue.put_nowait(
                tts.SynthesizedAudio(
                    segment_id=segment_id,
                    frame=rtc.AudioFrame(
                        data=bytearray(),
                        sample_rate=self._opts.sample_rate,
                        num_channels=1,
                        samples_per_channel=0,
                    ),
                    end_of_segment=True,
                )
            )
            return  # last message

        logger.error("unexpected 11labs message %s", data)

    async def __anext__(self) -> tts.SynthesizedAudio:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


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
