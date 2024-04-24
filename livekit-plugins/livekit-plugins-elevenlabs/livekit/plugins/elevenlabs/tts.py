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

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import AsyncIterable, List

import aiohttp
from livekit import rtc
from livekit.agents import aio, tts

from .log import logger
from .models import TTSModels


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
class TTSOptions:
    api_key: str
    voice: Voice
    model_id: TTSModels
    base_url: str
    sample_rate: int
    latency: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model_id: TTSModels = "eleven_turbo_v2",
        api_key: str | None = None,
        base_url: str | None = None,
        sample_rate: int = 24000,
        latency: int = 3,
    ) -> None:
        super().__init__(
            streaming_supported=True, sample_rate=sample_rate, num_channels=1
        )
        api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ELEVEN_API_KEY must be set")

        self._session = aiohttp.ClientSession()
        self._opts = TTSOptions(
            voice=voice,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            sample_rate=sample_rate,
            latency=latency,
        )

    async def list_voices(self) -> List[Voice]:
        async with self._session.get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            data = await resp.json()
            return dict_to_voices_list(data)

    def synthesize(
        self,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        voice = self._opts.voice
        url = f"{self._opts.base_url}/text-to-speech/{voice.id}?output_format=pcm_{self._opts.sample_rate}"

        async def generator():
            try:
                async with self._session.post(
                    url,
                    headers={AUTHORIZATION_HEADER: self._opts.api_key},
                    json=dict(
                        text=text,
                        model_id=self._opts.model_id,
                        voice_settings=dataclasses.asdict(voice.settings)
                        if voice.settings
                        else None,
                    ),
                ) as resp:
                    data = await resp.read()
                    yield tts.SynthesizedAudio(
                        text=text,
                        data=rtc.AudioFrame(
                            data=data,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,  # 16-bit
                        ),
                    )
            except Exception as e:
                logger.error(f"failed to synthesize: {e}")

        return generator()

    def stream(
        self,
    ) -> "SynthesizeStream":
        return SynthesizeStream(self._session, self._opts)


class SynthesizeStream(tts.SynthesizeStream):
    _STREAM_EOS = ""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        opts: TTSOptions,
        max_retry: int = 32,
    ):
        self._opts = opts
        self._session = session

        self._queue = asyncio.Queue[str | None]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent | None]()
        self._closed = False
        self._text = ""

        self._main_task = asyncio.create_task(self._run(max_retry))

    def _stream_url(self) -> str:
        base_url = self._opts.base_url
        voice_id = self._opts.voice.id
        model_id = self._opts.model_id
        sample_rate = self._opts.sample_rate
        latency = self._opts.latency
        return f"{base_url}/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format=pcm_{sample_rate}&optimize_streaming_latency={latency}"

    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if token is None:
            self._flush_if_needed()
            return

        if len(token) == 0:
            # 11labs marks the EOS with an empty string, avoid users from pushing empty strings
            return

        # TODO: Naive word boundary detection may not be good enough for all languages
        # fmt: off
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        # fmt: on

        self._text += token

        while True:
            last_split = -1
            for i, c in enumerate(self._text):
                if c in splitters:
                    last_split = i
                    break

            if last_split == -1:
                break

            seg = self._text[: last_split + 1]
            seg = seg.strip() + " "  # 11labs expects a space at the end
            self._queue.put_nowait(seg)
            self._text = self._text[last_split + 1 :]

    async def aclose(self, *, wait: bool = True) -> None:
        self._flush_if_needed()
        self._queue.put_nowait(None)
        self._closed = True

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    def _flush_if_needed(self) -> None:
        seg = self._text.strip()
        if len(seg) > 0:
            self._queue.put_nowait(seg + " ")

        self._text = ""
        self._queue.put_nowait(SynthesizeStream._STREAM_EOS)

    async def _run(self, max_retry: int) -> None:
        retry_count = 0
        ws: aiohttp.ClientWebSocketResponse | None = None
        ws_task: asyncio.Task | None = None
        data_tx: aio.ChanSender[str] | None = None

        try:
            while True:
                ws_connected = ws is not None and not ws.closed
                try:
                    data = await self._queue.get()

                    if data is None:
                        if ws_task is not None:
                            await ws_task
                        break

                    if not ws_connected:
                        if data == SynthesizeStream._STREAM_EOS:
                            continue

                        with contextlib.suppress(asyncio.CancelledError):
                            if ws_task is not None:
                                await ws_task

                        ws = await self._session.ws_connect(
                            self._stream_url(),
                            headers={AUTHORIZATION_HEADER: self._opts.api_key},
                        )
                        data_tx, data_rx = aio.channel()
                        ws_task = asyncio.create_task(self._run_ws(ws, data_rx))

                    assert data_tx is not None
                    assert ws_task is not None
                    assert ws is not None

                    data_tx.send_nowait(data)

                except Exception:
                    if retry_count >= max_retry:
                        logger.exception(
                            f"failed to connect to 11labs after {max_retry} retries"
                        )
                        break

                    retry_delay = min(retry_count * 5, 5)  # max 5s
                    retry_count += 1

                    logger.warning(
                        f"failed to connect to 11labs, retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)

        except Exception:
            logger.exception("11labs task failed")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                if ws_task is not None:
                    ws_task.cancel()
                    await ws_task

            self._event_queue.put_nowait(None)

    async def _run_ws(
        self, ws: aiohttp.ClientWebSocketResponse, data_rx: aio.ChanReceiver[str]
    ) -> None:
        closing_ws = False

        self._event_queue.put_nowait(
            tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
        )

        async def send_task():
            nonlocal closing_ws

            # 11labs stream must be initialized with a space
            voice = self._opts.voice
            voice_settings = (
                dataclasses.asdict(voice.settings) if voice.settings else None
            )
            init_pkt = dict(
                text=" ",
                voice_settings=voice_settings,
            )
            await ws.send_str(json.dumps(init_pkt))

            while True:
                data = await data_rx.recv()
                data_pkt = dict(
                    text=data,
                    try_trigger_generation=False,
                )
                await ws.send_str(json.dumps(data_pkt))

                if data == SynthesizeStream._STREAM_EOS:
                    closing_ws = True
                    return

        async def recv_task():
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected
                        return

                    raise Exception("11labs connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue

                try:
                    data: dict = json.loads(msg.data)
                    if data.get("audio"):
                        b64data = base64.b64decode(data["audio"])
                        frame = rtc.AudioFrame(
                            data=b64data,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                        self._event_queue.put_nowait(
                            tts.SynthesisEvent(
                                type=tts.SynthesisEventType.AUDIO,
                                audio=tts.SynthesizedAudio(text="", data=frame),
                            )
                        )
                    elif data.get("isFinal"):
                        return
                except Exception:
                    logger.exception("failed to process 11labs message")

        try:
            await asyncio.gather(send_task(), recv_task())
        except Exception:
            logger.exception("11labs connection failed")
        finally:
            self._event_queue.put_nowait(
                tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
            )

    async def __anext__(self) -> tts.SynthesisEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


def dict_to_voices_list(data: dict) -> List[Voice]:
    voices = []
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
