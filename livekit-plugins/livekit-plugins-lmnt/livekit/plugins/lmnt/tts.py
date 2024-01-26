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

import aiohttp
import asyncio
import base64
import contextlib
import io
import json
import logging
import numpy as np
import os
import torchaudio
from dataclasses import dataclass
from livekit import rtc
from livekit.agents import tts
from typing import List, Optional


@dataclass
class Voice:
    id: str
    name: str
    owner: str


DEFAULT_VOICE = Voice(
    id="lily",
    name="Lily",
    owner="system"
)

API_BASE_URL_V1 = "https://api.lmnt.com/v1"
AUTHORIZATION_HEADER = "X-API-Key"
STREAM_EOS = ""
LMNT_SAMPLE_RATE = 24000


@dataclass
class TTSOptions:
    api_key: str
    voice: Voice
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("LMNT_API_KEY")
        if not api_key:
            raise ValueError("LMNT_API_KEY must be set")

        self._session = aiohttp.ClientSession()
        self._config = TTSOptions(
            voice=voice,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
        )

    async def list_voices(self) -> List[Voice]:
        async with self._session.get(
            f"{self._config.base_url}/ai/voice/list",
            headers={AUTHORIZATION_HEADER: self._config.api_key},
        ) as resp:
            data = await resp.json()
            return list_to_voices_list(data)

    async def synthesize(
        self,
        *,
        text: str,
    ) -> tts.SynthesizedAudio:
        voice = self._config.voice
        async with self._session.post(
            f"{self._config.base_url}/ai/speech",
            headers={AUTHORIZATION_HEADER: self._config.api_key},
            json=dict(
                text=text,
                voice=voice.id
            ),
        ) as resp:
            data = await resp.read()
            msg = json.loads(data)
            audio = base64.b64decode(msg["audio"])
            audio_frame = decode_mp3_to_frame(audio)
            return tts.SynthesizedAudio(
                text=text,
                data=audio_frame
            )

    def stream(
        self,
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(self._session, self._config)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: TTSOptions,
    ):
        self._config = config
        self._session = session

        self._queue = asyncio.Queue[str]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent]()
        self._closed = False

        self._main_task = asyncio.create_task(self._run(max_retry=32))

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(
                    f"lmnt synthesis task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)
        self._text = ""

    def push_text(self, token: str) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if not token or len(token) == 0:
            return

        # TODO: Native word boundary detection may not be good enough for all languages
        # fmt: off
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        # fmt: on

        self._text += token
        if token[-1] in splitters:
            self._queue.put_nowait(self._text)
            self._text = ""

    async def _run(self, max_retry: int) -> None:
        retry_count = 0
        listen_task: Optional[asyncio.Task] = None
        ws: Optional[aiohttp.ClientWebSocketResponse] = None
        while True:
            try:
                ws = await self._try_connect()
                retry_count = 0  # reset retry count

                listen_task = asyncio.create_task(self._listen_task(ws))

                # forward queued text to LMNT
                started = False
                while not ws.closed:
                    text = await self._queue.get()
                    if not started:
                        self._event_queue.put_nowait(
                            tts.SynthesisEvent(
                                type=tts.SynthesisEventType.STARTED)
                        )
                        started = True
                    if text != STREAM_EOS:
                        await ws.send_str(json.dumps({"text": text}))
                        self._queue.task_done()
                    else:
                        await ws.send_str(json.dumps({"eof": True}))
                        self._queue.task_done()
                        await listen_task
                        # LMNT closes the socket after we send EOF.
                        self._event_queue.put_nowait(
                            tts.SynthesisEvent(
                                type=tts.SynthesisEventType.FINISHED)
                        )
                        break

            except asyncio.CancelledError:
                if ws:
                    await ws.close()
                    if listen_task:
                        await asyncio.shield(listen_task)
                break
            except Exception as e:
                if retry_count > max_retry and max_retry > 0:
                    logging.error(f"failed to connect to LMNT: {e}")
                    break

                retry_delay = min(retry_count * 5, 5)  # max 5s
                retry_count += 1
                logging.warning(
                    f"failed to connect to LMNT: {e} - retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

        self._closed = True

    async def _try_connect(self) -> aiohttp.ClientWebSocketResponse:
        ws = await self._session.ws_connect(f"{self._config.base_url}/ai/speech/stream")

        init_packet = {
            AUTHORIZATION_HEADER: self._config.api_key,
            "voice": self._config.voice.id,
        }
        await ws.send_str(json.dumps(init_packet))
        return ws

    async def _listen_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

            if msg.type != aiohttp.WSMsgType.BINARY:
                continue

            audio_frame = decode_mp3_to_frame(msg.data)
            self._event_queue.put_nowait(
                tts.SynthesisEvent(
                    type=tts.SynthesisEventType.AUDIO,
                    audio=tts.SynthesizedAudio(text="", data=audio_frame),
                )
            )

    async def flush(self) -> None:
        self._queue.put_nowait(self._text + " ")
        self._text = ""
        self._queue.put_nowait(STREAM_EOS)
        await self._queue.join()

    async def aclose(self) -> None:
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def __anext__(self) -> tts.SynthesisEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()


def list_to_voices_list(data: list) -> List[Voice]:
    voices = []
    for voice in data:
        voices.append(
            Voice(
                id=voice["id"],
                name=voice["name"],
                owner=voice["owner"],
            )
        )
    return voices


def _float_to_int16(array):
    return np.clip(np.floor(array * 32767.5), -32768, 32767).astype(np.int16)


def decode_mp3_to_frame(audio: bytes) -> rtc.AudioFrame:
    tensor, _ = torchaudio.load(io.BytesIO(audio), format="mp3")
    pcm16_audio = _float_to_int16(tensor.numpy())

    return rtc.AudioFrame(
        data=pcm16_audio.tobytes(),
        sample_rate=LMNT_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=tensor.shape[-1],
    )
