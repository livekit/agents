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

import contextlib
import asyncio
import logging
import base64
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import List, Optional
import aiohttp
from livekit import rtc
from livekit.agents import tts
from .models import TTSModels


@dataclass
class Voice:
    id: str
    name: str
    category: str
    settings: Optional["VoiceSettings"] = None


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: Optional[float] = None  # [0.0 - 1.0]
    use_speaker_boost: Optional[bool] = False


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
STREAM_EOS = ""


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
        model_id: TTSModels = "eleven_multilingual_v2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        sample_rate: int = 24000,
        latency: int = 2,
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ELEVEN_API_KEY must be set")

        self._session = aiohttp.ClientSession()
        self._config = TTSOptions(
            voice=voice,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            sample_rate=sample_rate,
            latency=latency,
        )

    async def list_voices(self) -> List[Voice]:
        async with self._session.get(
            f"{self._config.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._config.api_key},
        ) as resp:
            data = await resp.json()
            return dict_to_voices_list(data)

    async def synthesize(
        self,
        *,
        text: str,
    ) -> tts.SynthesizedAudio:
        voice = self._config.voice
        async with self._session.post(
            f"{self._config.base_url}/text-to-speech/{voice.id}?output_format=pcm_44100",
            headers={AUTHORIZATION_HEADER: self._config.api_key},
            json=dict(
                text=text,
                model_id=self._config.model_id,
                voice_settings=dataclasses.asdict(voice.settings)
                if voice.settings
                else None,
            ),
        ) as resp:
            data = await resp.read()
            return tts.SynthesizedAudio(
                text=text,
                data=rtc.AudioFrame(
                    data=data,
                    sample_rate=44100,
                    num_channels=1,
                    samples_per_channel=len(data) // 2,  # 16-bit
                ),
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
                logging.error(f"elevenlabs synthesis task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)
        self._text = ""

    def _stream_url(self) -> str:
        base_url = self._config.base_url
        voice_id = self._config.voice.id
        model_id = self._config.model_id
        return f"{base_url}/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format=pcm_{self._config.sample_rate}&optimize_streaming_latency={self._config.latency}"

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
        retry_text_queue: asyncio.Queue[str] = asyncio.Queue()
        while True:
            try:
                ws = await self._try_connect()
                retry_count = 0  # reset retry count

                listen_task = asyncio.create_task(self._listen_task(ws))

                # forward queued text to 11labs
                started = False
                while not ws.closed:
                    text = None
                    if not retry_text_queue.empty():
                        text = await retry_text_queue.get()
                        retry_text_queue.task_done()
                    else:
                        text = await self._queue.get()

                    if not started:
                        self._event_queue.put_nowait(
                            tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
                        )
                        started = True
                    text_packet = dict(
                        text=text,
                        try_trigger_generation=True,
                    )

                    # This case can happen in normal operation because 11labs will not
                    # keep connections open indefinitely if we are not sending data.
                    try:
                        await ws.send_str(json.dumps(text_packet))
                    except Exception:
                        await retry_text_queue.put(text)
                        break

                    # We call self._queue.task_done() even if we are retrying the text because
                    # all text has gone through self._queue. An exception may have short-circuited
                    # out of the loop so task_done() will not have already been called on text that
                    # is being retried.
                    self._queue.task_done()
                    if text == STREAM_EOS:
                        await listen_task
                        # We know 11labs is closing the stream after each request/flush
                        self._event_queue.put_nowait(
                            tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
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
                    logging.error(f"failed to connect to ElevenLabs: {e}")
                    break

                retry_delay = min(retry_count * 5, 5)  # max 5s
                retry_count += 1
                logging.warning(
                    f"failed to connect to ElevenLabs: {e} - retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

        self._closed = True

    async def _try_connect(self) -> aiohttp.ClientWebSocketResponse:
        ws = await self._session.ws_connect(
            self._stream_url(),
            headers={AUTHORIZATION_HEADER: self._config.api_key},
        )

        voice = self._config.voice
        voice_settings = dataclasses.asdict(voice.settings) if voice.settings else None

        init_packet = dict(
            text=" ",
            voice_settings=voice_settings,
        )
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

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            msg = json.loads(msg.data)
            if msg.get("audio"):
                data = base64.b64decode(msg["audio"])
                audio_frame = rtc.AudioFrame(
                    data=data,
                    sample_rate=self._config.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(data) // 2,
                )
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(
                        type=tts.SynthesisEventType.AUDIO,
                        audio=tts.SynthesizedAudio(text="", data=audio_frame),
                    )
                )
            elif msg.get("isFinal"):
                break
            else:
                logging.error(f"Unhandled message from ElevenLabs: {msg}")

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
