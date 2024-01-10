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
    voice_id: str
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
    voice_id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    category="premade",
    settings=VoiceSettings(
        stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
    ),
)

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
STREAM_EOS = json.dumps(dict(text=""))


class TTS(tts.TTS):
    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ELEVEN_API_KEY must be set")

        base_url = base_url or os.environ.get("ELEVEN_BASE_URL", API_BASE_URL_V1)
        self._base_url = base_url
        self._api_key = api_key

        self._session = aiohttp.ClientSession()

    async def list_voices(self) -> List[Voice]:
        async with self._session.get(
            f"{self._base_url}/voices", headers={AUTHORIZATION_HEADER: self._api_key}
        ) as resp:
            data = await resp.json()
            return dict_to_voices_list(data)

    async def synthesize(
        self,
        *,
        text: str,
        model_id: TTSModels = "eleven_multilingual_v2",
        voice: Voice = DEFAULT_VOICE,
    ) -> tts.SynthesizedAudio:
        async with self._session.post(
            f"{self._base_url}/text-to-speech/{voice.voice_id}?output_format=pcm_44100",
            headers={AUTHORIZATION_HEADER: self._api_key},
            json=dict(
                text=text,
                model_id=model_id,
                voice_settings=dataclasses.asdict(voice.settings)
                if voice.settings
                else None,
            ),
        ) as resp:
            data = await resp.read()
            return rtc.AudioFrame(
                data=data,
                sample_rate=44100,
                num_channels=1,
                samples_per_channel=len(data) // 2,  # 16-bit
            )

    def stream(
        self,
        *,
        model_id: TTSModels = "eleven_multilingual_v2",
        voice: Voice = DEFAULT_VOICE,
        latency: int = 2,
    ) -> tts.SynthesizeStream:
        return Stream(
            url=self._base_url,
            api_key=self._api_key,
            voice_id=voice.voice_id,
            model_id=model_id,
            latency=latency,
        )


def dict_to_voices_list(data: dict) -> List[Voice]:
    voices = []
    for voice in data["voices"]:
        voices.append(
            Voice(
                voice_id=voice["voice_id"],
                name=voice["name"],
                category=voice["category"],
                settings=None,
            )
        )
    return voices


class Stream(tts.SynthesizeStream):
    class _Session:
        def __init__(self, ws: aiohttp.ClientWebSocketResponse):
            self.ws = ws
            self.final_future = asyncio.Future()

    def __init__(
        self, url: str, api_key: str, voice_id: str, model_id: str, latency: int
    ):
        self._api_key = api_key
        self._uri = f"{url}/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format=pcm_44100&optimize_streaming_latency={latency}"
        self._http_session = aiohttp.ClientSession()
        self._session: Optional[Stream._Session] = None
        self._session_lock = asyncio.Lock()
        self._tasks = set()
        self._output_lock = asyncio.Lock()
        self._output_queue = asyncio.Queue()

    async def warmup(self):
        await self._new_session_if_needed()

    async def _new_session_if_needed(self):
        async with self._session_lock:
            if self._session is not None:
                return

            ws = await self._http_session.ws_connect(
                self._uri, headers={AUTHORIZATION_HEADER: self._api_key}
            )
            payload = json.dumps(dict(text=" ", try_trigger_generation=False))
            await ws.send_str(payload)
            self._session = Stream._Session(ws=ws)
            t = asyncio.create_task(self._result_loop(self._session))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    def push_text(self, token: str):
        asyncio.create_task(self._push_text(token))

    async def _push_text(self, token: str):
        if self._session is None:
            await self._new_session_if_needed()

        payload = json.dumps(dict(text=token + " ", try_trigger_generation=False))
        await self._session.ws.send_str(payload)

    async def flush(self) -> None:
        flush_session: Optional[Stream._Session] = None
        async with self._session_lock:
            if self._session is None:
                raise RuntimeError("Haven't started a session, cannot flush")

            flush_session = self._session
            self._session = None

        if flush_session is not None:
            await flush_session.ws.send_str(STREAM_EOS)
            await flush_session.final_future
            await flush_session.ws.close()

    async def close(self) -> None:
        if self._session is not None:
            await self.flush()

        await self._output_queue.put(None)

    async def _result_loop(self, session: "Stream._Session"):
        async with self._output_lock:
            while True:
                msg = await session.ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data["isFinal"]:
                        session.final_future.set_result(None)
                        break
                    bytes_audio = base64.b64decode(data["audio"])
                    audio_frame = rtc.AudioFrame(
                        data=bytes_audio,
                        sample_rate=44100,
                        num_channels=1,
                        samples_per_channel=len(bytes_audio) // 2,
                    )
                    await self._output_queue.put(
                        tts.SynthesisEvent(
                            audio=tts.SynthesizedAudio(text="", data=audio_frame)
                        )
                    )

                else:
                    break

        if not session.final_future.done():
            session.final_future.set_result(None)

        # If this session is still the current session, clear it
        async with self._session_lock:
            if self._session == session:
                self._session = None

    async def __anext__(self) -> tts.SynthesisEvent:
        item = await self._output_queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def __aiter__(self) -> "Stream":
        return self
