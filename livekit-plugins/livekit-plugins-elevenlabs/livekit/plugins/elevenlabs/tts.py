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

import json
import os
from livekit import rtc
from livekit.agents import tts
import dataclasses
from dataclasses import dataclass
from typing import List, Optional
import aiohttp
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
STREAM_EOS = json.dumps(dict(test=""))


class TTS(tts.TTS):
    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        super().__init__(streaming_supported=False)
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
                samples_per_channel=len(data) // 2, # 16-bit
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
