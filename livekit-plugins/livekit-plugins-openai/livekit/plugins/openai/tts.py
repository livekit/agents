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

from dataclasses import dataclass
import os
import io
import torchaudio
import torch
from typing import Optional
from livekit import rtc
from livekit.agents import tts
import openai
from .models import TTSModels, TTSVoices


@dataclass
class SynthesisOptions:
    model: TTSModels = "tts-1"
    voice: TTSVoices = "alloy"


class TTS(tts.TTS):
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(streaming_supported=False)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def synthesize(
        self, text: str, opts: tts.SynthesisOptions = tts.SynthesisOptions()
    ) -> tts.SynthesizedAudio:
        speech_res = await self._client.audio.speech.create(
            model=getattr(opts, "model", "tts-1"),
            voice=getattr(opts, "voice", "alloy"),
            response_format="mp3",
            input=text,
        )

        data = await speech_res.aread()
        tensor, sample_rate = torchaudio.load(io.BytesIO(data), format="mp3")

        with io.BytesIO() as buffer:
            torch.save(tensor, buffer)
            data = buffer.getvalue()

        num_channels = tensor.shape[0]
        frame = rtc.AudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=tensor.shape[-1],
        )

        return tts.SynthesizedAudio(text=text, data=frame)
