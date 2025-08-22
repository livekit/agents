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

import json
from dataclasses import dataclass, replace
from typing import Final

from livekit.agents import APIConnectionError, APIConnectOptions, tts, utils
from livekit.agents.job import get_job_context
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
)

from .models import HG_MODEL, TTSModels, TTSVoices, resolve_model_name
from .runner import _KittenRunner

SAMPLE_RATE: Final[int] = 24000
NUM_CHANNELS: Final[int] = 1


@dataclass
class _TTSOptions:
    model_name: TTSModels | str
    voice: TTSVoices | str
    speed: float


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model_name: TTSModels | str = HG_MODEL,
        voice: TTSVoices | str = "expr-voice-5-m",
        speed: float = 1.0,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model_name=resolve_model_name(model_name),
            voice=voice,
            speed=speed,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        executor = get_job_context().inference_executor

        try:
            payload = {
                "text": self._input_text,
                "voice": self._opts.voice,
                "speed": self._opts.speed,
            }
            data = json.dumps(payload).encode()
            result = await executor.do_inference(_KittenRunner.INFERENCE_METHOD, data)
            if result is None:
                raise APIConnectionError("Failed to get result from inference executor")

            output_emitter.push(result)
            output_emitter.flush()
        except Exception as e:
            raise APIConnectionError(f"Kitten TTS synthesis failed: {e}") from e
