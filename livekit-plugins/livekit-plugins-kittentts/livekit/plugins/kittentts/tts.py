# Copyright 2026 LiveKit, Inc.
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
import uuid
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_MODEL = "KittenML/kitten-tts-nano-0.8"
DEFAULT_VOICE = "expr-voice-5-m"
DEFAULT_SPEED = 1.0


@dataclass
class _TTSOptions:
    model: str
    voice: str
    speed: float
    clean_text: bool
    cache_dir: str | None
    backend: str | None


def _audio_to_pcm16(audio: Any) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).squeeze()
    if samples.size == 0:
        return b""
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def _next_chunk(iterator: Any) -> Any | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        clean_text: bool = True,
        cache_dir: str | None = None,
        backend: str | None = None,
    ) -> None:
        """Create a KittenTTS text-to-speech instance."""
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            speed=speed,
            clean_text=clean_text,
            cache_dir=cache_dir,
            backend=backend,
        )
        self._model: Any | None = None
        self._model_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "KittenML"

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        clean_text: NotGivenOr[bool] = NOT_GIVEN,
        cache_dir: NotGivenOr[str | None] = NOT_GIVEN,
        backend: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        reset_model = False
        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            reset_model = True
        if is_given(cache_dir) and cache_dir != self._opts.cache_dir:
            self._opts.cache_dir = cache_dir
            reset_model = True
        if is_given(backend) and backend != self._opts.backend:
            self._opts.backend = backend
            reset_model = True
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(clean_text):
            self._opts.clean_text = clean_text
        if reset_model:
            self._model = None

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is not None:
                return self._model
            opts = replace(self._opts)

            def load_model() -> Any:
                try:
                    from kittentts import KittenTTS
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "KittenTTS is required. Install it with "
                        "`pip install "
                        "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/"
                        "kittentts-0.8.1-py3-none-any.whl`."
                    ) from e

                return KittenTTS(opts.model, cache_dir=opts.cache_dir, backend=opts.backend)

            self._model = await asyncio.to_thread(load_model)
            return self._model

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        model = await self._tts._ensure_model()
        iterator = model.generate_stream(
            self.input_text,
            voice=self._opts.voice,
            speed=self._opts.speed,
            clean_text=self._opts.clean_text,
        )

        while True:
            chunk = await asyncio.to_thread(_next_chunk, iterator)
            if chunk is None:
                break
            pcm = _audio_to_pcm16(chunk)
            if pcm:
                output_emitter.push(pcm)

        output_emitter.flush()
