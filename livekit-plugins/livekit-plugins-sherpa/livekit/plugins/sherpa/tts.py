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
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sherpa_onnx

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    tts,
    utils,
)
from livekit.agents.types import NotGiven
from livekit.agents.utils import is_given

from .models import (
    TTS_MODELS,
    KittenModel,
    KokoroModel,
    KokoroVoice,
    TTSModel,
    TTSModels,
    VitsModel,
    local_model_dir,
)

ChunkCallback = Callable[[npt.NDArray[np.float32], float], int]


class GeneratedAudio(Protocol):
    samples: npt.NDArray[np.float32]
    sample_rate: int


class OfflineTts(Protocol):
    """The subset of `sherpa_onnx.OfflineTts` this plugin uses."""

    @property
    def sample_rate(self) -> int: ...
    def generate(
        self, text: str, sid: int = 0, speed: float = 1.0, callback: ChunkCallback | None = None
    ) -> GeneratedAudio: ...


Voice = KokoroVoice | str | int
"""A speaker name from the model's table (Kokoro names are typed) or a speaker id."""


@dataclass
class _TTSOptions:
    sid: int
    speed: float


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice | None = None,
        speed: float = 1.0,
        model: TTSModels = "kokoro-int8-multi-lang-v1_0",
        num_threads: int = 2,
        engine: OfflineTts | None = None,
        speakers: Sequence[str] | None = None,
    ) -> None:
        """Fully offline TTS run by sherpa-onnx: Kokoro by default, or Kitten, Piper/VITS
        and any other sherpa-onnx TTS model through `engine=`.

        Non-streaming: the agent wraps it in `tts.StreamAdapter`, which splits the LLM output
        into sentences. Audio of a sentence is pushed as soon as sherpa-onnx produces it.

        Args:
            voice: A speaker name from the model's table (e.g. "af_heart" for Kokoro,
                "expr-voice-2-f" for Kitten) or a speaker id. Defaults to the model's default.
            speed: Speaking rate. Larger is faster.
            model: Built-in model to load. Fetch it first with `download-files`.
            num_threads: ONNX Runtime threads.
            engine: Use an already built `sherpa_onnx.OfflineTts` instead of `model`.
            speakers: The speaker names of `engine`, in speaker id order, so `voice` can be a
                name. Without it, `voice` must be a speaker id.
        """
        spec = TTS_MODELS[model]
        self._speakers = tuple(speakers or ()) if engine is not None else spec.speakers
        default = 0 if engine is not None else spec.default_voice
        self._engine = engine or _load_engine(spec, num_threads=num_threads)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._engine.sample_rate,
            num_channels=1,
        )
        self._model_name = model if engine is None else "custom"
        self._opts = _TTSOptions(
            sid=_speaker_id(voice if voice is not None else default, self._speakers), speed=speed
        )
        # sherpa-onnx makes no promise that one engine can run two generations at once.
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "sherpa-onnx"

    def update_options(
        self,
        *,
        voice: NotGivenOr[Voice] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if not isinstance(voice, NotGiven):
            self._opts.sid = _speaker_id(voice, self._speakers)
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        if not self._input_text.strip():
            output_emitter.flush()
            return

        loop = asyncio.get_running_loop()
        chunks = asyncio.Queue[bytes | None]()
        cancelled = threading.Event()

        def on_chunk(samples: npt.NDArray[np.float32], progress: float) -> int:
            loop.call_soon_threadsafe(chunks.put_nowait, _to_pcm16(samples))
            return 1 if cancelled.is_set() else 0

        def generate() -> None:
            try:
                with self._tts._lock:
                    self._tts._engine.generate(
                        self._input_text,
                        sid=self._opts.sid,
                        speed=self._opts.speed,
                        callback=on_chunk,
                    )
            finally:
                loop.call_soon_threadsafe(chunks.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(generate))
        try:
            while (chunk := await chunks.get()) is not None:
                output_emitter.push(chunk)
            await task
        finally:
            # Stops sherpa-onnx at its next chunk if the agent was interrupted.
            cancelled.set()
            await utils.aio.cancel_and_wait(task)
        output_emitter.flush()


def _speaker_id(voice: Voice, speakers: Sequence[str]) -> int:
    if isinstance(voice, int):
        if voice < 0 or (speakers and voice >= len(speakers)):
            raise ValueError(f"speaker id {voice} is out of range for {len(speakers)} speakers")
        return voice
    if voice not in speakers:
        known = ", ".join(speakers) if speakers else "none (pass speakers= or a speaker id)"
        raise ValueError(f"unknown voice {voice!r}; this model has: {known}")
    return speakers.index(voice)


def _load_engine(spec: TTSModel, *, num_threads: int) -> OfflineTts:
    root = Path(local_model_dir(spec))
    model = sherpa_onnx.OfflineTtsModelConfig(num_threads=num_threads)
    if isinstance(spec, KokoroModel):
        model.kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(root / spec.model),
            voices=str(root / spec.voices),
            tokens=str(root / spec.tokens),
            lexicon=str(root / spec.lexicon),
            data_dir=str(root / spec.data_dir),
            lang=spec.language,
        )
    elif isinstance(spec, KittenModel):
        model.kitten = sherpa_onnx.OfflineTtsKittenModelConfig(
            model=str(root / spec.model),
            voices=str(root / spec.voices),
            tokens=str(root / spec.tokens),
            data_dir=str(root / spec.data_dir),
        )
    elif isinstance(spec, VitsModel):
        model.vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(root / spec.model),
            tokens=str(root / spec.tokens),
            data_dir=str(root / spec.data_dir),
        )
    else:
        raise TypeError(f"unsupported TTS model spec: {type(spec).__name__}")
    engine: OfflineTts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1)
    )
    return engine


def _to_pcm16(samples: npt.NDArray[np.float32]) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
