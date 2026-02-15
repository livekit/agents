# Copyright 2024 LiveKit, Inc.
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
import contextlib
import queue
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from pocket_tts import TTSModel

from livekit.agents import APIConnectionError, APITimeoutError, tts
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)
from livekit.agents.utils import shortuuid

from .log import logger

DEFAULT_VOICE = "alba"
NATIVE_SAMPLE_RATE = 24000


class TTSMetricsCallback(Protocol):
    def __call__(self, *, ttfb: float, duration: float, audio_duration: float) -> None: ...


OptionalTTSMetricsCallback = TTSMetricsCallback | None


@dataclass
class _GenerationError:
    error: Exception


class _GenerationDone:
    pass


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        temperature: float = 0.7,
        lsd_decode_steps: int = 1,
        sample_rate: int = NATIVE_SAMPLE_RATE,
        metrics_callback: OptionalTTSMetricsCallback = None,
    ) -> None:
        """
        Create a new instance of Pocket TTS.

        Args:
            voice: Built-in voice name or path to an audio prompt file.
            temperature: Sampling temperature used by Pocket TTS.
            lsd_decode_steps: Number of LSD decode steps.
            sample_rate: Requested output sample rate. Pocket runs at 24 kHz native.
            metrics_callback: Optional callback for per-segment generation metrics.
        """
        if sample_rate != NATIVE_SAMPLE_RATE:
            logger.warning(
                "Pocket TTS emits native 24kHz audio. Ignoring sample_rate=%s and using %s.",
                sample_rate,
                NATIVE_SAMPLE_RATE,
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=NATIVE_SAMPLE_RATE,
            num_channels=1,
        )

        self._voice = voice
        self._temperature = temperature
        self._lsd_decode_steps = lsd_decode_steps
        self._metrics_callback = metrics_callback

        self._model: Any = TTSModel.load_model(temp=temperature, lsd_decode_steps=lsd_decode_steps)
        self._voice_state: Any = self._load_voice_state(voice)
        self._generation_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return "pocket-tts"

    @property
    def provider(self) -> str:
        return "Kyutai"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return PocketChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        return PocketSynthesizeStream(tts=self, conn_options=conn_options)

    def _load_voice_state(self, voice: str) -> Any:
        try:
            return self._model.get_state_for_audio_prompt(voice, truncate=True)
        except FileNotFoundError as e:
            raise ValueError(f"Failed to load voice '{voice}': {e}") from e
        except Exception as e:
            if voice == DEFAULT_VOICE:
                raise ValueError(f"Failed to initialize Pocket TTS voice '{voice}': {e}") from e

            logger.warning(
                "Failed to load voice '%s' (%s). Falling back to '%s'.",
                voice,
                e,
                DEFAULT_VOICE,
            )
            try:
                self._voice = DEFAULT_VOICE
                return self._model.get_state_for_audio_prompt(DEFAULT_VOICE, truncate=True)
            except Exception as fallback_error:
                raise ValueError(
                    f"Failed to initialize Pocket TTS fallback voice '{DEFAULT_VOICE}': {fallback_error}"
                ) from fallback_error

    async def _generate_audio_stream(
        self,
        *,
        text: str,
        conn_options: APIConnectOptions,
    ) -> AsyncIterator[bytes]:
        items: queue.Queue[bytes | _GenerationError | _GenerationDone] = queue.Queue()

        def _producer() -> None:
            try:
                for audio_chunk in self._model.generate_audio_stream(
                    self._voice_state,
                    text,
                    copy_state=True,
                ):
                    chunk = _tensor_to_pcm_bytes(audio_chunk)
                    if chunk:
                        items.put(chunk)
            except Exception as e:
                items.put(_GenerationError(error=e))
            finally:
                items.put(_GenerationDone())

        producer_task = asyncio.create_task(
            asyncio.to_thread(_producer),
            name="PocketTTS._producer",
        )

        timeout = _timeout_value(conn_options.timeout)
        deadline = time.perf_counter() + timeout if timeout is not None else None

        try:
            while True:
                item: bytes | _GenerationError | _GenerationDone
                if deadline is None:
                    item = await asyncio.to_thread(items.get)
                else:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise APITimeoutError(
                            f"Pocket TTS synthesis timed out after {conn_options.timeout}s"
                        )
                    try:
                        item = await asyncio.wait_for(asyncio.to_thread(items.get), timeout=remaining)
                    except asyncio.TimeoutError as e:
                        raise APITimeoutError(
                            f"Pocket TTS synthesis timed out after {conn_options.timeout}s"
                        ) from e

                if isinstance(item, _GenerationDone):
                    return

                if isinstance(item, _GenerationError):
                    raise APIConnectionError("Pocket TTS synthesis failed") from item.error

                yield item
        finally:
            if not producer_task.done():
                producer_task.cancel()
            with contextlib.suppress(BaseException):
                await producer_task

    async def _push_generated_audio(
        self,
        *,
        text: str,
        conn_options: APIConnectOptions,
        output_emitter: tts.AudioEmitter,
    ) -> tuple[float, float, float]:
        start_time = time.perf_counter()
        first_chunk_ttfb = -1.0
        total_bytes = 0

        async with self._generation_lock:
            async for chunk in self._generate_audio_stream(text=text, conn_options=conn_options):
                if first_chunk_ttfb < 0:
                    first_chunk_ttfb = time.perf_counter() - start_time
                total_bytes += len(chunk)
                output_emitter.push(chunk)

        generation_duration = time.perf_counter() - start_time
        audio_duration = _bytes_to_duration(total_bytes=total_bytes, sample_rate=self.sample_rate)
        return first_chunk_ttfb, generation_duration, audio_duration


class PocketChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        pocket_tts = cast(TTS, self._tts)
        output_emitter.initialize(
            request_id=shortuuid("TTS_"),
            sample_rate=pocket_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )

        first_chunk_ttfb, generation_duration, audio_duration = await pocket_tts._push_generated_audio(
            text=self._input_text,
            conn_options=self._conn_options,
            output_emitter=output_emitter,
        )

        output_emitter.flush()

        if pocket_tts._metrics_callback:
            pocket_tts._metrics_callback(
                ttfb=first_chunk_ttfb,
                duration=generation_duration,
                audio_duration=audio_duration,
            )


class PocketSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("TTS_"),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        text_buffer = ""
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                await self._flush_text_buffer(
                    text_buffer=text_buffer, output_emitter=output_emitter
                )
                text_buffer = ""
                continue

            text_buffer += data

        await self._flush_text_buffer(text_buffer=text_buffer, output_emitter=output_emitter)

    async def _flush_text_buffer(
        self, *, text_buffer: str, output_emitter: tts.AudioEmitter
    ) -> None:
        if not text_buffer.strip():
            return

        segment_id = shortuuid("SEG_")
        output_emitter.start_segment(segment_id=segment_id)
        await self._synthesize_segment(text_buffer, output_emitter)
        output_emitter.end_segment()

    async def _synthesize_segment(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        self._mark_started()
        pocket_tts = cast(TTS, self._tts)
        first_chunk_ttfb, generation_duration, audio_duration = await pocket_tts._push_generated_audio(
            text=text,
            conn_options=self._conn_options,
            output_emitter=output_emitter,
        )

        if pocket_tts._metrics_callback:
            pocket_tts._metrics_callback(
                ttfb=first_chunk_ttfb,
                duration=generation_duration,
                audio_duration=audio_duration,
            )


def _tensor_to_pcm_bytes(audio_chunk: Any) -> bytes:
    audio = audio_chunk
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()

    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.size == 0:
        return b""

    if audio_np.ndim > 1:
        if audio_np.ndim != 2:
            raise ValueError(f"unsupported audio tensor shape: {audio_np.shape}")
        # Common layouts are [channels, samples] or [samples, channels].
        if audio_np.shape[0] <= audio_np.shape[1]:
            audio_np = np.mean(audio_np, axis=0)
        else:
            audio_np = np.mean(audio_np, axis=1)

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16, copy=False)
    return audio_int16.tobytes()


def _bytes_to_duration(*, total_bytes: int, sample_rate: int) -> float:
    samples = total_bytes / 2.0
    if sample_rate <= 0:
        return 0.0
    return samples / sample_rate


def _timeout_value(timeout: float) -> float | None:
    if timeout <= 0:
        return None
    return timeout


PocketTTS = TTS
