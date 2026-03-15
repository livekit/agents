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
import logging
import re
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from pocket_tts import TTSModel

from livekit.agents import APIConnectionError, APITimeoutError, tts
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)
from livekit.agents.utils import shortuuid

from .log import logger

# Reduce verbosity of pocket_tts library to avoid console spam
logging.getLogger("pocket_tts").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.conditioners.text").setLevel(logging.WARNING)

DEFAULT_VOICE = "alba"
NATIVE_SAMPLE_RATE = 24000
MAX_TTS_SEGMENT_CHARS = 220

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+[\.\)]))\s+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class _GenerationError:
    error: Exception


class _GenerationDone:
    pass


class PocketTTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        temperature: float = 0.7,
        lsd_decode_steps: int = 1,
        sample_rate: int = NATIVE_SAMPLE_RATE,
        max_concurrent_generations: int = 1,
    ) -> None:
        """Create a new instance of Pocket TTS.

        Args:
            voice: Built-in voice name or path to an audio prompt file.
            temperature: Sampling temperature used by Pocket TTS.
            lsd_decode_steps: Number of LSD decode steps.
            sample_rate: Output sample rate. Only native 24kHz is supported.
            max_concurrent_generations: Maximum number of concurrent synthesis tasks
                for this PocketTTS instance.
        """
        if max_concurrent_generations < 1:
            raise ValueError(
                "max_concurrent_generations must be >= 1: "
                f"{max_concurrent_generations}"
            )
        if sample_rate != NATIVE_SAMPLE_RATE:
            raise ValueError(
                "PocketTTS only supports native sample rate "
                f"{NATIVE_SAMPLE_RATE}Hz; received {sample_rate}Hz"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._voice = voice
        self._temperature = temperature
        self._lsd_decode_steps = lsd_decode_steps
        self._max_concurrent_generations = max_concurrent_generations
        self._model: Any = TTSModel.load_model(temp=temperature, lsd_decode_steps=lsd_decode_steps)
        self._voice_state: Any = self._load_voice_state(voice)
        self._generation_semaphore = asyncio.Semaphore(max_concurrent_generations)

        logger.info(
            "PocketTTS configured: sample_rate=%sHz max_concurrent_generations=%s",
            self.sample_rate,
            self._max_concurrent_generations,
        )

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
                raise ValueError(
                    f"Failed to initialize Pocket TTS voice '{voice}': {e}"
                ) from e

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
                    f"Failed to initialize Pocket TTS fallback voice "
                    f"'{DEFAULT_VOICE}': {fallback_error}"
                ) from fallback_error

    async def _generate_audio_stream(
        self,
        *,
        text: str,
        conn_options: APIConnectOptions,
    ) -> AsyncIterator[bytes]:
        items: asyncio.Queue[bytes | _GenerationError | _GenerationDone] = asyncio.Queue()
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()

        def _put_item(item: bytes | _GenerationError | _GenerationDone) -> None:
            try:
                loop.call_soon_threadsafe(items.put_nowait, item)
            except RuntimeError:
                # The event loop is shutting down; drop late producer output.
                return

        def _producer() -> None:
            try:
                for audio_chunk in self._model.generate_audio_stream(
                    self._voice_state,
                    text,
                    copy_state=True,
                ):
                    if stop_event.is_set():
                        break

                    chunk = _tensor_to_pcm_bytes(audio_chunk=audio_chunk)
                    if stop_event.is_set():
                        break
                    if chunk:
                        _put_item(chunk)
            except Exception as e:
                if not stop_event.is_set():
                    _put_item(_GenerationError(error=e))
            finally:
                _put_item(_GenerationDone())

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
                    item = await items.get()
                else:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise APITimeoutError(
                            f"Pocket TTS synthesis timed out after {conn_options.timeout}s"
                        )
                    try:
                        item = await asyncio.wait_for(items.get(), timeout=remaining)
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
            stop_event.set()
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
        chunk_count = 0
        total_bytes = 0

        async with self._generation_semaphore:
            async for chunk in self._generate_audio_stream(
                text=text, conn_options=conn_options
            ):
                if first_chunk_ttfb < 0:
                    first_chunk_ttfb = time.perf_counter() - start_time
                chunk_count += 1
                total_bytes += len(chunk)
                output_emitter.push(chunk)

        # Wall-clock synth duration (queueing + model generation + conversion + emit).
        synth_wall_time = time.perf_counter() - start_time
        audio_duration = _bytes_to_duration(total_bytes=total_bytes, sample_rate=self.sample_rate)
        wall_to_audio_ratio = (
            synth_wall_time / audio_duration if audio_duration > 0 else 0.0
        )

        logger.debug(
            "TTS segment telemetry: chars=%s chunks=%s bytes=%s sample_rate=%sHz "
            "audio_duration=%.3fs synth_wall_time=%.3fs "
            "wall_to_audio_ratio=%.3f",
            len(text),
            chunk_count,
            total_bytes,
            self.sample_rate,
            audio_duration,
            synth_wall_time,
            wall_to_audio_ratio,
        )
        if audio_duration > 0 and wall_to_audio_ratio > 2.0:
            logger.warning(
                "TTS generation slower than realtime: wall_to_audio_ratio=%.3f "
                "(sample_rate=%sHz)",
                wall_to_audio_ratio,
                self.sample_rate,
            )

        return first_chunk_ttfb, synth_wall_time, audio_duration

    def _prepare_text_segments(self, text: str) -> list[str]:
        """Normalize text for TTS and split into short chunks for lower tail latency."""
        cleaned = _sanitize_tts_text(text)
        if not cleaned:
            return []
        return _chunk_tts_text(cleaned, max_chars=MAX_TTS_SEGMENT_CHARS)


class PocketChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: PocketTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        pocket_tts = cast(PocketTTS, self._tts)
        output_emitter.initialize(
            request_id=shortuuid("TTS_"),
            sample_rate=pocket_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )

        text_segments = pocket_tts._prepare_text_segments(self._input_text)
        if not text_segments:
            output_emitter.flush()
            return

        first_chunk_ttfb = -1.0
        total_synth_wall_time = 0.0
        audio_duration = 0.0
        for text_segment in text_segments:
            (
                segment_ttfb,
                segment_synth_wall_time,
                segment_audio_duration,
            ) = await pocket_tts._push_generated_audio(
                text=text_segment,
                conn_options=self._conn_options,
                output_emitter=output_emitter,
            )
            if first_chunk_ttfb < 0 and segment_ttfb >= 0:
                first_chunk_ttfb = segment_ttfb
            total_synth_wall_time += segment_synth_wall_time
            audio_duration += segment_audio_duration

        output_emitter.flush()


class PocketSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: PocketTTS, conn_options: APIConnectOptions) -> None:
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
        pocket_tts = cast(PocketTTS, self._tts)
        text_segments = pocket_tts._prepare_text_segments(text_buffer)
        if not text_segments:
            return

        # LiveKit expects one segment per flushed text buffer in streaming mode.
        output_emitter.start_segment(segment_id=shortuuid("SEG_"))
        try:
            for text_segment in text_segments:
                await self._synthesize_segment(text_segment, output_emitter)
        finally:
            output_emitter.end_segment()

    async def _synthesize_segment(
        self, text: str, output_emitter: tts.AudioEmitter
    ) -> tuple[float, float, float]:
        self._mark_started()
        pocket_tts = cast(PocketTTS, self._tts)
        return await pocket_tts._push_generated_audio(
            text=text,
            conn_options=self._conn_options,
            output_emitter=output_emitter,
        )


def _sanitize_tts_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _MARKDOWN_LINK_RE.sub(r"\1", normalized)

    cleaned_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = _BULLET_PREFIX_RE.sub("", line)
        line = line.lstrip("#> ").strip()
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = line.replace("`", "")
        line = line.replace("*", "")
        line = line.replace("|", " ")
        cleaned_lines.append(line)

    cleaned = " ".join(cleaned_lines)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _chunk_tts_text(text: str, *, max_chars: int) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        for sentence_part in _split_overlong_text(sentence, max_chars=max_chars):
            if not current:
                current = sentence_part
                continue
            candidate = f"{current} {sentence_part}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = sentence_part

    if current:
        chunks.append(current)
    return chunks


def _split_overlong_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0
    for word in words:
        additional_len = len(word) if not current_words else len(word) + 1
        if current_words and current_len + additional_len > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)
            continue

        current_words.append(word)
        current_len += additional_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _tensor_to_pcm_bytes(
    *,
    audio_chunk: Any,
) -> bytes:
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
        dim0, dim1 = audio_np.shape
        dim0_looks_like_channels = dim0 <= 8
        dim1_looks_like_channels = dim1 <= 8

        # Prefer the axis with a "small" channel-like size. For ambiguous cases,
        # default to [samples, channels]. If neither axis looks channel-like,
        # pick the smaller axis as channels.
        if dim0_looks_like_channels and not dim1_looks_like_channels:
            channel_axis = 0
        elif dim1_looks_like_channels and not dim0_looks_like_channels:
            channel_axis = 1
        elif dim0_looks_like_channels and dim1_looks_like_channels:
            channel_axis = 1
        else:
            channel_axis = 1 if dim1 < dim0 else 0

        audio_np = np.mean(audio_np, axis=channel_axis)

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


TTS = PocketTTS
