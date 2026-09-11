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

"""Vui Nano TTS — local, streaming, in-process (CUDA or MLX)."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    tokenize,
    tts,
    utils,
)

from .log import logger

#: Voices shipped with the model on the Hugging Face Hub (fluxions/vui).
BUILTIN_VOICES = ("maeve", "abraham", "rhian", "harry")
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    checkpoint: str
    voice: str
    temperature: float
    max_secs: float


class TTS(tts.TTS):
    """Vui Nano text-to-speech.

    A small (219M active / 305M total parameters, Apache 2.0) context-aware TTS
    model trained on real conversations, running in-process: on CUDA, or on
    MLX on Apple Silicon. Weights and voice prompts download from Hugging Face
    on first use. Audio is emitted frame by frame as it is decoded.

    Streaming input is supported: sentences from the LLM are rendered as they
    complete, in one conversation row, so prosody carries across the reply.
    """

    def __init__(
        self,
        *,
        checkpoint: str = "vui-nano-1.1",
        voice: str = "maeve",
        temperature: float = 0.7,
        max_secs: float = 30.0,
    ) -> None:
        """Create a new instance of Vui TTS.

        Args:
            checkpoint: A name from ``vui.engine.Engine.NAMES`` (``"vui-nano-1.1"``,
                ``"vui-190k"``, ``"vui-nano"``), a file in the ``fluxions/vui`` Hub
                repo, or a local path.
            voice: One of the shipped voices (``"maeve"``, ``"abraham"``, ``"rhian"``,
                ``"harry"``), a path to a prompt ``.safetensors`` baked with Vui's
                ``scripts/build_prompts.py``, or a path to a ``.wav`` reference clip
                (with a sibling ``.txt`` transcript, else transcribed on the fly).
            temperature: Sampling temperature.
            max_secs: Longest audio a single sentence may produce.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            checkpoint=checkpoint, voice=voice, temperature=temperature, max_secs=max_secs
        )
        self._engine: Any = None
        self._row: Any = None
        self._gen_config: Any = None
        self._voice_loaded: str | None = None
        # One engine, one row, one decode at a time: renders are serialised on
        # a single worker thread and callers queue behind the lock.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vui-tts")
        self._lock = asyncio.Lock()
        self._sentence_tokenizer = tokenize.basic.SentenceTokenizer()

    @property
    def model(self) -> str:
        return self._opts.checkpoint

    @property
    def provider(self) -> str:
        return "Vui"

    def update_options(self, *, voice: str | None = None, temperature: float | None = None) -> None:
        """Change the voice or sampling temperature for subsequent synthesis."""
        if voice is not None:
            self._opts.voice = voice
        if temperature is not None:
            self._opts.temperature = temperature
            self._gen_config = None

    # ------------------------------------------------------------------
    # Engine (lazy — the first call loads weights; prewarm() does it early)
    # ------------------------------------------------------------------

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        from vui.engine import Engine

        logger.info("loading Vui checkpoint %s", self._opts.checkpoint)
        self._engine = Engine(self._opts.checkpoint, max_rows=1)
        self._row = self._engine.new_row()

    def _config(self) -> Any:
        if self._gen_config is None:
            from vui.engine import GenConfig

            self._gen_config = GenConfig(
                temperature=self._opts.temperature, max_secs=self._opts.max_secs
            )
        return self._gen_config

    def _resolve_voice(self, voice: str) -> Any:
        """Turn a voice spec into a ``vui.engine.Segment`` (transcript + codec codes)."""
        from safetensors.torch import load_file

        from vui.engine import Segment
        from vui.prompt_files import hub_prompt, hub_prompt_transcript, prompt_transcript

        path = Path(voice)
        if voice in BUILTIN_VOICES and not path.exists():
            st = hub_prompt(voice, self._engine.checkpoint)
            return Segment(hub_prompt_transcript(voice, st), load_file(st)["codes"].long())
        if path.suffix == ".safetensors":
            text = prompt_transcript(path)
            if not text:
                raise ValueError(f"{path}: no transcript in metadata or sibling .txt")
            return Segment(text, load_file(str(path))["codes"].long())
        if path.suffix.lower() == ".wav":
            return self._encode_wav(path)
        raise ValueError(
            f"unknown Vui voice {voice!r}: expected one of {BUILTIN_VOICES}, "
            "a prompt .safetensors, or a .wav"
        )

    def _encode_wav(self, path: Path) -> Any:
        import torch
        from julius.resample import resample_frac
        from torchcodec.decoders import AudioDecoder

        from vui.engine import Segment
        from vui.prompt_files import prompt_transcript
        from vui.qwen_codec import QwenCodecEncoder

        wav_16k = (
            AudioDecoder(str(path), sample_rate=16000, num_channels=1)
            .get_all_samples()
            .data.squeeze(0)
        )
        text = prompt_transcript(path)
        if not text:
            from vui.inference import asr  # openai-whisper: `vui-tts[server]`

            text = asr(wav_16k)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        enc = QwenCodecEncoder.from_pretrained().to(dev).float().eval()
        with torch.inference_mode():
            wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, SAMPLE_RATE)
            codes = enc.encode(wav_24k.float().to(dev).unsqueeze(0))
        return Segment(text, codes[0, : self._engine.Q].T.long().cpu())

    def _ensure_voice(self) -> None:
        import torch

        self._ensure_engine()
        if self._voice_loaded == self._opts.voice:
            return
        segment = self._resolve_voice(self._opts.voice)
        with torch.inference_mode():
            self._row.reset()
            self._row.prefill([segment])
        self._voice_loaded = self._opts.voice

    def _render_blocking(
        self, text: str, cancel: threading.Event, on_pcm: Any, rewind: bool
    ) -> None:
        """Worker-thread body: stream one sentence, hand int16 PCM to ``on_pcm``.

        Everything that touches the engine — load, prefill, decode, rewind —
        happens here, on the one worker thread: the CUDA graphs and MLX's
        streams are bound to the thread that created them.
        """
        import torch

        self._ensure_voice()
        try:
            with torch.inference_mode():
                for frame in self._row.stream(text, self._config(), cancel=cancel):
                    if cancel.is_set():
                        break
                    on_pcm(
                        frame.reshape(-1)
                        .float()
                        .clamp(-1.0, 1.0)
                        .mul(32767.0)
                        .to(torch.int16)
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
        finally:
            if rewind:
                self._row.rewind()

    async def _render(self, text: str, output_emitter: tts.AudioEmitter, *, rewind: bool) -> None:
        """Render ``text`` on the worker thread, pushing PCM into ``output_emitter``.

        With ``rewind`` the KV cache goes back to the end of the voice prompt
        afterwards (one-shot synthesis); without it the sentence stays in
        context so the next one is conditioned on it (streaming input).
        """
        loop = asyncio.get_running_loop()
        cancel = threading.Event()

        def on_pcm(pcm: bytes) -> None:
            loop.call_soon_threadsafe(output_emitter.push, pcm)

        async with self._lock:
            fut = loop.run_in_executor(
                self._executor, self._render_blocking, text, cancel, on_pcm, rewind
            )
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                cancel.set()
                await fut
                raise

    async def _rewind(self) -> None:
        """Rewind the row to the voice prompt (on the worker thread)."""
        if self._row is None:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(self._executor, self._row.rewind)

    def prewarm(self) -> None:
        """Load the checkpoint and the voice prompt ahead of the first request."""
        self._executor.submit(self._ensure_voice).result()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize one complete text (rewinds to the voice prompt afterwards)."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        await self._tts._render(self._input_text, output_emitter, rewind=True)
        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    """Synthesize streamed text sentence by sentence in one conversation row.

    Each completed sentence is rendered as soon as the tokenizer yields it and
    stays in the model's context, so the next sentence is conditioned on what
    was just said. The row is rewound to the voice prompt when the input ends.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        sent_stream = self._tts._sentence_tokenizer.stream()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _render_task() -> None:
            segment_open = False
            try:
                async for ev in sent_stream:
                    sentence = ev.token.strip()
                    if not sentence:
                        continue
                    if not segment_open:
                        output_emitter.start_segment(segment_id=utils.shortuuid())
                        segment_open = True
                    self._mark_started()
                    await self._tts._render(sentence, output_emitter, rewind=False)
            finally:
                if segment_open:
                    output_emitter.end_segment()
                await self._tts._rewind()

        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(_render_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
        output_emitter.end_input()
