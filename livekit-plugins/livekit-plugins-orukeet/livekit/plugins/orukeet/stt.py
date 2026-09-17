# Copyright 2026 Oruk AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import numpy as np

from livekit.agents import APIConnectOptions, LanguageCode, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from ._model import Model


class STT(stt.STT):
    """Local Orukeet INT8 recognition for completed utterances.

    Recognition runs on CPU and detects language automatically. It provides final
    text, without word timing, interim hypotheses, or diarization. Use LiveKit's
    StreamAdapter with a VAD for continuous audio.

    Args:
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Require already cached model files; do not download.
    """

    def __init__(self, *, cache_dir: str | None = None, local_files_only: bool = False) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._runtime = Model(cache_dir=cache_dir, local_files_only=local_files_only)

    @property
    def model(self) -> str:
        return "oruk/orukeet"

    @property
    def provider(self) -> str:
        return "oruk"

    def prewarm(self) -> None:
        """Load the model in the worker's synchronous prewarm hook."""
        self._runtime.prewarm()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if utils.is_given(language) and language not in ("", "auto"):
            raise ValueError("Orukeet detects language automatically; omit the language argument.")
        if isinstance(buffer, list) and not buffer:
            audio = np.empty(0, dtype=np.float32)
            sample_rate = 16000
        else:
            frame = utils.merge_frames(buffer)
            audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
            audio = audio.reshape(-1, frame.num_channels).mean(axis=1) / 32768.0
            sample_rate = frame.sample_rate
        text = await asyncio.to_thread(self._runtime.recognize, audio, sample_rate)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode(""), text=text)],
        )

    async def aclose(self) -> None:
        """Release the model after any in-flight native inference has finished."""
        await asyncio.to_thread(self._runtime.close)
