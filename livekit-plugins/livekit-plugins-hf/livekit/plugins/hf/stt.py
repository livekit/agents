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

import asyncio
import copy
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from livekit import rtc
from livekit.agents import APIConnectionError, stt
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given, shortuuid

from ._cache import get_or_load, not_cached_error
from .log import logger

DEFAULT_STT_MODEL = "openai/whisper-large-v3-turbo"


@dataclass(frozen=True)
class _STTKey:
    model: str
    revision: str | None
    device: str
    dtype: str
    trust_remote_code: bool


@dataclass
class _LoadedSTT:
    pipeline: Any
    sample_rate: int
    is_whisper: bool
    # a pipeline is not safe to call concurrently, and jobs in a thread executor share
    # it across event loops, so this is a thread lock
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _STTOptions:
    model: str
    revision: str | None
    device: str
    dtype: str
    language: str | None
    generate_kwargs: dict[str, Any]
    language_token_ids: dict[str, int]
    trust_remote_code: bool
    local_files_only: bool

    @property
    def key(self) -> _STTKey:
        return _STTKey(
            model=self.model,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = DEFAULT_STT_MODEL,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        language: str | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        language_token_ids: dict[str, int] | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
    ) -> None:
        """Run a Hugging Face speech recognition model locally with ``transformers``.

        Works with the models the ``automatic-speech-recognition`` pipeline loads:
        Whisper and its fine-tunes, Moonshine, wav2vec2 and other CTC models. This is a
        batch STT: the ``AgentSession`` VAD cuts each utterance and the model transcribes
        it in one pass, so pass a ``vad`` to the session.

        Weights load in a background thread when the session prewarms the STT, or
        synchronously with :meth:`load`. They are cached per process.

        Args:
            model: Hugging Face Hub repo id or path to a local model directory.
            revision: Hub revision (branch, tag or commit) to load.
            device: ``device_map`` passed to ``from_pretrained``, e.g. ``"auto"``,
                ``"cuda"``, ``"mps"`` or ``"cpu"``.
            dtype: Torch dtype name, e.g. ``"auto"``, ``"float16"`` or ``"bfloat16"``.
            language: Language passed to Whisper models, e.g. ``"en"`` or ``"french"``.
                ``None`` lets Whisper detect it. Ignored by models without language
                selection.
            generate_kwargs: Extra arguments for ``generate``, e.g. ``num_beams`` or a
                fine-tune's own language tokens (see the README).
            trust_remote_code: Allow models that ship their own modeling code.
            local_files_only: Load only from the local Hugging Face cache and never
                download at runtime.
        """
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._opts = _STTOptions(
            model=model,
            revision=revision,
            device=device,
            dtype=dtype,
            language=language,
            generate_kwargs=dict(generate_kwargs or {}),
            language_token_ids=dict(language_token_ids or {}),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self._prewarm_task: asyncio.Future[Any] | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "huggingface"

    def load(self) -> None:
        """Load the model weights into this process, blocking until they are ready."""
        self._get_model()

    def prewarm(self) -> None:
        # prewarm() runs on the event loop and must not block it
        if self._prewarm_task is None:
            self._prewarm_task = asyncio.ensure_future(asyncio.to_thread(self._get_model))
            self._prewarm_task.add_done_callback(_log_prewarm_error)

    def _get_model(self) -> _LoadedSTT:
        opts = self._opts
        return get_or_load(
            ("stt", opts.key), lambda: _load_stt(opts.key, local_files_only=opts.local_files_only)
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language
        try:
            loaded = await asyncio.to_thread(self._get_model)
            audio = _to_mono_float32(buffer, loaded.sample_rate)
            text = await asyncio.to_thread(self._transcribe, loaded, audio, lang)
        except Exception as e:
            raise APIConnectionError(
                f"local transcription failed: {type(e).__name__}", retryable=False
            ) from e

        duration = len(audio) / loaded.sample_rate
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=shortuuid("hf_stt_"),
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=LanguageCode(lang or ""),
                    end_time=duration,
                )
            ],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )

    def _transcribe(self, loaded: _LoadedSTT, audio: np.ndarray, language: str | None) -> str:
        if audio.size == 0:
            return ""
        generate_kwargs = dict(self._opts.generate_kwargs)
        # a fine-tune that forces its own decoder prefix owns the language and task
        if loaded.is_whisper and "forced_decoder_ids" not in generate_kwargs:
            generate_kwargs.setdefault("task", "transcribe")
            if language and "language" not in generate_kwargs:
                if (token_id := self._opts.language_token_ids.get(language)) is not None:
                    # Whisper looks languages up in lang_to_id, so add the fine-tune's
                    # token there instead of overwriting the shared generation config
                    token = f"<|{language}|>"
                    generation_config = copy.deepcopy(loaded.pipeline.model.generation_config)
                    generation_config.lang_to_id = {
                        **(getattr(generation_config, "lang_to_id", None) or {}),
                        token: token_id,
                    }
                    generate_kwargs["generation_config"] = generation_config
                    generate_kwargs["language"] = token
                else:
                    generate_kwargs["language"] = language

        call_kwargs: dict[str, Any] = {}
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs
        with loaded.lock:
            result = loaded.pipeline(
                {"raw": audio, "sampling_rate": loaded.sample_rate}, **call_kwargs
            )
        return str(result.get("text", "")).strip()


def _log_prewarm_error(fut: asyncio.Future[Any]) -> None:
    if not fut.cancelled() and (exc := fut.exception()) is not None:
        logger.error("failed to load STT model", exc_info=exc)


def _to_mono_float32(buffer: AudioBuffer, sample_rate: int) -> np.ndarray:
    """Merge the buffer, downmix to mono and resample to the model's rate."""
    if isinstance(buffer, list) and not buffer:
        return np.zeros(0, dtype=np.float32)
    frame = rtc.combine_audio_frames(buffer)
    pcm = np.frombuffer(frame.data, dtype=np.int16)
    if frame.num_channels > 1:
        pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)

    if frame.sample_rate != sample_rate and pcm.size:
        mono = rtc.AudioFrame(
            data=pcm.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=pcm.size,
        )
        resampler = rtc.AudioResampler(frame.sample_rate, sample_rate, num_channels=1)
        frames = [*resampler.push(mono), *resampler.flush()]
        pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])

    return pcm.astype(np.float32) / 32768.0


def _load_stt(key: _STTKey, *, local_files_only: bool) -> _LoadedSTT:
    from transformers import (
        AutoConfig,
        AutoModelForCTC,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline,
    )

    logger.info("loading STT model", extra={"model": key.model, "device": key.device})
    kwargs: dict[str, Any] = {
        "revision": key.revision,
        "trust_remote_code": key.trust_remote_code,
        "local_files_only": local_files_only,
    }
    try:
        config: Any = AutoConfig.from_pretrained(key.model, **kwargs)
        processor: Any = AutoProcessor.from_pretrained(key.model, **kwargs)  # type: ignore[no-untyped-call]
        is_ctc = any(a.endswith("ForCTC") for a in getattr(config, "architectures", None) or [])
        model_cls: Any = AutoModelForCTC if is_ctc else AutoModelForSpeechSeq2Seq
        model: Any = model_cls.from_pretrained(
            key.model, dtype=key.dtype, device_map=key.device, **kwargs
        )
    except OSError as e:
        if not local_files_only:
            raise
        raise not_cached_error(key.model) from e

    model.eval()
    feature_extractor = getattr(processor, "feature_extractor", processor)
    asr: Any = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=getattr(processor, "tokenizer", None),
        feature_extractor=feature_extractor,
    )
    sample_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
    loaded = _LoadedSTT(
        pipeline=asr,
        sample_rate=sample_rate,
        is_whisper=getattr(config, "model_type", "") == "whisper",
    )
    # the first pass pays for lazy weight paging and kernel setup; do it before a caller
    with loaded.lock:
        asr({"raw": np.zeros(sample_rate // 2, dtype=np.float32), "sampling_rate": sample_rate})
    logger.info("STT model loaded", extra={"model": key.model})
    return loaded
