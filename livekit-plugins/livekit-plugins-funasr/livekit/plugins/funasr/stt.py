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
import re
from dataclasses import dataclass

import numpy as np

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, LanguageCode, stt
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

try:
    from funasr import AutoModel  # type: ignore
    from funasr.utils.postprocess_utils import rich_transcription_postprocess  # type: ignore
except ImportError as e:
    raise ImportError(
        "funasr is required for the FunASR plugin. Install it with: pip install funasr"
    ) from e

# Languages natively supported by SenseVoice; anything else falls back to auto-detect.
_FUNASR_LANGUAGES = {"zh", "en", "ja", "ko", "yue", "nospeech"}
_SAMPLE_RATE = 16000
_LANG_TAG_RE = re.compile(r"<\|([a-z]+)\|>")


def _normalize_language(language: NotGivenOr[str]) -> str:
    if not is_given(language) or not language:
        return "auto"
    code = str(language).split("-")[0].lower()
    return code if code in _FUNASR_LANGUAGES else "auto"


@dataclass
class _STTOptions:
    language: str = "auto"
    use_itn: bool = True


class FunASRSTT(stt.STT):
    """Local speech-to-text using a FunASR model such as SenseVoice.

    SenseVoice is an open-source, fully-local, non-autoregressive multilingual ASR
    model (Chinese, Cantonese, English, Japanese, Korean and more) with strong
    Chinese accuracy and fast inference. The model runs locally; no API key needed.
    """

    def __init__(
        self,
        *,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        language: NotGivenOr[str] = NOT_GIVEN,
        use_itn: bool = True,
    ) -> None:
        """Create a FunASR STT instance.

        Args:
            model: FunASR model id on ModelScope/Hugging Face (default
                ``"iic/SenseVoiceSmall"``).
            device: Inference device, ``"cpu"`` or ``"cuda"``.
            language: Default language. When not given, the language is
                auto-detected per utterance.
            use_itn: Apply inverse text normalization (e.g. "nine" -> "9").
        """
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._opts = _STTOptions(language=_normalize_language(language), use_itn=use_itn)
        logger.info(f"loading FunASR model {model} on {device}...")
        self._model = AutoModel(model=model, device=device, disable_update=True)
        logger.info("FunASR model loaded")

    @property
    def model(self) -> str:
        return "SenseVoice"

    @property
    def provider(self) -> str:
        return "FunASR"

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        use_itn: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = _normalize_language(language)
        if is_given(use_itn):
            self._opts.use_itn = use_itn

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = _normalize_language(language) if is_given(language) else self._opts.language

        combined = rtc.combine_audio_frames(buffer)
        channels = combined.num_channels
        if combined.sample_rate != _SAMPLE_RATE:
            resampler = rtc.AudioResampler(
                combined.sample_rate, _SAMPLE_RATE, num_channels=channels
            )
            frames = list(resampler.push(combined)) + list(resampler.flush())
            data = b"".join(bytes(f.data) for f in frames)
        else:
            data = bytes(combined.data)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        def _run() -> str:
            result = self._model.generate(
                input=samples,
                cache={},
                language=lang,
                use_itn=self._opts.use_itn,
            )
            return result[0]["text"] if result else ""

        try:
            raw = await asyncio.to_thread(_run)
        except Exception as e:
            raise APIConnectionError("failed to run FunASR inference") from e

        text = rich_transcription_postprocess(raw).strip()
        m = _LANG_TAG_RE.match(raw)
        detected = m.group(1) if m and m.group(1) in _FUNASR_LANGUAGES else ""

        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=LanguageCode(detected))],
        )
