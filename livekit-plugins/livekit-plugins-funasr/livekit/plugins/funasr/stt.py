from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

_DEFAULT_MODEL = "iic/SenseVoiceSmall"
_TARGET_SR = 16000


@dataclass
class _STTOptions:
    model: str = _DEFAULT_MODEL
    language: str = "auto"
    device: str = "cpu"
    hub: str = "ms"
    use_itn: bool = True


class STT(stt.STT):
    """FunASR self-hosted speech-to-text.

    Runs FunASR models (SenseVoice, Paraformer, Fun-ASR-Nano) locally — no cloud
    API. Non-streaming; LiveKit wraps it with a VAD StreamAdapter for agents.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        language: str = "auto",
        device: str = "cpu",
        hub: str = "ms",
        use_itn: bool = True,
        vad_model: str | None = "fsmn-vad",
    ) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._opts = _STTOptions(model=model, language=language, device=device, hub=hub, use_itn=use_itn)
        self._vad_model = vad_model
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from funasr import AutoModel

            kwargs = dict(model=self._opts.model, device=self._opts.device, hub=self._opts.hub, disable_update=True)
            if self._vad_model:
                kwargs.update(vad_model=self._vad_model, vad_kwargs={"max_single_segment_time": 30000})
            logger.info("loading FunASR model %s on %s", self._opts.model, self._opts.device)
            self._model = AutoModel(**kwargs)
        return self._model

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        def _run() -> str:
            import soundfile as sf
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            model = self._ensure_model()
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != _TARGET_SR:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
            gen_kwargs = dict(input=audio, cache={}, use_itn=self._opts.use_itn, batch_size_s=300)
            if "SenseVoice" in self._opts.model or (lang and lang != "auto"):
                gen_kwargs["language"] = lang
            res = model.generate(**gen_kwargs)
            text = res[0]["text"] if res else ""
            return rich_transcription_postprocess(text)

        try:
            text = await asyncio.get_event_loop().run_in_executor(None, _run)
        except Exception as e:  # noqa: BLE001
            raise APIConnectionError() from e

        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=str(lang))],
        )
