import asyncio
import dataclasses
import os
from dataclasses import dataclass

from fish_audio_sdk import ASRRequest, Session

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.stt import stt
from livekit.agents.utils import AudioBuffer

FISHAUDIO_API_KEY = os.getenv("FISHAUDIO_API_KEY")


@dataclass
class _STTOptions:
    language: str


class STT(stt.STT):
    def __init__(self, *, language: str = "en") -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._opts = _STTOptions(language=language)
        if not FISHAUDIO_API_KEY:
            raise APIConnectionError("FISHAUDIO_API_KEY not set")
        self._session = Session(FISHAUDIO_API_KEY)

    def update_options(self, language: str):
        self._opts.language = language or self._opts.language

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            config = self._sanitize_options(language=language or None)
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            # fish-audio-sdk is sync, so run in thread
            loop = asyncio.get_running_loop()

            def run_asr():
                return self._session.asr(ASRRequest(audio=data, language=config.language))

            response = await loop.run_in_executor(None, run_asr)
            text = response.text or ""
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(text=text, language=config.language or ""),
                ],
            )
        except Exception as e:
            raise APIConnectionError() from e
