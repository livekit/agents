import dataclasses
from dataclasses import dataclass

import httpx

import spitch
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    NotGivenOr,
)
from livekit.agents.stt import stt
from livekit.agents.utils import AudioBuffer
from spitch import AsyncSpitch


@dataclass
class _STTOptions:
    language: str


class STT(stt.STT):
    def __init__(
            self,
            *,
            language: str = "en"
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))

        self._opts = _STTOptions(language=language)
        self._client = AsyncSpitch()

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
            config = self._sanitize_options(language=language)
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            resp = await self._client.speech.transcribe(
                language=config.language,
                content=data,
                timeout=httpx.Timeout(30, connect=conn_options.timeout),
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(text=resp.text or "", language=config.language or ""),
                ],
            )
        except spitch.APITimeoutError as e:
            raise APITimeoutError() from e
        except spitch.APIStatusError as e:
            raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e
