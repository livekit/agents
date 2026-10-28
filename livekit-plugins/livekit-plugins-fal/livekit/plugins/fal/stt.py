from __future__ import annotations

import os
from dataclasses import dataclass

import fal_client
import httpx

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given


@dataclass
class _STTOptions:
    language: LanguageCode = LanguageCode("en")
    task: str = "transcribe"
    chunk_level: str = "segment"
    version: str = "3"


class WizperSTT(stt.STT):
    def __init__(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
    ):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False, interim_results=True, aligned_transcript=False
            )
        )
        self._api_key = api_key if is_given(api_key) else os.getenv("FAL_KEY")
        if not self._api_key:
            raise ValueError("fal AI API key is required. It should be set with env FAL_KEY")
        self._opts = _STTOptions(
            language=LanguageCode(language) if is_given(language) else LanguageCode("en")
        )
        self._client = httpx.AsyncClient(headers={"Authorization": f"Key {self._api_key}"})

    @property
    def model(self) -> str:
        return "Wizper"

    @property
    def provider(self) -> str:
        return "Fal"

    def update_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> None:
        if is_given(language):
            self._opts.language = LanguageCode(language)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            request_language = LanguageCode(language) if is_given(language) else self._opts.language
            data_uri = fal_client.encode(
                rtc.combine_audio_frames(buffer).to_wav_bytes(), "audio/x-wav"
            )
            response = await self._client.post(
                "https://fal.run/fal-ai/wizper",
                json={
                    "audio_url": data_uri,
                    "task": self._opts.task,
                    "language": request_language.language,
                    "chunk_level": self._opts.chunk_level,
                    "version": self._opts.version,
                },
                timeout=httpx.Timeout(
                    connect=conn_options.connect_timeout,
                    pool=conn_options.pool_timeout,
                    write=conn_options.write_timeout,
                    read=conn_options.read_timeout,
                ),
            )
            response.raise_for_status()
            text = response.json().get("text", "")
            return self._transcription_to_speech_event(text=text, language=request_language)
        except httpx.TimeoutException as e:
            raise APITimeoutError() from e
        except httpx.HTTPStatusError as e:
            raise APIStatusError(
                str(e),
                status_code=e.response.status_code,
                body=e.response.text,
                request_id=e.response.headers.get("x-fal-request-id"),
            ) from e
        except httpx.RequestError as e:
            raise APIConnectionError() from e

    def _transcription_to_speech_event(self, text: str, language: LanguageCode) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language)],
        )

    async def aclose(self) -> None:
        await self._client.aclose()
