from __future__ import annotations

import os
from dataclasses import dataclass

import fal_client

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, stt
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given


@dataclass
class _STTOptions:
    language: str = "en"
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
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=True))
        self._api_key = api_key if is_given(api_key) else os.getenv("FAL_KEY")
        if not self._api_key:
            raise ValueError("fal AI API key is required. It should be set with env FAL_KEY")
        self._opts = _STTOptions(language=language)
        self._fal_client = fal_client.AsyncClient(key=self._api_key)

    def update_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> None:
        if is_given(language):
            self._opts.language = language

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            if is_given(language):
                self._opts.language = language
            data_uri = fal_client.encode(
                rtc.combine_audio_frames(buffer).to_wav_bytes(), "audio/x-wav"
            )
            response = await self._fal_client.run(
                "fal-ai/wizper",
                arguments={
                    "audio_url": data_uri,
                    "task": self._opts.task,
                    "language": self._opts.language,
                    "chunk_level": self._opts.chunk_level,
                    "version": self._opts.version,
                },
                timeout=conn_options.timeout,
            )
            text = response.get("text", "")
            return self._transcription_to_speech_event(text=text)
        except fal_client.client.FalClientError as e:
            raise APIConnectionError() from e

    def _transcription_to_speech_event(self, text: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=self._opts.language)],
        )

    async def aclose(self) -> None:
        await self._fal_client._client.aclose()
