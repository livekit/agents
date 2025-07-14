from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Optional

import fal_client
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.utils import AudioBuffer


@dataclass
class _STTOptions:
    language: str
    task: str
    chunk_level: str
    version: str


class WizperSTT(stt.STT):
    def __init__(
        self,
        *,
        language: Optional[str] = "en",
        task: Optional[str] = "transcribe",
        chunk_level: Optional[str] = "segment",
        version: Optional[str] = "3",
    ):
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=True)
        )
        self._api_key = os.getenv("FAL_KEY")
        self._opts = _STTOptions(
            language=language or "en",
            task=task or "transcribe",
            chunk_level=chunk_level or "segment",
            version=version or "3",
        )
        self._fal_client = fal_client.AsyncClient()

        if not self._api_key:
            raise ValueError(
                "fal AI API key is required. It should be set with env FAL_KEY"
            )

    def update_options(self, *, language: Optional[str] = None) -> None:
        self._opts.language = language or self._opts.language

    def _sanitize_options(
        self,
        *,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_level: Optional[str] = None,
        version: Optional[str] = None,
    ) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        config.task = task or config.task
        config.chunk_level = chunk_level or config.chunk_level
        config.version = version or config.version
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            config = self._sanitize_options(language=language)
            data_uri = fal_client.encode(
                rtc.combine_audio_frames(buffer).to_wav_bytes(), "audio/x-wav"
            )
            response = await self._fal_client.run(
                "fal-ai/wizper",
                arguments={
                    "audio_url": data_uri,
                    "task": config.task,
                    "language": config.language,
                    "chunk_level": config.chunk_level,
                    "version": config.version,
                },
                timeout=conn_options.timeout,
            )
            text = response.get("text", "")
            return self._transcription_to_speech_event(text=text)
        except fal_client.client.FalClientError as e:
            raise APIConnectionError() from e

    def _transcription_to_speech_event(
        self, event_type=SpeechEventType.FINAL_TRANSCRIPT, text=None
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=event_type,
            alternatives=[stt.SpeechData(text=text, language=self._opts.language)],
        )

    async def aclose(self) -> None:
        await self._fal_client._client.aclose()
