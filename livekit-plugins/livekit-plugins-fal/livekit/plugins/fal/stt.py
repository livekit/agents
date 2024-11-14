import dataclasses
import os
from dataclasses import dataclass

import fal_client
from livekit.agents import stt
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.utils import AudioBuffer
from livekit.rtc import AudioFrame

from .log import logger


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
        language="en",
        task="transcribe",
        chunk_level="segment",
        version="3",
    ):
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=True)
        )
        self._api_key = os.getenv("FAL_KEY")
        self._opts = _STTOptions(
            language=language, task=task, chunk_level=chunk_level, version=version
        )

        if not self._api_key:
            raise ValueError(
                "FAL AI API key is required. It should be set with env FAL_KEY"
            )

    def _sanitize_options(
        self,
        *,
        language: str = None,
        task: str = None,
        chunk_level: str = None,
        version: str = None,
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
        language: str = None,
        task: str = None,
        chunk_level: str = None,
        version: str = None,
    ) -> stt.SpeechEvent:
        try:
            if buffer is None:
                raise ValueError("AudioBuffer input is required")

            config = self._sanitize_options(
                language=language, task=task, chunk_level=chunk_level, version=version
            )

            wav_bytes = AudioFrame.to_wav_bytes(buffer)
            data_uri = fal_client.encode(wav_bytes, "audio/x-wav")

            handler = await fal_client.submit_async(
                "fal-ai/wizper",
                arguments={
                    "audio_url": data_uri,
                    "task": config.task,
                    "language": config.language,
                    "chunk_level": config.chunk_level,
                    "version": config.version,
                },
            )

            try:
                result = await fal_client.result_async(
                    "fal-ai/wizper", handler.request_id
                )
                text = result.get("text", "")
                return self._transcription_to_speech_event(text=text)
            except fal_client.client.FalClientError as e:
                logger.error(f"FAL AI API error: {e}")
                return self._transcription_to_speech_event(text="")

        except Exception as ex:
            logger.error(f"Error during recognition: {ex}", exc_info=True)
            return self._transcription_to_speech_event(text="")

    def _transcription_to_speech_event(
        self, event_type=SpeechEventType.FINAL_TRANSCRIPT, text=None
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=event_type,
            alternatives=[stt.SpeechData(text=text, language=self._opts.language)],
        )
