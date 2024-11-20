from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .. import utils
from .stt import STT, SpeechStream, STTCapabilities


@dataclass
class AvailabilityChangedEvent:
    stt: STT
    available: bool


class FallbackAdapter(
    STT[Literal["tts_availability_changed"]],
):
    def __init__(self, stt: list[STT]) -> None:
        if len(stt) < 1:
            raise ValueError("At least one STT instance must be provided.")

        super().__init__(
            capabilities=STTCapabilities(
                streaming=all(t.capabilities.streaming for t in stt),
                interim_results=all(t.capabilities.interim_results for t in stt),
            )
        )

    async def _recognize_impl(
        self, buffer: utils.AudioBuffer, *, language: str | None = None
    ):
        return await self._stt.recognize(buffer=buffer, language=language)

    def stream(self, *, language: str | None = None) -> SpeechStream:
        return StreamAdapterWrapper(
            self, vad=self._vad, wrapped_stt=self._stt, language=language
        )
