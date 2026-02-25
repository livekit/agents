from __future__ import annotations

from livekit import rtc

from ..llm import ChatContext


class MultiModalTurnDetector:
    def __init__(self):
        pass

    @property
    def model(self) -> str:
        return "multimodal"

    @property
    def provider(self) -> str:
        return "livekit"

    async def unlikely_threshold(self, language: str | None) -> float | None:
        raise NotImplementedError("unlikely_threshold is not implemented")

    async def supports_language(self, language: str | None) -> bool:
        raise NotImplementedError("supports_language is not implemented")

    async def predict_end_of_turn(
        self,
        chat_ctx: ChatContext,
        *,
        timeout: float | None = None,
        audio: rtc.AudioFrame | None = None,
    ) -> float:
        raise NotImplementedError("predict_end_of_turn is not implemented")
