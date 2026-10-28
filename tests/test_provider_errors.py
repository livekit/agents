"""Provider error reports retain the original exception for the final caller."""

from __future__ import annotations

from typing import Never

import pytest

from livekit.agents import APIConnectOptions, APIError, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.audio import AudioBuffer

pytestmark = pytest.mark.unit


class _FailingSTT(stt.STT[Never]):
    def __init__(self, failure: APIError) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self.failure = failure

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise self.failure

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return _FailingSpeech(stt=self, conn_options=conn_options)


class _FailingSpeech(stt.RecognizeStream):
    async def _run(self) -> None:
        async for _ in self.input:
            pass
        self.send(stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT))
        assert isinstance(self._stt, _FailingSTT)
        raise self._stt.failure


async def test_fallback_error_report_preserves_the_original_raised_exception() -> None:
    original = APIError("the provider's own failure", retryable=False)
    provider = _FailingSTT(original)
    reports: list[Exception] = []
    provider.on("error", lambda event: reports.append(event.error))
    fallback = stt.FallbackAdapter([provider], max_retry_per_stt=0)
    try:
        with pytest.raises(APIError) as raised:
            async with fallback.stream(conn_options=APIConnectOptions(max_retry=0)) as source:
                source.end_input()
                async for _ in source:
                    pass
        assert raised.value is original
        assert len(reports) == 1 and reports[0] is original
    finally:
        await fallback.aclose()
        await provider.aclose()
