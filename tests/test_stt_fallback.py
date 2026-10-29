"""Automatic streaming adaptation preserves a repeated recognizer's health identity."""

import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN, APIError, LanguageCode, NotGivenOr, stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class BatchStt(stt.STT):
    def __init__(self, *, fail: bool) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.fail = fail

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if self.fail:
            raise APIError("recognizer failed", retryable=True)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode("en"), text="Recovered.")],
        )


async def test_repeated_batch_recognizers_share_one_health_transition_after_adaptation() -> None:
    primary, backup = BatchStt(fail=True), BatchStt(fail=False)
    adapter = stt.FallbackAdapter(
        [primary, primary, backup], vad=FakeVAD(), max_retry_per_stt=0, retry_interval=0
    )
    unavailable: list[stt.STT] = []

    @adapter.on("stt_availability_changed")
    def changed(event: stt.AvailabilityChangedEvent) -> None:
        if not event.available:
            unavailable.append(event.stt)

    try:
        result = await adapter.recognize(rtc.AudioFrame.create(16_000, 1, 160))
        assert result.alternatives[0].text == "Recovered."
    finally:
        await adapter.aclose()
    assert len(unavailable) == 1, "automatic wrappers must share a repeated backend's health"
