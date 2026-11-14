"""The speech a stream's reports make: a provider's end held for the final of its words."""

from __future__ import annotations

import asyncio

from livekit.agents import APIConnectOptions, LanguageCode, _core, stt
from tests.fakes import until

SOS, EOS = stt.SpeechEventType.START_OF_SPEECH, stt.SpeechEventType.END_OF_SPEECH
INTERIM, FINAL = stt.SpeechEventType.INTERIM_TRANSCRIPT, stt.SpeechEventType.FINAL_TRANSCRIPT
EN = LanguageCode("en")


class _Provider(stt.STT):
    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.reports: asyncio.Queue[stt.Report] = asyncio.Queue()

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(self, **_: object) -> stt.RecognizeStream:
        return _Stream(stt=self, conn_options=APIConnectOptions(max_retry=0))


class _Stream(stt.RecognizeStream):
    provider_vad = True

    async def _run(self) -> None:
        assert isinstance(self._stt, _Provider)
        while True:
            self.report(await self._stt.reports.get())


# Catches: a provider's end that came before the final of the words it heard closing the turn
# without them, or a pause, however long, cutting the speech before that final.
async def test_an_end_before_the_final_waits_for_it_however_long() -> None:
    provider = _Provider()
    stream = provider.stream()
    events: list[stt.SpeechEvent] = []
    reader = asyncio.create_task(_collect(stream, events))
    try:
        heard: list[stt.Report] = [
            stt.SpeechStarted(speech_start_time=12.5),
            stt.Transcript("book a flight to Par", EN, request_id="r1"),
            stt.SpeechEnded(),
        ]
        for report in heard:
            provider.reports.put_nowait(report)
        await until(lambda: len(events) == 2, "the interim")
        # a timer releasing the end would make the interim final before the provider's final
        _core.skip_host_time(5.0)
        assert stream.speaking, "the end waits for the final"
        provider.reports.put_nowait(
            stt.Transcript("book a flight to Paris", EN, final=True, request_id="r1")
        )
        await until(lambda: len(events) == 4, "the final and the end")
    finally:
        await stream.aclose()
        await reader

    assert [(e.type, e.alternatives[0].text if e.alternatives else "") for e in events] == [
        (SOS, ""),
        (INTERIM, "book a flight to Par"),
        (FINAL, "book a flight to Paris"),
        (EOS, ""),
    ]
    assert events[0].speech_start_time == 12.5
    assert {e.utterance for e in events} == {1}, "the late final is the same utterance's"
    assert events[2].request_id == events[3].request_id == "r1"


async def _collect(stream: stt.RecognizeStream, events: list[stt.SpeechEvent]) -> None:
    async for event in stream:
        events.append(event)
