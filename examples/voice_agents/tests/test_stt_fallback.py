from __future__ import annotations

import asyncio

import pytest

from livekit.agents import APIConnectionError, utils
from livekit.agents.stt import STT, AvailabilityChangedEvent, FallbackAdapter
from livekit.agents.utils.aio.channel import ChanEmpty

from .fake_stt import FakeSTT


class FallbackAdapterTester(FallbackAdapter):
    def __init__(
        self,
        stt: list[STT],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        super().__init__(
            stt,
            attempt_timeout=attempt_timeout,
            max_retry_per_stt=max_retry_per_stt,
            retry_interval=retry_interval,
        )

        self.on("stt_availability_changed", self._on_stt_availability_changed)

        self._availability_changed_ch: dict[int, utils.aio.Chan[AvailabilityChangedEvent]] = {
            id(t): utils.aio.Chan[AvailabilityChangedEvent]() for t in stt
        }

    def _on_stt_availability_changed(self, ev: AvailabilityChangedEvent) -> None:
        self._availability_changed_ch[id(ev.stt)].send_nowait(ev)

    def availability_changed_ch(
        self,
        tts: STT,
    ) -> utils.aio.ChanReceiver[AvailabilityChangedEvent]:
        return self._availability_changed_ch[id(tts)]


async def test_stt_fallback() -> None:
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])
    ev = await fallback_adapter.recognize([])
    assert ev.alternatives[0].text == "hello world"

    assert fake1.recognize_ch.recv_nowait()
    assert fake2.recognize_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    fake2.update_options(fake_exception=APIConnectionError("fake2 failed"))

    with pytest.raises(APIConnectionError):
        await fallback_adapter.recognize([])

    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    await fallback_adapter.aclose()

    # stream
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream() as stream:
        stream.end_input()

        last_alt = ""

        async for ev in stream:
            last_alt = ev.alternatives[0].text

        assert last_alt == "hello world"

    await fallback_adapter.aclose()


async def test_stt_stream_fallback() -> None:
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream() as stream:
        stream.end_input()

        async for _ in stream:
            pass

        assert fake1.stream_ch.recv_nowait()
        assert fake2.stream_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    await fallback_adapter.aclose()


async def test_stt_recover() -> None:
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_exception=APIConnectionError("fake2 failed"), fake_timeout=0.5)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    with pytest.raises(APIConnectionError):
        await fallback_adapter.recognize([])

    fake2.update_options(fake_exception=None, fake_transcript="hello world")

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    assert (
        await asyncio.wait_for(fallback_adapter.availability_changed_ch(fake2).recv(), 1.0)
    ).available, "fake2 should have recovered"

    await fallback_adapter.recognize([])

    assert fake1.recognize_ch.recv_nowait()
    assert fake2.recognize_ch.recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake1).recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake2).recv_nowait()

    await fallback_adapter.aclose()
