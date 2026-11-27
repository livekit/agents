import asyncio
import time
from dataclasses import dataclass

from playout_hold import hold_playout


@dataclass(frozen=True)
class Frame:
    name: str
    duration: float = 0.05


class FakeSource:
    def __init__(self) -> None:
        self.queued = 0.0
        self.pushed: list[Frame] = []
        self.cleared = 0

    @property
    def queued_duration(self) -> float:
        return self.queued

    async def capture_frame(self, frame: Frame) -> None:
        self.pushed.append(frame)

    def clear_queue(self) -> None:
        self.cleared += 1


def patched() -> type[FakeSource]:
    class Source(FakeSource):
        pass

    hold_playout(Source, hold=0.05)
    return Source


async def test_a_burst_starts_after_the_hold_and_then_flows_straight_through():
    source = patched()()
    started = time.monotonic()
    await source.capture_frame(Frame("first"))
    assert time.monotonic() - started >= 0.05
    source.queued = 0.08
    started = time.monotonic()
    await source.capture_frame(Frame("second"))
    assert time.monotonic() - started < 0.02
    assert [f.name for f in source.pushed] == ["first", "second"]


async def test_a_held_frame_counts_as_queued_until_it_goes_in():
    source = patched()()
    push = asyncio.create_task(source.capture_frame(Frame("first", 0.05)))
    await asyncio.sleep(0.01)
    assert source.queued_duration == 0.05
    await push
    assert source.queued_duration == 0.0


async def test_a_frame_held_when_the_caller_interrupts_is_dropped():
    source = patched()()
    push = asyncio.create_task(source.capture_frame(Frame("stale")))
    await asyncio.sleep(0.01)
    source.clear_queue()
    await push
    assert source.pushed == [] and source.cleared == 1
    assert source.queued_duration == 0.0


def test_installing_twice_keeps_one_hold():
    source_cls = patched()
    once = source_cls.capture_frame
    hold_playout(source_cls, hold=0.05)
    assert source_cls.capture_frame is once
