import asyncio
import time

from playout_hold import hold_playout


class FakeSource:
    def __init__(self) -> None:
        self.queued_duration = 0.0
        self.pushed: list[str] = []
        self.cleared = 0

    async def capture_frame(self, frame: str) -> None:
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
    await source.capture_frame("first")
    assert time.monotonic() - started >= 0.05
    source.queued_duration = 0.08
    started = time.monotonic()
    await source.capture_frame("second")
    assert time.monotonic() - started < 0.02
    assert source.pushed == ["first", "second"]


async def test_a_frame_held_when_the_caller_interrupts_is_dropped():
    source = patched()()
    push = asyncio.create_task(source.capture_frame("stale"))
    await asyncio.sleep(0.01)
    source.clear_queue()
    await push
    assert source.pushed == [] and source.cleared == 1


def test_installing_twice_keeps_one_hold():
    source_cls = patched()
    once = source_cls.capture_frame
    hold_playout(source_cls, hold=0.05)
    assert source_cls.capture_frame is once
