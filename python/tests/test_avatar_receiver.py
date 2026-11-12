"""A custom ``AudioReceiver`` drives ``AvatarRunner`` through its iterator and playback hooks."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import (
    AudioReceiver,
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    VideoGenerator,
)

pytestmark = pytest.mark.unit


def frame():
    return rtc.AudioFrame(b"\0\0" * 160, 16000, 1, 160)


class Receiver(AudioReceiver):
    def __init__(self, items):
        super().__init__()
        self.items = items
        self.reports = []

    def notify_playback_started(self, *, segment):
        self.reports.append(("started", segment))

    async def notify_playback_finished(self, playback_position, interrupted, *, segment):
        self.reports.append(("finished", segment, interrupted))

    async def __aiter__(self):
        for item in self.items:
            yield item


class Generator(VideoGenerator):
    def __init__(self):
        self.received = asyncio.Queue()
        self.frames = utils.aio.Chan()

    async def push_audio(self, item):
        self.received.put_nowait(item)
        if isinstance(item, AudioSegmentEnd):
            self.frames.send_nowait(item)

    def clear_buffer(self):
        pass

    def __aiter__(self):
        return self.frames


async def test_a_custom_receiver_feeds_each_segment_and_hears_its_playback():
    receiver = Receiver([frame(), frame(), AudioSegmentEnd(), frame(), AudioSegmentEnd()])
    generator = Generator()
    runner = AvatarRunner(
        rtc.Room(),
        audio_recv=receiver,
        video_gen=generator,
        options=AvatarOptions(16, 16, 25, 16000, 1),
    )
    try:
        await runner.start()
        kinds = [type(await asyncio.wait_for(generator.received.get(), 1)) for _ in range(5)]
        assert kinds == [
            rtc.AudioFrame,
            rtc.AudioFrame,
            AudioSegmentEnd,
            rtc.AudioFrame,
            AudioSegmentEnd,
        ]
        async with asyncio.timeout(1):
            while len(receiver.reports) < 2:
                await asyncio.sleep(0.01)
        finished = [report for report in receiver.reports if report[0] == "finished"]
        assert [interrupted for _, _, interrupted in finished] == [False, False]
        first, second = (segment for _, segment, _ in finished)
        assert first != second
    finally:
        generator.frames.close()
        await runner.aclose()
