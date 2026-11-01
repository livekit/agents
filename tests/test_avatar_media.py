"""The native avatar publishes and captures through shared RTC capabilities."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice import room_io
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    QueueAudioOutput,
    VideoGenerator,
)

pytestmark = pytest.mark.unit


class Generator(VideoGenerator):
    def __init__(self):
        self.frames = utils.aio.Chan()

    async def push_audio(self, frame):
        pass

    def clear_buffer(self):
        pass

    def __aiter__(self):
        return self.frames


class Source:
    def __init__(self, events, kind):
        self.events = events
        self.kind = kind

    async def _core_capture(self, frame):
        self.events.append(("capture", self.kind))

    def _core_clear_queue(self):
        pass

    def _core_queued_duration(self):
        return 0.0

    async def _core_wait_for_playout(self):
        from livekit.agents._generated.rtc_wire import PlaybackInfo

        return PlaybackInfo(played=0.0, discarded=0.0)

    def _core_playback_info(self):
        from livekit.agents._generated.rtc_wire import PlaybackInfo

        return PlaybackInfo(played=0.0, discarded=0.0)

    async def _core_close(self):
        self.events.append(("close", self.kind))


class VideoSource(Source):
    def _core_capture(self, frame):
        self.events.append(("capture", self.kind))


class Publication:
    def __init__(self, events, kind, subscribed):
        self.events, self.kind, self.subscribed = events, kind, subscribed

    def _core_sid(self):
        return self.kind

    def _core_name(self):
        return f"avatar_{self.kind}"

    def _core_source(self):
        return "microphone" if self.kind == "audio" else "camera"

    def _core_kind(self):
        return 1 if self.kind == "audio" else 2

    def _core_audio_features(self):
        return []

    async def _core_wait_for_subscription(self):
        self.events.append(("subscribe", self.kind))
        await self.subscribed.wait()


@pytest.mark.parametrize("fail_video", [False, True])
async def test_publication_order_rollback_and_sdk_source_reuse(monkeypatch, fail_video):
    events = []
    subscribed = asyncio.Event()
    audio_source, video_source = Source(events, "audio"), VideoSource(events, "video")
    monkeypatch.setattr(room_io.Room, "_core_connection_state", lambda self: "connected")
    monkeypatch.setattr(room_io.Room, "_core_create_audio_source", lambda self, opts: audio_source)
    monkeypatch.setattr(room_io.Room, "_core_create_video_source", lambda self, opts: video_source)

    async def publish_audio(self, source, options):
        assert source is audio_source
        assert options.name == "avatar_audio"
        events.append(("publish", "audio"))
        return Publication(events, "audio", subscribed)

    async def publish_video(self, source, options):
        assert source is video_source
        assert (options.name, options.width, options.height) == ("avatar_video", 16, 16)
        events.append(("publish", "video"))
        if fail_video:
            raise ValueError("video publication failed")
        return Publication(events, "video", subscribed)

    async def unpublish(self, sid):
        events.append(("unpublish", sid))

    monkeypatch.setattr(room_io.LocalParticipant, "_core_publish_audio_source", publish_audio)
    monkeypatch.setattr(room_io.LocalParticipant, "_core_publish_video_source", publish_video)
    monkeypatch.setattr(room_io.LocalParticipant, "_core_unpublish_track", unpublish)
    audio = QueueAudioOutput(sample_rate=16000)
    generator = Generator()
    runner = AvatarRunner(
        rtc.Room(),
        audio_recv=audio,
        video_gen=generator,
        options=AvatarOptions(16, 16, 25, 16000, 1),
        _lazy_publish=False,
    )
    starting = asyncio.create_task(runner.start())
    try:
        async with asyncio.timeout(2):
            while ("subscribe", "audio") not in events:
                await asyncio.sleep(0)
        assert events == [("publish", "audio"), ("subscribe", "audio")]
        subscribed.set()
        if fail_video:
            with pytest.raises(ValueError, match="video publication failed"):
                await asyncio.wait_for(starting, 2)
            assert ("unpublish", "audio") in events
        else:
            await asyncio.wait_for(starting, 2)
            await runner.av_sync.push(
                rtc.VideoFrame(16, 16, rtc.VideoBufferType.RGBA, b"\0" * 1024)
            )
            await asyncio.wait_for(runner.av_sync.wait_for_playout(), 2)
            assert ("capture", "video") in events
    finally:
        subscribed.set()
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)
        await asyncio.wait_for(runner.aclose(), 2)
        await audio.aclose()
    assert events.count(("close", "audio")) == 1
    assert events.count(("close", "video")) == 1
    assert events.count(("unpublish", "audio")) == 1
    assert events.count(("unpublish", "video")) == (0 if fail_video else 1)
