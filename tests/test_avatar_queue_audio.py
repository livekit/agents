"""Queue-backed avatar audio cuts release playout without inventing provider reports."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents._generated.io_events import AvatarPlaybackSegment
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput

pytestmark = pytest.mark.unit


def frame():
    return rtc.AudioFrame(b"\0\0" * 160, 16000, 1, 160)


async def test_delayed_interruption_report_does_not_finish_the_next_segment():
    audio = QueueAudioOutput(sample_rate=16000)
    finished = []
    audio.on("playback_finished", finished.append)
    try:
        await audio.capture_frame(frame())
        old = audio.playback_segment
        assert old is not None
        assert isinstance(await anext(audio), rtc.AudioFrame)
        audio.clear_buffer()
        await audio.capture_frame(frame())
        new = audio.playback_segment
        assert new is not None
        audio.flush()
        assert isinstance(await anext(aiter(audio)), rtc.AudioFrame)
        assert isinstance(await anext(aiter(audio)), AudioSegmentEnd)
        waiting = asyncio.create_task(audio.wait_for_playout())
        await asyncio.sleep(0)
        audio.notify_playback_finished(
            0.0, True, segment=AvatarPlaybackSegment(playback=old, stream_id=None)
        )
        await asyncio.sleep(0)
        assert not waiting.done(), "the interrupted report belongs to the cleared segment"
        assert len(finished) == 1
        audio.notify_playback_finished(
            0.01, False, segment=AvatarPlaybackSegment(playback=new, stream_id=None)
        )
        result = await asyncio.wait_for(waiting, 1)
        assert result.interrupted is False
        assert result.playback_position == 0.01
        assert len(finished) == 2
    finally:
        await audio.aclose()


async def test_close_discards_queued_audio_and_releases_readers_and_playout():
    audio = QueueAudioOutput(sample_rate=16000)
    await audio.capture_frame(frame())
    waiting = asyncio.create_task(audio.wait_for_playout())
    await asyncio.sleep(0)
    try:
        await audio.aclose()
        result = await asyncio.wait_for(waiting, 1)
        assert result.interrupted is True
        assert [item async for item in audio] == []
        await audio.aclose()
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_native_output_close_does_not_wait_for_the_avatar_to_join():
    from livekit.agents.voice.avatar import DataStreamAudioOutput

    output = DataStreamAudioOutput(rtc.Room(), destination_identity="avatar")
    await asyncio.wait_for(output.aclose(), 1)
    await asyncio.wait_for(output.aclose(), 1)


async def test_clear_only_requests_a_stop_when_new_audio_has_been_captured():
    audio = QueueAudioOutput(sample_rate=16000)
    clear = []
    audio.on("clear_buffer", lambda: clear.append(True))
    try:
        audio.clear_buffer()
        assert clear == []
        await audio.capture_frame(frame())
        audio.flush()
        audio.clear_buffer()
        audio.clear_buffer()
        assert clear == [True]
        await audio.capture_frame(frame())
        audio.clear_buffer()
        assert clear == [True, True]
    finally:
        await audio.aclose()


@pytest.mark.parametrize(
    "segments_read, clear_during_delivery", [(0, False), (2, False), (0, True)]
)
async def test_clear_accounting_matches_the_real_runner(
    monkeypatch, segments_read, clear_during_delivery
):
    from livekit.agents import utils
    from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, VideoGenerator

    class Generator(VideoGenerator):
        def __init__(self):
            self.received = asyncio.Queue()
            self.frames = utils.aio.Chan()

        async def push_audio(self, item):
            self.received.put_nowait(item)

        def clear_buffer(self):
            pass

        def __aiter__(self):
            return self.frames

    class Media:
        def __init__(self, *args):
            pass

        async def _core_publish(self):
            pass

        async def _core_push(self, item):
            pass

        async def _core_aclose(self):
            pass

    monkeypatch.setattr("livekit.agents.voice.avatar._AvatarMedia", Media)

    class ClearOnDelivery(QueueAudioOutput):
        first = True

        async def _core_next_audio(self):
            observation = await super()._core_next_audio()
            if self.first and isinstance(observation[2], rtc.AudioFrame):
                self.first = False
                # Exactly the race: the host yielded a tagged frame, then clear reaches
                # the native owner before this asynchronous result crosses the boundary.
                self.clear_buffer()
            return observation

    audio_type = ClearOnDelivery if clear_during_delivery else QueueAudioOutput
    audio = audio_type(sample_rate=16000)
    generator = Generator()
    runner = AvatarRunner(
        room=rtc.Room(),
        audio_recv=audio,
        video_gen=generator,
        options=AvatarOptions(16, 16, 25, 16000, 1),
    )
    try:
        if segments_read:
            for _ in range(segments_read):
                await audio.capture_frame(frame())
                audio.flush()
            await runner.start()
            for _ in range(segments_read * 2):
                await asyncio.wait_for(generator.received.get(), 1)
        if clear_during_delivery:
            await runner.start()
            await audio.capture_frame(frame())
        else:
            # This newest segment has never reached the runner and cannot receive its ack.
            await audio.capture_frame(frame())
            audio.clear_buffer()
        result = await asyncio.wait_for(audio.wait_for_playout(), 1)
        assert result.interrupted
        if clear_during_delivery:
            assert generator.received.empty(), "the cleared frame reached the generator"
        if not segments_read and not clear_during_delivery:
            await runner.start()
        await audio.capture_frame(frame())
        audio.flush()
        await asyncio.wait_for(generator.received.get(), 1)
        await asyncio.wait_for(generator.received.get(), 1)
        waiting = asyncio.create_task(audio.wait_for_playout())
        await asyncio.sleep(0)
        assert not waiting.done()
        generator.frames.send_nowait(AudioSegmentEnd())
        result = await asyncio.wait_for(waiting, 1)
        assert not result.interrupted
    finally:
        await runner.aclose()
        await audio.aclose()
