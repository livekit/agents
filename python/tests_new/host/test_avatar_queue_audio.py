"""The in-process avatar: ``QueueAudioOutput`` as the agent's audio output and the avatar's
``AudioReceiver``, driven by ``AvatarRunner``. A clear interrupts only the audio it cut, and each
segment finishes when the video generator ends it.

The room is never connected, so the runner feeds the generator but publishes nothing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import NamedTuple

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import (
    AudioReceiver,
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioOutput,
    QueueAudioOutput,
    VideoGenerator,
)
from livekit.agents.voice.io import PlaybackFinishedEvent
from tests_new.fakes import until

OPTIONS = AvatarOptions(16, 16, 25, 16000, 1)


def frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(b"\0\0" * 160, 16000, 1, 160)


class Generator(VideoGenerator):
    """A video generator that ends each segment only on ``end()``. It takes audio only while
    ``delivering`` is set; its ``clear_buffer`` waits for ``clearing`` to be set, and
    ``clear_on_delivery`` clears that output when audio arrives."""

    def __init__(self) -> None:
        self.received: list[rtc.AudioFrame | AudioSegmentEnd] = []
        self.clearing = asyncio.Event()
        self.clearing.set()
        self.delivering = asyncio.Event()
        self.delivering.set()
        self.clear_on_delivery: QueueAudioOutput | None = None
        self._ends: asyncio.Queue[AudioSegmentEnd] = asyncio.Queue()

    def ends_received(self) -> int:
        return sum(isinstance(item, AudioSegmentEnd) for item in self.received)

    def end(self) -> None:
        self._ends.put_nowait(AudioSegmentEnd())

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        await self.delivering.wait()
        self.received.append(frame)
        if (audio := self.clear_on_delivery) and isinstance(frame, rtc.AudioFrame):
            self.clear_on_delivery = None
            audio.clear_buffer()

    async def clear_buffer(self) -> None:
        await self.clearing.wait()

    def __aiter__(self) -> AsyncIterator[AudioSegmentEnd]:
        return self._frames()

    async def _frames(self) -> AsyncIterator[AudioSegmentEnd]:
        while True:
            yield await self._ends.get()


class Avatar(NamedTuple):
    audio: QueueAudioOutput
    generator: Generator
    runner: AvatarRunner
    finished: list[PlaybackFinishedEvent]


@pytest.fixture
async def avatar() -> AsyncIterator[Avatar]:
    audio = QueueAudioOutput(sample_rate=16000)
    generator = Generator()
    runner = AvatarRunner(rtc.Room(), audio_recv=audio, video_gen=generator, options=OPTIONS)
    finished: list[PlaybackFinishedEvent] = []
    audio.on("playback_finished", finished.append)
    yield Avatar(audio, generator, runner, finished)
    await runner.aclose()
    await audio.aclose()


async def say(audio: QueueAudioOutput) -> None:
    """One segment of one frame."""
    await audio.capture_frame(frame())
    audio.flush()


async def clear_before_the_runner_starts(avatar: Avatar) -> PlaybackFinishedEvent:
    await avatar.audio.capture_frame(frame())
    avatar.audio.clear_buffer()
    result = await avatar.audio.wait_for_playout()
    await avatar.runner.start()
    return result


async def clear_after_segments_the_generator_never_ended(avatar: Avatar) -> PlaybackFinishedEvent:
    await say(avatar.audio)
    await say(avatar.audio)
    await avatar.runner.start()
    await until(lambda: avatar.generator.ends_received() == 2, "both segments are delivered")
    await avatar.audio.capture_frame(frame())
    avatar.audio.clear_buffer()
    return await avatar.audio.wait_for_playout()


async def clear_while_the_runner_delivers_a_frame(avatar: Avatar) -> PlaybackFinishedEvent:
    avatar.generator.clear_on_delivery = avatar.audio
    await avatar.runner.start()
    await avatar.audio.capture_frame(frame())
    return await avatar.audio.wait_for_playout()


CLEARS: list[Callable[[Avatar], Awaitable[PlaybackFinishedEvent]]] = [
    clear_before_the_runner_starts,
    clear_after_segments_the_generator_never_ended,
    clear_while_the_runner_delivers_a_frame,
]


@pytest.mark.parametrize("clear", CLEARS)
async def test_a_clear_interrupts_the_audio_it_cut(
    avatar: Avatar, clear: Callable[[Avatar], Awaitable[PlaybackFinishedEvent]]
) -> None:
    """Clearing audio the avatar has not finished playing settles its playout as interrupted."""
    result = await clear(avatar)

    assert result.interrupted


@pytest.mark.parametrize("clear", CLEARS)
async def test_after_a_clear_the_next_segment_finishes_at_the_generators_end(
    avatar: Avatar, clear: Callable[[Avatar], Awaitable[PlaybackFinishedEvent]]
) -> None:
    """After a clear, the next segment's playout waits for the generator to end that segment,
    whatever the clear left unacknowledged."""
    await clear(avatar)
    delivered = avatar.generator.ends_received()
    await say(avatar.audio)
    await until(lambda: avatar.generator.ends_received() > delivered, "the segment is delivered")
    waiting = asyncio.ensure_future(avatar.audio.wait_for_playout())
    pending_before_end = not waiting.done()

    avatar.generator.end()

    result = await waiting
    assert pending_before_end
    assert not result.interrupted


async def test_a_clear_during_delivery_never_reaches_the_generator_as_audio(
    avatar: Avatar,
) -> None:
    """A clear that lands while the runner hands a frame to the generator stops the segment
    there: the generator gets that frame and nothing after it."""
    await clear_while_the_runner_delivers_a_frame(avatar)

    assert [type(item) for item in avatar.generator.received] == [rtc.AudioFrame]


async def test_a_late_interruption_report_does_not_finish_the_next_segment(
    avatar: Avatar,
) -> None:
    """When the generator's clear completes only after the next segment was captured, the
    interruption it reports settles the cleared segment, never the next one."""
    avatar.generator.clearing.clear()
    await avatar.runner.start()
    await avatar.audio.capture_frame(frame())
    await until(lambda: avatar.generator.received, "the first segment reaches the generator")
    avatar.audio.clear_buffer()
    await say(avatar.audio)
    waiting = asyncio.ensure_future(avatar.audio.wait_for_playout())

    avatar.generator.clearing.set()
    await until(lambda: avatar.generator.ends_received() == 1, "the next segment is delivered")

    assert [event.interrupted for event in avatar.finished] == [True]
    assert not waiting.done()
    avatar.generator.end()
    assert not (await waiting).interrupted


async def test_close_discards_queued_audio_and_releases_readers_and_playout() -> None:
    """Closing the queue settles a pending playout as interrupted, drops the audio no reader
    took and ends iteration; a second close is harmless."""
    audio = QueueAudioOutput(sample_rate=16000)
    await audio.capture_frame(frame())
    waiting = asyncio.ensure_future(audio.wait_for_playout())

    await audio.aclose()
    await audio.aclose()

    assert (await waiting).interrupted
    assert [item async for item in audio] == []


async def test_clear_asks_the_avatar_to_stop_only_when_new_audio_was_captured() -> None:
    """``clear_buffer`` emits ``"clear_buffer"`` to the avatar once per stretch of newly captured
    audio: never on an empty queue, never twice for the same audio."""
    audio = QueueAudioOutput(sample_rate=16000)
    requests: list[None] = []
    audio.on("clear_buffer", lambda: requests.append(None))

    audio.clear_buffer()
    await say(audio)
    audio.clear_buffer()
    audio.clear_buffer()
    await audio.capture_frame(frame())
    audio.clear_buffer()

    assert len(requests) == 2
    await audio.aclose()


async def test_closing_the_data_stream_output_does_not_wait_for_the_avatar_to_join() -> None:
    """The agent's avatar output closes at once, twice over, even though no avatar ever joined
    its room."""
    output = DataStreamAudioOutput(rtc.Room(), destination_identity="avatar")

    await output.aclose()
    await output.aclose()


class Receiver(AudioReceiver):
    """An application's own audio receiver: it yields scripted audio, then waits for more until
    closed (or ends there when ``runs_out``), and records the playback reports the runner sends
    back."""

    def __init__(
        self, items: list[rtc.AudioFrame | AudioSegmentEnd], *, runs_out: bool = False
    ) -> None:
        super().__init__()
        self.items = items
        self.runs_out = runs_out
        self.ran_out = False
        self.finished: list[tuple[object, bool]] = []
        self.closed = asyncio.Event()

    def notify_playback_started(self, *, segment: object) -> None:
        pass

    async def notify_playback_finished(
        self, playback_position: float, interrupted: bool, *, segment: object
    ) -> None:
        self.finished.append((segment, interrupted))

    async def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        for item in self.items:
            yield item
        self.ran_out = True
        if not self.runs_out:
            await self.closed.wait()

    async def aclose(self) -> None:
        self.closed.set()


async def test_a_custom_receiver_feeds_each_segment_and_hears_its_playback() -> None:
    """``AvatarRunner`` feeds a custom ``AudioReceiver``'s segments to the generator in order and
    reports each one finished back to that receiver, under its own segment."""
    receiver = Receiver([frame(), frame(), AudioSegmentEnd(), frame(), AudioSegmentEnd()])
    generator = Generator()
    runner = AvatarRunner(rtc.Room(), audio_recv=receiver, video_gen=generator, options=OPTIONS)
    await runner.start()

    await until(lambda: generator.ends_received() == 2, "both segments are delivered")
    generator.end()
    generator.end()
    await until(lambda: len(receiver.finished) == 2, "both segments are reported")

    await runner.aclose()
    kinds = [type(item) for item in generator.received]
    assert kinds == [
        rtc.AudioFrame,
        rtc.AudioFrame,
        AudioSegmentEnd,
        rtc.AudioFrame,
        AudioSegmentEnd,
    ]
    (first, first_cut), (second, second_cut) = receiver.finished
    assert not first_cut and not second_cut
    assert first != second


async def test_a_receiver_that_runs_out_still_delivers_its_last_segment() -> None:
    """A custom ``AudioReceiver`` whose iterator ends while the generator is still behind has
    all of its last segment's audio reach the generator, and its playback reported."""
    last_segment = [frame() for _ in range(50)]
    receiver = Receiver(
        [frame(), AudioSegmentEnd(), *last_segment, AudioSegmentEnd()], runs_out=True
    )
    generator = Generator()
    generator.delivering.clear()
    runner = AvatarRunner(rtc.Room(), audio_recv=receiver, video_gen=generator, options=OPTIONS)
    await runner.start()

    await until(lambda: receiver.ran_out, "the receiver runs out while the generator is behind")
    generator.delivering.set()
    await until(lambda: generator.ends_received() == 2, "both segments are delivered")
    generator.end()
    generator.end()
    await until(lambda: len(receiver.finished) == 2, "both segments are reported")

    await runner.aclose()
    delivered = [i.samples_per_channel for i in generator.received if isinstance(i, rtc.AudioFrame)]
    assert sum(delivered) == 51 * frame().samples_per_channel
