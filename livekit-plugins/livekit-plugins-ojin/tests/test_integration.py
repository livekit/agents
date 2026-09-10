"""End-to-end over the real QueueAudioOutput, without a room.

The claim this plugin exists to make is that multi-turn conversations complete:
Ojin never emits a segment marker of its own, and the agent session blocks on
``wait_for_playout()`` after every utterance, so a missing marker wedges the
conversation permanently. Everything else is detail.

``AvatarRunner`` needs an ``rtc.Room`` only to publish tracks; the part that
matters here is its ~15-line marker branch, reproduced faithfully below so the
claim is covered by a regression test rather than by a manual check.
"""

from __future__ import annotations

import asyncio

import pytest
from fake_stv import FakeSTVClient
from ojin.stv import FrameType, STVEvent

from livekit import rtc
from livekit.agents.voice.avatar import AudioReceiver, AudioSegmentEnd, QueueAudioOutput
from livekit.plugins.ojin.avatar import OjinVideoGenerator, _FrameSink

# Hermetic: driven by a fake Ojin client, no network and no credentials.
pytestmark = pytest.mark.unit


class RunnerHarness:
    """Mirrors AvatarRunner's two loops and its playback-finished protocol."""

    def __init__(self) -> None:
        self.sink = _FrameSink()
        self.client = FakeSTVClient(output=self.sink)
        self.generator = OjinVideoGenerator(self.client, self.sink)
        self.audio_output = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)

        # The same wiring AvatarSession installs: the stop edge is what closes an
        # output segment.
        self.client.add_listener(STVEvent.BOT_STOPPED_SPEAKING, self.sink.on_bot_stopped_speaking)

        receiver: AudioReceiver = self.audio_output
        receiver.on("clear_buffer", self._on_clear_buffer)

        self._playback_position = 0.0
        self._audio_playing = False
        self._tasks: list[asyncio.Task[None]] = []
        self.markers_seen = 0  # segment completions the plugin handed the runner

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._read_audio()),
            asyncio.create_task(self._forward()),
        ]

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _read_audio(self) -> None:
        # AvatarRunner._read_audio
        async for frame in self.audio_output:
            if isinstance(frame, rtc.AudioFrame):
                self._audio_playing = True
            await self.generator.push_audio(frame)

    async def _forward(self) -> None:
        # AvatarRunner._forward_video, minus the track publishing
        async for frame in self.generator:
            if isinstance(frame, AudioSegmentEnd):
                self.markers_seen += 1
                if self._audio_playing:
                    self.audio_output.notify_playback_finished(
                        playback_position=self._playback_position, interrupted=False
                    )
                    self._audio_playing = False
                    self._playback_position = 0.0
                continue

            if isinstance(frame, rtc.AudioFrame):
                if self._playback_position == 0.0 and frame.duration > 0.0:
                    self.audio_output.notify_playback_started()
                self._playback_position += frame.duration

    def _on_clear_buffer(self) -> None:
        # AvatarRunner._on_clear_buffer
        self.sink.note_clear_pending()
        audio_playing = self._audio_playing
        self._audio_playing = False

        async def handle() -> None:
            await self.generator.clear_buffer()
            if audio_playing:
                self.audio_output.notify_playback_finished(
                    playback_position=self._playback_position, interrupted=True
                )
                self._playback_position = 0.0

        self._tasks.append(asyncio.create_task(handle()))


async def until(predicate, timeout: float = 3.0, what: str = "condition") -> None:
    """Wait for an observable condition instead of guessing at a delay.

    The harness runs its read and forward loops as independent tasks, so a fixed
    sleep is a bet on scheduling that a loaded machine will eventually lose.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


def tts_chunk(ms: int = 200) -> rtc.AudioFrame:
    samples = int(24000 * ms / 1000)
    return rtc.AudioFrame(
        data=b"\x11\x22" * samples,
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=samples,
    )


@pytest.fixture
async def harness():
    h = RunnerHarness()
    await h.start()
    yield h
    await h.aclose()


async def speak(h: RunnerHarness, *, chunks: int = 3) -> None:
    """One agent utterance: TTS in, then the server's echo back."""
    for _ in range(chunks):
        await h.audio_output.capture_frame(tts_chunk())
    h.audio_output.flush()
    await until(
        lambda: len(h.client.sent) >= chunks and h.sink._segments.input_closed,
        what="the read loop to forward the utterance and close the input",
    )

    # A turn's first server frame is START_OF_SPEECH; that is what lifts the mute
    # after a barge-in, so the echo has to carry it.
    for i in range(chunks):
        await h.client.push_tick(
            frame_type=FrameType.START_OF_SPEECH if i == 0 else FrameType.SPEECH
        )
    await until(lambda: h.sink._segments.output_open, what="the echo to open an output segment")
    await h.client.emit_stopped_speaking()


async def test_one_turn_completes(harness: RunnerHarness) -> None:
    """Without the synthesized marker this call never returns."""
    await speak(harness)

    ev = await asyncio.wait_for(harness.audio_output.wait_for_playout(), 2)

    assert ev.interrupted is False
    assert ev.playback_position > 0.0


async def test_three_consecutive_turns_complete(harness: RunnerHarness) -> None:
    for _ in range(3):
        await speak(harness)
        ev = await asyncio.wait_for(harness.audio_output.wait_for_playout(), 2)
        assert ev.interrupted is False


async def test_underrun_does_not_end_the_turn_early(harness: RunnerHarness) -> None:
    """Slow TTS drains the buffer mid-utterance; the session must not proceed."""
    await harness.audio_output.capture_frame(tts_chunk())
    await until(lambda: bool(harness.client.sent), what="the chunk to reach the client")
    await harness.client.push_tick()
    await harness.client.emit_stopped_speaking()  # spurious drain edge

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(harness.audio_output.wait_for_playout(), 0.2)

    # the rest of the utterance arrives and really ends
    await harness.audio_output.capture_frame(tts_chunk())
    harness.audio_output.flush()
    await until(lambda: harness.sink._segments.input_closed, what="the input segment to close")
    await harness.client.push_tick()
    await harness.client.emit_stopped_speaking()

    ev = await asyncio.wait_for(harness.audio_output.wait_for_playout(), 2)
    assert ev.interrupted is False


async def test_barge_in_recovers_into_a_following_turn(harness: RunnerHarness) -> None:
    """The turn after a barge-in must still produce its completion marker.

    Asserted on the marker the plugin hands the runner, not on wait_for_playout:
    the framework never resets AudioOutput.__capturing in clear_buffer, so the
    next capture_frame opens no new segment and the follow-up report is dropped
    by io.py's count guard. That accounting is identical for every avatar plugin;
    what this plugin owns is emitting the completion marker at all.
    """
    events: list[bool] = []
    harness.audio_output.on("playback_finished", lambda ev: events.append(ev.interrupted))

    await harness.audio_output.capture_frame(tts_chunk())
    await until(lambda: bool(harness.client.sent), what="the chunk to reach the client")
    await harness.client.push_tick()
    await until(lambda: harness.sink._segments.output_open, what="the echo to start playing")

    harness.audio_output.clear_buffer()
    await until(lambda: events == [True], what="the interrupted report")
    assert harness.markers_seen == 0, "an interrupted turn must not also emit a marker"

    await speak(harness)
    await until(lambda: harness.markers_seen == 1, what="the following turn to complete")


async def test_silent_utterance_still_completes(harness: RunnerHarness) -> None:
    """An all-zero utterance never echoes back, so nothing would close it."""
    silent = rtc.AudioFrame(
        data=bytes(9600), sample_rate=24000, num_channels=1, samples_per_channel=4800
    )
    await harness.audio_output.capture_frame(silent)
    harness.audio_output.flush()

    ev = await asyncio.wait_for(harness.audio_output.wait_for_playout(), 2)

    assert ev.interrupted is False


async def test_idle_ticks_do_not_complete_a_captured_segment(harness: RunnerHarness) -> None:
    """Ojin streams silence forever between turns; none of it is a segment."""
    await harness.audio_output.capture_frame(tts_chunk())
    harness.audio_output.flush()
    await until(lambda: harness.sink._segments.input_closed, what="the input segment to close")

    for _ in range(10):
        await harness.client.push_tick(silent=True, frame_type=FrameType.IDLE)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(harness.audio_output.wait_for_playout(), 0.2)
