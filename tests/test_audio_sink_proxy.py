from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents.voice.io import (
    AgentOutput,
    AudioOutput,
    AudioOutputCapabilities,
    PlaybackFinishedEvent,
    PlaybackProgressedEvent,
    _AudioSinkProxy,
)

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit


def _make_agent_output() -> AgentOutput:
    return AgentOutput(lambda: None, lambda: None, lambda: None)


def _silence(duration_s: float = 0.01, sample_rate: int = 16000) -> rtc.AudioFrame:
    n = int(sample_rate * duration_s)
    return rtc.AudioFrame(
        data=b"\x00\x00" * n,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=n,
    )


class _PassthroughWrapper(AudioOutput):
    """Minimal wrapper that forwards capture_frame/flush/clear_buffer through
    next_in_chain. Used to exercise the auto-wrap and swap mechanics without
    pulling in TranscriptSynchronizer or RecorderAudioOutput.
    """

    def __init__(self, *, next_in_chain: AudioOutput) -> None:
        super().__init__(
            label="Passthrough",
            capabilities=AudioOutputCapabilities(pause=True),
            next_in_chain=next_in_chain,
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        assert self.next_in_chain is not None
        await self.next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        assert self.next_in_chain is not None
        self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        assert self.next_in_chain is not None
        self.next_in_chain.clear_buffer()


class _TrackingSink(AudioOutput):
    """Leaf sink that records attach/detach calls for assertion."""

    def __init__(self) -> None:
        super().__init__(label="TrackingSink", capabilities=AudioOutputCapabilities(pause=True))
        self.attached_calls = 0
        self.detached_calls = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass

    def on_attached(self) -> None:
        self.attached_calls += 1
        super().on_attached()

    def on_detached(self) -> None:
        self.detached_calls += 1
        super().on_detached()


# ---------- auto-wrap ----------


def test_auto_wrap_inserts_proxy_above_bare_leaf() -> None:
    leaf = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf)

    assert isinstance(wrapper.next_in_chain, _AudioSinkProxy)
    assert wrapper.next_in_chain.next_in_chain is leaf


def test_auto_wrap_skipped_when_passed_an_existing_proxy() -> None:
    leaf = FakeAudioOutput()
    proxy = _AudioSinkProxy(leaf)
    wrapper = _PassthroughWrapper(next_in_chain=proxy)

    assert wrapper.next_in_chain is proxy


def test_auto_wrap_skipped_when_wrapping_a_non_leaf() -> None:
    leaf = FakeAudioOutput()
    inner = _PassthroughWrapper(next_in_chain=leaf)
    outer = _PassthroughWrapper(next_in_chain=inner)

    # outer should reference inner directly — no proxy interposed between them
    assert outer.next_in_chain is inner


# ---------- replace_audio_tail ----------


def test_replace_audio_tail_swaps_proxy_inner() -> None:
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    output = _make_agent_output()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    output.audio = wrapper

    output.replace_audio_tail(leaf_b)

    # wrapper chain intact; only the leaf swapped
    assert output.audio is wrapper
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)
    assert proxy.next_in_chain is leaf_b


def test_replace_audio_tail_falls_back_when_no_proxy() -> None:
    leaf = FakeAudioOutput()
    output = _make_agent_output()
    # no wrapper chain yet

    output.replace_audio_tail(leaf)

    assert output.audio is leaf


# ---------- proxy invariants ----------


@pytest.mark.asyncio
async def test_proxy_accepts_wrapper_chain_as_inner() -> None:
    leaf = FakeAudioOutput()
    wrapped_sink = _PassthroughWrapper(next_in_chain=leaf)
    proxy = _AudioSinkProxy(wrapped_sink)

    assert proxy.next_in_chain is wrapped_sink

    # events from the inner leaf still bubble up through proxy
    received: list[PlaybackFinishedEvent] = []
    proxy.on("playback_finished", received.append)

    await proxy.capture_frame(_silence())
    leaf.on_playback_finished(playback_position=1.0, interrupted=False)

    assert len(received) == 1
    assert received[0].playback_position == 1.0


# ---------- swap routing ----------


@pytest.mark.asyncio
async def test_swap_routes_playback_events_from_new_leaf() -> None:
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackFinishedEvent] = []
    wrapper.on("playback_finished", received.append)

    proxy.set_next_in_chain(leaf_b)
    # capture after the swap so leaf_b has a segment to mark as finished
    await wrapper.capture_frame(_silence())

    leaf_b.on_playback_finished(playback_position=1.0, interrupted=False)
    assert len(received) == 1
    assert received[0].playback_position == 1.0


async def test_swap_routes_playback_progress_from_new_leaf() -> None:
    """A wrapper above a swapped leaf keeps hearing where the audio went."""
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackProgressedEvent] = []
    wrapper.on("playback_progressed", received.append)

    leaf_a.on_playback_progressed(started_at=100.0, offset=0.0, duration=0.5)
    proxy.set_next_in_chain(leaf_b)
    leaf_b.on_playback_progressed(started_at=101.0, offset=0.0, duration=0.25)

    assert [(ev.started_at, ev.duration) for ev in received] == [(100.0, 0.5), (101.0, 0.25)]


async def test_swap_stops_playback_progress_from_the_old_leaf() -> None:
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackProgressedEvent] = []
    wrapper.on("playback_progressed", received.append)

    proxy.set_next_in_chain(leaf_b)
    leaf_a.on_playback_progressed(started_at=100.0, offset=0.0, duration=0.5)

    assert received == []


class _ReportingSink(FakeAudioOutput):
    """A sink with a local playhead, which reports it as it is cleared."""

    def __init__(self, played: float = 0.4, started_at: float = 100.0) -> None:
        super().__init__()
        self._played = played
        self._started_at = started_at

    def clear_buffer(self) -> None:
        self.on_playback_progressed(started_at=self._started_at, offset=0.0, duration=self._played)
        super().clear_buffer()


async def test_a_swap_still_hears_where_the_old_sink_got_to() -> None:
    """The clear happens while the proxy is still listening for progress."""
    leaf_a, leaf_b = _ReportingSink(), FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackProgressedEvent] = []
    wrapper.on("playback_progressed", received.append)

    await wrapper.capture_frame(_silence(duration_s=1.0))
    wrapper.flush()
    proxy.set_next_in_chain(leaf_b)

    assert [(ev.offset, ev.duration) for ev in received] == [(0.0, 0.4)]


async def test_a_mid_segment_swap_keeps_progress_offsets_on_the_segment() -> None:
    """A sink attached mid-segment counts from its own zero, the segment is already further in."""
    leaf_a, leaf_b = _ReportingSink(played=2.0, started_at=100.0), FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackProgressedEvent] = []
    wrapper.on("playback_progressed", received.append)

    # the avatar sink arrives mid-segment; what leaf_a still held never plays
    await wrapper.capture_frame(_silence(duration_s=5.0))
    proxy.set_next_in_chain(leaf_b)
    await wrapper.capture_frame(_silence(duration_s=5.0))
    leaf_b.on_playback_progressed(started_at=105.0, offset=0.0, duration=5.0)

    assert [(ev.offset, ev.duration) for ev in received] == [(0.0, 2.0), (5.0, 5.0)]


async def test_a_detached_sink_that_reported_nothing_keeps_its_stretch() -> None:
    """It cannot be asked now, so all it was given counts as played."""
    leaf_a, leaf_b = FakeAudioOutput(), FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackProgressedEvent] = []
    wrapper.on("playback_progressed", received.append)

    await wrapper.capture_frame(_silence(duration_s=5.0))
    proxy.set_next_in_chain(leaf_b)

    assert [(ev.offset, ev.duration) for ev in received] == [(0.0, 5.0)]


async def test_a_sink_that_only_reports_a_position_is_placed_at_its_offset() -> None:
    """All a remote sink can say is how much played, so the segment supplies the where."""
    leaf_a, leaf_b = _ReportingSink(played=0.03), FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackProgressedEvent] = []
    wrapper.on("playback_progressed", received.append)

    await wrapper.capture_frame(_silence(duration_s=0.05))
    proxy.set_next_in_chain(leaf_b)
    await wrapper.capture_frame(_silence(duration_s=0.02))
    wrapper.flush()
    await asyncio.wait_for(wrapper.wait_for_playout(), timeout=2.0)

    # leaf_b reports no runs, only that 0.02s played, and the segment puts that at 0.05
    assert [(round(ev.offset, 3), round(ev.duration, 3)) for ev in received] == [
        (0.0, 0.03),
        (0.05, 0.02),
    ]


async def test_the_next_segment_counts_from_its_own_start_again() -> None:
    leaf_a, leaf_b = _ReportingSink(played=0.05), FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackProgressedEvent] = []
    wrapper.on("playback_progressed", received.append)

    await wrapper.capture_frame(_silence(duration_s=0.05))
    proxy.set_next_in_chain(leaf_b)
    await wrapper.capture_frame(_silence(duration_s=0.02))
    wrapper.flush()
    await asyncio.wait_for(wrapper.wait_for_playout(), timeout=2.0)

    await wrapper.capture_frame(_silence(duration_s=0.02))
    leaf_b.on_playback_progressed(started_at=200.0, offset=0.0, duration=0.02)

    # leaf_a's stretch, leaf_b's stretch in the same segment, then a segment from zero again
    assert [round(ev.offset, 3) for ev in received] == [0.0, 0.05, 0.0]


@pytest.mark.asyncio
async def test_swap_disconnects_old_leaf() -> None:
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    # give leaf_a a segment so its on_playback_finished would normally emit
    await wrapper.capture_frame(_silence())

    received: list[PlaybackFinishedEvent] = []
    wrapper.on("playback_finished", received.append)

    proxy.set_next_in_chain(leaf_b)

    # leaf_a is detached: any event it fires must not reach the wrapper
    leaf_a.on_playback_finished(playback_position=0.5, interrupted=False)
    assert received == []


# ---------- swap with in-flight playback ----------


class _ClearCountingSink(FakeAudioOutput):
    def __init__(self) -> None:
        super().__init__()
        self.clear_calls = 0
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1
        super().flush()

    def clear_buffer(self) -> None:
        self.clear_calls += 1
        super().clear_buffer()


async def test_a_mid_capture_swap_interrupts_the_old_sink() -> None:
    """It sees a flush and a clear, the shape every other interruption gives it."""
    leaf_a, leaf_b = _ClearCountingSink(), FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    await wrapper.capture_frame(_silence(duration_s=0.05))
    proxy.set_next_in_chain(leaf_b)

    assert (leaf_a.flush_calls, leaf_a.clear_calls) == (1, 1)


async def test_a_swap_after_a_flush_does_not_flush_again() -> None:
    """A second flush cancels the playout wait that is already draining the sink."""
    leaf_a, leaf_b = _ClearCountingSink(), FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    await wrapper.capture_frame(_silence(duration_s=1.0))
    wrapper.flush()
    proxy.set_next_in_chain(leaf_b)

    assert (leaf_a.flush_calls, leaf_a.clear_calls) == (1, 1)


@pytest.mark.asyncio
async def test_swap_finishes_pending_segment_as_interrupted() -> None:
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    # a flushed segment still playing out on leaf_a (frames are pushed faster than realtime)
    await wrapper.capture_frame(_silence(duration_s=1.0))
    wrapper.flush()

    received: list[PlaybackFinishedEvent] = []
    wrapper.on("playback_finished", received.append)

    proxy.set_next_in_chain(leaf_b)

    # the pending segment must be finished as interrupted so wait_for_playout() doesn't hang
    ev = await asyncio.wait_for(wrapper.wait_for_playout(), timeout=0.5)
    assert ev.interrupted is True
    assert ev.playback_position == pytest.approx(1.0)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_swap_clears_old_sink_with_inflight_audio() -> None:
    leaf_a = _ClearCountingSink()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    await wrapper.capture_frame(_silence(duration_s=1.0))
    wrapper.flush()

    proxy.set_next_in_chain(leaf_b)

    assert leaf_a.clear_calls == 1


def test_idle_swap_does_not_clear_old_sink() -> None:
    leaf_a = _ClearCountingSink()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    proxy.set_next_in_chain(leaf_b)

    assert leaf_a.clear_calls == 0


@pytest.mark.asyncio
async def test_swap_mid_capture_segment_finishes_on_new_leaf() -> None:
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)

    received: list[PlaybackFinishedEvent] = []
    wrapper.on("playback_finished", received.append)

    # swap in the middle of a segment, before flush
    await wrapper.capture_frame(_silence(duration_s=0.05))
    proxy.set_next_in_chain(leaf_b)

    # no synthesized event: the segment continues on leaf_b, which reports it
    assert received == []

    await wrapper.capture_frame(_silence(duration_s=0.05))
    wrapper.flush()

    ev = await asyncio.wait_for(wrapper.wait_for_playout(), timeout=1.0)
    assert ev.interrupted is False
    assert len(received) == 1


# ---------- attached state ----------


def test_proxy_propagates_on_attached_to_current_inner() -> None:
    sink = _TrackingSink()
    proxy = _AudioSinkProxy(sink)

    proxy.on_attached()

    assert sink.attached_calls == 1
    assert sink.detached_calls == 0


def test_proxy_propagates_on_detached_to_current_inner() -> None:
    sink = _TrackingSink()
    proxy = _AudioSinkProxy(sink)

    proxy.on_attached()
    proxy.on_detached()

    assert sink.attached_calls == 1
    assert sink.detached_calls == 1


def test_swap_while_attached_attaches_new_and_detaches_old() -> None:
    sink_a = _TrackingSink()
    sink_b = _TrackingSink()
    proxy = _AudioSinkProxy(sink_a)
    proxy.on_attached()

    assert sink_a.attached_calls == 1

    proxy.set_next_in_chain(sink_b)

    assert sink_a.detached_calls == 1
    assert sink_b.attached_calls == 1


def test_swap_while_detached_does_not_fire_lifecycle_hooks() -> None:
    sink_a = _TrackingSink()
    sink_b = _TrackingSink()
    proxy = _AudioSinkProxy(sink_a)
    # never attached

    proxy.set_next_in_chain(sink_b)

    assert sink_a.detached_calls == 0
    assert sink_b.attached_calls == 0
