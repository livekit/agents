from __future__ import annotations

import pytest

from livekit import rtc
from livekit.agents.voice.io import (
    AgentOutput,
    AudioOutput,
    AudioOutputCapabilities,
    PlaybackFinishedEvent,
    _AudioSinkProxy,
)

from .fake_io import FakeAudioOutput


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


# ---------- swap_audio_endpoint ----------


def test_swap_audio_endpoint_swaps_proxy_inner() -> None:
    leaf_a = FakeAudioOutput()
    leaf_b = FakeAudioOutput()
    output = _make_agent_output()
    wrapper = _PassthroughWrapper(next_in_chain=leaf_a)
    output.audio = wrapper

    output.swap_audio_endpoint(leaf_b)

    # wrapper chain intact; only the leaf swapped
    assert output.audio is wrapper
    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)
    assert proxy.next_in_chain is leaf_b


def test_swap_audio_endpoint_falls_back_when_no_proxy() -> None:
    leaf = FakeAudioOutput()
    output = _make_agent_output()
    # no wrapper chain yet

    output.swap_audio_endpoint(leaf)

    assert output.audio is leaf


def test_swap_audio_endpoint_none_detaches_proxy_inner() -> None:
    leaf = FakeAudioOutput()
    output = _make_agent_output()
    wrapper = _PassthroughWrapper(next_in_chain=leaf)
    output.audio = wrapper

    output.swap_audio_endpoint(None)

    proxy = wrapper.next_in_chain
    assert isinstance(proxy, _AudioSinkProxy)
    assert proxy.next_in_chain is None


# ---------- proxy invariants ----------


@pytest.mark.asyncio
async def test_proxy_accepts_wrapper_chain_as_inner() -> None:
    leaf = FakeAudioOutput()
    wrapped_sink = _PassthroughWrapper(next_in_chain=leaf)
    proxy = _AudioSinkProxy()
    proxy.set_next_in_chain(wrapped_sink)

    assert proxy.next_in_chain is wrapped_sink

    # events from the inner leaf still bubble up through proxy
    received: list[PlaybackFinishedEvent] = []
    proxy.on("playback_finished", received.append)

    await proxy.capture_frame(_silence())
    leaf.on_playback_finished(playback_position=1.0, interrupted=False)

    assert len(received) == 1
    assert received[0].playback_position == 1.0


@pytest.mark.asyncio
async def test_proxy_synthesizes_playback_finished_when_detached() -> None:
    proxy = _AudioSinkProxy()
    received: list[PlaybackFinishedEvent] = []
    proxy.on("playback_finished", received.append)

    await proxy.capture_frame(_silence())
    proxy.flush()

    # detached proxy treats the segment as interrupted (no real sink played it)
    assert len(received) == 1
    assert received[0].interrupted is True
    assert received[0].playback_position == 0.0


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
