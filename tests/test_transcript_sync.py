from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit import rtc
from livekit.agents import Agent, tokenize
from livekit.agents.types import TimedString
from livekit.agents.voice import io
from livekit.agents.voice.agent_activity import _aligned_transcript_or_text
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.transcription._speaking_rate import SpeakingRateDetector
from livekit.agents.voice.transcription.synchronizer import (
    TranscriptSynchronizer,
    _SegmentSynchronizerImpl,
    _TextSyncOptions,
)

from .fake_io import FakeAudioOutput, FakeTextOutput
from .fake_session import FakeActions, create_session, run_session
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SAMPLE_RATE = 16000


class _Collector(io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="collector", next_in_chain=None)
        self.words: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.words)

    async def capture_text(self, text: str) -> None:
        self.words.append(str(text))

    def flush(self) -> None:
        pass


def _silent_frames(duration: float) -> list[rtc.AudioFrame]:
    samples = SAMPLE_RATE // 100  # 10 ms
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )
    return [frame] * int(duration * 100)


@pytest.fixture
async def segment() -> AsyncIterator[tuple[_SegmentSynchronizerImpl, _Collector]]:
    collector = _Collector()
    opts = _TextSyncOptions(
        speed=1.0,
        hyphenate_word=tokenize.basic.hyphenate_word,
        word_tokenizer=tokenize.basic.WordTokenizer(
            retain_format=True, ignore_punctuation=False, split_character=True
        ),
        speaking_rate_detector=SpeakingRateDetector(),
    )
    impl = _SegmentSynchronizerImpl(opts, next_in_chain=collector)
    try:
        yield impl, collector
    finally:
        await impl.aclose()


async def _await_text(collector: _Collector, expected: str) -> str:
    deadline = time.monotonic() + 3.0
    while collector.text != expected and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    return collector.text


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_markup_fragments_add_no_pacing_delay(segment) -> None:
    """Expressive markup is forwarded verbatim but paced as zero spoken words."""
    impl, collector = segment
    # ~10 visible hyphens of speech against ~40 hyphens of markup fragments
    marked_up = (
        '<expr type="expression" label="speak with warm surprise and bright energy"/> '
        "Hello there my friend! "
        '<expr type="sound" label="laugh"/> '
        '<expr type="expression" label="speak calmly and evenly, unhurried"/> '
        "How are you today?"
    )
    for frame in _silent_frames(3.0):
        impl.push_audio(frame)
    impl.end_audio_input()
    impl.push_text(marked_up)
    impl.end_text_input()

    start = time.monotonic()
    impl.on_playback_started(time.time())
    await impl._main_atask
    await impl._capture_atask

    assert collector.text == marked_up
    # paced as speech, the markup alone would take over 12s
    assert time.monotonic() - start < 6.0


async def test_markup_only_turn_keeps_nonzero_speed(segment) -> None:
    impl, collector = segment
    markup_only = '<expr type="sound" label="laugh"/>'
    for frame in _silent_frames(1.0):
        impl.push_audio(frame)
    impl.end_audio_input()
    impl.push_text(markup_only)
    impl.end_text_input()

    impl.on_playback_started(time.time())
    await impl._main_atask
    await impl._capture_atask

    assert impl._speed > 0
    # no visible syllables to estimate from, so the speaking-unit speed stays unset
    assert impl._speed_on_speaking_unit is None
    assert collector.text == markup_only


@pytest.mark.parametrize(
    "fragments",
    [
        [("It is", 0.0, 0.2), (" sunny", 0.2, 0.4)],
        # a span ending mid-word is split across two deltas the output concatenates
        [("It is curren", 0.0, 0.2), ("tly sunny", 0.2, 0.4)],
    ],
)
async def test_timed_fragments_forward_the_last_word_while_the_segment_is_open(
    segment, fragments: list[tuple[str, float, float]]
) -> None:
    impl, collector = segment
    for frame in _silent_frames(0.4):
        impl.push_audio(frame)
    impl.end_audio_input()
    impl.on_playback_started(time.time())

    for text, start_time, end_time in fragments:
        impl.push_text(io.TimedString(text, start_time=start_time, end_time=end_time))

    expected = "".join(text for text, _, _ in fragments)
    assert await _await_text(collector, expected) == expected


async def test_speaking_rate_stream_end_input_twice_does_not_raise() -> None:
    stream = SpeakingRateDetector().stream()
    try:
        for frame in _silent_frames(0.5):
            stream.push_frame(frame)
        stream.end_input()
        stream.end_input()
        stream.flush()
        async for _ in stream:
            pass
    finally:
        await stream.aclose()


async def test_segment_end_audio_input_twice_keeps_speed_and_still_forwards(
    segment, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    impl, collector = segment
    text = "Hello there my friend!"
    for frame in _silent_frames(1.0):
        impl.push_audio(frame)
    impl.push_text(text)
    impl.end_text_input()
    impl.end_audio_input()
    speed, speed_on_unit = impl._speed, impl._speed_on_speaking_unit

    impl.end_audio_input()

    assert (impl._speed, impl._speed_on_speaking_unit) == (speed, speed_on_unit)
    impl.on_playback_started(time.time())
    assert await _await_text(collector, text) == text
    assert not _warnings(caplog)


async def test_synced_audio_output_flushed_twice_after_frames(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A wrapper flushes a segment early, then the end of the reply flushes it again."""
    caplog.set_level(logging.INFO, logger="livekit.agents")
    sync = TranscriptSynchronizer(
        next_in_chain_audio=FakeAudioOutput(sample_rate=SAMPLE_RATE),
        next_in_chain_text=FakeTextOutput(),
    )
    try:
        await sync.text_output.capture_text("hello")
        sync.text_output.flush()
        for frame in _silent_frames(0.5):
            await sync.audio_output.capture_frame(frame)

        sync.audio_output.flush()
        sync.audio_output.flush()

        sync.audio_output.on_playback_finished(playback_position=0.5, interrupted=False)
        await sync.barrier()
        assert not _warnings(caplog)
    finally:
        await sync.aclose()


async def test_barrier_cancellation_does_not_cancel_segment_rotation() -> None:
    sync = TranscriptSynchronizer(
        next_in_chain_text=FakeTextOutput(), next_in_chain_audio=FakeAudioOutput()
    )
    release = asyncio.Event()
    rotation = asyncio.create_task(release.wait())
    sync._rotate_segment_atask = rotation
    waiter = asyncio.create_task(sync.barrier())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not rotation.cancelled()
    release.set()
    await sync.barrier()
    await sync.aclose()


async def test_cancelled_close_does_not_replace_closed_segment() -> None:
    sync = TranscriptSynchronizer(
        next_in_chain_text=FakeTextOutput(), next_in_chain_audio=FakeAudioOutput()
    )
    release = asyncio.Event()
    original_impl = sync._impl

    async def delayed_close() -> None:
        await release.wait()

    original_impl.aclose = delayed_close  # type: ignore[method-assign]
    sync.rotate_segment()
    close = asyncio.create_task(sync.aclose())
    await asyncio.sleep(0)
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close
    release.set()
    await sync.barrier()

    assert sync._impl is original_impl
    del original_impl.aclose
    await original_impl.aclose()


# --- aligned transcript fallback ---


async def _stream(*items: str) -> AsyncIterable[str]:
    for item in items:
        yield item


async def _collect(source: AsyncIterable[str]) -> list[str]:
    return [chunk async for chunk in source]


async def test_aligned_transcript_is_forwarded_when_it_arrives() -> None:
    timed = _stream(
        TimedString("hello ", start_time=0.0, end_time=0.5),
        TimedString("there", start_time=0.5, end_time=1.0),
    )
    text = _stream("hello there")

    forwarded = await _collect(_aligned_transcript_or_text(timed, text))

    assert forwarded == ["hello ", "there"]
    assert all(isinstance(chunk, TimedString) for chunk in forwarded)
    assert await _collect(text) == ["hello there"]


async def test_falls_back_to_the_spoken_text_when_no_timings_arrive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        forwarded = await _collect(
            _aligned_transcript_or_text(_stream(), _stream("hello ", "there"))
        )

    assert forwarded == ["hello ", "there"]
    assert any("no aligned transcript" in record.message for record in caplog.records)


async def test_partial_alignment_is_not_duplicated(caplog: pytest.LogCaptureFixture) -> None:
    # one timed word proves the provider aligns, so the spoken text is not replayed on top
    timed = _stream(TimedString("hello", start_time=0.0, end_time=0.5))

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        forwarded = await _collect(_aligned_transcript_or_text(timed, _stream("hello there")))

    assert forwarded == ["hello"]
    assert not [r for r in caplog.records if "no aligned transcript" in r.message]


async def test_turn_keeps_its_transcript_when_the_tts_sends_no_timings() -> None:
    """A TTS that claims alignment but sends audio only still yields an agent transcript."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.1)
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.2)
    actions.add_tts(2.0, ttfb=0.2, duration=0.2)

    session = create_session(actions, extra_kwargs={"use_tts_aligned_transcript": True})
    assert isinstance(session.tts, FakeTTS)
    session.tts._capabilities.aligned_transcript = True

    conversation: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", conversation.append)

    await asyncio.wait_for(run_session(session, Agent(instructions="test")), timeout=30)

    agent_messages = [
        ev.item for ev in conversation if ev.item.type == "message" and ev.item.role == "assistant"
    ]
    assert agent_messages and agent_messages[0].text_content
