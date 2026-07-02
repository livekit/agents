"""Tests for STT RecognizeStream input buffering and replay on reconnect."""

import pytest

from livekit import rtc
from livekit.agents._exceptions import APIError
from livekit.agents.stt.stt import RecognizeStream, SpeechEventType
from livekit.agents.types import APIConnectOptions

from .fake_stt import FakeSTT


def _make_audio_frame(
    duration_ms: int = 100,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> rtc.AudioFrame:
    """Create a silent audio frame with the given duration."""
    samples_per_channel = int(sample_rate * duration_ms / 1000)
    data = b"\x00\x00" * samples_per_channel * num_channels
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


class TestInputBuffer:
    """Tests for the input buffer append and eviction logic."""

    @pytest.mark.asyncio
    async def test_append_audio_frame(self):
        """Audio frames are appended and duration is tracked."""
        stream = FakeSTT(fake_transcript="hello").stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        frame = _make_audio_frame(duration_ms=100)
        stream._append_to_buffer(frame)

        assert len(stream._input_buffer) == 1
        assert stream._input_buffer[0] is frame
        assert stream._input_buffer_duration == pytest.approx(frame.duration, abs=1e-6)
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_append_flush_sentinel(self):
        """Flush sentinels are appended but don't affect duration."""
        stream = FakeSTT(fake_transcript="hello").stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        sentinel = RecognizeStream._FlushSentinel()
        stream._append_to_buffer(sentinel)

        assert len(stream._input_buffer) == 1
        assert isinstance(stream._input_buffer[0], RecognizeStream._FlushSentinel)
        assert stream._input_buffer_duration == 0.0
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_eviction_over_max_duration(self):
        """Oldest frames are evicted when buffer exceeds max duration."""
        stream = FakeSTT(fake_transcript="hello").stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        stream._max_buffer_duration = 1.0  # 1 second max

        # Push 15 frames of 100ms each = 1.5s total, should evict ~5 oldest
        frames = []
        for _ in range(15):
            frame = _make_audio_frame(duration_ms=100)
            frames.append(frame)
            stream._append_to_buffer(frame)

        assert stream._input_buffer_duration <= stream._max_buffer_duration
        assert len(stream._input_buffer) < 15
        # The most recent frames should still be in the buffer
        assert stream._input_buffer[-1] is frames[-1]
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_eviction_preserves_sentinels_duration(self):
        """When sentinels are evicted, duration doesn't go negative."""
        stream = FakeSTT(fake_transcript="hello").stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        stream._max_buffer_duration = 0.5

        # Push sentinel + frames that exceed max
        stream._append_to_buffer(RecognizeStream._FlushSentinel())
        for _ in range(10):
            stream._append_to_buffer(_make_audio_frame(duration_ms=100))

        assert stream._input_buffer_duration >= 0.0
        assert stream._input_buffer_duration <= stream._max_buffer_duration
        await stream.aclose()


class TestPushFrameBuffering:
    """Tests that push_frame() and flush() populate the input buffer."""

    @pytest.mark.asyncio
    async def test_push_frame_buffers(self):
        """push_frame() adds to both _input_ch and _input_buffer."""
        stream = FakeSTT(fake_transcript="hello").stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        frame = _make_audio_frame()

        stream.push_frame(frame)
        assert len(stream._input_buffer) == 1
        assert isinstance(stream._input_buffer[0], rtc.AudioFrame)

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_flush_buffers_sentinel(self):
        """flush() adds a FlushSentinel to _input_buffer."""
        stream = FakeSTT(fake_transcript="hello").stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        stream.push_frame(_make_audio_frame())
        stream.flush()

        sentinel_count = sum(
            1 for item in stream._input_buffer if isinstance(item, RecognizeStream._FlushSentinel)
        )
        assert sentinel_count == 1

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_end_input_sets_flag(self):
        """end_input() sets _input_ended flag."""
        stream = FakeSTT(fake_transcript="hello").stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        stream.push_frame(_make_audio_frame())
        stream.end_input()

        assert stream._input_ended is True
        # wait for task completion
        events = []
        async for ev in stream:
            events.append(ev)


class TestReconnectReplay:
    """Tests for channel replay during retry."""

    @pytest.mark.asyncio
    async def test_replay_on_retry(self):
        """After a retryable error, the input buffer is replayed into a fresh channel."""
        fail_count = 0

        class FailOnceSpeechStream(RecognizeStream):
            def __init__(self, *, stt, conn_options):
                super().__init__(stt=stt, conn_options=conn_options)
                self._frames_received: list[rtc.AudioFrame] = []

            async def _run(self) -> None:
                nonlocal fail_count
                fail_count += 1
                if fail_count == 1:
                    # consume some input, then fail
                    async for data in self._input_ch:
                        if isinstance(data, rtc.AudioFrame):
                            self._frames_received.append(data)
                        if isinstance(data, RecognizeStream._FlushSentinel):
                            break
                    raise APIError("transient failure")
                else:
                    # on retry, should receive replayed frames
                    count = 0
                    async for data in self._input_ch:
                        if isinstance(data, rtc.AudioFrame):
                            count += 1
                        if isinstance(data, RecognizeStream._FlushSentinel):
                            break
                    # emit a final transcript to indicate success
                    from livekit.agents.stt.stt import SpeechData, SpeechEvent

                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[
                                SpeechData(
                                    text="test",
                                    language="en",
                                )
                            ],
                        )
                    )
                    async for _ in self._input_ch:
                        pass

        class FailOnceSTT(FakeSTT):
            def stream(self, *, language=None, conn_options=None):
                if conn_options is None:
                    conn_options = APIConnectOptions()
                return FailOnceSpeechStream(stt=self, conn_options=conn_options)

        stt = FailOnceSTT(fake_transcript="test")
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0))

        # Push audio frames then end input
        for _ in range(5):
            stream.push_frame(_make_audio_frame())
        stream.end_input()

        events = []
        async for ev in stream:
            events.append(ev)

        # Should have gotten a final transcript on the retry
        assert fail_count == 2
        assert any(ev.type == SpeechEventType.FINAL_TRANSCRIPT for ev in events)
