import logging

import pytest

from livekit.agents.utils.audio import AudioByteStream

pytestmark = pytest.mark.unit

SAMPLE_RATE = 8000
SAMPLES_PER_FRAME = 1600  # 200ms
BYTES_PER_SAMPLE = 2  # mono, 16 bit


def _stream() -> AudioByteStream:
    return AudioByteStream(
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLES_PER_FRAME,
    )


class TestAudioByteStreamFlush:
    """flush() must not lose complete samples to a partial trailing one."""

    def test_flush_returns_buffered_samples(self):
        stream = _stream()
        assert stream.push(b"\x11\x22" * 1500) == []

        frames = stream.flush()
        assert len(frames) == 1
        assert frames[0].samples_per_channel == 1500
        assert len(stream._buf) == 0

    def test_flush_keeps_complete_samples_when_buffer_ends_mid_sample(self):
        stream = _stream()
        stream.push(b"\x11\x22" * 1500 + b"\x33")

        frames = stream.flush()
        assert len(frames) == 1, "the 1500 complete samples must survive the partial one"
        assert frames[0].samples_per_channel == 1500

    def test_flush_drops_only_the_trailing_partial_sample(self):
        stream = _stream()
        body = b"\x11\x22" * 3
        stream.push(body + b"\x33")

        frames = stream.flush()
        assert bytes(frames[0].data)[: len(body)] == body

    def test_flush_empties_the_buffer_after_a_partial_sample(self):
        """A leftover byte would offset every sample of a later push."""
        stream = _stream()
        stream.push(b"\x11\x22" * 10 + b"\x33")
        stream.flush()
        assert len(stream._buf) == 0

    def test_flush_warns_with_the_dropped_byte_count(self, caplog):
        stream = _stream()
        stream.push(b"\x11\x22" * 10 + b"\x33")

        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            stream.flush()

        assert "dropping the trailing 1 byte(s)" in caplog.text

    def test_flush_of_a_lone_partial_sample_returns_nothing(self):
        stream = _stream()
        stream.push(b"\x33")

        assert stream.flush() == []
        assert len(stream._buf) == 0

    def test_flush_of_an_empty_buffer_returns_nothing(self, caplog):
        stream = _stream()
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            assert stream.flush() == []
        assert caplog.text == ""

    def test_flush_after_whole_frames_were_emitted(self):
        stream = _stream()
        frames = stream.push(b"\x11\x22" * (SAMPLES_PER_FRAME + 500) + b"\x33")
        assert len(frames) == 1
        assert frames[0].samples_per_channel == SAMPLES_PER_FRAME

        remaining = stream.flush()
        assert len(remaining) == 1
        assert remaining[0].samples_per_channel == 500

    def test_flush_is_idempotent(self):
        stream = _stream()
        stream.push(b"\x11\x22" * 10 + b"\x33")
        assert len(stream.flush()) == 1
        assert stream.flush() == []

    def test_stereo_partial_sample_keeps_whole_frames(self):
        """bytes_per_sample is 4 for stereo, so a 3 byte tail is partial."""
        stream = AudioByteStream(sample_rate=SAMPLE_RATE, num_channels=2, samples_per_channel=1600)
        stream.push(b"\x11\x22\x33\x44" * 100 + b"\x55\x66\x77")

        frames = stream.flush()
        assert len(frames) == 1
        assert frames[0].samples_per_channel == 100
        assert len(stream._buf) == 0
