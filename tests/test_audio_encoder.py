from __future__ import annotations

import array
import io
import math
from collections.abc import Iterable
from typing import get_args

import av
import numpy as np
import pytest

from livekit import rtc
from livekit.agents.utils.codecs import AudioStreamDecoder
from livekit.agents.utils.codecs.encoder import (
    _SUPPORTED_CODECS,
    AudioStreamEncoder,
    EncodedAudioData,
    _CompactableBuffer,
)

_CODECS: list[str] = get_args(_SUPPORTED_CODECS)


def _mime_for(codec: str) -> str:
    _TABLE: dict[str, str] = {
        "opus": "audio/ogg",
        "mp3": "audio/mpeg",
        "pcm": "audio/wav",
    }
    return _TABLE[codec]


def _find_byte_sync(data: bytes, mask: int, value: int) -> int:
    for off in range(len(data) - 3):
        if (data[off] & mask) == value:
            return off
    return -1


def _make_encoder(
    codec: str,
    *,
    num_channels: int = 1,
    sample_rate: int = 48000,
    bit_rate: int = 24000,
) -> AudioStreamEncoder:
    """Construct an encoder or skip the test if the codec isn't available in av/ffmpeg."""
    try:
        return AudioStreamEncoder(
            codec=codec,
            sample_rate=sample_rate,
            num_channels=num_channels,
            bit_rate=bit_rate,
        )
    except (av.FFmpegError, ValueError) as e:
        pytest.skip(f"codec {codec!r} not available in av/ffmpeg build: {e}")


def _silence_frame(
    *,
    sample_rate: int = 48000,
    num_channels: int = 1,
    samples_per_channel: int = 480,
) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes(samples_per_channel * num_channels * 2),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


def _sine_frames(
    *,
    sample_rate: int = 48000,
    num_channels: int = 1,
    duration_s: float = 0.2,
    chunk_samples: int = 480,
    freq_hz: float = 440.0,
    amplitude: int = 16000,
) -> list[rtc.AudioFrame]:
    total_samples = int(sample_rate * duration_s)
    pcm = array.array("h")
    two_pi_f_over_sr = 2.0 * math.pi * freq_hz / sample_rate
    for n in range(total_samples):
        s = int(amplitude * math.sin(two_pi_f_over_sr * n))
        for _ in range(num_channels):
            pcm.append(s)

    raw = pcm.tobytes()
    bytes_per_chunk = chunk_samples * num_channels * 2
    frames: list[rtc.AudioFrame] = []
    for off in range(0, len(raw), bytes_per_chunk):
        chunk = raw[off : off + bytes_per_chunk]
        if len(chunk) < bytes_per_chunk:
            break
        frames.append(
            rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=chunk_samples,
            )
        )
    return frames


def _encode_all(enc: AudioStreamEncoder, frames: Iterable[rtc.AudioFrame]) -> tuple[bytes, int]:
    buf = bytearray()
    total_samples = 0
    for f in frames:
        out = enc.push(f)
        buf.extend(out.data)
        total_samples += out.num_samples
    closed = enc.close()
    buf.extend(closed.data)
    total_samples += closed.num_samples
    return bytes(buf), total_samples


async def _decode_all(
    data: bytes, *, mime: str, sample_rate: int = 48000, num_channels: int = 1
) -> list[rtc.AudioFrame]:
    dec = AudioStreamDecoder(sample_rate=sample_rate, num_channels=num_channels, format=mime)
    dec.push(data)
    dec.end_input()
    out: list[rtc.AudioFrame] = []
    async for frame in dec:
        out.append(frame)
    await dec.aclose()
    return out


def _frames_to_i16(frames: Iterable[rtc.AudioFrame]) -> np.ndarray:
    buf = bytearray()
    for f in frames:
        buf.extend(bytes(f.data))
    return np.frombuffer(bytes(buf), dtype=np.int16)


def _align(ref: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align ``dec`` against ``ref`` via cross-correlation and return equal-length slices.

    Absorbs encoder priming / decoder delay (Opus ~312, MP3 ~576-1152 samples).
    """
    ref_f = ref.astype(np.float64)
    dec_f = dec.astype(np.float64)
    corr = np.correlate(dec_f, ref_f, mode="full")
    lag = int(np.argmax(corr)) - (len(ref_f) - 1)
    if lag >= 0:
        dec_a = dec_f[lag:]
        ref_a = ref_f[: len(dec_a)]
    else:
        ref_a = ref_f[-lag:]
        dec_a = dec_f[: len(ref_a)]
    n = min(len(ref_a), len(dec_a))
    return ref_a[:n], dec_a[:n]


def _snr_db(ref: np.ndarray, dec: np.ndarray) -> float:
    ref_f = ref.astype(np.float64)
    err = dec.astype(np.float64) - ref_f
    sig_p = float(np.mean(ref_f**2)) + 1e-12
    err_p = float(np.mean(err**2)) + 1e-12
    return 10.0 * math.log10(sig_p / err_p)


def _dominant_freq(x: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Return (peak frequency in Hz, fraction of total energy within ±2 FFT bins of the peak)."""
    xf = x.astype(np.float64) * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(xf))
    k = int(np.argmax(mag))
    freq = k * sample_rate / len(x)
    total = float(np.sum(mag**2)) + 1e-12
    band = mag[max(0, k - 2) : k + 3]
    concentration = float(np.sum(band**2)) / total
    return freq, concentration


class TestAudioStreamEncoder:
    def test_unsupported_codec_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported codec"):
            AudioStreamEncoder(codec="not-a-real-codec")

    def test_push_after_close_raises(self) -> None:
        enc = _make_encoder("opus")
        enc.close()
        with pytest.raises(RuntimeError, match="encoder is closed"):
            enc.push(_silence_frame())

    def test_close_is_idempotent(self) -> None:
        enc = _make_encoder("opus")
        for f in _sine_frames(duration_s=0.1):
            enc.push(f)
        first = enc.close()
        assert isinstance(first, EncodedAudioData)

        second = enc.close()
        assert second.data == b""
        assert second.num_samples == 0

    def test_first_non_empty_push_contains_opus_container_headers(self) -> None:
        """The first non-empty emission must carry the OGG page magic and the OpusHead
        identification header, ahead of any audio content (docstring guarantee on L92)."""
        enc = _make_encoder("opus")
        try:
            first_non_empty = b""
            # 20ms frames at 48kHz; push up to 1s to guarantee the muxer flushes a page.
            for f in _sine_frames(duration_s=1.0, chunk_samples=960):
                out = enc.push(f)
                if out.data:
                    first_non_empty = out.data
                    break
        finally:
            enc.close()

        assert first_non_empty, "encoder produced no output across 1s of input"
        assert first_non_empty.startswith(b"OggS"), (
            f"expected OGG magic on first emission, got {first_non_empty[:8]!r}"
        )
        assert b"OpusHead" in first_non_empty, (
            "first emission should include the OpusHead identification header"
        )

    def test_codec_options(self) -> None:
        frames = _sine_frames(duration_s=1.0, chunk_samples=960)

        def _encode_with_frame_duration(ms: str) -> bytes:
            enc = AudioStreamEncoder(
                codec="opus",
                sample_rate=48000,
                num_channels=1,
                bit_rate=24000,
                codec_options={"frame_duration": ms, "application": "audio"},
            )
            buf = bytearray()
            for f in frames:
                buf.extend(enc.push(f).data)
            buf.extend(enc.close().data)
            return bytes(buf)

        small = _encode_with_frame_duration("10")
        large = _encode_with_frame_duration("60")

        def _packet_count(data: bytes) -> int:
            with av.open(io.BytesIO(data), mode="r") as container:
                return sum(1 for _ in container.demux(audio=0) if _.dts is not None)

        small_packets = _packet_count(small)
        large_packets = _packet_count(large)

        # 10ms frames produce roughly 6x more packets than 60ms for the same input.
        assert small_packets > large_packets * 3, (
            f"codec_options frame_duration did not propagate: "
            f"frame_duration=10 produced {small_packets} packets, "
            f"frame_duration=60 produced {large_packets}"
        )

    def test_drain_cursor_concatenation(self) -> None:
        """All bytes emitted across push() + close() must concatenate into a valid, re-parseable container."""
        enc = _make_encoder("opus")
        frames = _sine_frames(duration_s=1.0, chunk_samples=960)

        collected = bytearray()
        for f in frames:
            out = enc.push(f)
            collected.extend(out.data)
        final = enc.close()
        collected.extend(final.data)

        full = bytes(collected)
        assert len(full) > 0

        with av.open(io.BytesIO(full), mode="r") as container:
            assert len(container.streams.audio) == 1
            decoded_any = False
            for _ in container.decode(audio=0):
                decoded_any = True
            assert decoded_any, "re-parse of concatenated bytes produced no frames"

    def _assert_magic(self, codec: str, data: bytes) -> None:
        if codec == "opus":
            assert data[:4] == b"OggS", f"opus: expected OggS magic, got {data[:8]!r}"
        elif codec == "pcm":
            assert data[:4] == b"RIFF", f"pcm: expected RIFF, got {data[:4]!r}"
            assert data[8:12] == b"WAVE", f"pcm: expected WAVE at offset 8, got {data[8:12]!r}"
        elif codec == "mp3":
            # MP3 stream may start with an ID3 tag or directly with a frame sync (0xFFE_).
            if data[:3] == b"ID3":
                return
            off = _find_byte_sync(data, mask=0xFF, value=0xFF)
            assert off != -1, "mp3: no 0xFF sync byte found in output"
            assert (data[off + 1] & 0xE0) == 0xE0, (
                f"mp3: bad frame sync high bits at offset {off + 1}: {data[off + 1]:#x}"
            )
        else:
            pytest.fail(f"no magic-byte assertion defined for codec {codec!r}")

    @pytest.mark.parametrize("codec", _CODECS)
    def test_output_has_valid_magic(self, codec: str) -> None:
        enc = _make_encoder(codec)
        data, _ = _encode_all(enc, _sine_frames(duration_s=0.2))
        assert len(data) > 0, f"{codec}: produced empty output"
        self._assert_magic(codec, data)

    @pytest.mark.parametrize("codec", _CODECS)
    def test_output_is_reparseable(self, codec: str) -> None:
        enc = _make_encoder(codec)
        data, _ = _encode_all(enc, _sine_frames(duration_s=0.3))
        assert len(data) > 0, f"{codec}: produced empty output"

        with av.open(io.BytesIO(data), mode="r") as container:
            assert len(container.streams.audio) >= 1, f"{codec}: no audio stream found"
            decoded_frames = 0
            for _ in container.decode(audio=0):
                decoded_frames += 1
            assert decoded_frames > 0, f"{codec}: container produced no decoded frames"

    # (codec, min_snr_db, freq_tol_hz, min_energy_concentration)
    #
    # Thresholds are calibrated for a 1 s / 48 kHz / mono 440 Hz sine at 64 kbps,
    # with ~20% headroom below measured values. SNR is measured post-alignment
    # (cross-correlation) to absorb codec priming. Energy concentration is the
    # fraction of spectral energy within ±2 FFT bins of the peak. Pure tones are
    # deceptively hard for Opus/MP3 — realistic inputs (speech, noise) typically
    # measure noticeably higher SNR.
    @pytest.mark.parametrize(
        "codec, min_snr_db, freq_tol_hz, min_energy_conc",
        [
            ("pcm", 80.0, 0.5, 0.99),
            ("opus", 12.0, 2.0, 0.95),
            ("mp3", 13.0, 2.0, 0.95),
        ],
    )
    async def test_roundtrip_fidelity(
        self,
        codec: str,
        min_snr_db: float,
        freq_tol_hz: float,
        min_energy_conc: float,
    ) -> None:
        sample_rate = 48000
        num_channels = 1
        freq_hz = 440.0
        input_frames = _sine_frames(
            sample_rate=sample_rate,
            num_channels=num_channels,
            duration_s=1.0,
            chunk_samples=960,
            freq_hz=freq_hz,
        )
        ref = _frames_to_i16(input_frames)
        total_samples = len(ref)

        enc = _make_encoder(
            codec,
            sample_rate=sample_rate,
            num_channels=num_channels,
            bit_rate=64000,
        )
        encoded, reported_samples = _encode_all(enc, input_frames)
        assert len(encoded) > 0, f"{codec}: encoder produced empty output"
        assert reported_samples > 0, f"{codec}: encoder reported zero samples across packets"

        decoded_frames = await _decode_all(
            encoded,
            mime=_mime_for(codec),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        dec = _frames_to_i16(decoded_frames)

        # (a) Sample count: decoded and encoder-reported lengths within 100 ms of input.
        # Absolute tolerance covers encoder priming / padding (Opus ~312, MP3 ~1152)
        # without hiding missing-tail bugs at the ~5000+ sample scale.
        abs_tol = sample_rate // 10
        assert reported_samples == pytest.approx(total_samples, abs=abs_tol)
        assert len(dec) == pytest.approx(total_samples, abs=abs_tol)

        # (b) Spectral fidelity: the 440 Hz tone survives encoding/decoding.
        # Insensitive to priming drift — a good floor check for every codec.
        peak_hz, concentration = _dominant_freq(dec, sample_rate)
        assert abs(peak_hz - freq_hz) < freq_tol_hz, (
            f"{codec}: dominant frequency {peak_hz:.2f} Hz, expected {freq_hz} Hz "
            f"(tol ±{freq_tol_hz} Hz)"
        )
        assert concentration >= min_energy_conc, (
            f"{codec}: only {concentration:.3f} of energy near {freq_hz} Hz, "
            f"expected ≥{min_energy_conc}"
        )

        # (c) Waveform fidelity post-alignment. PCM is effectively lossless;
        # Opus/MP3 thresholds are empirical at 64 kbps for a pure tone.
        ref_a, dec_a = _align(ref, dec)
        snr = _snr_db(ref_a, dec_a)
        assert snr >= min_snr_db, f"{codec}: post-align SNR {snr:.1f} dB, expected ≥{min_snr_db} dB"


class TestCompactableBuffer:
    def test_reports_write_only_non_seekable(self) -> None:
        """libav picks streaming muxer paths (no header back-patching) based on these flags."""
        buf = _CompactableBuffer()
        assert buf.writable()
        assert not buf.readable()
        assert not buf.seekable()

    def test_write_returns_byte_count(self) -> None:
        buf = _CompactableBuffer()
        assert buf.write(b"hello") == 5
        assert buf.write(b"") == 0

    def test_drain_returns_all_bytes_written_since_last_drain(self) -> None:
        buf = _CompactableBuffer()
        buf.write(b"abc")
        buf.write(b"def")
        assert buf.drain() == b"abcdef"

        buf.write(b"xyz")
        assert buf.drain() == b"xyz"

    def test_drain_returns_empty_when_no_new_bytes(self) -> None:
        buf = _CompactableBuffer()
        assert buf.drain() == b""

        buf.write(b"data")
        buf.drain()
        assert buf.drain() == b""

    def test_concatenated_drains_equal_all_writes(self) -> None:
        """The core invariant: every byte written is returned exactly once, in order."""
        buf = _CompactableBuffer()
        chunks = [b"a" * 100, b"b" * 200, b"c" * 300, b"d" * 50]

        collected = bytearray()
        for chunk in chunks:
            buf.write(chunk)
            collected.extend(buf.drain())

        assert bytes(collected) == b"".join(chunks)

    def test_compaction_reclaims_memory_past_threshold(self) -> None:
        buf = _CompactableBuffer()
        chunk = b"x" * (buf._COMPACT_THRESHOLD + 1)
        buf.write(chunk)

        drained = buf.drain()
        assert drained == chunk
        assert len(buf._buf) == 0, "backing storage should be reclaimed after compaction"
        assert buf._read_pos == 0

    def test_no_compaction_below_threshold(self) -> None:
        buf = _CompactableBuffer()
        chunk = b"x" * 1024
        buf.write(chunk)
        buf.drain()

        assert len(buf._buf) == 1024
        assert buf._read_pos == 1024

    def test_drain_after_compaction_returns_only_new_bytes(self) -> None:
        """After compaction, subsequent writes must not re-emit or drop any byte."""
        buf = _CompactableBuffer()
        big = b"a" * (buf._COMPACT_THRESHOLD + 1)
        buf.write(big)
        assert buf.drain() == big
        assert len(buf._buf) == 0

        buf.write(b"tail")
        assert buf.drain() == b"tail"

    def test_repeated_compaction_bounds_memory(self) -> None:
        """Peak backing-storage size stays near the threshold across many write/drain cycles."""
        buf = _CompactableBuffer()
        threshold = buf._COMPACT_THRESHOLD
        chunk = b"x" * (threshold // 4)

        peak = 0
        for _ in range(20):
            buf.write(chunk)
            buf.drain()
            peak = max(peak, len(buf._buf))

        # Each drain sets _read_pos == len(_buf). When that crosses threshold, compaction
        # clears _buf. So peak is bounded by (threshold - 1) + one chunk's worth.
        assert peak < threshold + len(chunk)

    def test_plugs_into_av_open(self) -> None:
        """The buffer must satisfy libav's file-like contract for streaming muxers."""
        buf = _CompactableBuffer()
        try:
            container = av.open(buf, mode="w", format="ogg")
        except (av.FFmpegError, ValueError) as e:
            pytest.skip(f"libopus not available: {e}")

        stream = container.add_stream("libopus", rate=48000, layout="mono")
        frame = av.AudioFrame(format="s16", layout="mono", samples=480)
        frame.rate = 48000
        frame.planes[0].update(bytes(960))
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()

        data = buf.drain()
        assert data.startswith(b"OggS"), f"expected OGG magic, got {data[:8]!r}"
        with av.open(io.BytesIO(data), mode="r") as c:
            assert len(c.streams.audio) == 1
