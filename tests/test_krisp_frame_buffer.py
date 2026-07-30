"""Regression tests for the Krisp license-mode frame buffering.

These exercise the adaptive frame-size buffering in
``_KrispLicenseFrameProcessor._process`` in isolation, with the Krisp SDK
replaced by an identity "session". The property under test: the processor must
never inject silence *between* real audio. Doing so produces audible gaps
(a "cutting in and out" artifact) once frames are concatenated downstream, and
happens whenever the input frame size is not a clean multiple of the internal
chunk size.

With an identity session the output is a pure subsequence of the input — no
zeros are ever inserted — so the whole output stream must equal the input,
minus the tail still buffered in flight.

The tests are hermetic — they do not import or require the ``krisp_audio`` wheel.
"""

from __future__ import annotations

import numpy as np
import pytest

from livekit import rtc
from livekit.plugins.krisp._krisp import _KrispLicenseFrameProcessor

pytestmark = pytest.mark.unit


class _IdentitySession:
    """Stand-in for a Krisp NC session that returns its input unchanged.

    With an identity session, the processor's output (minus any warm-up prefix)
    must equal its input exactly and in order — so any zero inserted mid-stream
    is unambiguously an artifact of the buffering, not of the model.
    """

    def process(self, chunk_in: np.ndarray, level: int) -> np.ndarray:
        return np.asarray(chunk_in, dtype=np.int16)


def _make_processor(sample_rate: int, chunk_samples: int) -> _KrispLicenseFrameProcessor:
    # Build the object without going through __init__ so we don't touch the SDK.
    proc = object.__new__(_KrispLicenseFrameProcessor)
    proc._filtering_enabled = True
    proc._warned_channels = False
    proc._noise_suppression_level = 100
    proc._session = _IdentitySession()
    proc._sample_rate = sample_rate
    proc._chunk_samples = chunk_samples
    proc._frame_duration_ms = int(chunk_samples * 1000 / sample_rate)
    proc._in_buf = np.empty(0, dtype=np.int16)
    proc._out_buf = np.empty(0, dtype=np.int16)
    return proc


def _feed(
    proc: _KrispLicenseFrameProcessor,
    sample_rate: int,
    frame_sizes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Feed a monotonic ramp through the processor one frame at a time.

    Returns (input_stream, output_stream) as flat int16 arrays. The ramp starts
    at 1 so that any 0 in the output is necessarily injected silence, never a
    genuine input sample.
    """
    counter = 1
    fed: list[np.ndarray] = []
    out: list[np.ndarray] = []
    for fs in frame_sizes:
        in_arr = np.arange(counter, counter + fs, dtype=np.int16)
        counter += fs
        fed.append(in_arr)
        frame = rtc.AudioFrame(
            data=in_arr.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=fs,
        )
        processed = proc._process(frame)
        out.append(np.frombuffer(processed.data, dtype=np.int16))
    return np.concatenate(fed), np.concatenate(out)


def _interior_zeros(stream: np.ndarray) -> int:
    """Count zeros that appear after the first non-zero sample (injected silence)."""
    nonzero = np.flatnonzero(stream)
    if nonzero.size == 0:
        return 0
    return int((stream[nonzero[0] :] == 0).sum())


def test_no_interior_silence_when_frame_smaller_than_chunk() -> None:
    # 16kHz, 10ms chunk => 160 samples per chunk; feed 100-sample frames.
    sample_rate, chunk = 16000, 160
    proc = _make_processor(sample_rate, chunk)
    in_stream, out_stream = _feed(proc, sample_rate, [100] * 40)

    assert _interior_zeros(out_stream) == 0, (
        "silence was injected between real audio — this is the cutting in/out artifact"
    )
    # With an identity session and no injected silence, the output is exactly
    # the input in order, minus the tail still buffered in flight.
    assert out_stream.size > 0
    assert np.array_equal(out_stream, in_stream[: out_stream.size])


def test_no_interior_silence_with_variable_frame_sizes() -> None:
    sample_rate, chunk = 16000, 160
    proc = _make_processor(sample_rate, chunk)
    sizes = [137, 53, 200, 80, 160, 45, 300, 10, 90, 160] * 4
    _, out_stream = _feed(proc, sample_rate, sizes)
    assert _interior_zeros(out_stream) == 0


def test_frame_equals_chunk_is_exact_passthrough() -> None:
    # The common case: input frame size matches the chunk size exactly.
    sample_rate, chunk = 16000, 160
    proc = _make_processor(sample_rate, chunk)
    in_stream, out_stream = _feed(proc, sample_rate, [chunk] * 20)
    assert np.array_equal(out_stream, in_stream)
