"""The vectorized spectral flux must match the frame-by-frame STFT it replaced."""

from __future__ import annotations

import numpy as np
import pytest

from livekit.agents.voice.transcription._speaking_rate import SpeakingRateDetector

pytestmark = pytest.mark.unit


def _reference_spectral_flux(audio: np.ndarray, sample_rate: int) -> float:
    # the previous implementation: one rfft per frame, then a loop over frame pairs
    frame_length = int(sample_rate * 0.025)
    hop_length = frame_length // 2
    num_frames = (len(audio) - frame_length) // hop_length + 1
    if num_frames < 2:
        return 0.0

    window = np.hanning(frame_length)
    scale_factor = 1.0 / np.sqrt(np.sum(window**2))
    zxx = np.zeros((frame_length // 2 + 1, num_frames), dtype=np.complex128)
    for i in range(num_frames):
        start = i * hop_length
        zxx[:, i] = np.fft.rfft(audio[start : start + frame_length] * window) * scale_factor

    magnitudes = np.abs(zxx)
    flux = [
        np.sum(np.abs(magnitudes[:, i] - magnitudes[:, i - 1]))
        for i in range(1, magnitudes.shape[1])
    ]
    return float(np.mean(flux))


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 24000, 44100, 48000])
@pytest.mark.parametrize("duration", [0.0, 0.03, 0.5, 1.0])
async def test_spectral_flux_matches_frame_by_frame(sample_rate: int, duration: float) -> None:
    stream = SpeakingRateDetector().stream()
    rng = np.random.default_rng(sample_rate + int(duration * 1000))
    audio = (rng.standard_normal(int(sample_rate * duration)) * 0.1).astype(np.float32)

    try:
        result = stream._spectral_flux(audio, sample_rate)
    finally:
        await stream.aclose()
    expected = _reference_spectral_flux(audio, sample_rate)

    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)
