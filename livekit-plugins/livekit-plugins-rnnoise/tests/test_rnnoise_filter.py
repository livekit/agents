# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np

import livekit.plugins.rnnoise as rn
from livekit import rtc
from livekit.plugins.rnnoise import RNNoise


def _make_frame(samples: np.ndarray, sample_rate: int) -> rtc.AudioFrame:
    samples = samples.astype(np.int16)
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(samples),
    )


def _white_noise(n: int, amplitude: int = 8000, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-amplitude, amplitude, size=n, dtype=np.int16)


def _frame_energy(frame: rtc.AudioFrame) -> float:
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float64)
    return float(np.mean(samples**2))


def test_enabled_property_roundtrip():
    processor = RNNoise()
    assert processor.enabled is True

    processor.enabled = False
    assert processor.enabled is False

    processor.enabled = True
    assert processor.enabled is True


def test_passthrough_when_disabled():
    processor = RNNoise(enabled=False)
    samples = _white_noise(160, seed=1)
    frame = _make_frame(samples, 16000)

    out = processor._process(frame)

    assert out.sample_rate == frame.sample_rate
    assert out.num_channels == frame.num_channels
    assert out.samples_per_channel == frame.samples_per_channel
    assert bytes(out.data) == bytes(frame.data)


def test_same_length_contract_at_48k():
    processor = RNNoise()
    samples = _white_noise(480, seed=2)
    frame = _make_frame(samples, 48000)

    out = processor._process(frame)

    assert out.sample_rate == 48000
    assert out.num_channels == 1
    assert out.samples_per_channel == 480


def test_same_length_contract_at_16k():
    processor = RNNoise()
    samples = _white_noise(160, seed=3)
    frame = _make_frame(samples, 16000)

    out = processor._process(frame)

    # The resampler/denoiser pipeline has warm-up latency, so the very first
    # call may return silence -- only rate + length are guaranteed per call.
    assert out.sample_rate == 16000
    assert out.num_channels == 1
    assert out.samples_per_channel == 160


def test_stream_length_conservation():
    processor = RNNoise()
    total_in = 0
    total_out = 0

    for i in range(50):
        samples = _white_noise(160, seed=100 + i)
        frame = _make_frame(samples, 16000)

        out = processor._process(frame)

        assert out.sample_rate == 16000
        assert out.num_channels == 1
        total_in += frame.samples_per_channel
        total_out += out.samples_per_channel

    assert total_out == total_in


def test_noise_reduction_at_steady_state():
    processor = RNNoise()
    last_in_energy = None
    last_out_frame = None

    # 48kHz frames aligned to RNNoise's 480-sample chunk size need no
    # resampler warm-up, so a handful of frames is enough to reach steady
    # state before we compare energies.
    for i in range(60):
        samples = _white_noise(480, amplitude=8000, seed=200 + i)
        frame = _make_frame(samples, 48000)

        out = processor._process(frame)

        last_in_energy = _frame_energy(frame)
        last_out_frame = out

    assert last_out_frame is not None
    out_energy = _frame_energy(last_out_frame)
    assert out_energy < last_in_energy


def test_noise_reduction_through_resampled_16k_path():
    # Exercises the actual resample(16k->48k) -> denoise -> resample(48k->16k)
    # round-trip, not just the no-resampling 48kHz case above. Measured
    # empirically: the resampler/denoiser warm-up clears well within 80
    # frames (energy becomes nonzero around frame 7-9, steady state by ~70),
    # after which output energy stays several orders of magnitude below the
    # white-noise input -- so a broken/silent pipeline would show energy 0
    # here, and a passthrough-without-denoising pipeline would show energy
    # close to the input.
    processor = RNNoise()
    last_in_energy = None
    last_out_frame = None

    for i in range(80):
        samples = _white_noise(160, amplitude=8000, seed=300 + i)
        frame = _make_frame(samples, 16000)

        out = processor._process(frame)

        last_in_energy = _frame_energy(frame)
        last_out_frame = out

    assert last_out_frame is not None
    out_energy = _frame_energy(last_out_frame)
    assert out_energy > 0
    assert out_energy < last_in_energy


def test_close_resets_state_and_is_idempotent():
    processor = RNNoise()
    samples = _white_noise(480, seed=4)
    frame = _make_frame(samples, 48000)

    processor._process(frame)
    processor._close()
    processor._close()  # must be safe to call twice

    out = processor._process(frame)

    assert out.sample_rate == 48000
    assert out.num_channels == 1
    assert out.samples_per_channel == 480


def test_public_api_and_type():
    processor = rn.RNNoise()

    assert isinstance(processor, rtc.FrameProcessor)
    assert isinstance(processor, rn.RNNoiseFrameProcessor)
