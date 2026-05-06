#!/usr/bin/env python3
"""
Minimal Example: Test Krisp Audio Filter Directly

This is a minimal example to test that Krisp filtering works without a full
agent setup — useful for debugging the C extension and audio plumbing.

Because this script runs outside an agent context, it cannot use the default
``LiveKitCloudAuthProvider`` (that path needs the framework to push the room's
JWT into the FrameProcessor at runtime). Instead, it uses
``KrispLicenseAuthProvider`` with a Krisp license key + ``.kef`` model file.

Prerequisites:
    1. Set ``KRISP_VIVA_SDK_LICENSE_KEY`` and ``KRISP_VIVA_FILTER_MODEL_PATH``
       env vars (or pass them explicitly to ``KrispLicenseAuthProvider``).
    2. Install: ``pip install livekit-plugins-krisp krisp-audio numpy``

Usage:
    python krisp_minimal_example.py
"""

import asyncio
import logging

import numpy as np

from livekit import rtc
from livekit.plugins.krisp import KrispLicenseAuthProvider, KrispVivaFilterFrameProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("krisp-minimal-test")


async def test_krisp_filter():
    """Test Krisp filter with synthetic audio."""

    logger.info("Testing Krisp Audio Filter")

    # Create the frame processor with explicit license auth (no framework
    # available in this standalone script).
    krisp_processor = KrispVivaFilterFrameProcessor(
        auth_provider=KrispLicenseAuthProvider(),
        noise_suppression_level=100,
        frame_duration_ms=10,
        sample_rate=16000,
    )
    logger.info("Krisp frame processor created")

    # Create a test audio frame (10ms @ 16kHz = 160 samples)
    sample_rate = 16000
    frame_duration_ms = 10
    num_samples = int(sample_rate * frame_duration_ms / 1000)

    # Generate synthetic audio (white noise + sine wave)
    t = np.linspace(0, frame_duration_ms / 1000, num_samples)
    sine_wave = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    noise = np.random.normal(0, 0.1, num_samples)
    audio_signal = (sine_wave + noise) * 0.5
    audio_int16 = (audio_signal * 32767).astype(np.int16)

    test_frame = rtc.AudioFrame(
        data=audio_int16.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )
    logger.info(f"Test frame created: {num_samples} samples @ {sample_rate}Hz")

    # Process the frame through Krisp
    filtered_frame = krisp_processor.process(test_frame)
    logger.info("Frame processed")
    logger.info(f"   Input samples:  {test_frame.samples_per_channel}")
    logger.info(f"   Output samples: {filtered_frame.samples_per_channel}")
    assert filtered_frame.samples_per_channel == test_frame.samples_per_channel

    # Process multiple frames
    for _ in range(10):
        krisp_processor.process(test_frame)
    logger.info("Processed 10 additional frames")

    krisp_processor.close()
    logger.info("Frame processor closed — all checks passed")


async def main():
    await test_krisp_filter()


if __name__ == "__main__":
    asyncio.run(main())
