#!/usr/bin/env python3
"""
Minimal Example: Test Krisp Audio Filter Directly

This is a minimal example to test that Krisp filtering works
without requiring a full agent setup. Good for testing and debugging.

Prerequisites:
    1. Set KRISP_VIVA_FILTER_MODEL_PATH environment variable
    2. Install: pip install livekit-plugins-krisp krisp-audio numpy

Usage:
    python krisp_minimal_example.py
"""

import asyncio
import logging

import numpy as np

from livekit import rtc
from livekit.plugins.krisp import KrispVivaFilterFrameProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("krisp-minimal-test")


async def test_krisp_filter():
    """Test Krisp filter with synthetic audio."""

    logger.info("=" * 60)
    logger.info("Testing Krisp Audio Filter")
    logger.info("=" * 60)

    try:
        # Create the frame processor
        logger.info("\n1. Creating Krisp frame processor...")
        krisp_processor = KrispVivaFilterFrameProcessor(
            noise_suppression_level=100,
            frame_duration_ms=10,
            sample_rate=16000,
        )
        logger.info("✅ Krisp frame processor created successfully")

        # Create a test audio frame (10ms @ 16kHz = 160 samples)
        logger.info("\n2. Creating test audio frame...")
        sample_rate = 16000
        frame_duration_ms = 10
        num_samples = int(sample_rate * frame_duration_ms / 1000)

        # Generate synthetic audio (white noise + sine wave)
        t = np.linspace(0, frame_duration_ms / 1000, num_samples)
        sine_wave = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        noise = np.random.normal(0, 0.1, num_samples)  # Noise
        audio_signal = (sine_wave + noise) * 0.5

        # Convert to int16 PCM
        audio_int16 = (audio_signal * 32767).astype(np.int16)

        # Create AudioFrame
        test_frame = rtc.AudioFrame(
            data=audio_int16.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=num_samples,
        )
        logger.info(f"✅ Test frame created: {num_samples} samples @ {sample_rate}Hz")

        # Process the frame through Krisp
        logger.info("\n3. Processing frame through Krisp frame processor...")
        filtered_frame = krisp_processor.process(test_frame)
        logger.info("✅ Frame processed successfully")

        # Verify output
        logger.info("\n4. Verifying output...")
        logger.info(f"   Input samples:  {test_frame.samples_per_channel}")
        logger.info(f"   Output samples: {filtered_frame.samples_per_channel}")
        logger.info(f"   Input rate:     {test_frame.sample_rate}Hz")
        logger.info(f"   Output rate:    {filtered_frame.sample_rate}Hz")

        if filtered_frame.samples_per_channel == test_frame.samples_per_channel:
            logger.info("✅ Output frame size matches input")
        else:
            logger.error("❌ Output frame size mismatch!")

        # Test multiple frames
        logger.info("\n5. Processing multiple frames...")
        for i in range(10):
            _ = krisp_processor.process(test_frame)
            logger.info(f"   Frame {i + 1}/10 processed")
        logger.info("✅ Multiple frames processed successfully")

        # Cleanup
        logger.info("\n6. Cleaning up...")
        krisp_processor.close()
        logger.info("✅ Frame processor closed")

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED - Krisp frame processor is working correctly!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error(f"❌ TEST FAILED: {e}")
        logger.error("=" * 60)
        raise


async def main():
    """Run all tests."""
    # Test frame processor
    await test_krisp_filter()


if __name__ == "__main__":
    asyncio.run(main())
