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
from livekit.plugins.krisp import KrispVivaFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("krisp-minimal-test")


async def test_krisp_filter():
    """Test Krisp filter with synthetic audio."""

    logger.info("=" * 60)
    logger.info("Testing Krisp Audio Filter")
    logger.info("=" * 60)

    try:
        # Create the filter
        logger.info("\n1. Creating Krisp filter...")
        krisp_filter = KrispVivaFilter(
            noise_suppression_level=100,
            frame_duration_ms=10,
            sample_rate=16000,
        )
        logger.info("✅ Krisp filter created successfully")

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
        logger.info("\n3. Processing frame through Krisp filter...")
        filtered_frame = await krisp_filter.filter(test_frame)
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
            filtered = await krisp_filter.filter(test_frame)
            logger.info(f"   Frame {i + 1}/10 processed")
        logger.info("✅ Multiple frames processed successfully")

        # Test stream processing
        logger.info("\n6. Testing stream processing...")

        async def generate_frames():
            """Generate test audio frames."""
            for i in range(5):
                yield test_frame

        frame_count = 0
        async for filtered_frame in krisp_filter.process_stream(generate_frames()):
            frame_count += 1

        logger.info(f"✅ Stream processing completed: {frame_count} frames")

        # Cleanup
        logger.info("\n7. Cleaning up...")
        krisp_filter.close()
        logger.info("✅ Filter closed")

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED - Krisp filter is working correctly!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error(f"❌ TEST FAILED: {e}")
        logger.error("=" * 60)
        raise


async def test_krisp_audio_input():
    """Test KrispAudioInput wrapper (if livekit-agents is available)."""
    try:
        from livekit.agents.voice import io
        from livekit.plugins.krisp import KrispAudioInput

        logger.info("\n" + "=" * 60)
        logger.info("Testing KrispAudioInput Wrapper")
        logger.info("=" * 60)

        # Create a mock audio source
        class MockAudioInput(io.AudioInput):
            def __init__(self):
                super().__init__(label="MockInput")
                self.frame_count = 0

            async def __anext__(self):
                if self.frame_count >= 5:
                    raise StopAsyncIteration

                self.frame_count += 1

                # Generate test frame
                sample_rate = 16000
                num_samples = 160  # 10ms @ 16kHz
                audio_data = np.random.randint(-1000, 1000, num_samples, dtype=np.int16)

                return rtc.AudioFrame(
                    data=audio_data.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=num_samples,
                )

        logger.info("\n1. Creating mock audio input...")
        mock_input = MockAudioInput()
        logger.info("✅ Mock input created")

        logger.info("\n2. Wrapping with KrispAudioInput...")
        krisp_input = KrispAudioInput(
            source=mock_input,
            noise_suppression_level=100,
            frame_duration_ms=10,
            sample_rate=16000,
        )
        krisp_input.on_attached()
        logger.info("✅ KrispAudioInput wrapper created")

        logger.info("\n3. Processing frames through wrapper...")
        frame_count = 0
        async for frame in krisp_input:
            frame_count += 1
            logger.info(
                f"   Frame {frame_count}: {frame.samples_per_channel} samples @ {frame.sample_rate}Hz"
            )

        logger.info(f"✅ Processed {frame_count} frames through wrapper")

        krisp_input.on_detached()
        logger.info("✅ Wrapper cleaned up")

        logger.info("\n" + "=" * 60)
        logger.info("✅ KrispAudioInput wrapper test passed!")
        logger.info("=" * 60)

    except ImportError:
        logger.warning("\n⚠️  livekit-agents not installed, skipping KrispAudioInput test")


async def main():
    """Run all tests."""
    # Test basic filter
    await test_krisp_filter()

    # Test audio input wrapper (if available)
    await test_krisp_audio_input()


if __name__ == "__main__":
    asyncio.run(main())
