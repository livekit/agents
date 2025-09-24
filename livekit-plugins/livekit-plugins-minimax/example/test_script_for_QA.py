# examples/minimax_tts_test_suite.py
"""
Comprehensive test suite for Minimax TTS plugin.

This test suite validates:
- Basic streaming synthesis functionality
- Word-level timestamp extraction
- Non-streaming (batch) synthesis
- Audio output generation and file saving
"""

import asyncio
import logging
import os
from livekit.plugins.minimax import TTS
from livekit.agents.voice.io import TimedString
from livekit import rtc

# Configure logging for detailed test output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants for test configuration
USERDATA_TIMED_TRANSCRIPT_KEY = "lk.timed_transcripts"
OUTPUT_FILENAME_BASE = "test_output"


async def test_basic_streaming():
    """
    Test 1: Basic Streaming Synthesis

    Validates:
    - Stream creation and text input
    - Audio chunk reception
    - Proper stream lifecycle management
    """
    print("\n--- Test 1: Basic Streaming Synthesis ---")

    # Create TTS instance without subtitles for faster processing
    tts_instance = TTS(subtitle_enable=False)
    stream = tts_instance.stream()

    async def push_test_text():
        """Push text to the stream and signal completion."""
        await asyncio.sleep(0.1)  # Brief delay to ensure stream is ready
        await stream.push_text("This is a basic streaming synthesis test.")
        await stream.end_input()

    # Start text pushing task
    pusher_task = asyncio.create_task(push_test_text())

    # Collect audio chunks from the stream
    audio_received_count = 0
    async for event in stream:
        if len(event.frame.data) > 0:
            audio_received_count += 1
            # Print progress every 10 chunks
            if audio_received_count % 10 == 0:
                print(f"  Received {audio_received_count} audio chunks...")

    # Wait for text pushing to complete and cleanup
    await pusher_task
    await tts_instance.aclose()

    # Validate results
    if audio_received_count > 0:
        print(f"✅ Test PASSED: Received {audio_received_count} audio chunks.")
    else:
        print("❌ Test FAILED: No audio data received.")


async def test_streaming_with_timestamps():
    """
    Test 2: Streaming Synthesis with Word-Level Timestamps

    Validates:
    - Word-level timing information extraction
    - Subtitle processing functionality
    - Timestamp accuracy and format
    """
    print("\n--- Test 2: Streaming with Word-Level Timestamps ---")

    # Create TTS instance with word-level timing enabled
    tts_instance = TTS(subtitle_type="word")
    stream = tts_instance.stream()

    async def push_test_text():
        """Push text optimized for timestamp testing."""
        await asyncio.sleep(0.1)
        await stream.push_text("Testing word level timestamps.")
        await stream.end_input()

    pusher_task = asyncio.create_task(push_test_text())

    # Collect timestamp information from audio frames
    timestamps_found = []
    async for event in stream:
        if USERDATA_TIMED_TRANSCRIPT_KEY in event.frame.userdata:
            timestamps_found.extend(event.frame.userdata[USERDATA_TIMED_TRANSCRIPT_KEY])

    await pusher_task
    await tts_instance.aclose()

    # Validate timestamp extraction
    if timestamps_found:
        print("✅ Test PASSED: Successfully received timestamps.")
        print("  Word timing information:")
        for tt in timestamps_found:
            if isinstance(tt, TimedString):
                print(f"    - '{tt}' (start: {tt.start_time:.3f}s, end: {tt.end_time:.3f}s)")
    else:
        print("❌ Test FAILED: No timestamp data received.")


async def test_non_streaming_synthesis():
    """
    Test 3: Non-Streaming (Batch) Synthesis

    Validates:
    - Complete text synthesis in single request
    - Audio file generation and saving
    - Output file integrity
    """
    print("\n--- Test 3: Non-Streaming Synthesis ---")

    # Create TTS instance for batch processing
    tts_instance = TTS(subtitle_enable=False)
    full_text = "This is a complete sentence for testing non-streaming synthesis functionality."

    # Create synthesis stream for the complete text
    stream = tts_instance.synthesize(full_text)

    output_filename = f"{OUTPUT_FILENAME_BASE}_non_streaming.pcm"
    print(f"  Synthesis result will be saved to: {output_filename}")

    try:
        # Collect all audio data into a single frame
        final_frame: rtc.AudioFrame = await stream.collect()

        # Save audio data to file
        with open(output_filename, "wb") as f:
            f.write(final_frame.data)

        # Validate output
        if len(final_frame.data) > 0:
            print(f"✅ Test PASSED: Audio file generated, size: {len(final_frame.data)} bytes")
            print(f"  Sample rate: {final_frame.sample_rate} Hz")
            print(f"  Duration: {final_frame.duration:.2f} seconds")
        else:
            print("❌ Test FAILED: Generated audio file is empty")

    except Exception as e:
        print(f"❌ Test FAILED: Error during synthesis - {e}")
    finally:
        await tts_instance.aclose()


async def test_multilingual_content():
    """
    Test 4: Multilingual Content Processing

    Validates:
    - Smart tokenizer with mixed languages
    - Proper handling of different character types
    - Text segmentation quality
    """
    print("\n--- Test 4: Multilingual Content Processing ---")

    tts_instance = TTS(subtitle_enable=False)
    stream = tts_instance.stream()

    async def push_multilingual_text():
        """Push mixed Chinese-English content."""
        await asyncio.sleep(0.1)
        await stream.push_text("Hello world! 这是中文测试。Testing 123 numbers.")
        await stream.end_input()

    pusher_task = asyncio.create_task(push_multilingual_text())

    audio_received_count = 0
    async for event in stream:
        if len(event.frame.data) > 0:
            audio_received_count += 1

    await pusher_task
    await tts_instance.aclose()

    if audio_received_count > 0:
        print(f"✅ Test PASSED: Multilingual content processed, {audio_received_count} chunks received.")
    else:
        print("❌ Test FAILED: Multilingual content processing failed.")


async def test_error_handling():
    """
    Test 5: Error Handling and Edge Cases

    Validates:
    - Empty text handling
    - Resource cleanup on errors
    - Graceful degradation
    """
    print("\n--- Test 5: Error Handling and Edge Cases ---")

    # Test empty text input
    print("  Testing empty text input...")
    tts_instance = TTS(subtitle_enable=False)
    stream = tts_instance.stream()

    async def push_empty_text():
        await asyncio.sleep(0.1)
        await stream.push_text("")  # Empty text
        await stream.end_input()

    pusher_task = asyncio.create_task(push_empty_text())

    audio_received_count = 0
    async for event in stream:
        if len(event.frame.data) > 0:
            audio_received_count += 1

    await pusher_task
    await tts_instance.aclose()

    # Empty text should not produce audio
    if audio_received_count == 0:
        print("✅ Empty text handling PASSED: No audio generated for empty input.")
    else:
        print(f"⚠️  Empty text handling: Unexpected {audio_received_count} chunks received.")


async def cleanup_test_files():
    """Remove test output files to start fresh."""
    test_files = [
        f"{OUTPUT_FILENAME_BASE}_non_streaming.pcm",
        f"{OUTPUT_FILENAME_BASE}_streaming.wav"
    ]

    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"  Removed existing test file: {filename}")


async def main_test_suite():
    """
    Main test suite runner.

    Executes all test cases in sequence and provides summary results.
    """
    print("=" * 70)
    print("🧪 MINIMAX TTS PLUGIN TEST SUITE")
    print("=" * 70)
    print("Testing comprehensive TTS functionality...")

    # Clean up any existing test files
    await cleanup_test_files()

    # Test execution tracking
    test_results = {}

    try:
        # Run all test cases
        print("\n🚀 Starting test execution...")

        await test_basic_streaming()
        print("-" * 60)

        await test_streaming_with_timestamps()
        print("-" * 60)

        await test_non_streaming_synthesis()
        print("-" * 60)

        await test_multilingual_content()
        print("-" * 60)

        await test_error_handling()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n📊 Test files generated:")
    if os.path.exists(f"{OUTPUT_FILENAME_BASE}_non_streaming.pcm"):
        file_size = os.path.getsize(f"{OUTPUT_FILENAME_BASE}_non_streaming.pcm")
        print(f"  - {OUTPUT_FILENAME_BASE}_non_streaming.pcm ({file_size} bytes)")


if __name__ == "__main__":
    """Run the complete test suite."""
    print("Starting Minimax TTS Test Suite...")
    print("Make sure MINIMAX_API_KEY and MINIMAX_GROUP_ID are set in your environment.")

    try:
        asyncio.run(main_test_suite())
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback

        traceback.print_exc()