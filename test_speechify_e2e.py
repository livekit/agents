#!/usr/bin/env python3
"""
E2E test for Speechify TTS plugin with streaming endpoint migration.

Tests that the new /v1/audio/stream/with-timestamps implementation works correctly.
"""

import asyncio
import os
import sys

# Add the plugin to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "livekit-plugins/livekit-plugins-speechify",
    ),
)

from livekit.plugins.speechify import TTS


async def test_synthesize():
    """Test the synthesize() method."""
    print("=== Testing synthesize() ===")

    tts = TTS(voice_id="dominic_32", model="simba-3.2")

    test_text = "Hello world! This is a test of the Speechify TTS plugin."
    print(f"Input: {test_text}")

    audio_frames = []
    transcript_segments = []

    stream = tts.synthesize(test_text)

    async for event in stream:
        audio_frames.append(event.frame)
        if event.timed_transcripts:
            transcript_segments.extend(event.timed_transcripts)
            for seg in event.timed_transcripts:
                print(
                    f"  Transcript: '{seg.text}' @ {seg.start_time:.3f}s - {seg.end_time:.3f}s"
                )

    total_audio_bytes = sum(len(frame.data) for frame in audio_frames)
    print(f"✓ Total audio: {total_audio_bytes} bytes")
    print(f"✓ Transcript segments: {len(transcript_segments)}")

    assert total_audio_bytes > 0, "No audio generated"
    assert len(transcript_segments) > 0, "No transcript segments"

    await tts.aclose()
    print("✓ synthesize() test passed\n")


async def test_stream():
    """Test the stream() method."""
    print("=== Testing stream() ===")

    tts = TTS(voice_id="dominic_32", model="simba-3.2")

    sentences = [
        "First sentence for streaming test.",
        "Second sentence for streaming test.",
    ]

    audio_frames = []
    transcript_segments = []

    stream = tts.stream()

    async def push_input():
        for sentence in sentences:
            print(f"Input: {sentence}")
            stream.push_text(sentence)
            await asyncio.sleep(0.1)
        stream.end_input()

    async def consume_output():
        async for event in stream:
            audio_frames.append(event.frame)
            if event.timed_transcripts:
                transcript_segments.extend(event.timed_transcripts)
                for seg in event.timed_transcripts:
                    print(
                        f"  Transcript: '{seg.text}' @ {seg.start_time:.3f}s - {seg.end_time:.3f}s"
                    )

    await asyncio.gather(push_input(), consume_output())

    total_audio_bytes = sum(len(frame.data) for frame in audio_frames)
    print(f"✓ Total audio: {total_audio_bytes} bytes")
    print(f"✓ Transcript segments: {len(transcript_segments)}")

    assert total_audio_bytes > 0, "No audio generated"
    assert len(transcript_segments) > 0, "No transcript segments"

    await tts.aclose()
    print("✓ stream() test passed\n")


async def main():
    if not os.environ.get("SPEECHIFY_API_KEY"):
        print("ERROR: SPEECHIFY_API_KEY environment variable not set")
        sys.exit(1)

    print("Starting Speechify TTS e2e tests...\n")

    try:
        await test_synthesize()
        await test_stream()
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
