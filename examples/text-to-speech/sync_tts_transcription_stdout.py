import asyncio

from livekit import rtc
from livekit.agents.transcription.tts_forwarder import TTSStdoutForwarder
from livekit.agents.transcription.tts_segments_sync import TTSSegmentsSyncOptions


async def push_text(forwarder: TTSStdoutForwarder, text: str):
    """Simulate text being generated faster than audio."""
    # Push text in chunks to simulate real-time generation
    words = text.split(" ")
    chunk_size = 5  # words per chunk

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        forwarder.push_text(chunk + " ")
        # Text generation is faster than audio
        await asyncio.sleep(0.01)

    forwarder.mark_text_segment_end()


async def push_audio(
    forwarder: TTSStdoutForwarder,
    audio_data: bytes,
    sample_rate: int,
    num_channels: int,
    frame_duration: float = 0.01,  # 10ms frames
    realtime_factor: float = 0.3,  # Push audio faster than real-time
):
    """Push audio frames at simulated real-time rate."""
    frame_size = int(sample_rate * frame_duration)
    bytes_per_sample = 2
    total_bytes = len(audio_data)

    for offset in range(0, total_bytes, frame_size * bytes_per_sample * num_channels):
        frame_data = audio_data[
            offset : offset + frame_size * bytes_per_sample * num_channels
        ]
        if not frame_data:
            break

        frame = rtc.AudioFrame(
            frame_data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=len(frame_data) // (bytes_per_sample * num_channels),
        )
        forwarder.push_audio(frame)
        await asyncio.sleep(frame_duration * realtime_factor)

    forwarder.mark_audio_segment_end()


async def main():
    # Mock audio data
    audio_duration = 14  # seconds
    sample_rate = 24000
    num_channels = 1
    audio_data = b"\x00\x00" * (sample_rate * num_channels * audio_duration)

    # Test transcript
    transcript = (
        "Let's find a time that works for you. When would you like to schedule your appointment? "
        "I'm flexible with my availability and can adjust to fit your needs. Please let me know "
        "your preferred days and times, and I'll do my best to accommodate. Looking forward to "
        "hearing from you soon!"
    )

    # Create forwarder
    opts = TTSSegmentsSyncOptions(
        language="en",
        speed=1.0,
    )
    tts_forwarder = TTSStdoutForwarder(opts, show_timing=True)

    try:
        # Start playout
        tts_forwarder.segment_playout_started()

        # Start text and audio push tasks
        total_duration = len(audio_data) / (sample_rate * 2 * num_channels)
        print(f"Audio duration: {total_duration:.2f} seconds")

        start_time = asyncio.get_running_loop().time()

        # Run text and audio push concurrently
        await asyncio.gather(
            push_text(tts_forwarder, transcript),
            push_audio(tts_forwarder, audio_data, sample_rate, num_channels),
        )

        # Wait for playout to complete
        curr_time = asyncio.get_running_loop().time()
        await asyncio.sleep(total_duration - (curr_time - start_time))

        tts_forwarder.segment_playout_finished()
        duration = asyncio.get_running_loop().time() - start_time

        # Wait for final processing
        await asyncio.sleep(1)
        print(f"Playout finished, duration: {duration:.2f}s")

    finally:
        await tts_forwarder.aclose()


if __name__ == "__main__":
    asyncio.run(main())
