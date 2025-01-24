import asyncio
import wave
from pathlib import Path

from livekit import rtc
from livekit.agents.transcription.tts_segments_sync import (
    TTSSegmentsSync,
    TTSSegmentsSyncOptions,
)


async def main():
    # Load audio file
    audio_path = Path("examples/test_audio.wav")
    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        num_channels = wav.getnchannels()
        audio_data = wav.readframes(wav.getnframes())

    # Test transcript
    transcript = (
        "Let's find a time that works for you. When would you like to schedule your appointment? "
        "I'm flexible with my availability and can adjust to fit your needs. Please let me know "
        "your preferred days and times, and I’ll do my best to accommodate. Looking forward to "
        "hearing from you soon!"
    )

    # Create syncer
    opts = TTSSegmentsSyncOptions(
        language="en",
        speed=1.0,  # Normal speed
    )
    sync = TTSSegmentsSync(opts)

    # Start background task to consume transcription segments
    async def print_segments():
        try:
            async for segment in sync.segment_stream:
                print(f"{'FINAL' if segment.final else 'INTERIM'}: {segment.text}")
        except asyncio.CancelledError:
            return

    segment_task = asyncio.create_task(print_segments())

    try:
        # Push text
        sync.push_text(transcript)
        sync.mark_text_segment_end()

        # Simulate audio frame pushing (assuming 20ms frames)
        frame_duration = 0.01
        frame_size = int(sample_rate * frame_duration)  # 20ms
        bytes_per_sample = 2  # 16-bit audio
        total_bytes = len(audio_data)
        total_duration = total_bytes / (sample_rate * bytes_per_sample * num_channels)
        print(f"Total duration: {total_duration:.2f} seconds")

        # Notify playout started
        sync.segment_playout_started()

        # Push audio frames
        t_start = asyncio.get_running_loop().time()
        for offset in range(
            0, total_bytes, frame_size * bytes_per_sample * num_channels
        ):
            frame_data = audio_data[
                offset : offset + frame_size * bytes_per_sample * num_channels
            ]
            if not frame_data:
                break

            frame = rtc.AudioFrame(
                frame_data,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=len(frame_data)
                // (bytes_per_sample * num_channels),
            )
            sync.push_audio(frame)
            await asyncio.sleep(
                frame_duration * 0.3
            )  # audio push is faster than playout

        sync.mark_audio_segment_end()

        t_end = asyncio.get_running_loop().time()
        print(f"Time taken for audio push: {t_end - t_start:.2f} seconds")

        await asyncio.sleep(total_duration - (t_end - t_start))

        # Notify playout finished
        sync.segment_playout_finished()
        t_end = asyncio.get_running_loop().time()
        print("Playout finished, duration: ", t_end - t_start)

        # Wait a bit to ensure all segments are processed
        await asyncio.sleep(2)
    finally:
        # Cleanup
        segment_task.cancel()
        await sync.aclose()
        try:
            await segment_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
