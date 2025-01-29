import asyncio
import logging
import sys

from livekit import rtc
from livekit.agents.pipeline.io import AudioSink, TextSink
from livekit.agents.transcription import (
    TranscriptionSyncIO,
    TranscriptionSyncOptions,
    TTSStreamForwarder,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


class MockAudioSink(AudioSink):
    """Mock audio sink that drops frames."""

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)  # Emit event and update state

    def flush(self) -> None:
        super().flush()  # Emit event and update state

    def clear_buffer(self) -> None:
        super().clear_buffer()  # Emit event and update state


async def push_text(text_sink: TextSink, text: str):
    """Simulate text being generated faster than audio."""
    # Push text in chunks to simulate real-time generation
    words = text.split(" ")
    chunk_size = 5  # words per chunk

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        await text_sink.capture_text(chunk + " ")
        # Text generation is faster than audio
        await asyncio.sleep(0.01)
    text_sink.flush()


async def push_audio(
    audio_sink: AudioSink,
    audio_data: bytes,
    sample_rate: int,
    num_channels: int,
    frame_duration: float = 1,  # 10ms frames
    realtime_factor: float = 0.3,  # Push audio faster than real-time
):
    """Push audio frames at simulated real-time rate."""
    frame_size = int(sample_rate * frame_duration)
    bytes_per_sample = 2
    total_bytes = len(audio_data)
    audio_duration = total_bytes / (sample_rate * 2 * num_channels)
    print(f"\n[START] Audio duration: {audio_duration:.2f} seconds")

    start_time = asyncio.get_running_loop().time()
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
        await audio_sink.capture_frame(frame)
        await asyncio.sleep(frame_duration * realtime_factor)

    audio_sink.flush()

    # Wait for playout to complete
    wait_time = audio_duration - (asyncio.get_running_loop().time() - start_time)
    await asyncio.sleep(wait_time)
    audio_sink.on_playback_finished(playback_position=audio_duration, interrupted=False)

    await asyncio.sleep(0.5)
    print("[END] Playout finished")


async def main():
    sample_rate = 24000
    num_channels = 1

    # Test transcript (text, audio duration)
    transcripts = [
        (
            "Hi there! I was just calling to check in - how have you been doing lately?",
            4,
        ),
        (
            "Let's find a time that works for you. When would you like to schedule your appointment? "
            "I'm flexible with my availability and can adjust to fit your needs. Please let me know "
            "your preferred days and times, and I'll do my best to accommodate. Looking forward to "
            "hearing from you soon!",
            14,
        ),
    ]

    # Create sinks and forwarder
    audio_sink = MockAudioSink(sample_rate=sample_rate)

    opts = TranscriptionSyncOptions(language="en", speed=1.0)
    transcript_sync = TranscriptionSyncIO(audio_sink, sync_options=opts)
    tts_forwarder = TTSStreamForwarder(stream=sys.stdout)
    transcript_sync.on("transcription_segment", tts_forwarder)

    # Run forwarder and push data concurrently
    for transcript, audio_duration in transcripts:
        audio_data = b"\x00\x00" * (sample_rate * num_channels * audio_duration)
        await asyncio.gather(
            push_text(transcript_sync.text_output, transcript),
            push_audio(
                transcript_sync.audio_output, audio_data, sample_rate, num_channels
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
