"""
ElevenLabs TTS with Character-Level Alignment for Lip Sync

This example demonstrates how to use ElevenLabs TTS with character-level
alignment data for accurate lip sync animation in avatar applications.

The alignment data is published to LiveKit clients via data channels on
the topic "tts.alignment" with JSON payload containing character timings.
"""

import asyncio
import json
import logging

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.plugins import elevenlabs

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("elevenlabs-lipsync")
logger.setLevel(logging.INFO)

server = AgentServer()


@server.rtc_session()
async def entrypoint(job: JobContext):
    """
    Main entrypoint for the lip sync demo agent.

    This agent:
    1. Creates an ElevenLabs TTS instance with alignment forwarding enabled
    2. Listens for alignment events from the TTS engine
    3. Publishes alignment data to clients via LiveKit data channels
    4. Synthesizes sample text with timing information
    """
    logger.info("Starting ElevenLabs lip sync example agent")

    # Initialize ElevenLabs TTS with alignment forwarding enabled
    tts = elevenlabs.TTS(
        model="eleven_turbo_v2_5",
        enable_alignment_forwarding=True,  # Enable character-level alignment
        sync_alignment=True,  # Required for alignment data
    )

    # Set up audio track for TTS output
    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    logger.info("Audio track published and subscribed")

    # Event handler for alignment data
    def on_alignment_received(event_data):
        """
        Handle alignment events from ElevenLabs TTS.

        This callback receives character-level timing data and publishes it
        to all clients in the room via the data channel.
        """
        alignment = event_data["alignment"]
        segment_id = event_data["segment_id"]

        logger.info(
            f"Received alignment for segment {segment_id}: {len(alignment.characters)} characters"
        )

        # Prepare alignment payload for clients
        alignment_payload = {
            "type": "elevenlabs_alignment",
            "segment_id": segment_id,
            "characters": alignment.characters,
            "start_times_seconds": alignment.start_times_seconds,
            "end_times_seconds": alignment.end_times_seconds,
            "char_count": len(alignment.characters),
        }

        # Log first few characters for debugging
        if len(alignment.characters) > 0:
            sample_size = min(5, len(alignment.characters))
            logger.debug(f"First {sample_size} characters: {alignment.characters[:sample_size]}")
            logger.debug(f"Start times: {alignment.start_times_seconds[:sample_size]}")

        # Publish alignment data to all clients via data channel
        asyncio.create_task(
            job.room.local_participant.publish_data(
                payload=json.dumps(alignment_payload).encode("utf-8"),
                topic="tts.alignment",
                destination_identities=None,  # Broadcast to all participants
            )
        )
        logger.info("Published alignment data to topic 'tts.alignment'")

    # Register alignment event handler
    tts.on("alignment_received", on_alignment_received)
    logger.info("Registered alignment event handler")

    # Create TTS stream
    stream = tts.stream()

    # Playback task: forward synthesized audio to the audio source
    async def _playback_task():
        async for audio in stream:
            await source.capture_frame(audio.frame)

    playback_task = asyncio.create_task(_playback_task())

    # Sample text for demonstration
    text = (
        "Hello! This is a demonstration of character-level alignment for lip sync. "
        "Each character is precisely timed for accurate visual synchronization."
    )

    logger.info(f"Synthesizing text: {text}")

    # Push text in chunks to simulate streaming (like from an LLM)
    words = text.split()
    for i in range(0, len(words), 3):  # 3 words at a time
        chunk = " ".join(words[i : i + 3])
        if chunk:
            logger.info(f'Pushing text chunk: "{chunk}"')
            stream.push_text(chunk + " ")
            await asyncio.sleep(0.1)  # Small delay between chunks

    # Mark end of input
    stream.flush()
    stream.end_input()

    logger.info("Waiting for synthesis to complete...")
    await playback_task
    logger.info("Synthesis complete")


if __name__ == "__main__":
    cli.run_app(server)
