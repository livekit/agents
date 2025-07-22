import logging
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe
from livekit.plugins import rime
from livekit import rtc

# Initialize environment and logging
load_dotenv()
logger = logging.getLogger("rime-tts-demo")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext) -> None:
    """
    Main entrypoint for the TTS demo agent.

    This function:
    1. Initializes connection to LiveKit room
    2. Sets up Rime TTS with specific configuration
    3. Creates and publishes an audio track
    4. Synthesizes and streams sample text

    Args:
        ctx (JobContext): The job context providing access to LiveKit room and other utilities

    Raises:
        Exception: If any error occurs during setup or streaming
    """
    try:
        # Initialize LiveKit connection with no auto-subscription
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
        await ctx.wait_for_participant()
        logger.info(
            "Connected to LiveKit room successfully And participant joined")
        # Initialize Rime TTS with specific voice and generation parameters
        tts = rime.TTS(
            model="arcana",          # The TTS model to use
            # The voice profile you can look at list of voices here https://docs.rime.ai/api-reference/voices
            speaker="astra",
            repetition_penalty=1.5,  # Prevents repetitive speech patterns
            temperature=0.5,         # Controls randomness in generation
            top_p=1.0,              # Nucleus sampling parameter
            max_tokens=5000,         # Maximum tokens for generation

        )

        logger.info("TTS initialized successfully")

        # Set up audio streaming infrastructure
        source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE

        # Connect to LiveKit and publish audio track
        publication = await ctx.room.local_participant.publish_track(track, options)
        await publication.wait_for_subscription()
        logger.info("Audio track published successfully")

        # Sample text to demonstrate TTS capabilities
        text = (
            """Hi! I'm using Rime's Arcana model with the Astra voice. I can speak naturally with different tones, paces, and emotions. Want to try other voices like Luna, Celeste, or Orion?"""
        )

        # Stream synthesized audio
        logger.info("Starting audio synthesis...")
        async for output in tts.synthesize(text):
            await source.capture_frame(output.frame)

        logger.info("Audio synthesis completed")

    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    # Run the application using LiveKit's CLI utilities
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
