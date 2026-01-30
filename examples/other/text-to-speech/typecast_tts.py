import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import typecast

load_dotenv()

logger = logging.getLogger("typecast-tts-demo")
logger.setLevel(logging.INFO)


async def entrypoint(job: JobContext) -> None:
    """Run the Typecast TTS demonstration agent.

    This entrypoint demonstrates various Typecast TTS features including
    listing available voices, basic synthesis, emotional expression,
    audio adjustments, and reproducible synthesis with seeds.

    Args:
        job: The LiveKit job context providing room connection and lifecycle.
    """
    logger.info("starting Typecast TTS example agent")

    # Example 0: List available voices (optional)
    logger.info("Listing available Typecast voices...")

    # Create TTS instance with default voice
    # You can also specify a voice ID: typecast.TTS(voice="tc_your_voice_id")
    tts = typecast.TTS(language="eng")  # Uses DEFAULT_VOICE_ID

    try:
        try:
            voices = await tts.list_voices()
            logger.info(f"Found {len(voices)} available voices")

            # Display first 3 voices as examples
            for i, voice in enumerate(voices[:3], 1):
                logger.info(
                    f"  {i}. {voice.name} ({voice.id}) - Emotions: {', '.join(voice.emotions[:3])}..."
                )

            # You can filter by model
            # voices_filtered = await tts.list_voices(model="ssfm-v21")
        except Exception as e:
            logger.warning(f"Could not list voices: {e}")

        # Optionally, select a specific voice from the list
        # For this demo, we'll use the default voice
        logger.info(f"Using default voice: {typecast.DEFAULT_VOICE_ID}")

        source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE

        await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
        publication = await job.room.local_participant.publish_track(track, options)
        await publication.wait_for_subscription()

        # Example 1: Basic synthesis
        logger.info("Example 1: Basic synthesis")
        text1 = "Hello! Welcome to Typecast text-to-speech demonstration."
        async for output in tts.synthesize(text1):
            await source.capture_frame(output.frame)

        await asyncio.sleep(1)

        # Example 2: Happy emotion
        logger.info("Example 2: Synthesizing with happy emotion")
        tts.update_options(
            prompt_options=typecast.PromptOptions(
                emotion_preset="happy",
                emotion_intensity=1.5,
            )
        )
        text2 = "This is great! I'm so excited to demonstrate emotional expression!"
        async for output in tts.synthesize(text2):
            await source.capture_frame(output.frame)

        await asyncio.sleep(1)

        # Example 3: Sad emotion
        logger.info("Example 3: Synthesizing with sad emotion")
        tts.update_options(
            prompt_options=typecast.PromptOptions(
                emotion_preset="sad",
                emotion_intensity=1.2,
            )
        )
        text3 = "Sometimes things don't go as planned, and that's okay."
        async for output in tts.synthesize(text3):
            await source.capture_frame(output.frame)

        await asyncio.sleep(1)

        # Example 4: Back to normal with audio adjustments
        logger.info("Example 4: Normal emotion with audio adjustments")
        tts.update_options(
            prompt_options=typecast.PromptOptions(
                emotion_preset="normal",
                emotion_intensity=1.0,
            ),
            output_options=typecast.OutputOptions(
                volume=110,  # Louder volume
                audio_pitch=1,  # Higher pitch
                audio_tempo=1.1,  # Faster tempo
            ),
        )
        text4 = "Now I'm speaking faster with a higher pitch!"
        async for output in tts.synthesize(text4):
            await source.capture_frame(output.frame)

        await asyncio.sleep(1)

        # Example 5: Reproducible synthesis with seed
        logger.info("Example 5: Using seed for reproducible synthesis")
        tts.update_options(
            seed=42,  # Same seed will produce the same output
            prompt_options=typecast.PromptOptions(
                emotion_preset="normal",
                emotion_intensity=1.0,
            ),
            output_options=typecast.OutputOptions(
                volume=100,
                audio_pitch=0,
                audio_tempo=1.0,
            ),
        )
        text5 = "This synthesis can be reproduced with the same seed value."
        async for output in tts.synthesize(text5):
            await source.capture_frame(output.frame)

        logger.info("Typecast TTS demonstration completed!")
    finally:
        await tts.close()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
