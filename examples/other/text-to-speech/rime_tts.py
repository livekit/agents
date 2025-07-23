import logging
import re
from dotenv import load_dotenv
from typing import List
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import rime


"""
This script demonstrates text-to-speech capabilities using Rime's TTS service with LiveKit.

Required Environment Variables:
    LIVEKIT_URL: The URL of your LiveKit server
    LIVEKIT_API_KEY: Your LiveKit API key
    LIVEKIT_API_SECRET: Your LiveKit API secret
    RIME_API_KEY: Your Rime API key
"""


# Initialize environment and logging
load_dotenv()
logger = logging.getLogger("rime-tts-demo")
logger.setLevel(logging.INFO)


_sentence_pattern = re.compile(r".+?[,，.。!！?？:：]", re.DOTALL)


class TextSegmenter:
    """Utility class for segmenting text into natural chunks for TTS processing."""

    @staticmethod
    def sentence_segmentation(text: str) -> List[str]:
        """
        Segments text into natural sentences.

        Args:
            text (str): Input text to be segmented

        Returns:
            List[str]: List of segmented sentences
        """
        # Clean up text by replacing smart quotes and removing asterisks
        text = text.replace("\u2018", "'").replace("\u2019", "'").replace("*", "")
        result = []
        start_pos = 0

        # Find sentence boundaries using regex pattern
        for match in _sentence_pattern.finditer(text):
            sentence = match.group(0)
            end_pos = match.end()
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
            start_pos = end_pos

        # Handle any remaining text
        if start_pos < len(text):
            sentence = text[start_pos:].strip()
            if sentence:
                result.append(sentence)

        return result


async def stream_text_chunks():
    """Generator that yields properly segmented text chunks for natural TTS output."""

    # Example text demonstrating various sentence structures and punctuation
    text = """
    Welcome to the Rime Text-to-Speech demonstration! This example shows how to properly segment text 
    for natural-sounding speech synthesis. We handle various punctuation marks, including commas, 
    periods, and question marks. Have you noticed how the voice maintains proper intonation? This is 
    achieved through careful text segmentation. The TTS engine processes each segment independently, 
    ensuring optimal timing and natural flow. Would you like to try different voices like Astra, Luna, 
    or Celeste? Each voice has its own unique characteristics!
    """

    # Create segmenter instance and process text
    segmenter = TextSegmenter()
    segments = segmenter.sentence_segmentation(text)

    # Yield each segment
    for sentence in segments:
        logger.debug("Processing segment: %s", sentence)
        yield sentence


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
        logger.info("Connected to LiveKit room successfully And participant joined")
        # Initialize Rime TTS with specific voice and generation parameters
        # For available models: https://docs.rime.ai/api-reference/models
        # For available voices: https://docs.rime.ai/api-reference/voices
        tts = rime.TTS(
            model="arcana",  # The TTS model to use
            speaker="astra",  # Voice ID to use for synthesis
            temperature=0.5,  # Controls speech randomness
            repetition_penalty=1.2,  # Prevents repetitive patterns
            top_p=1.0,  # Controls sound diversity
            max_tokens=5000,  # Maximum tokens for generation
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
        text = """Hi! I'm using Rime's Arcana model with the Astra voice. I can speak naturally with different tones, paces, and emotions. Want to try other voices like Luna, Celeste, or Orion?"""

        # Stream synthesized audio
        logger.info("Starting audio synthesis...")
        async for output in tts.synthesize(text):
            await source.capture_frame(output.frame)

        logger.info("Audio synthesis completed successfully")

        logger.info("Audio Streaming Simulation Example")
        logger.info("Starting streaming text chunks...")
        async for text_chunk in stream_text_chunks():
            logger.info("Processing chunk: %s...", text_chunk[:50])
            # Synthesize each chunk separately
            async for output in tts.synthesize(text_chunk):
                await source.capture_frame(output.frame)
        logger.info("Streaming completed successfully")

    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    # Run the application using LiveKit's CLI utilities
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
