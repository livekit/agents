import asyncio
import logging
import random

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, llm, stv, utils
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import elevenlabs, openai, silero
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
load_dotenv()

logger = logging.getLogger("livekit.examples.speech-to-video")

WIDTH = 360
HEIGHT = 360


class IdleStream(stv.IdleStream):
    async def __anext__(self) -> rtc.VideoFrame:
        await asyncio.sleep(0.5)
        # Create a new random color
        r, g, b = [random.randint(0, 255) for _ in range(3)]
        color = bytes([r, g, b, 255])

        # Fill the frame with the new random color
        argb_frame = bytearray(WIDTH * HEIGHT * 4)
        argb_frame[:] = color * WIDTH * HEIGHT
        return rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)


class SpeechStream(stv.SpeechStream):
    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        font = ImageFont.load_default().font_variant(size=120)
        start_time = 0
        async for audio_chunk in self._input_ch:
            if isinstance(audio_chunk, self._FlushSentinel):
                return

            if not audio_chunk.alignment:
                continue

            aligned_chars = zip(
                audio_chunk.alignment.chars, 
                audio_chunk.alignment.start_times, 
                audio_chunk.alignment.durations
            )
            for char, _, duration in aligned_chars:
                try:
                    # Create a white image
                    image = Image.new('RGBA', (WIDTH, HEIGHT), color='white')
                    draw = ImageDraw.Draw(image)
                
                    # Draw the character
                    bbox = draw.textbbox((0, 0), char, font=font)
                    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    position = ((WIDTH - text_width) / 2, (HEIGHT - text_height) / 2)
                    draw.text(position, char, font=font, fill='black')
                except Exception as e:
                    logger.error(f"Error drawing text: {e}")
                
                # Create and send video frame event
                start_time += duration
                self._event_ch.send_nowait(
                    rtc.VideoFrameEvent(
                        frame=rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, image.tobytes()),
                        timestamp_us=start_time,
                        rotation=rtc.VideoRotation.VIDEO_ROTATION_0,
                    )
                )


class SimpleSpeechToVideo(stv.STV):
    def idle_stream(self) -> stv.IdleStream:
        return IdleStream()

    def speech_stream(self) -> stv.SpeechStream:
        return SpeechStream()


# This function is the entrypoint for the agent.
async def entrypoint(ctx: JobContext):
    # Connect to the LiveKit room
    await ctx.connect()

    # VoiceAssistant is a class that creates a full conversational AI agent.
    # See https://github.com/livekit/agents/tree/main/livekit-agents/livekit/agents/voice_assistant
    # for details on how it works.
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=elevenlabs.TTS(),
        stv=SimpleSpeechToVideo(width=WIDTH, height=HEIGHT),
        chat_ctx=llm.ChatContext(),
    )

    # Start the voice assistant with the LiveKit room
    assistant.start(ctx.room)
    await asyncio.sleep(8)

    # Greets the user with an initial message
    async def greeting_generator():
        yield "Hey there! "
        yield "Look, "
        yield "I can draw "
        yield "letters!"

    await assistant.say(greeting_generator(), allow_interruptions=True)


if __name__ == "__main__":
    # Initialize the worker with the entrypoint
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))