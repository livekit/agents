import asyncio
import logging
import os
import random
import string
import sys

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, llm, stv
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import elevenlabs, openai, silero
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
load_dotenv()

WIDTH = 360
HEIGHT = 360


class IdleStream(stv.IdleStream):
    async def __anext__(self) -> rtc.VideoFrame:
        await asyncio.sleep(1)
        # Create a new random color
        r, g, b = [random.randint(0, 255) for _ in range(3)]
        color = bytes([r, g, b, 255])

        # Fill the frame with the new random color
        argb_frame = bytearray(WIDTH * HEIGHT * 4)
        argb_frame[:] = color * WIDTH * HEIGHT
        return rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)


class SpeechStream(stv.SynthesizeStream):
    def _main_task(self):
        pass


class SimpleSpeechToVideo(stv.STV):
    def idle_stream(self) -> stv.IdleStream:
        return IdleStream()

    def stream(self) -> stv.SynthesizeStream:
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

    # Greets the user with an initial message
    # await assistant.say("Hey there! Look, I have a face now!", allow_interruptions=True)


if __name__ == "__main__":
    # Initialize the worker with the entrypoint
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))