"""noise_cancellation_with_volume.py

Demonstrates post-processing audio with ``VolumeAmplifierProcessor`` after
Krisp BVCTelephony noise cancellation.

Audio pipeline for each incoming frame:

  1. Noise cancellation (FFI ``NoiseCancellationOptions`` or Python ``FrameProcessor``)
  2. VolumeAmplifierProcessor (``audio_post_processor``) for final level tuning
  3. STT / VAD

Requirements:
  pip install livekit-agents livekit-plugins-noise-cancellation
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
)
from livekit.agents.utils.audio_processors import VolumeAmplifierProcessor
from livekit.agents.voice import room_io
from livekit.plugins import noise_cancellation

logger = logging.getLogger("nc-volume-agent")
logger.setLevel(logging.INFO)

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant.",
            stt=inference.STT("deepgram/nova-3"),
            llm=inference.LLM("openai/gpt-4o-mini"),
            tts=inference.TTS("cartesia/sonic-3"),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Native FFI-level NC: Krisp BVCTelephony removes background noise.
                noise_cancellation=noise_cancellation.BVCTelephony(),
                # Runs last – final level adjustment after all other processing.
                audio_post_processor=VolumeAmplifierProcessor(gain_db=3.0),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
