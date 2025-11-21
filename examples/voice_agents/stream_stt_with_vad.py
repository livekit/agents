import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    stt,
)
from livekit.plugins import deepgram, silero

logger = logging.getLogger("stream-stt-with-vad")

# This example shows how to use a streaming STT with a VAD.
# Only the audio frames which are detected as speech by the VAD will be sent to the STT.
# This requires the STT to support streaming and flush, e.g. deepgram, cartesia, etc.,
# check the `STT.capabilities` for more details.

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt.StreamAdapter(
            stt=deepgram.STT(),
            vad=ctx.proc.userdata["vad"],
            use_streaming=True,  # use streaming mode of the wrapped STT with VAD
        ),
        llm="openai/gpt-4.1-mini",
        tts="elevenlabs",
    )
    await session.start(
        agent=Agent(instructions="You are a somewhat helpful assistant."), room=ctx.room
    )

    await session.say("Hello, how can I help you?")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


if __name__ == "__main__":
    cli.run_app(server)
