import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli, stt
from livekit.plugins import elevenlabs, silero

logger = logging.getLogger("realtime-scribe-v2")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt.StreamAdapter(  # use local VAD with the STT
            stt=elevenlabs.STT(
                use_realtime=True,
                server_vad=None,  # disable server-side VAD
                language_code="en",
            ),
            vad=ctx.proc.userdata["vad"],
            use_streaming=True,
        ),
        llm="openai/gpt-4.1-mini",
        tts="elevenlabs",
    )
    await session.start(
        agent=Agent(instructions="You are a somewhat helpful assistant."), room=ctx.room
    )

    await session.say("Hello, how can I help you?", allow_interruptions=False)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


if __name__ == "__main__":
    cli.run_app(server)
