import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli, inference
from livekit.plugins import elevenlabs, silero

logger = logging.getLogger("realtime-scribe-v2")
logger.setLevel(logging.INFO)

load_dotenv()


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    # Using ElevenLabs STT plugin directly for realtime mode support
    stt = elevenlabs.STT(
        use_realtime=True,
        server_vad={
            "vad_silence_threshold_secs": 0.5,
            "vad_threshold": 0.5,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 300,
        },
        model_id="scribe_v2_realtime",
    )

    session: AgentSession = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )
    await session.start(
        agent=Agent(instructions="You are a somewhat helpful assistant."), room=ctx.room
    )

    await session.say("Hello, how can I help you?", allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
