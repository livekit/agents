import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)
from livekit.plugins import reson8, silero

logger = logging.getLogger("reson8-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. Keep your responses concise and "
                "conversational, and do not use markdown or emojis."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself")


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        # Reson8 streams with server-side turn detection and any-language
        # auto-detection. Omit `language` to auto-detect, or pass any code
        # (e.g. reson8.STT(language="nl")) to pin it.
        stt=reson8.STT(),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
