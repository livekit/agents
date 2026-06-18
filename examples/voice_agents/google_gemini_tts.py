import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.plugins import deepgram, google
from livekit.plugins.google.beta import GeminiTTS

logger = logging.getLogger("gemini-tts-agent")
load_dotenv()


class GeminiTTSAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. Respond briefly and concisely using voice conversation.",
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=deepgram.STT(),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=GeminiTTS(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            voice_name="Kore",
            model="gemini-3.1-flash-tts-preview",
        ),
    )
    await session.start(agent=GeminiTTSAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
