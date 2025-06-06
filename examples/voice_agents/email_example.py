import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    workflows,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, cartesia, silero

logger = logging.getLogger("get-email-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a voice assistant that helps users register to attend an event. "
                "Wait for the user to indicate that they want to register or attend the event before starting the registration process. "
                "Once the user expresses their intent to register, guide them through the process smoothly. "
                "Your main goal is to confirm their spot for the event and let them know they are successfully registered. "
                "Keep the interaction simple, clear, and friendly, ensuring the user feels confident that their attendance is confirmed. "
                "Avoid providing information or starting processes that the user has not explicitly requested."
            )
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def register_for_event(self, context: RunContext):
        "Start the registration process for the event."

        email_result = await workflows.GetEmailAgent()
        email_address = email_result.email_address

        logger.info(f"User's email address: {email_address}")

        return "The user is confirmed for seat 23 in group LK1. "


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
    )

    await session.start(agent=MyAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
