import logging
from dataclasses import dataclass

from api_setup import setup_event_types
from dotenv import load_dotenv
from pydantic import BaseModel
from tasks import Messenger, Receptionist, Scheduler

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import cartesia, deepgram, openai, silero


class UserInfo(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    message: str | None = None


@dataclass
class Agents:
    @property
    def receptionist(self) -> Agent:
        return Receptionist()

    @property
    def messenger(self) -> Agent:
        return Messenger()

    def scheduler(self, service: str) -> Agent:
        return Scheduler(service=service)


load_dotenv()

logger = logging.getLogger("dental-scheduler")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    event_ids = await setup_event_types()
    userdata = {"event_ids": event_ids, "userinfo": UserInfo(), "agents": Agents()}

    session = AgentSession(
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await ctx.connect()
    await session.start(agent=userdata["agents"].receptionist, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
