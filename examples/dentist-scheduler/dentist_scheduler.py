import logging
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.agents.voice import VoiceAgent
from livekit.plugins import openai, cartesia, deepgram, silero
from livekit.agents.voice import AgentTask

from api_setup import setup_event_types
from tasks import *


@dataclass
class UserInfo:
    name: str = "not given"
    email: str = "not given"
    phone: str = "not given"
    message: str | None = None

@dataclass
class Tasks:
    @property
    def receptionist(self) -> AgentTask:
        return Receptionist()
    
    @property
    def messenger(self) -> AgentTask:
        return Messenger()
    
    def scheduler(self, service: str) -> AgentTask:
        return Scheduler(service=service)

load_dotenv()

logger = logging.getLogger("dental-scheduler")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    event_ids = await setup_event_types()
    userdata = {"event_ids": event_ids, 
                "userinfo": UserInfo(),
                "tasks": Tasks()
                }

    agent = VoiceAgent(
        task=Receptionist(),
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await agent.start(room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
