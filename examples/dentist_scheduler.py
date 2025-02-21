import asyncio
import logging
import datetime
import aiohttp
import os
from typing import Literal, Annotated

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)

from livekit.agents.llm import ai_function, ChatContext
from livekit.agents.pipeline import AgentContext, AgentTask, PipelineAgent
from livekit.plugins import openai


class Receptionist(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy, a receptionist scheduling dentist appointments for users. Always speak in English.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    async def on_enter(self) -> None:
        speech_handle = await self.agent.generate_reply(
            instructions="Welcome the user to the LiveKit Dental Office and ask how you could assist."
        )

    async def on_exit(self) -> None:
        self.agent.say("Thank you for calling the LiveKit Dental Office, have a wonderful day!")

    @ai_function()
    async def callback_request(self, message: str):
        """ Allows the user to request a call back and leave a message for the dentist """
        ...
    
    @ai_function()
    async def services_inquiry(self):
        """ Answers questions regarding hours of operation, services offered, and location """
        ...

    @ai_function()
    async def scheduler(self, context: AgentContext):
        """ Handles appointments: scheduling, rescheduling, and canceling """
        return Scheduler(chat_ctx=self.llm.chat_ctx)

class Scheduler(AgentTask):
    def __init__(self, *, chat_ctx: llm.ChatContext) -> None:
        super().__init__(
            instructions="You are Echo, a scheduler managing appointments for the LiveKit dental office.",
            llm=openai.realtime.RealtimeModel(voice="echo"),
            chat_ctx=chat_ctx,
        )
    
    @ai_function
    async def schedule(self) -> None:
        """ Schedules a new appointment """
        ...
    
    @ai_function
    async def cancel(self) -> None:
        """ Cancels an existing appointment """
        ...
    
    @ai_function
    async def reschedule(self) -> None:
        """ Reschedules an existing appointment """
        ...
    
class Messenger(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Shimmer, an assistant taking messages for the LiveKit dental office.",
            llm=openai.realtime.RealtimeModel(voice="shimmer"),
        )

    async def on_enter(self) -> None:
        speech_handle = await self.agent.generate_reply(
            instructions="Ask the user for their message and their contact information."
        )

load_dotenv()

logger = logging.getLogger("dental-scheduler")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    agent = PipelineAgent(task=Receptionist(), llm=openai.realtime.RealtimeModel())

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    agent.start(ctx.room, participant)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
