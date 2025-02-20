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


class DentistScheduler(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are an assistant scheduling dentist appointments for users. Always speak in English.",
            llm=openai.realtime.RealtimeModel(),
        )

    async def on_enter(self) -> None:
        speech_handle = await self.agent.generate_reply(
            instructions="Welcome the user to the LiveKit Dental Office and ask how you could assist."
        )

    @ai_function(description="This functions allows the user to leave a message for the dentist.")
    async def callback_request(self, message: str):...
    
    @ai_function(description="This function answers inquiries about services offered.")
    async def services_inquiry(self):...

    @ai_function()
    async def schedule_appointment(self, context: AgentContext):
        return CollectUserData(chat_ctx=self.llm.chat_ctx)

class CollectUserData(AgentTask):
    def __init__(self, *, chat_ctx: llm.ChatContext) -> None:
        super().__init__(
            instructions="Retrieve the user's name and then their email.",
            llm=openai.realtime.RealtimeModel(),
            chat_ctx=chat_ctx,
        )
        self._appointment = {}

    @ai_function
    async def record_data(self): ...

    # add email + name property to dictionary from transcript

class CollectReason(AgentTask):
    def __init__(self, *, appointment: dict) -> None:
        super().__init__(
            instructions="Ask the user why they are scheduling an appointment. The options are: Routine checkup, tooth extraction, and other.",
            llm=openai.realtime.RealtimeModel(),
        )
        self._appointment = appointment

    @ai_function
    async def routine_checkup(self, context: AgentContext): ...

    # call SelectTime()

    @ai_function
    async def tooth_extraction(self, context: AgentContext): ...

    # specify which tooth

    @ai_function
    async def other(self, context: AgentContext): ...

    # ask for details


class SelectTime(AgentTask):
    def __init__(self, *, appointment: dict) -> None:
        super().__init__(
            instructions="You are a dental assistant checking what times are available for appointments.",
            llm=openai.realtime.RealtimeModel(),
        )
        self._appointment = appointment
    
    async def on_enter(self) -> None:
        await self.agent.say("Let me check what time slots are available.")
    
    async def get_availability(self) -> None:
        token = os.getenv("CALENDLY_TOKEN")
        user = os.getenv("CALENDLY_USER_ID")
        url = f"https://api.calendly.com/user_availability_schedules?user={user}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status== 200:
                    data = await response.data
                    # iterate through availabilities 
                    # add case for when no days work

                else:
                    raise Exception(
                        f"Failed to access Calendly API. Status code: {response.status}"
                    )


class FinalizeAppointment(AgentTask): ...

# confirms details and schedules via Calendly API

load_dotenv()

logger = logging.getLogger("dental-scheduler")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    agent = PipelineAgent(task=DentistScheduler(), llm=openai.realtime.RealtimeModel())

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    agent.start(ctx.room, participant)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
