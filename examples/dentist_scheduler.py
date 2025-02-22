import asyncio
import logging
import datetime
import aiohttp
from enum import Enum
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    NotGivenOr,
    NOT_GIVEN,
)

from livekit.agents.llm import ai_function
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.plugins import openai


@dataclass
class UserInfo:
    name: str
    email: NotGivenOr[str] = NOT_GIVEN
    phone: NotGivenOr[str] = NOT_GIVEN
    message: NotGivenOr[str] = NOT_GIVEN


class ServicesOffered(Enum):
    MESSENGER = "messenger"
    SCHEDULER = "scheduler"


class Receptionist(AgentTask):
    def __init__(self, userdata: NotGivenOr[UserInfo] = NOT_GIVEN) -> None:
        super().__init__(
            instructions="""You are Alloy, a receptionist at the LiveKit Dental Office scheduling dentist appointments for users. Always speak in English.
                            The office is closed on Sundays but operates from 10AM to 12PM and 1PM to 4PM PT otherwise. The services offered are routine checkups and tooth extractions.
                            The location of the office is 123 LiveKit Lane, California.""",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )
        self.agent.userdata = userdata or None
        self._userdata_event = asyncio.Event()

    async def on_enter(self) -> None:
        speech_handle = await self.agent.generate_reply(
            instructions="Welcome the user to the LiveKit Dental Office and ask how you can assist."
        )

    async def on_exit(self) -> None:
        self.agent.say(
            "Thank you for calling the LiveKit Dental Office, have a wonderful day!"
        )

    @ai_function()
    async def servicer(self, service_requested: ServicesOffered):
        """Performs a service for the user, either managing appointments or leaving a message

        Args:
            service_requested (ServicesOffered): either "messenger", which allows the user to leave a message, or "scheduler",
            which allows the user to schedule a new appointment and reschedule or cancel an existing appointment.
        """
        if self.agent.userdata is None:
            self.agent.say("What is your name and contact information?")
            await self._userdata_event.wait()

        if service_requested.value == "messenger":
            return Messenger(), "I'll be transferring you to Shimmer."

        if service_requested.value == "scheduler":
            # perhaps would be better to also pass schedule/reschedule/cancel to prevent repetitiveness
            return Scheduler(), "I'll be transferring you to Echo."

    @ai_function()
    async def get_userdata(
        self, name: str | None, phone: str | None, email: str | None
    ) -> None:
        """Records user data.

        Args:
            name (str | None): User's name
            phone (str | None): User's phone number
            email (str | None): User's email
        """
        # to do: confirm that details are correct
        if name and self.agent.userdata is None:
            self.agent.userdata = UserInfo(name=name)
        else:
            self.agent.userdata.name = name
        if phone and self.agent.userdata is not None:
            self.agent.userdata.phone = phone
        if email and self.agent.userdata is not None:
            self.agent.userdata.email = email
        else:
            self.agent.say("Sorry, could you please repeat your name and contact info?")

        self._userdata_event.set()


class Scheduler(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo, a scheduler managing appointments for the LiveKit dental office.",
            llm=openai.realtime.RealtimeModel(voice="echo"),
        )
        self._payload_queue = asyncio.Queue(dict)

    async def on_enter(self) -> None:
        speech_handle = await self.agent.generate_reply(
            instructions="Introduce yourself and ask the user if they'd like to schedule a new appointment, reschedule, or cancel an existing appointment."
        )

    async def send_requests(self) -> None: ...

    # continuously sends requests from queue?

    @ai_function
    async def schedule(self) -> None:
        """Schedules a new appointment"""
        ...

    @ai_function
    async def cancel(self) -> None:
        """Cancels an existing appointment"""
        ...

    @ai_function
    async def reschedule(self) -> None:
        """Reschedules an existing appointment"""
        ...


class Messenger(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Shimmer, an assistant taking messages for the LiveKit dental office.",
            llm=openai.realtime.RealtimeModel(voice="shimmer"),
        )

    async def on_enter(self) -> None:
        self.agent.say(
            f"Alright {self.agent.userdata.name}, please state your message whenever you're ready."
        )

    # either directly use the transcript or function calling


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
