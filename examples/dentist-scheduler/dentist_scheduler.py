import logging
import aiohttp
import os
from enum import Enum
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.agents.llm import ai_function
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.plugins import openai, cartesia, deepgram, silero

from api_setup import setup_event_types


@dataclass
class UserInfo:
    name: str = "not given"
    email: str = "not given"
    phone: str = "not given"
    message: str | None = None


class APIRequests(Enum):
    GET_APPTS = "get_appts"
    GET_APPT = "get_appt"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    SCHEDULE = "schedule"


class Receptionist(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are Alloy, a receptionist at the LiveKit Dental Office who answers inquiries and manages appointments for users. 
                            When the user provides their name, number, or email, spell it out to confirm it. Always speak in English.""",
            tts=cartesia.TTS(emotion=["positivity:high"]),
        )

    async def on_enter(self) -> None:
        self._userinfo = self.agent.userdata["userinfo"]
        await self.agent.generate_reply(
            instructions=f"""Welcome the user to the LiveKit Dental Office and ask how you can assist. The user's name is {self._userinfo.name}. 
            If the user wants to manage an appointment or leave a message and their name is not given, ask for it before proceeding."""
        )

    @ai_function()
    async def hours_inquiry(self):
        """ Answers user inquiries about the LiveKit dental office's hours of operation. """
        await self.agent.current_speech.wait_for_playout()
        await self.agent.generate_reply(instructions="Inform the user that the LiveKit dental office is closed on Sundays but open 10 AM to 12 PM, and 1 PM to 4 PM otherwise.")

    @ai_function()
    async def location_inquiry(self):
        """ Answers user inquiries about the LiveKit dental office's location and parking"""
        await self.agent.current_speech.wait_for_playout()
        await self.agent.generate_reply(instructions="Inform the user that the LiveKit dental office is located at 123 LiveKit Lane and there is free parking.")

    @ai_function()
    async def appointment(self, service: str):
        """This function allows for users to schedule, reschedule, or cancel an appointment.

        Args:
            service (str): Either "schedule", "reschedule", or "cancel"
        """
        return Scheduler(service=service), "I'll be transferring you to our scheduler, Echo!"

    @ai_function()
    async def take_message(self):
        """This function allows for users to leave a message for the office."""
        return Messenger(), "I'll be transferring you to Shimmer."

    @ai_function()
    async def get_name(self, name: str) -> None:
        """Records the user's name

        Args:
            name (str): User's name
        """
        self.agent.userdata["userinfo"].name = name

    
    @ai_function()
    async def update_email(self, email: str) -> None:
        """ Updates email associated with the user 

            Args:
                email (str): The user's email
        """
        self.agent.userdata["userinfo"].email = email
    
    @ai_function()
    async def update_phone_number(self, phone_number: str) -> None:
        """ Updates phone number associated with the user 

            Args:
                phone number (str): The user's phone number
        """
        self.agent.userdata["userinfo"].phone = phone_number

class Scheduler(AgentTask):
    def __init__(self, *, service: str) -> None:
        super().__init__(
            instructions="""You are Echo, a scheduler managing appointments for the LiveKit dental office. If the user wants to schedule a new appointment and
                            their email is not given, ask for it. When given any information, confirm with the user that it is correct. """
        )
        self._service_requested = service

    async def on_enter(self) -> None:
        self._userinfo = self.agent.userdata["userinfo"]
        self._event_ids = self.agent.userdata["event_ids"]

        await self.agent.generate_reply(
            instructions=f"""Introduce yourself and ask {self._userinfo.name} to confirm that they would like to {self._service_requested} an appointment. 
                            Their email is {self._userinfo.email}."""
        )

    async def send_request(
        self, *, request: APIRequests, uid: str = "", time: str = "", slug: str = ""
    ) -> dict:
        headers = {
            "cal-api-version": "2024-08-13",
            "Authorization": "Bearer " + os.getenv("CAL_API_KEY"),
        }
        async with aiohttp.ClientSession() as session:
            params = {}
            if request.value == "get_appts":
                payload = {"attendeeName": self._userdata.name}
                params = {
                    "url": "https://api.cal.com/v2/bookings",
                    "params": payload,
                    "headers": headers,
                }

            if request.value == "cancel":
                params = {
                    "url": f"https://api.cal.com/v2/bookings/{uid}/cancel",
                    "headers": headers,
                }

            if request.value == "schedule":
                attendee_details = {
                    "name": self._userinfo.name,
                    "email": self._userinfo.email,
                    "timeZone": "America/Los_Angeles",
                }
                payload = {
                    "start": time,
                    "eventTypeId": self._event_ids[slug],
                    "attendee": attendee_details,
                }
                params = {
                    "url": "https://api.cal.com/v2/bookings",
                    "json": payload,
                    "headers": headers,
                }

            if request.value == "reschedule":
                payload = {"start": time}
                params = {
                    "url": f"https://api.cal.com/v2/bookings/{uid}/reschedule",
                    "headers": headers,
                    "json": payload,
                }

            if request.value == "get_appt":
                params = {
                    "url": f"https://api.cal.com/v2/bookings/{uid}",
                    "headers": headers,
                }
            else:
                raise Exception(f"APIRequest not valid: {request}")
            try:
                async with session.post(**params) as response:
                    data = await response.json()
                return data
            except Exception as e:
                print(f"Error occurred: {e}")

    @ai_function()
    async def schedule(self, description: str, date: str) -> None:
        """Schedules a new appointment for users.
        Args:
            description (str): Reason for scheduling appointment, either "routine-checkup" or "tooth-extraction"
            date (str): Date and time for the appointment in ISO 8601 format in UTC timezone.
        """
        response = await self.send_request(
            request=APIRequests.SCHEDULE, time=date, slug=description
        )
        if response.status == "success":
            await self.agent.generate_reply(
                instructions="Tell the user you were able to schedule the appointment successfully."
            )
        else:
            raise Exception("Error occurred when attempting to schedule")

    @ai_function()
    async def cancel(self) -> None:
        """Cancels an existing appointment"""
        response = await self.send_request(request=APIRequests.GET_APPTS)
        if response.data:
            await self.agent.generate_reply(
                instructions=f"Inform the user that you are canceling the appointment found on {response.data.start} for {response.data.title}."
            )
            confirmation = await self.send_request(
                request=APIRequests.CANCEL, uid=response.data.uid
            )
            if confirmation.status == "success":
                return Receptionist(), "You're all set, transferring you back to Alloy."
        else:
            await self.agent.generate_reply(
                instructions="Inform the user that there are no appointments under their name and ask to create one."
            )

    @ai_function()
    async def reschedule(self, new_time: str) -> None:
        """Reschedules an existing appointment
        Args:
            new_time (str): New time for the appointment to be rescheduled to, in ISO 8601 format in UTC timezone.
        """
        response = await self.send_request(request=APIRequests.GET_APPTS)
        if response.data:
            await self.agent.generate_reply(
                instructions=f"Inform the user that you are rescheduling the appointment found on {response.data.start} for {response.data.title} to {new_time}."
            )
            confirmation = await self.send_request(
                request=APIRequests.RESCHEDULE, uid=response.data.uid, time=new_time
            )
            if confirmation.status == "success":
                return Receptionist(), "You're all set, transferring you back to Alloy."
        else:
            await self.agent.generate_reply(
                instructions="Inform the user that there are no appointments under their name and ask to create one."
            )

    @ai_function()
    async def get_email(self, email: str) -> None:
        """Records the user's email.

        Args:
            email (str): The user's email
        """
        self.agent.userdata.email = email


class Messenger(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Shimmer, an assistant taking messages for the LiveKit dental office.",
            llm=openai.realtime.RealtimeModel(voice="shimmer"),
        )

    async def on_enter(self) -> None:
        self.agent.say(
            f"Alright {self.agent.userdata.name}, please state your phone number and then your message."
        )

    @ai_function()
    async def get_phone_number(self, phone_number: str) -> None:
        """Records the user's phone number.

        Args:
            phone_number (str): The user's phone number
        """
        self.agent.userdata.phone = phone_number
        # confirm correct phone number

    @ai_function()
    async def record_message(self, message: str) -> None:
        """Records the user's message to be left for the office.

        Args:
            message (str): The user's message to be left for the office
        """
        self.agent.userdata.message = message
        # send to supabase
        return (
            Receptionist(),
            f"Got it {self.agent.userdata.name}, transferring you to Alloy!",
        )


load_dotenv()

logger = logging.getLogger("dental-scheduler")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    event_ids = await setup_event_types()
    userdata = {"event_ids": event_ids, "userinfo": UserInfo()}

    agent = PipelineAgent(
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
