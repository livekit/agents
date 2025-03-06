import os
import logging
import aiohttp
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
from livekit.agents.voice import AgentTask, VoiceAgent
from livekit.plugins import openai, cartesia, deepgram, silero
from supabase import create_client, Client
from api_setup import setup_event_types


@dataclass
class UserInfo:
    name: str = "not given"
    email: str = "not given"
    phone: str = "not given"
    message: str | None = None


class APIRequests(Enum):
    GET_APPTS = "get_appts"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    SCHEDULE = "schedule"


class Receptionist(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Alloy, a receptionist at the LiveKit Dental Office who answers inquiries and manages appointments for users. 
                            When the user provides their name, number, or email, spell it out to confirm it. Always speak in English. Be brief and concise.""",
        )

    async def on_enter(self) -> None:
        self._userinfo = self.agent.userdata["userinfo"]
        await self.agent.generate_reply(
            instructions=f"""Welcome the user to the LiveKit Dental Office and ask how you can assist. The user's name is {self._userinfo.name}. 
            If the user wants to manage an appointment or leave a message and their name is not given, ask for it before proceeding."""
        )

    @ai_function()
    async def hours_inquiry(self):
        """Answers user inquiries about the LiveKit dental office's hours of operation."""
        await self.agent.current_speech.wait_for_playout()
        await self.agent.generate_reply(
            instructions="Inform the user that the LiveKit dental office is closed on Sundays but open 10 AM to 12 PM, and 1 PM to 4 PM otherwise."
        )

    @ai_function()
    async def location_inquiry(self):
        """Answers user inquiries about the LiveKit dental office's location and parking"""
        await self.agent.current_speech.wait_for_playout()
        await self.agent.generate_reply(
            instructions="Inform the user that the LiveKit dental office is located at 123 LiveKit Lane and there is free parking."
        )

    @ai_function()
    async def appointment(self, service: str):
        """
        This function allows for users to schedule, reschedule, or cancel an appointment.
        Args:
            service: Either "schedule", "reschedule", or "cancel"
        """
        return Scheduler(
            service=service
        ), "I'll be transferring you to our scheduler, Echo!"

    @ai_function()
    async def take_message(self):
        """This function allows for users to leave a message for the office."""
        return Messenger(), "I'll be transferring you to Shimmer."

    @ai_function()
    async def update_name(self, name: str) -> None:
        """Updates the user's name on record

        Args:
            name: User's name
        """
        self.agent.userdata["userinfo"].name = name

    @ai_function()
    async def update_email(self, email: str) -> None:
        """Updates email associated with the user

        Args:
            email: The user's email
        """
        self.agent.userdata["userinfo"].email = email

    @ai_function()
    async def update_phone_number(self, phone_number: str) -> None:
        """Updates phone number associated with the user

        Args:
            phone number: The user's phone number
        """
        self.agent.userdata["userinfo"].phone = phone_number


class Scheduler(AgentTask):
    def __init__(self, *, service: str) -> None:
        super().__init__(
            instructions="""You are Echo, a scheduler managing appointments for the LiveKit dental office. If the user's email is not given, ask for it before 
                            proceeding. Always double check details with the user. """,
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
                payload = {
                    "attendeeEmail": self.agent.userdata["userinfo"].email,
                    "attendeeName": self.agent.userdata["userinfo"].name,
                    "status": "upcoming",
                }
                params = {
                    "url": "https://api.cal.com/v2/bookings",
                    "params": payload,
                    "headers": headers,
                }

            elif request.value == "cancel":
                payload = {"cancellationReason": "User requested cancellation"}
                params = {
                    "url": f"https://api.cal.com/v2/bookings/{uid}/cancel",
                    "json": payload,
                    "headers": headers,
                }

            elif request.value == "schedule":
                attendee_details = {
                    "name": self.agent.userdata["userinfo"].name,
                    "email": self.agent.userdata["userinfo"].email,
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

            elif request.value == "reschedule":
                payload = {"start": time}
                params = {
                    "url": f"https://api.cal.com/v2/bookings/{uid}/reschedule",
                    "headers": headers,
                    "json": payload,
                }

            else:
                raise Exception(f"APIRequest not valid: {request}, {request.value}")
            try:
                if request.value in ["schedule", "reschedule", "cancel"]:
                    async with session.post(**params) as response:
                        data = await response.json()
                else:
                    async with session.get(**params) as response:
                        data = await response.json()
                return data
            except Exception as e:
                print(f"Error occurred: {e}")

    @ai_function()
    async def schedule(self, email: str, description: str, date: str) -> None:
        """
        Schedules a new appointment for users.
        Args:
            email: The user's email, in the format local-part@domain
            description: Reason for scheduling appointment, either "routine-checkup" or "tooth-extraction"
            date: Date and time for the appointment, convert it to ISO 8601 format in UTC timezone.
        """
        self.agent.userdata["userinfo"].email = email
        response = await self.send_request(
            request=APIRequests.SCHEDULE, time=date, slug=description
        )
        if response["status"] == "success":
            await self.agent.generate_reply(
                instructions="Tell the user you were able to schedule the appointment successfully."
            )
        if (
            response["status"] == "error"
            and response["status"]["error"]["message"]
            == "User either already has booking at this time or is not available"
        ):
            if self.agent.current_speech:
                await self.agent.current_speech.wait_for_playout()
            await self.agent.generate_reply(
                instructions="Inform the user that the date and time specified are unavailable, and ask the user to choose another date."
            )
        else:
            raise Exception("Error occurred when attempting to schedule")

    @ai_function()
    async def cancel(self, email: str) -> None:
        """
        Cancels an existing appointment
        Args:
            email: The user's email formatted local-part@domain
        """
        self.agent.userdata["userinfo"].email = email
        response = await self.send_request(request=APIRequests.GET_APPTS)
        if response["data"]:
            if self.agent.current_speech:
                await self.agent.current_speech.wait_for_playout()
            await self.agent.generate_reply(
                instructions=f"Confirm with the user that they'd like to cancel the appointment found on {response["data"][0]["start"]} for {response["data"][0]["title"]}."
            )
            cancel_response = await self.send_request(
                request=APIRequests.CANCEL, uid=response["data"][0]["uid"]
            )
            if cancel_response["status"] == "success":
                if self.agent.current_speech:
                    await self.agent.current_speech.wait_for_playout()
                await self.agent.generate_reply(
                    instructions="Inform the user that they are all set."
                )
                return Receptionist(), "You're all set, transferring you back to Alloy."
        else:
            await self.agent.generate_reply(
                instructions="Inform the user that there are no appointments under their name and ask to create one."
            )

    @ai_function()
    async def reschedule(self, email: str, new_time: str) -> None:
        """
        Reschedules an existing appointment.
        Args:
            email: The user's email formatted local-part@domain
            new_time: New time for the appointment to be rescheduled to in ISO 8601 format in UTC timezone
        """
        self.agent.userdata["userinfo"].email = email
        response = await self.send_request(request=APIRequests.GET_APPTS)
        if response["data"]:
            if self.agent.current_speech:
                await self.agent.current_speech.wait_for_playout()
            await self.agent.generate_reply(
                instructions=f"Confirm with the user that you are rescheduling the appointment found on {response["data"][0]["start"]} for {response["data"][0]["title"]} to {new_time}."
            )

            reschedule_response = await self.send_request(
                request=APIRequests.RESCHEDULE,
                uid=response["data"][0]["uid"],
                time=new_time,
            )
            if reschedule_response["status"] == "success":
                if self.agent.current_speech:
                    await self.agent.current_speech.wait_for_playout()
                await self.agent.generate_reply(
                    instructions="Inform the user that they are all set."
                )
            if (
                response["status"] == "error"
                and response["status"]["error"]["message"]
                == "User either already has booking at this time or is not available"
            ):
                if self.agent.current_speech:
                    await self.agent.current_speech.wait_for_playout()
                await self.agent.generate_reply(
                    instructions="Inform the user that the date and time specified are unavailable, and ask the user to choose another date."
                )
                return Receptionist(), "You're all set, transferring you back to Alloy."
        else:
            await self.agent.generate_reply(
                instructions="Inform the user that there are no appointments under their name and ask to create one."
            )

    @ai_function()
    async def transfer_to_receptionist(self) -> None:
        """Transfers the user to the receptionist"""
        return Receptionist(), "Transferring you to our receptionist!"

    @ai_function()
    async def transfer_to_messenger(self) -> None:
        """Transfers the user to the messenger"""
        return Messenger(), "Transferring you to our message taker!"


class Messenger(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Shimmer, an assistant taking messages for the LiveKit dental office. If the user's number is not given, ask for it before proceeding.
            Be sure to confirm details such as phone numbers with the user.""",
        )
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self._supabase: Client = create_client(url, key)

    async def on_enter(self) -> None:
        self._userinfo = self.agent.userdata["userinfo"]

        await self.agent.generate_reply(
            instructions=f"""Introduce yourself and ask {self._userinfo.name} to confirm that they would like to {self._service_requested} an appointment. 
                            Their phone number is {self._userinfo.phone}."""
        )

    @ai_function()
    async def get_phone_number(self, phone_number: str) -> None:
        """
        Records the user's phone number.
        Args:
            phone_number: The user's phone number
        """
        self.agent.userdata["userinfo"].phone = phone_number

    @ai_function()
    async def record_message(self, message: str) -> None:
        """Records the user's message to be left for the office.

        Args:
            message: The user's message to be left for the office
        """
        self.agent.userdata["userinfo"].message = message
        param = {"name": self.agent.userdata["userinfo"].name, "message": message}
        response = self._supbase.table("messages").insert(param).execute()

        if response["data"]:
            if self.agent.current_speech:
                await self.agent.current_speech.wait_for_playout()
            await self.agent.generate_reply(
                instructions="Inform the user that their message has been submitted."
            )
        else:
            raise Exception("Error sending data to Supabase")


load_dotenv()

logger = logging.getLogger("dental-scheduler")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    event_ids = await setup_event_types()
    userdata = {"event_ids": event_ids, "userinfo": UserInfo()}

    agent = VoiceAgent(
        task=Receptionist(),
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.realtime.RealtimeModel(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await agent.start(room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
