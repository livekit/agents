from __future__ import annotations

from enum import Enum
import aiohttp
import os
from enum import Enum

from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask
from livekit.plugins import openai, cartesia, deepgram, silero


class APIRequests(Enum):
    GET_APPTS = "get_appts"
    GET_APPT = "get_appt"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    SCHEDULE = "schedule"


class Scheduler(AgentTask):
    def __init__(self, *, service: str) -> None:
        super().__init__(
            instructions="""You are Echo, a scheduler managing appointments for the LiveKit dental office. If the user's email is not given, ask for it before 
                            proceeding. Always confirm details with the user. """,
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
        Schedules a new appointment for users. The email should be confirmed by spelling it out to the user.
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
        else:
            await self.agent.generate_reply(
                instructions="Inform the user that there are no appointments under their name and ask to create one."
            )

    @ai_function()
    async def transfer_to_receptionist(self) -> None:
        """Transfers the user to the receptionist"""
        return self.agent.userdata["tasks"].receptionist, "Transferring you to our receptionist!"

    @ai_function()
    async def transfer_to_messenger(self) -> None:
        """Transfers the user to the messenger"""
        return self.agent.userdata["tasks"].messenger, "Transferring you to our messenger!"
