import os
from enum import Enum
from typing import Annotated

import aiohttp
from global_functions import (
    transfer_to_messenger,
    transfer_to_receptionist,
    update_information,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent
from livekit.plugins import cartesia
from pydantic import Field


class APIRequests(Enum):
    GET_APPTS = "get_appts"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    SCHEDULE = "schedule"


class Scheduler(Agent):
    def __init__(self, *, service: str) -> None:
        super().__init__(
            instructions="""You are a scheduler managing appointments for the LiveKit dental office. If the user's email is not given, ask for it before 
                            scheduling/rescheduling/canceling. When calling functions, return the user's email if already known.
                            Always confirm details with the user. Convert all times given by the user to ISO 8601 format in UTC timezone,
                            assuming the user is in America/Los Angeles, and do not mention the conversion or the UTC timezone to the user. Avoiding repeating words.""",
            tts=cartesia.TTS(voice="729651dc-c6c3-4ee5-97fa-350da1f88600"),
            tools=[update_information, transfer_to_receptionist, transfer_to_messenger],
        )
        self._service_requested = service

    async def on_enter(self) -> None:
        self._event_ids = self.session.userdata["event_ids"]

        await self.session.generate_reply(
            instructions=f"""Introduce yourself and ask {self.session.userdata["userinfo"].name} to confirm that they would like to {self._service_requested} an appointment. 
                            Their email is {self.session.userdata["userinfo"].email}."""
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
                    "attendeeEmail": self.session.userdata["userinfo"].email,
                    "attendeeName": self.session.userdata["userinfo"].name,
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
                    "name": self.session.userdata["userinfo"].name,
                    "email": self.session.userdata["userinfo"].email,
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
                raise Exception(f"API Communication Error: {e}")

    @function_tool()
    async def schedule(
        self,
        email: Annotated[
            str, Field(description="The user's email, in the format local-part@domain")
        ],
        description: Annotated[
            str,
            Field(
                description="Reason for scheduling appointment, either 'routine-checkup' or 'tooth-extraction'"
            ),
        ],
        date: Annotated[
            str,
            Field(
                description="Formatted and converted date and time for the appointment, in ISO 8601 format in UTC timezone assuming the user is in Los Angeles."
            ),
        ],
    ) -> None:
        """
        Schedules a new appointment for users. The email should be confirmed by spelling it out to the user.
        """
        self.session.userdata["userinfo"].email = email
        response = await self.send_request(
            request=APIRequests.SCHEDULE, time=date, slug=description
        )

        if response["status"] == "success":
            await self.session.generate_reply(
                instructions="Tell the user you were able to schedule the appointment successfully."
            )
        elif (
            response["status"] == "error"
            and response["error"]["message"]
            == "User either already has booking at this time or is not available"
        ):
            if self.session.current_speech:
                await self.session.current_speech.wait_for_playout()
            await self.session.generate_reply(
                instructions="Inform the user that the date and time specified are unavailable, and ask the user to choose another date."
            )
        else:
            await self.session.generate_reply(
                instructions="There was an error with scheduling, suggest to leave a message about it."
            )

    @function_tool()
    async def cancel(
        self,
        email: Annotated[
            str, Field(description="The user's email, in the format local-part@domain")
        ],
    ) -> None:
        """
        Cancels an existing appointment.
        """
        self.session.userdata["userinfo"].email = email
        response = await self.send_request(request=APIRequests.GET_APPTS)
        if response["data"]:
            cancel_response = await self.send_request(
                request=APIRequests.CANCEL, uid=response["data"][0]["uid"]
            )
            if cancel_response["status"] == "success":
                if self.session.current_speech:
                    await self.session.current_speech.wait_for_playout()
                await self.session.generate_reply(
                    instructions="Inform the user that they are all set."
                )
        else:
            if self.session.current_speech:
                await self.session.current_speech.wait_for_playout()
            await self.session.generate_reply(
                instructions="Inform the user that there are no appointments under their name and ask to create one."
            )

    @function_tool()
    async def reschedule(
        self,
        email: Annotated[
            str, Field(description="The user's email, in the format local-part@domain")
        ],
        new_time: Annotated[
            str,
            Field(
                description="the new time and day for the appointment to be rescheduled to"
            ),
        ],
    ) -> None:
        """
        Reschedules an appointment to a new date specified by the user
        """
        self.session.userdata["userinfo"].email = email
        response = await self.send_request(request=APIRequests.GET_APPTS)
        if response["data"]:
            reschedule_response = await self.send_request(
                request=APIRequests.RESCHEDULE,
                uid=response["data"][0]["uid"],
                time=new_time,
            )
            if (
                reschedule_response["status"] == "error"
                and reschedule_response["error"]["message"]
                == "User either already has booking at this time or is not available"
            ):
                if self.session.current_speech:
                    await self.session.current_speech.wait_for_playout()
                await self.session.generate_reply(
                    instructions="Tell the user that the office is unavailable at the time specified. You were unable to reschedule, so ask the user to choose another time to try again."
                )

            elif reschedule_response["status"] == "success":
                if self.session.current_speech:
                    await self.session.current_speech.wait_for_playout()
                await self.session.generate_reply(
                    instructions="Inform the user that they are all set and confirm that the appointment was moved."
                )

        else:
            if self.session.current_speech:
                await self.session.current_speech.wait_for_playout()
            await self.session.generate_reply(
                instructions="Inform the user that there are no appointments under their name and ask to create one."
            )
