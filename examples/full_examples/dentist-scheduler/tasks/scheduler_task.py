import os
from enum import Enum
from typing import Annotated

import aiohttp
from pydantic import Field

from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, RunContext
from livekit.plugins import cartesia

from .global_functions import (
    get_date_today,
    get_user_info,
    transfer_to_messenger,
    transfer_to_receptionist,
    update_information,
)


class APIRequests(Enum):
    GET_APPTS = "get_appts"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    SCHEDULE = "schedule"


class Scheduler(Agent):
    def __init__(self, *, service: str) -> None:
        super().__init__(
            instructions="""You are Echo, a scheduler managing appointments for the LiveKit dental
                            office. If the user's email is not given, ask for it before
                            scheduling/rescheduling/canceling. Assume the letters are lowercase
                            unless specified otherwise. When calling functions, return the user's
                            email if already given. Always confirm details with the user.
                            Convert all times given by the user to ISO 8601 format in UTC
                            timezone, assuming the user is in America/Los Angeles,
                            and do not mention the conversion or the UTC timezone to the user.
                            Avoiding repeating words. When scheduling appointments, ensure that
                            the dates are in the future.""",
            tts=cartesia.TTS(voice="729651dc-c6c3-4ee5-97fa-350da1f88600"),
            tools=[
                update_information,
                get_user_info,
                transfer_to_receptionist,
                transfer_to_messenger,
                get_date_today,
            ],
        )
        self._service_requested = service

    async def on_enter(self) -> None:
        self._event_ids = self.session.userdata["event_ids"]
        await self.session.generate_reply(
            instructions=f"""Introduce yourself and confirm that they would like to
            {self._service_requested} an appointment. The information given is:
            {self.session.userdata["userinfo"].json()}"""
        )

    async def send_request(
        self,
        *,
        request: APIRequests,
        uid: str = "",
        time: str = "",
        slug: str = "",
        context: RunContext,
    ) -> dict:
        headers = {
            "cal-api-version": "2024-08-13",
            "Authorization": "Bearer " + os.getenv("CAL_API_KEY"),
        }
        async with aiohttp.ClientSession() as session:
            params = {}
            if request.value == "get_appts":
                payload = {
                    "attendeeEmail": context.userdata["userinfo"].email,
                    "attendeeName": context.userdata["userinfo"].name,
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
                    "name": context.userdata["userinfo"].name,
                    "email": context.userdata["userinfo"].email,
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
            if request.value in ["schedule", "reschedule", "cancel"]:
                async with session.post(**params) as response:
                    data = await response.json()
            elif request.value == "get_appts":
                async with session.get(**params) as response:
                    data = await response.json()
            else:
                raise Exception("Cal.com API Communication Error")
            return data

    @function_tool()
    async def schedule(
        self,
        email: Annotated[
            str, Field(description="The user's email, in the format local-part@domain")
        ],
        description: Annotated[
            str,
            Field(
                description="""Reason for scheduling appointment,
                                either 'routine-checkup' or 'tooth-extraction'"""
            ),
        ],
        date: Annotated[
            str,
            Field(
                description="""Formatted and converted date and time for the new appointment,
                in ISO 8601 format in UTC timezone assuming the user is in Los Angeles."""
            ),
        ],
        context: RunContext,
    ) -> str:
        """
        Schedules a new appointment for users.
        The email should be confirmed by spelling it out to the user.
        """
        context.userdata["userinfo"].email = email
        response = await self.send_request(
            request=APIRequests.SCHEDULE, time=date, slug=description, context=context
        )
        print(response)
        if response["status"] == "success":
            return "Appointment has been successfully scheduled!"
        elif (
            response["status"] == "error"
            and response["error"]["message"]
            == "User either already has booking at this time or is not available"
        ):
            return "The date and time specified are unavailable, please choose another date."

    @function_tool()
    async def cancel(
        self,
        email: Annotated[
            str, Field(description="The user's email, in the format local-part@domain")
        ],
        context: RunContext,
    ) -> str:
        """
        Cancels an existing appointment.
        """
        context.userdata["userinfo"].email = email
        response = await self.send_request(request=APIRequests.GET_APPTS, context=context)
        if response["data"]:
            cancel_response = await self.send_request(
                request=APIRequests.CANCEL, uid=response["data"][0]["uid"], context=context
            )
            if cancel_response["status"] == "success":
                return "You're all set!"
        else:
            return "There are no appointments under your name, perhaps you should create one."

    @function_tool()
    async def reschedule(
        self,
        email: Annotated[
            str, Field(description="The user's email, in the format local-part@domain")
        ],
        new_time: Annotated[
            str,
            Field(description="the new time and day for the appointment to be rescheduled to"),
        ],
        context: RunContext,
    ) -> str:
        """
        Reschedules an appointment to a new date specified by the user
        """
        context.userdata["userinfo"].email = email
        response = await self.send_request(request=APIRequests.GET_APPTS, context=context)
        if response["data"]:
            reschedule_response = await self.send_request(
                request=APIRequests.RESCHEDULE,
                uid=response["data"][0]["uid"],
                time=new_time,
                context=context,
            )
            if (
                reschedule_response["status"] == "error"
                and reschedule_response["error"]["message"]
                == "User either already has booking at this time or is not available"
            ):
                return """The office is unavailable at the time specified.
                        Please choose another time to try again."""

            elif reschedule_response["status"] == "success":
                return "You are all set, the appointment was moved."

        else:
            return "There are no appointments under your name, perhaps you should create one."
