from typing import Annotated

from global_functions import update_information
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent
from livekit.plugins import cartesia
from pydantic import Field


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a receptionist at the LiveKit Dental Office who answers inquiries and manages appointments for users. 
            If there is an inquiry that can't be answered, suggest to leave a message. Be brief and efficient, do not ask for unnecessary details. 
            When handling appointments or taking a message, you will transfer the user to another agent.""",
            tts=cartesia.TTS(emotion=["positivity:high"]),
            tools=[update_information],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=f"""Welcome the user to the LiveKit Dental Office and ask how you can assist. The user's name is {self.session.userdata["userinfo"].name}."""
        )

    @function_tool()
    async def hours_inquiry(self) -> None:
        """Answers user inquiries about the LiveKit dental office's hours of operation."""
        await self.session.current_speech.wait_for_playout()
        await self.session.generate_reply(
            instructions="Inform the user that the LiveKit dental office is closed on Sundays but open 10 AM to 12 PM, and 1 PM to 4 PM otherwise."
        )

    @function_tool()
    async def location_inquiry(self) -> None:
        """Answers user inquiries about the LiveKit dental office's location and parking"""
        await self.session.current_speech.wait_for_playout()
        await self.session.generate_reply(
            instructions="Inform the user that the LiveKit dental office is located at 123 LiveKit Lane and there is free parking."
        )

    @function_tool()
    async def manage_appointment(
        self,
        name: Annotated[str, Field(description="The user's name")],
        action: Annotated[
            str,
            Field(
                description="The appointment action requested, either 'schedule', 'reschedule', or 'cancel'"
            ),
        ],
    ) -> tuple[Agent, str]:
        """
        This function allows for users to schedule, reschedule, or cancel an appointment by transferring to the scheduler.
        """
        self.session.userdata["userinfo"].name = name
        return self.session.userdata["agents"].scheduler(
            service=action
        ), "I'll be transferring you to our scheduler, Echo!"

    @function_tool()
    async def leave_message(
        self,
        name: Annotated[str, Field(description="The user's name")],
    ) -> tuple[Agent, str]:
        """
        This function allows users to leave a message for the office by transferring to the messenger.
        """
        self.session.userdata["userinfo"].name = name
        return self.session.userdata[
            "agents"
        ].messenger, "I'll be transferring you to Shimmer."
