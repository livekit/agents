from __future__ import annotations

from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask
from livekit.plugins import openai, cartesia, deepgram, silero


class Receptionist(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are Alloy, a receptionist at the LiveKit Dental Office who answers inquiries and manages appointments for users. 
                            Confirm names by spelling them out. Always speak in English. Be brief and concise.""",
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
    async def manage_appointment(self, name: str, service: str):
        """
        This function allows for users to schedule, reschedule, or cancel an appointment.
        The user's name will be confirmed by spelling it out to the user.

        Args:
            name: The user's name
            service: Either "schedule", "reschedule", or "cancel"
        """
        self.agent.userdata["userinfo"].name = name
        return self.agent.userdata["tasks"].scheduler(
            service=service
        ), "I'll be transferring you to our scheduler, Echo!"

    @ai_function()
    async def leave_message(self, name: str):
        """
        This function allows for users to leave a message for the office.
        The user's name will be confirmed by spelling it out to the user.

        Args:
            name: The user's name
        """
        self.agent.userdata["userinfo"].name = name
        return self.agent.userdata[
            "tasks"
        ].messenger, "I'll be transferring you to Shimmer."

    @ai_function()
    async def update_name(self, name: str) -> None:
        """Updates the user's name

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
