from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask
from livekit.plugins import cartesia


class Receptionist(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Alloy, a receptionist at the LiveKit Dental Office who answers inquiries and manages appointments for users. 
            If there is an inquiry that can't be answered, suggest to leave a message. When calling functions, return the user's name if already known. Be brief and efficient.""",
            tts=cartesia.TTS(emotion=["positivity:high"]),
        )

    async def on_enter(self) -> None:
        await self.agent.generate_reply(
            instructions=f"""Welcome the user to the LiveKit Dental Office and ask how you can assist. The user's name is {self.agent.userdata["userinfo"].name}."""
        )

    @ai_function()
    async def hours_inquiry(self) -> None:
        """Answers user inquiries about the LiveKit dental office's hours of operation."""
        await self.agent.current_speech.wait_for_playout()
        await self.agent.generate_reply(
            instructions="Inform the user that the LiveKit dental office is closed on Sundays but open 10 AM to 12 PM, and 1 PM to 4 PM otherwise."
        )

    @ai_function()
    async def location_inquiry(self) -> None:
        """Answers user inquiries about the LiveKit dental office's location and parking"""
        await self.agent.current_speech.wait_for_playout()
        await self.agent.generate_reply(
            instructions="Inform the user that the LiveKit dental office is located at 123 LiveKit Lane and there is free parking."
        )

    @ai_function()
    async def manage_appointment(self, name: str, service: str):
        """
        This function allows for users to schedule, reschedule, or cancel an appointment by transferring to the scheduler.
        The user's name will be confirmed with the user by spelling it out.
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
        This function allows users to leave a message for the office by transferring to the messenger.
        The user's name will be confirmed with the user by spelling it out.

        Args:
            name: The user's name
        """
        self.agent.userdata["userinfo"].name = name
        return self.agent.userdata[
            "tasks"
        ].messenger, "I'll be transferring you to Shimmer."

    @ai_function()
    async def update_information(self, field: str, updated_info: str) -> str:
        """Updates the user's information on record.

        Args:
            field: Either "name", "email", or "phone_number"
            updated_info: The user's name, email, or phone number
        """
        if field == "name":
            self.agent.userdata["userinfo"].name = updated_info

        if field == "email":
            self.agent.userdata["userinfo"].email = updated_info

        if field == "phone_number":
            self.agent.userdata["userinfo"].phone = updated_info

        return "Got it, thank you!"
