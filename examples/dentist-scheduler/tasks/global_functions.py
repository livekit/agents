from livekit.agents.llm import function_tool


@function_tool()
async def update_information(self, field: str, info: str) -> str:
    """
    Updates information on record about the user. The only fields to update are names, phone numbers, and emails.

    Args:
        field: The type of information to be updated, either "phone_number", "email", or "name"
        info: The new user provided information

    """
    if field == "name":
        self.agent.userdata["userinfo"].name = info
    elif field == "phone_number":
        self.agent.userdata["userinfo"].phone = info
    elif field == "email":
        self.agent.userdata["userinfo"].email = info

    return "Got it, thank you!"


@function_tool()
async def transfer_to_receptionist(self):
    """Transfers the user to the receptionist for any office inquiries, user information updates, or when they are finished with managing appointments."""
    return self.agent.userdata[
        "tasks"
    ].receptionist, "Transferring you to our receptionist!"


@function_tool()
async def transfer_to_scheduler(self, service: str):
    """
    Transfers the user to the Scheduler.

    Args:
        service: Either "schedule", "reschedule", or "cancel"
    """
    return self.agent.userdata["tasks"].scheduler(
        service=service
    ), "Transferring you to our scheduler!"


@function_tool()
async def transfer_to_messenger(self):
    """Transfers the user to the messenger if they want to leave a message for the office."""
    return self.agent.userdata["tasks"].messenger, "Transferring you to our messenger!"
