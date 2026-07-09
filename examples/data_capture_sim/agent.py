import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    beta,
    cli,
    inference,
)
from livekit.agents.llm import function_tool

logger = logging.getLogger("data-capture-sim")

load_dotenv()

# Simulation harness for the beta data-capture workflows. Every capture runs
# with require_confirmation=True so the confirm_* tool paths (including the
# corrective branches that now return tool output instead of calling
# generate_reply) are exercised even in text-only simulations, where
# confirmation is otherwise skipped. Tasks are deliberately created
# without chat_ctx: they start from an empty context, so every scenario value
# must be delivered inside the running task (the scenarios gate values on
# being asked). Seed chat_ctx here if a scenario ever needs the task to see
# something said before the capture began.


class DataCaptureAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an intake operator collecting details from the user, one at a time. "
                "When the user offers a detail (email, phone number, mailing address, "
                "date of birth, name, or credit card), call the matching collect_* tool "
                "to run the capture flow. After a capture completes, read the captured "
                "value back to the user verbatim, then ask if there is anything else. "
                "Do not collect anything the user has not offered."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user and ask which detail they would like to provide."
        )

    @function_tool
    async def collect_email(self, context: RunContext) -> str:
        """Collect the user's email address."""
        result = await beta.workflows.GetEmailTask(require_confirmation=True)
        logger.info(f"captured email: {result.email_address}")
        return f"Email captured: {result.email_address}. Read it back to the user."

    @function_tool
    async def collect_phone_number(self, context: RunContext) -> str:
        """Collect the user's phone number."""
        result = await beta.workflows.GetPhoneNumberTask(require_confirmation=True)
        logger.info(f"captured phone: {result.phone_number}")
        return f"Phone number captured: {result.phone_number}. Read it back to the user."

    @function_tool
    async def collect_address(self, context: RunContext) -> str:
        """Collect the user's mailing address."""
        result = await beta.workflows.GetAddressTask(require_confirmation=True)
        logger.info(f"captured address: {result.address}")
        return f"Address captured: {result.address}. Read it back to the user."

    @function_tool
    async def collect_date_of_birth(self, context: RunContext) -> str:
        """Collect the user's date of birth (and time of birth if they know it)."""
        result = await beta.workflows.GetDOBTask(include_time=True, require_confirmation=True)
        logger.info(f"captured dob: {result.date_of_birth} time: {result.time_of_birth}")
        captured = result.date_of_birth.isoformat()
        if result.time_of_birth is not None:
            captured += f" at {result.time_of_birth.strftime('%H:%M')}"
        return f"Date of birth captured: {captured}. Read it back to the user."

    @function_tool
    async def collect_name(self, context: RunContext) -> str:
        """Collect the user's first and last name."""
        result = await beta.workflows.GetNameTask(last_name=True, require_confirmation=True)
        logger.info(f"captured name: {result.first_name} {result.last_name}")
        return f"Name captured: {result.first_name} {result.last_name}. Read it back to the user."

    @function_tool
    async def collect_credit_card(self, context: RunContext) -> str:
        """Collect the user's credit card details (number, expiration, security code, cardholder)."""
        result = await beta.workflows.GetCreditCardTask(require_confirmation=True)
        logger.info(f"captured card: {result}")
        return (
            f"Card captured: {result.issuer} ending {result.card_number[-4:]}, "
            f"expiring {result.expiration_date}, cardholder {result.cardholder_name}. "
            "Confirm completion to the user without repeating the full number."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=inference.LLM("openai/gpt-4.1-mini"),
        stt=inference.STT("deepgram/nova-3"),
        tts=inference.TTS("cartesia/sonic-3"),
    )

    await session.start(agent=DataCaptureAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
