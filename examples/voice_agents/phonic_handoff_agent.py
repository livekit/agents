import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins.phonic.realtime import RealtimeModel

logger = logging.getLogger("phonic-handoff")

load_dotenv()


class NameAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Alex, a friendly interviewer. You just started the call. "
                "Greet the user, then ask for their full name. "
                "Once you have it, thank the user and call record_name. (will transfer to another agent)"
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def record_name(self, context: RunContext, name: str) -> Agent:
        """Record the user's name and move on.

        Args:
            name: The user's full name
        """
        logger.info(f"Got name: {name}")
        context.session.userdata["name"] = name
        return EmailAgent()


class EmailAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Alex, continuing an interview. "
                "Ask the user for their email address. Be conversational. "
                "Once you have it, thank the user and call record_email. (will transfer to another agent)"
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Transition naturally and ask for their email address.",
        )

    @function_tool
    async def record_email(self, context: RunContext, email: str) -> Agent:
        """Record the user's email and move on.

        Args:
            email: The user's email address
        """
        logger.info(f"Got email: {email}")
        context.session.userdata["email"] = email
        return AddressAgent()


class AddressAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Alex, wrapping up an interview. "
                "Ask the user for their mailing address (city and state is fine). "
                "Once you have it, thank the user and call record_address. "
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Transition naturally and ask for their mailing address.",
        )

    @function_tool
    async def record_address(self, context: RunContext, address: str) -> None:
        """Record the user's address and finish.

        Args:
            address: The user's mailing address
        """
        logger.info(f"Got address: {address}")
        context.session.userdata["address"] = address
        ud = context.session.userdata
        logger.info(
            f"All collected: name={ud['name']}, email={ud['email']}, address={ud['address']}"
        )
        await context.session.generate_reply(
            instructions="Thank the user for their time. Let them know they're all set.",
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=RealtimeModel(
            voice="sabrina",
            audio_speed=1.2,
        ),
        userdata={"name": None, "email": None, "address": None},
    )

    async def log_usage():
        logger.info(f"Usage: {session.usage}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=NameAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(delete_room_on_close=True),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
