import logging
from typing import Any, override

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TextMessageContext,
    cli,
)
from livekit.agents.beta.workflows import GetEmailTask
from livekit.agents.llm import ToolFlag, function_tool
from livekit.durable import EffectCall
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self, *, text_mode: bool) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )
        self._text_mode = text_mode

    @override
    def get_init_kwargs(self) -> dict[str, Any]:
        return {
            "text_mode": self._text_mode,
        }

    async def on_enter(self):
        if not self._text_mode:
            logger.debug("greeting the user")
            self.session.generate_reply(allow_interruptions=False)

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        # this will create multiple responses to the user
        context.session.say("Let me check the weather for you")

        return "sunny with a temperature of 70 degrees."

    @function_tool(flags=ToolFlag.DURABLE)
    async def register_for_weather(self, context: RunContext):
        """Called when the user wants to register for the weather event."""
        logger.info("register_for_weather called")

        get_email_task = GetEmailTask(
            extra_instructions=(
                "You are communicate to the user via text messages, "
                "so there is no need to verify the email address with the user multiple times."
            )
            if self._text_mode
            else ""
        )
        get_email_task.configure(llm="openai/gpt-4.1")

        email_result = await EffectCall(get_email_task)
        email_address = email_result.email_address

        logger.info(f"User's email address: {email_address}")

        return "You are now registered for the weather event."


server = AgentServer(port=8081)


@server.text_handler()
async def text_handler(ctx: TextMessageContext):
    logger.info(f"text message received: {ctx.text}")

    session = AgentSession(
        llm="openai/gpt-4.1-mini",
        # state_passphrase="my-secret-passphrase",
    )
    if ctx.session_data:
        await session.rehydrate(ctx.session_data)
    else:
        await session.start(agent=MyAgent(text_mode=True))

    async for ev in session.run(user_input=ctx.text):
        await ctx.send_response(ev)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        preemptive_generation=True,
    )

    await session.start(agent=MyAgent(text_mode=False), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
