import logging

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
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self, *, greet_on_enter: bool = True) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )
        self._greet_on_enter = greet_on_enter

    async def on_enter(self):
        if self._greet_on_enter:
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


server = AgentServer()


@server.text_handler()
async def sms_handler(ctx: TextMessageContext):
    logger.info(f"SMS received: {ctx.text}")

    session = AgentSession(llm="openai/gpt-4.1-mini")
    if ctx.session_data:
        await session.rehydrate(ctx.session_data)
    else:
        await session.start(agent=MyAgent(greet_on_enter=False))

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

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
