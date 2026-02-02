import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    FunctionToolsExecutedEvent,
    JobContext,
    cli,
    inference,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero

logger = logging.getLogger("silent-function-call")
logger.setLevel(logging.INFO)

load_dotenv()

# This example shows how to execute function tools without generating a reply.
# A tool without a return value won't generate a reply automatically.
# If multiple tools are called in parallel, it creates a reply if any of them has a output,
# you can cancel the reply by calling `ev.cancel_tool_reply()` in the `function_tools_executed` event handler.


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice agent. Call the turn_on_light function when user asks to turn on the light."  # noqa: E501
            ),
        )
        self.light_on = False

    @function_tool()
    async def turn_on_light(self):
        """Called when user asks to turn on the light."""
        self.light_on = True
        logger.info("Light is now on")

        # a tool without a return value won't generate a reply automatically

    @function_tool()
    async def turn_off_light(self):
        """Called when user asks to turn off the light."""
        self.light_on = False
        logger.info("Light is now off")

        return "Light is now off"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/flux-general"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=silero.VAD.load(),
    )

    @session.on("function_tools_executed")
    def on_function_tools_executed(ev: FunctionToolsExecutedEvent):
        tools = (fnc.name for fnc in ev.function_calls)
        if "turn_off_light" in tools:
            # you can also prevent the tool reply after all tools executed
            ev.cancel_tool_reply()

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
