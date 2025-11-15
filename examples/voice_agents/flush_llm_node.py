import asyncio
import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    ModelSettings,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    llm,
    metrics,
)
from livekit.plugins import silero

logger = logging.getLogger("flush-llm-node")
logger.setLevel(logging.INFO)

load_dotenv()

## This example shows how to flush a fast response in `llm_node` to tts immediately.


class FastResponseAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
        )

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

        await asyncio.sleep(3)
        context.session.say("Okay I found what you were looking for...")

        return "sunny with a temperature of 70 degrees."

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | llm.FlushSentinel]:
        async for chunk in Agent.default.llm_node(
            agent=self,
            chat_ctx=chat_ctx,
            tools=tools,
            model_settings=model_settings,
        ):
            if isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.tool_calls:
                yield "One moment while I look that up."
                yield llm.FlushSentinel()

            yield chunk


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        llm="openai/gpt-4.1-mini",
        stt="assemblyai/universal-streaming",
        tts="elevenlabs",
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(agent=FastResponseAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
