import logging
from datetime import datetime, timezone
from typing import Annotated

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import silero
from livekit.plugins.ultravox.realtime import RealtimeModel

logger = logging.getLogger("ultravox-agent")
logger.setLevel(logging.INFO)

load_dotenv()


@function_tool
async def get_time(
    time_zone: Annotated[str, "The time zone to get the time in"],
) -> str:
    """
    Get the current time in the given time zone.

    Args:
        time_zone: The time zone to get the time in.

    Returns:
        The current time in the given time zone.
    """
    return datetime.now(timezone.utc).isoformat()


@function_tool(
    raw_schema={
        "name": "get_time_raw",
        "description": "Get the current time in the given time zone.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_zone": {"type": "string", "description": "The time zone to get the time in"},
            },
            "required": ["time_zone"],
        },
    }
)
async def get_time_raw(
    raw_arguments: dict[str, object],
) -> str:
    """
    Get the current time in the given time zone.

    Args:
        time_zone: The time zone to get the time in.

    Returns:
        The current time in the given time zone.
    """
    return datetime.now(timezone.utc).isoformat()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session: AgentSession[None] = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        llm=RealtimeModel(
            model_id="fixie-ai/ultravox",
        ),
    )

    @session.on("function_tools_executed")
    def on_function_tools_executed(ev: FunctionToolsExecutedEvent) -> None:
        logger.info(f"function_tools_executed: {ev}")

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        logger.info(f"conversation_item_added: {ev}")

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        logger.info(f"metrics_collected: {ev}")

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        logger.info(f"user_input_transcribed: {ev}")

    await session.start(
        agent=Agent(instructions="You are a helpful assistant.", tools=[get_time, get_time_raw]),
        room=ctx.room,
    )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
