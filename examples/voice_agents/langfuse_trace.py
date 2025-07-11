import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.debug import use_trace_langfuse
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()

# This example shows how to use the langfuse tracer to trace the agent session.
# To enable tracing, call `use_trace_langfuse` once before AgentSession starts or in global scope,
# you can set the `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` env vars
# or pass them as arguments to `use_trace_langfuse`


@function_tool
async def lookup_weather(context: RunContext, location: str) -> str:
    """Called when the user asks for weather related information.

    Args:
        location: The location they are asking for
    """

    logger.info(f"Looking up weather for {location}")

    return "sunny with a temperature of 70 degrees."


class Kelly(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly.",
            llm=openai.LLM(model="gpt-4o-mini"),
            stt=deepgram.STT(model="nova-3", language="multi"),
            tts=openai.TTS(voice="ash"),
            turn_detection=MultilingualModel(),
            tools=[lookup_weather],
        )

    async def on_enter(self):
        logger.info("Kelly is entering the session")
        self.session.generate_reply()

    @function_tool
    async def transfer_to_alloy(self) -> Agent:
        """Transfer the call to Alloy."""
        logger.info("Transferring the call to Alloy")
        return Alloy()


class Alloy(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
            tools=[lookup_weather],
        )

    async def on_enter(self):
        logger.info("Alloy is entering the session")
        self.session.generate_reply()

    @function_tool
    async def transfer_to_kelly(self) -> Agent:
        """Transfer the call to Kelly."""

        logger.info("Transferring the call to Kelly")
        return Kelly()


async def entrypoint(ctx: JobContext):
    # set up the langfuse tracer
    use_trace_langfuse()

    session = AgentSession(vad=ctx.proc.userdata["vad"])

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(agent=Kelly(), room=ctx.room)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
