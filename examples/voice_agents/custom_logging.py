"""Example: using configure_logging() to customise log output.

Call configure_logging() *before* cli.run_app() to control the log format.
The CLI will detect that logging has already been set up and will not
override your configuration.

Usage:
    # JSON logs (useful for log aggregators in development)
    python custom_logging.py dev

    # Switch to the default colored formatter by commenting out
    # the configure_logging() call and re-running.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    inference,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.log import configure_logging
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("custom-logging-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. "
            "Keep responses concise and conversational.",
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather information.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location
            longitude: The longitude of the location
        """
        logger.info("weather lookup", extra={"location": location})
        return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("session ended", extra={"usage": str(usage_collector.get_summary())})

    ctx.add_shutdown_callback(log_usage)

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    # ---- configure logging before starting the app ----
    # Option 1: JSON logs (great for structured logging / log aggregators)
    configure_logging(json=True, level=logging.DEBUG)

    # Option 2: custom formatter
    # configure_logging(
    #     formatter=logging.Formatter("[%(levelname)s] %(name)s | %(message)s"),
    #     level=logging.DEBUG,
    # )

    # Option 3: write logs to a file
    # file_handler = logging.FileHandler("agent.log")
    # configure_logging(handler=file_handler, json=True, level=logging.DEBUG)

    cli.run_app(server)
