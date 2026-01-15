import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    cli,
    inference,
    mcp,
    metrics,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a voice assistant created by LiveKit with Zapier integration via MCP. Your interface with users will be voice. "  # noqa: E501
            "You can help users with Zapier automations and workflows through the MCP server connection. "  # noqa: E501
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "  # noqa: E501
            "You were created as a demo to showcase the capabilities of LiveKit's agents framework with MCP integration.",  # noqa: E501
            stt=inference.STT("deepgram/nova-3"),
            llm=inference.LLM("google/gemini-2.5-flash"),
            tts=inference.TTS("rime/arcana"),
            # use LiveKit's transformer-based turn detector
            turn_detection=MultilingualModel(),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Hey, how can I help you today?", allow_interruptions=True
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    # Log metrics and collect usage data
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    # Get MCP server URL from environment variable
    zapier_mcp_server = os.getenv("ZAPIER_MCP_SERVER")
    mcp_servers = []
    if zapier_mcp_server:
        logger.info(f"Connecting to Zapier MCP server at {zapier_mcp_server}")
        mcp_servers.append(mcp.MCPServerHTTP(url=zapier_mcp_server))
    else:
        logger.warning("ZAPIER_MCP_SERVER environment variable not set. MCP integration disabled.")

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn  # noqa: E501
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn  # noqa: E501
        # this should be increased more if latency is an issue
        max_endpointing_delay=5.0,
        mcp_servers=mcp_servers,
    )

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )


if __name__ == "__main__":
    cli.run_app(server)
