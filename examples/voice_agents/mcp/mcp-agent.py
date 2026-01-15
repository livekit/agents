import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, mcp
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("mcp-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You can retrieve data via the MCP server. The interface is voice-based: "
                "accept spoken user queries and respond with synthesized speech."
            ),
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
        mcp_servers=[
            mcp.MCPServerHTTP(url="http://localhost:8000/sse"),
        ],
    )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
