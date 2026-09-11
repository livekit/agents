import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.plugins import openai, standin

load_dotenv()

server = AgentServer(
    api_key=os.environ["LIVEKIT_API_KEY"],
    api_secret=os.environ["LIVEKIT_API_SECRET"],
    ws_url=os.environ["LIVEKIT_URL"],
)


class MyAgent(Agent):
    def __init__(self, call: standin.CallInfo) -> None:
        # The caller arrives in the job metadata, so the agent is personalized
        # before the first word. The caller's voice is published into the room
        # by the plugin's "standin-bridge" participant.
        who = call.caller_name or "the caller"
        super().__init__(
            instructions=(
                f"You are a helpful colleague on a Microsoft Teams call with {who}. "
                "You are speaking out loud, so keep replies short and conversational. "
                "Do not use emojis, asterisks, markdown, or other characters that "
                "cannot be pronounced."
            )
        )
        self.call = call

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                f"Greet {self.call.caller_name} by name and ask how you can help."
                if self.call.caller_name
                else "Greet the caller and ask how you can help."
            )
        )


@server.rtc_session(agent_name="msteams-agent")
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        # Half-cascade: the realtime model handles comprehension only (text
        # modality) and its own server-side turn detection;
        llm=openai.realtime.RealtimeModel.with_azure(
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-realtime"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
        ),
    )

    # The plugin answered StandIn's dial, created this room, and dispatched this
    # job. TeamsCall binds what is Teams-specific: the caller identity and the
    # msteams.context / msteams.goodbye data topics.
    call = await standin.TeamsCall().start(session, ctx=ctx)

    ctx.log_context_fields = {
        "room": ctx.room.name,
        "call_id": call.call_id,
        "direction": call.direction,
    }

    await session.start(agent=MyAgent(call), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
