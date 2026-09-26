import json
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, mcp

logger = logging.getLogger("mcp-elicitation-agent")

load_dotenv()

# Run `python elicitation_server.py` first. Its reserve_seat tool asks the user for a seat
# with an MCP elicitation request while the tool call is running.
#
# The handler below forwards each request to the frontend over RPC. The frontend registers
# an "mcp.elicit" RPC method, shows `message` (and a form built from `requested_schema`,
# or a link to `url`), and answers with {"action": "accept" | "decline" | "cancel",
# "content": {...}}.


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    async def on_elicit(req: mcp.MCPElicitationContext) -> mcp.MCPElicitationResult:
        participant = session.room_io.linked_participant
        if participant is None:
            return mcp.MCPElicitationResult(action="cancel")

        try:
            response = await ctx.room.local_participant.perform_rpc(
                destination_identity=participant.identity,
                method="mcp.elicit",
                payload=json.dumps(
                    {
                        "mode": req.mode,
                        "message": req.message,
                        "requested_schema": req.requested_schema,
                        "url": req.url,
                    }
                ),
                response_timeout=60,
            )
        except rtc.RpcError as e:
            # e.g. the frontend doesn't implement "mcp.elicit"
            logger.warning("elicitation RPC failed: %s", e.message)
            return mcp.MCPElicitationResult(action="cancel")

        return mcp.MCPElicitationResult.model_validate_json(response)

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        tools=[
            mcp.MCPToolset(
                id="elicitation_demo",
                mcp_server=mcp.MCPServerHTTP(
                    url="http://localhost:8000/mcp",
                    elicitation_handler=on_elicit,
                    # unanswered requests are answered "cancel" after 60s; time spent
                    # waiting on the user doesn't count toward the tool call's timeout
                    elicitation_timeout=60,
                ),
            )
        ],
    )

    await session.start(
        agent=Agent(
            instructions=(
                "You are a friendly airline assistant that communicates via voice. "
                "Use the reserve_seat tool when the user wants to pick a seat; the user "
                "chooses the seat on their screen while the tool runs."
            )
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
