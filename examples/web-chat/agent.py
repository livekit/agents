from collections.abc import AsyncIterator

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, cli, inference
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

load_dotenv()

assistant = Agent(
    instructions=(
        "You are a friendly, helpful assistant in a web chat. "
        "Answer clearly and briefly, in plain text."
    ),
    llm=inference.LLM(model="google/gemma-4-31b-it"),
)

server = AgentServer(storage=AgentDBStorage(), agent_name="web-chat")


# One turn per message. The thread keeps the history; the answer streams back as it is written.
@server.a2a_session(endpoint="chat", description="A general chat assistant.")
async def chat(ctx: A2ASessionContext) -> AsyncIterator[str]:
    async with assistant.run_stream(ctx.message, message_history=ctx.message_history) as stream:
        async for delta in stream.stream_text(delta=True):
            yield delta


if __name__ == "__main__":
    cli.run_app(server)
