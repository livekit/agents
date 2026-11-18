from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, RunResult, cli, inference
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

load_dotenv()

assistant = Agent(
    instructions="You are a friendly, helpful assistant in a web chat. Answer clearly and "
    "briefly, in plain text.",
    llm=inference.LLM(model="google/gemma-4-31b-it"),
)

server = AgentServer(storage=AgentDBStorage(), agent_name="web-chat")


# the text door: one turn per message, with the thread's history; the answer's text streams
# to the caller as the model writes it, and the returned result finalizes it
@server.a2a_session(endpoint="chat", description="A general chat assistant.")
async def chat(ctx: A2ASessionContext) -> RunResult[str]:
    async with assistant.run_stream(ctx.message, message_history=ctx.message_history) as stream:
        await ctx.stream(stream.stream_text(delta=True))
        result = await stream.result()
    return result


if __name__ == "__main__":
    cli.run_app(server)
